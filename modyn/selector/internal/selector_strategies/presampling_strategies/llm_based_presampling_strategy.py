import time
import uuid

import openai
from sqlalchemy import Column, Integer, MetaData, Table, select
from sqlalchemy.orm.session import Session

from modyn.config.schema.pipeline import PresamplingConfig
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies.abstract_presampling_strategy import (
    AbstractPresamplingStrategy,
)
from modyn.selector.internal.storage_backend import AbstractStorageBackend


class LLMEvaluationPresamplingStrategy(AbstractPresamplingStrategy):
    def __init__(
        self,
        presampling_config: PresamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        storage_backend: AbstractStorageBackend,
    ):
        super().__init__(presampling_config, modyn_config, pipeline_id, storage_backend)
        self.batch_size = presampling_config.batch_size
        self.model_name = presampling_config.model_name
        self.ratio = presampling_config.ratio
        self.custom_prompt = presampling_config.custom_prompt
        self.dataset_id = presampling_config.dataset_id
        self.client = openai.Client(api_key=presampling_config.api_key, base_url=presampling_config.base_url)

    def evaluate_batch_quality(self, keys: list[int], model_name: str, dataset_id: str) -> list[bool]:
        """Retrieve sample texts from storage for the given keys and evaluate
        their quality using the LLM.

        Uses a custom prompt if provided, otherwise builds a default
        prompt. If ratio is not 100, instructs the LLM to only keep the
        top ratio percent.
        """

        # Retrieve samples directly via the storage backend.
        sample_map: dict[int, str] = {}
        iterator = self._storage_backend._get_data_from_storage(selector_keys=keys, dataset_id=dataset_id)  # type: ignore
        for ret_keys, samples, _, _, _ in iterator:
            for k, sample in zip(ret_keys, samples):
                text = sample.decode("utf-8") if isinstance(sample, bytes) else str(sample)
                sample_map[k] = text

        # Build the list of texts in the order of keys.
        batch_texts = [sample_map.get(key, "") for key in keys]

        # Use custom prompt if provided; otherwise, build the default prompt.
        if self.custom_prompt:
            eval_prompt = self.custom_prompt
        else:
            eval_prompt = (
                "Determine if the following texts are useful for training an LLM on Wikipedia fact updates.\n\n"
            )
            for i, text in enumerate(batch_texts):
                eval_prompt += f"{i+1}. {text}\n"
            eval_prompt += (
                "\nIf the text is informative and fact-based, return 'true'. "
                "If the text is meaningless, misleading, or repetitive, try to return false at least once per batch. "
                "If you are unsure, return 'false'. Return only 'true' or 'false' without numbering the responses.\n"
            )
        if self.ratio != 100:
            eval_prompt += f"\nAdditionally, only keep the top {self.ratio}% of the samples.\n"

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=model_name, messages=[{"role": "user", "content": eval_prompt}], max_tokens=512, stream=False
                )
                break
            except Exception:  # pylint:disable=broad-exception-caught
                time.sleep(5)

        content = response.choices[0].message.content.strip().lower()
        results = content.split("\n")
        return [res.strip() == "true" for res in results]

    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: int | None,
        limit: int | None,
        trigger_dataset_size: int | None,
    ) -> select:
        base_query = select(SelectorStateMetadata.sample_key).filter(
            SelectorStateMetadata.pipeline_id == self.pipeline_id,
            SelectorStateMetadata.seen_in_trigger_id == next_trigger_id,
        )

        def fetch_raw_keys(session: Session) -> list[int]:
            keys = session.execute(base_query).scalars().all()
            return keys

        raw_keys = self._storage_backend._execute_on_session(fetch_raw_keys)  # type: ignore
        filtered_keys = []

        for i in range(0, len(raw_keys), self.batch_size):
            batch_keys = raw_keys[i : i + self.batch_size]
            # Directly evaluate the quality by calling get_data_from_storage within evaluate_batch_quality.
            results = self.evaluate_batch_quality(batch_keys, model_name=self.model_name, dataset_id=self.dataset_id)  # type: ignore
            for idx, keep in enumerate(results):
                if keep:
                    filtered_keys.append(batch_keys[idx])

        temp_table_name = f"temp_llm_filter_{uuid.uuid4().hex}"
        metadata = MetaData()
        temp_table = Table(
            temp_table_name,
            metadata,  # type: ignore
            Column("sample_key", Integer, primary_key=True),
        )

        def create_temp_table_and_insert(session: Session) -> None:
            metadata.create_all(session.get_bind())
            if filtered_keys:
                session.execute(temp_table.insert(), [{"sample_key": key} for key in filtered_keys])
                session.commit()

        self._storage_backend._execute_on_session(create_temp_table_and_insert)  # type: ignore

        return select(temp_table.c.sample_key)
