import time
import uuid

import requests
from sqlalchemy import Column, Integer, MetaData, Table, select
from sqlalchemy.engine import Engine

from modyn.config.schema.pipeline import PresamplingConfig
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractPresamplingStrategy
from modyn.selector.internal.storage_backend.abstract_storage_backend import AbstractStorageBackend


def evaluate_batch_quality(texts: list[str], api_url: str, model_name: str) -> list[bool]:
    eval_prompt = "Determine if the following texts are useful for training an LLM on Wikipedia fact updates.\n\n"
    for i, text in enumerate(texts):
        eval_prompt += f"{i+1}. {text}\n"
    eval_prompt += (
        "\nIf the text is informative and fact-based, return 'true'. If the text is meaningless, misleading, or repetitive, "
        "return 'false'. Return only 'true' or 'false' do not number the responses and give as many as there are texts.\n"
    )

    while True:
        try:
            response = requests.post(
                api_url,
                json={"model": model_name, "messages": [{"content": eval_prompt, "role": "user"}]},
            )
            response.raise_for_status()
            break
        except Exception:
            time.sleep(5)

    results = response.json()["choices"][0]["message"]["content"].strip().lower().split("\n")
    return [res.strip() == "true" for res in results]


class LLMEvaluationPresamplingStrategy(AbstractPresamplingStrategy):
    def __init__(
        self,
        presampling_config: PresamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        storage_backend: AbstractStorageBackend,
        engine: Engine,
    ):
        super().__init__(presampling_config, modyn_config, pipeline_id, storage_backend)
        self._engine = engine
        self.batch_size = 10
        self.api_url = presampling_config.get("api_url", "")
        self.model_name = presampling_config.get("model_name", "meta-llama/Llama-3.3-70B-Instruct")

    def _get_sample_text(self, key: int) -> str:
        # Iterate over all data returned by the storage backend.
        # It is assumed that get_all_data() yields tuples of (list_of_keys, data_dict)
        # and that data_dict contains the sample under the "sample" key.
        for keys, data in self._storage_backend.get_all_data():
            if key in keys:
                sample = data.get("sample")
                if isinstance(sample, bytes):
                    return sample.decode("utf-8")
                return str(sample)
        raise KeyError(f"Key {key} not found in storage backend data.")

    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: int | None,
        limit: int | None,
        trigger_dataset_size: int | None,
    ) -> select:
        base_query = select(SelectorStateMetadata.sample_key).filter(
            SelectorStateMetadata.trigger_id <= next_trigger_id
        )
        if limit is not None and limit > 0:
            base_query = base_query.limit(limit)

        with self._engine.connect() as conn:
            raw_keys = [row[0] for row in conn.execute(base_query)]

        filtered_keys = []
        batch_texts, batch_ids = [], []
        for k in raw_keys:
            text = self._get_sample_text(k)
            batch_texts.append(text)
            batch_ids.append(k)
            if len(batch_texts) == self.batch_size:
                results = evaluate_batch_quality(batch_texts, self.api_url, self.model_name)
                for i, keep in enumerate(results):
                    if keep:
                        filtered_keys.append(batch_ids[i])
                batch_texts.clear()
                batch_ids.clear()

        if batch_texts:
            results = evaluate_batch_quality(batch_texts, self.api_url, self.model_name)
            for i, keep in enumerate(results):
                if keep:
                    filtered_keys.append(batch_ids[i])

        temp_table_name = f"temp_llm_filter_{uuid.uuid4().hex}"
        metadata = MetaData()
        temp_table = Table(
            temp_table_name,
            metadata,
            Column("sample_key", Integer, primary_key=True),
        )
        metadata.create_all(self._engine)

        with self._engine.connect() as conn:
            if filtered_keys:
                conn.execute(temp_table.insert(), [{"sample_key": key} for key in filtered_keys])

        return select(temp_table.c.sample_key)
