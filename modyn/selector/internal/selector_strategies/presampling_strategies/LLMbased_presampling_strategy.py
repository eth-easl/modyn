import uuid
import time
import logging
import openai
from sqlalchemy import MetaData, Table, Column, Integer, select
from sqlalchemy.orm.session import Session
from modyn.config.schema.pipeline import PresamplingConfig
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies.abstract_presampling_strategy import AbstractPresamplingStrategy
from modyn.selector.internal.storage_backend.abstract_storage_backend import AbstractStorageBackend
from modyn.common.benchmark.stopwatch import Stopwatch

class LLMEvaluationPresamplingStrategy(AbstractPresamplingStrategy):
    def __init__(
        self,
        presampling_config: PresamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        storage_backend: AbstractStorageBackend,
        batch_size: int = 10,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        ratio: int = 100,
        custom_prompt: str | None = None,
        api_key: str = "sk-rc-75HStaTc3UOSoVgyXSEU7w",
        base_url: str = "https://fmapi.swissai.cscs.ch",
        datatset_id: str = "abstracts_train_gen"
    ):
        super().__init__(presampling_config, modyn_config, pipeline_id, storage_backend)
        self.batch_size = batch_size
        self.model_name = model_name
        self.ratio = ratio
        self.custom_prompt = custom_prompt
        self.dataset_id=datatset_id
        # Create an OpenAI client with customizable API key and base URL.
        self.client = openai.Client(api_key=api_key, base_url=base_url)

    def evaluate_batch_quality(self, keys: list[int], model_name: str , dataset_id:str ) -> list[bool]:
        """
        Retrieve sample texts from storage for the given keys and evaluate their quality using the LLM.
        Uses a custom prompt if provided, otherwise builds a default prompt.
        If ratio is not 100, instructs the LLM to only keep the top ratio percent.
        """
        if model_name is None:
            model_name = self.model_name

        # Retrieve samples directly via the storage backend.
        sample_map: dict[int, str] = {}
        iterator = self._storage_backend._get_data_from_storage(keys, dataset_id)
        for ret_keys, samples, labels, targets, response_time in iterator:
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
                    model=model_name,
                    messages=[{"role": "user", "content": eval_prompt}],
                    max_tokens=512,
                    stream=False
                )
                break
            except Exception as e:
                # Debug: print(f"[DEBUG] Error evaluating batch quality: {e}")
                time.sleep(5)
        
        content = response.choices[0].message.content.strip().lower()
        results = content.split("\n")
        # Debug: print(f"[DEBUG] Evaluation prompt response: {results}")
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
            SelectorStateMetadata.seen_in_trigger_id == next_trigger_id
        )
        # Debug: print(f"[DEBUG] Base query: {base_query}")
        
        def fetch_raw_keys(session: Session) -> list[int]:
            keys = session.execute(base_query).scalars().all()
            # Debug: print(f"[DEBUG] Raw keys fetched from DB: {len(keys)}")
            return keys

        raw_keys = self._storage_backend._execute_on_session(fetch_raw_keys)
        # Debug: print(f"[DEBUG] Raw keys obtained: {len(raw_keys)}")
        filtered_keys = []
        
        for i in range(0, len(raw_keys), self.batch_size):
            batch_keys = raw_keys[i: i + self.batch_size]
            # Directly evaluate the quality by calling get_data_from_storage within evaluate_batch_quality.
            results = self.evaluate_batch_quality(batch_keys,model_name=self.model_name,dataset_id=self.dataset_id)
            for idx, keep in enumerate(results):
                if keep:
                    filtered_keys.append(batch_keys[idx])
                    # Debug: print(f"[DEBUG] Keeping key {batch_keys[idx]}")
                else:
                    # Debug: print(f"[DEBUG] Discarding key {batch_keys[idx]}")
                    pass
        # Debug: print(f"[DEBUG] Final filtered keys: {filtered_keys}")

        temp_table_name = f"temp_llm_filter_{uuid.uuid4().hex}"
        metadata = MetaData()
        temp_table = Table(
            temp_table_name,
            metadata,
            Column("sample_key", Integer, primary_key=True),
        )
        # Debug: print(f"[DEBUG] Creating temporary table: {temp_table_name}")

        def create_temp_table_and_insert(session: Session):
            metadata.create_all(session.get_bind())
            if filtered_keys:
                # Debug: print(f"[DEBUG] Inserting filtered keys into temporary table: {filtered_keys}")
                session.execute(temp_table.insert(), [{"sample_key": key} for key in filtered_keys])
                session.commit()

        self._storage_backend._execute_on_session(create_temp_table_and_insert)
        # Debug: print(f"[DEBUG] Returning presampling query based on temporary table: {temp_table_name}")
        return select(temp_table.c.sample_key)
    
   