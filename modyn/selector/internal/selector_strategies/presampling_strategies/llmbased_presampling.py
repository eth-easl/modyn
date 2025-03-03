import uuid
import time
import requests
from sqlalchemy import Column, Integer, MetaData, Table, select
from sqlalchemy.engine import Engine

from modyn.config.schema.pipeline import PresamplingConfig
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.presampling_strategies import AbstractPresamplingStrategy
from modyn.selector.internal.storage_backend.abstract_storage_backend import AbstractStorageBackend


def evaluate_batch_quality(texts: list[str], api_url: str, model_name: str) -> list[bool]:
    """
    Uses SwissAI API to evaluate a batch of texts.

    Args:
        texts (list[str]): List of text samples to evaluate.
        api_url (str): SwissAI API endpoint.
        model_name (str): SwissAI model to use for evaluation.

    Returns:
        list[bool]: List of True/False values indicating if each text is useful.
    """
    eval_prompt = "Determine if the following texts are useful for training an LLM on Wikipedia fact updates.\n\n"

    for i, text in enumerate(texts):
        eval_prompt += f"{i+1}. {text}\n"
    eval_prompt += "\nIf the text is informative and fact-based, return 'true'. If the text is meaningless, misleading, or repetitive, return 'false'. Return only 'true' or 'false' do not number the responses and give as many as there are texts.\n"

    while True:
        try:
            # pylint: disable=missing-timeout
            response = requests.post(
                api_url,
                json={"model": model_name, "messages": [{"content": eval_prompt, "role": "user"}]},
            )
            response.raise_for_status()
            break  # Exit loop if no error is encountered.
        except Exception: #pylint: disable=broad-exception-caught
            time.sleep(5)  # Wait 5 seconds before retrying on error.

    results = response.json()["choices"][0]["message"]["content"].strip().lower().split("\n")
    return [res.strip() == "true" for res in results]


class LLMEvaluationPresamplingStrategy(AbstractPresamplingStrategy):
    def __init__(
        self,
        presampling_config: PresamplingConfig,
        modyn_config: dict,
        pipeline_id: int,
        storage_backend: AbstractStorageBackend,
        engine: Engine,  # SQLAlchemy Engine to create temp tables
    ):
        super().__init__(presampling_config, modyn_config, pipeline_id, storage_backend)
        self._engine = engine
        self.batch_size = 10  # Adjust as needed
        self.api_url = presampling_config.get("api_url", "")
        self.model_name = presampling_config.get("model_name", "meta-llama/Llama-3.3-70B-Instruct")

    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: int | None,
        limit: int | None,
        trigger_dataset_size: int | None,
    ) -> select:
        """
        1) Query potential samples (sample_key).
        2) Retrieve text from storage backend.
        3) Filter them via SwissAI LLM in batches.
        4) Store kept sample_keys in a temporary table.
        5) Return SELECT from that temp table.
        """
        # Step 1: Basic query for candidate sample_keys
        base_query = select(SelectorStateMetadata.sample_key).filter(
            SelectorStateMetadata.trigger_id <= next_trigger_id
        )
        if limit is not None and limit > 0:
            base_query = base_query.limit(limit)

        # Execute the query to get all candidate sample_keys
        with self._engine.connect() as conn:
            raw_keys = [row[0] for row in conn.execute(base_query)]

        # Step 2 & 3: LLM-based filtering in batches
        filtered_keys = []
        batch_texts, batch_ids = [], []
        for k in raw_keys:
            # Fetch text for each sample
            text = self._storage_backend.get_sample_text(k)
            batch_texts.append(text)
            batch_ids.append(k)

            if len(batch_texts) == self.batch_size:
                results = evaluate_batch_quality(batch_texts, self.api_url, self.model_name)
                for i, keep in enumerate(results):
                    if keep:
                        filtered_keys.append(batch_ids[i])
                batch_texts.clear()
                batch_ids.clear()

        # Handle any leftover batch
        if batch_texts:
            results = evaluate_batch_quality(batch_texts, self.api_url, self.model_name)
            for i, keep in enumerate(results):
                if keep:
                    filtered_keys.append(batch_ids[i])

        # Step 4: Create a temp table with kept keys
        temp_table_name = f"temp_llm_filter_{uuid.uuid4().hex}"
        metadata = MetaData()
        temp_table = Table(
            temp_table_name,
            metadata,
            Column("sample_key", Integer, primary_key=True),
        )
        metadata.create_all(self._engine)

        # Insert filtered keys
        with self._engine.connect() as conn:
            if filtered_keys:
                conn.execute(temp_table.insert(), [{"sample_key": key} for key in filtered_keys])

        # Step 5: Return SELECT from the temp table
        return select(temp_table.c.sample_key)
