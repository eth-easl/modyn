import random
import uuid
from sqlalchemy import select, Table, Column, Integer, MetaData
from sqlalchemy.engine import Engine

from modyn.config.schema.pipeline import PresamplingConfig
from modyn.selector.internal.presampling_strategies.abstract_presampling_strategy import AbstractPresamplingStrategy
from modyn.selector.internal.database.models import SelectorStateMetadata
from modyn.selector.internal.storage_backend.abstract_storage_backend import AbstractStorageBackend

# Example SwissAI-based filtering call
def evaluate_batch_quality(texts: list[str]) -> list[bool]:
    """
    Replace this mock with your real SwissAI evaluation logic.
    """
    # For demo, randomly keep or discard:
    return [random.choice([True, False]) for _ in texts]

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

    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: int | None,
        limit: int | None,
        trigger_dataset_size: int | None,
    ):
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
            # Example: fetch the actual text (you must implement get_sample_text in your storage backend)
            text = self._storage_backend.get_sample_text(k)
            batch_texts.append(text)
            batch_ids.append(k)

            if len(batch_texts) == self.batch_size:
                results = evaluate_batch_quality(batch_texts)
                for i, keep in enumerate(results):
                    if keep:
                        filtered_keys.append(batch_ids[i])
                batch_texts.clear()
                batch_ids.clear()

        # Handle any leftover batch
        if batch_texts:
            results = evaluate_batch_quality(batch_texts)
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
                conn.execute(
                    temp_table.insert(),
                    [{"sample_key": key} for key in filtered_keys]
                )

        # Step 5: Return SELECT from the temp table
        return select(temp_table.c.sample_key)
