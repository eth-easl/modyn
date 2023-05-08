CREATE TABLE IF NOT EXISTS samples (
    sample_id BIGINT NOT NULL AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    file_id INTEGER,
    sample_index BIGINT,
    label BIGINT,
    PRIMARY KEY (sample_id, dataset_id),
    PARTITION BY LIST (dataset_id)
);