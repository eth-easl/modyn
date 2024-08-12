R"(CREATE TABLE IF NOT EXISTS samples (
    sample_id BIGSERIAL NOT NULL,
    dataset_id INTEGER NOT NULL,
    file_id INTEGER,
    sample_index BIGINT,
    label BIGINT,
    PRIMARY KEY (dataset_id, sample_id)

) PARTITION BY LIST (dataset_id))"
