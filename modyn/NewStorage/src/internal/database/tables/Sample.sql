CREATE TABLE IF NOT EXISTS samples (
    sample_id BIGINT NOT NULL AUTO_INCREMENT,
    dataset_id INTEGER NOT NULL,
    file_id INTEGER,
    index BIGINT,
    label BIGINT,
    PRIMARY KEY (sample_id, dataset_id)
);