CREATE TABLE datasets (
    dataset_id INTEGER PRIMARY KEY,
    name VARCHAR(80) NOT NULL,
    description VARCHAR(120),
    version VARCHAR(80),
    filesystem_wrapper_type VARCHAR(80),
    file_wrapper_type VARCHAR(80),
    base_path VARCHAR(120) NOT NULL,
    file_wrapper_config VARCHAR(240),
    last_timestamp BIGINT NOT NULL,
    ignore_last_timestamp BOOLEAN NOT NULL DEFAULT FALSE,
    file_watcher_interval BIGINT NOT NULL DEFAULT 5
);