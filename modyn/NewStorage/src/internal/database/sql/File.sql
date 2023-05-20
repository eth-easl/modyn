R"(CREATE TABLE IF NOT EXISTS files (
    file_id BIGINT NOT NULL AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    path VARCHAR(120) NOT NULL,
    updated_at BIGINT,
    number_of_samples INTEGER,
    PRIMARY KEY (file_id),
    INDEX (dataset_id),
    INDEX (updated_at)
);)"