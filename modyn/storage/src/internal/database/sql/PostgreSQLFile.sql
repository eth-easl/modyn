R"(CREATE TABLE IF NOT EXISTS files (
    file_id BIGSERIAL NOT NULL,
    dataset_id INTEGER NOT NULL,
    path VARCHAR(120) NOT NULL,
    updated_at BIGINT,
    number_of_samples INTEGER,
    PRIMARY KEY (file_id)
);

CREATE INDEX IF NOT EXISTS files_dataset_id ON files (dataset_id);

CREATE INDEX IF NOT EXISTS files_updated_at ON files (updated_at);)"