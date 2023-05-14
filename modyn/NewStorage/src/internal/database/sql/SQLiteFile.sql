R"(CREATE TABLE IF NOT EXISTS files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    path VARCHAR(120) NOT NULL,
    created_at BIGINT,
    updated_at BIGINT,
    number_of_samples INTEGER
);)"