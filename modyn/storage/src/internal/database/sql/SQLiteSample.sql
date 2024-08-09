R"(CREATE TABLE IF NOT EXISTS samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    file_id INTEGER,
    sample_index BIGINT,
    label BIGINT
))"
