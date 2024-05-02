R"(INSERT INTO datasets (
    name,
    base_path,
    filesystem_wrapper_type,
    file_wrapper_type,
    description,
    version,
    file_wrapper_config,
    ignore_last_timestamp,
    file_watcher_interval,
    last_timestamp
)
VALUES (
    :name,
    :base_path,
    :filesystem_wrapper_type,
    :file_wrapper_type,
    :description,
    :version,
    :file_wrapper_config,
    :ignore_last_timestamp,
    :file_watcher_interval,
    0
))"
