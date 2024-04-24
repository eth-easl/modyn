R"(UPDATE datasets 
SET base_path = :name,
    filesystem_wrapper_type = :filesystem_wrapper_type,
    file_wrapper_type = :file_wrapper_type,
    description = :description,
    version = :version,
    file_wrapper_config = :file_wrapper_config,
    ignore_last_timestamp = :ignore_last_timestamp,
    file_watcher_interval = :file_watcher_interval
WHERE dataset_id = :dataset_id)"
