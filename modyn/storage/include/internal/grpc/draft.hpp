//""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

// Adapted function to send sample data without labels
template <typename WriterT = ServerWriter<modyn::storage::GetResponseNoLabels>>
static void send_sample_data_for_keys_and_file_NL(  // NOLINT(readability-function-cognitive-complexity) //TODO Adaptar esto
    WriterT* writer, std::mutex& writer_mutex, const std::vector<int64_t>& sample_keys,
    const DatasetData& dataset_data, soci::session& session, int64_t /*sample_batch_size*/) {
  
  // Note that we currently ignore the sample batch size here, under the assumption that users do not request more
  // keys than this
  try {
    const uint64_t num_keys = sample_keys.size();

    if (num_keys == 0) {
      SPDLOG_ERROR("num_keys is 0, this should not have happened. Exiting send_sample_data_for_keys_and_file_NL");
      return;
    }

    // Removed labels-related vectors
    // std::vector<int64_t> sample_labels(num_keys);
    std::vector<uint64_t> sample_indices(num_keys);
    std::vector<int64_t> sample_fileids(num_keys);

    const std::string sample_query = fmt::format(
        "SELECT sample_index, file_id FROM samples WHERE dataset_id = :dataset_id AND sample_id IN ({}) ORDER BY file_id",
        fmt::join(sample_keys, ","));
    session << sample_query, 
              soci::into(sample_indices), 
              soci::into(sample_fileids),
              soci::use(dataset_data.dataset_id);

    if (sample_fileids.size() != num_keys) {
      SPDLOG_ERROR(fmt::format("Sample query is {}", sample_query));
      SPDLOG_ERROR(
          fmt::format("num_keys = {}\n sample_indices = [{}]\n sample_fileids = [{}]",
                      num_keys, fmt::join(sample_indices, ", "),
                      fmt::join(sample_fileids, ", ")));
      throw modyn::utils::ModynException(
          fmt::format("Got back {} samples from DB, while asking for {} keys. You might have asked for duplicate "
                      "keys, which is not supported.",
                      sample_fileids.size(), num_keys));
    }

    int64_t current_file_id = sample_fileids.at(0);
    uint64_t current_file_start_idx = 0;
    std::string current_file_path;
    session << "SELECT path FROM files WHERE file_id = :file_id AND dataset_id = :dataset_id",
        soci::into(current_file_path), soci::use(current_file_id), soci::use(dataset_data.dataset_id);

    if (current_file_path.empty() || current_file_path.find_first_not_of(' ') == std::string::npos) {
      SPDLOG_ERROR(fmt::format("Sample query is {}", sample_query));
      SPDLOG_ERROR(
          fmt::format("num_keys = {}, current_file_id = {}\n sample_indices = [{}]\n sample_fileids = [{}]",
                      num_keys, current_file_id,
                      fmt::join(sample_indices, ", "),
                      fmt::join(sample_fileids, ", ")));
      throw modyn::utils::ModynException(fmt::format("Could not obtain full path of file id {} in dataset {}",
                                                     current_file_id, dataset_data.dataset_id));
    }
    const YAML::Node file_wrapper_config_node = YAML::Load(dataset_data.file_wrapper_config);
    auto filesystem_wrapper =
        get_filesystem_wrapper(static_cast<FilesystemWrapperType>(dataset_data.filesystem_wrapper_type));

    auto file_wrapper =
        get_file_wrapper(current_file_path, static_cast<FileWrapperType>(dataset_data.file_wrapper_type),
                         file_wrapper_config_node, filesystem_wrapper);

    for (uint64_t sample_idx = 0; sample_idx < num_keys; ++sample_idx) {
      const int64_t& sample_fileid = sample_fileids.at(sample_idx);

      if (sample_fileid != current_file_id) {
        // 1. Prepare response without labels
        const std::vector<uint64_t> file_indexes(
            sample_indices.begin() + static_cast<int64_t>(current_file_start_idx),
            sample_indices.begin() + static_cast<int64_t>(sample_idx));
        std::vector<std::vector<unsigned char>> data = file_wrapper->get_samples_from_indices(file_indexes);

        // Protobuf expects the data as std::string...
        std::vector<std::string> stringified_data;
        stringified_data.reserve(data.size());
        for (const std::vector<unsigned char>& char_vec : data) {
          stringified_data.emplace_back(char_vec.begin(), char_vec.end());
        }
        data.clear();
        data.shrink_to_fit();

        // Changed GetResponse to GetResponseNoLabels
        modyn::storage::GetResponseNoLabels response; // <-- Changed from GetResponse
        response.mutable_samples()->Assign(stringified_data.begin(), stringified_data.end());
        response.mutable_keys()->Assign(sample_keys.begin() + static_cast<int64_t>(current_file_start_idx),
                                        sample_keys.begin() + static_cast<int64_t>(sample_idx));
        // Removed labels assignment
        // response.mutable_labels()->Assign(sample_labels.begin() + static_cast<int64_t>(current_file_start_idx),
        //                                   sample_labels.begin() + static_cast<int64_t>(sample_idx));

        // 2. Send response
        {
          const std::lock_guard<std::mutex> lock(writer_mutex);
          writer->Write(response); // <-- Correct type: GetResponseNoLabels
        }

        // 3. Update state
        current_file_id = sample_fileid;
        current_file_path = "";
        session << "SELECT path FROM files WHERE file_id = :file_id AND dataset_id = :dataset_id",
            soci::into(current_file_path), soci::use(current_file_id), soci::use(dataset_data.dataset_id);
        if (current_file_path.empty() || current_file_path.find_first_not_of(' ') == std::string::npos) {
          SPDLOG_ERROR(fmt::format("Sample query is {}", sample_query));
          const int64_t& previous_fid = sample_fileids.at(sample_idx - 1);
          SPDLOG_ERROR(
              fmt::format("num_keys = {}, sample_idx = {}, previous_fid = {}\n sample_indices = [{}]\n sample_fileids = [{}]",
                          num_keys, sample_idx, previous_fid,
                          fmt::join(sample_indices, ", "),
                          fmt::join(sample_fileids, ", ")));
          throw modyn::utils::ModynException(fmt::format("Could not obtain full path of file id {} in dataset {}",
                                                         current_file_id, dataset_data.dataset_id));
        }
        file_wrapper->set_file_path(current_file_path);
        current_file_start_idx = sample_idx;
      }
    }

    // Send leftovers without labels
    const std::vector<uint64_t> file_indexes(sample_indices.begin() + static_cast<int64_t>(current_file_start_idx),
                                             sample_indices.end());
    const std::vector<std::vector<unsigned char>> data = file_wrapper->get_samples_from_indices(file_indexes);
    // Protobuf expects the data as std::string...
    std::vector<std::string> stringified_data;
    stringified_data.reserve(data.size());
    for (const std::vector<unsigned char>& char_vec : data) {
      stringified_data.emplace_back(char_vec.begin(), char_vec.end());
    }

    // Changed GetResponse to GetResponseNoLabels
    modyn::storage::GetResponseNoLabels response; // <-- Changed from GetResponse
    response.mutable_samples()->Assign(stringified_data.begin(), stringified_data.end());
    response.mutable_keys()->Assign(sample_keys.begin() + static_cast<int64_t>(current_file_start_idx),
                                    sample_keys.end());
    // Removed labels assignment
    // response.mutable_labels()->Assign(sample_labels.begin() + static_cast<int64_t>(current_file_start_idx),
    //                                   sample_labels.end());

    {
      const std::lock_guard<std::mutex> lock(writer_mutex);
      writer->Write(response); // <-- Correct type: GetResponseNoLabels
    }
  } catch (const std::exception& e) {
    SPDLOG_ERROR("Error in send_sample_data_for_keys_and_file_NL: {}", e.what());
    SPDLOG_ERROR("Propagating error up the call chain to handle gRPC calls.");
    throw;
  }
}
