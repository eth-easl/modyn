#pragma once

#include <gmock/gmock.h>
#include <grpcpp/support/sync_stream.h>
#include <yaml-cpp/yaml.h>

#include <fstream>

namespace modyn::storage {

class StorageTestUtils {
 public:
  static YAML::Node get_dummy_file_wrapper_config(const std::string& byteorder = "little");
  static std::string get_dummy_file_wrapper_config_inline(const std::string& byteorder = "little");
  static YAML::Node get_dummy_file_wrapper_config_unlabeled(const std::string& byteorder = "little");
  static std::string get_dummy_file_wrapper_config_inline_unlabeled(const std::string& byteorder = "little");
};

template <typename T>
class MockServerWriter : public grpc::ServerWriterInterface<T> {
 public:
  MockServerWriter() = default;

  MockServerWriter(grpc::internal::Call* call, grpc::ServerContext* ctx) : call_(call), ctx_(ctx) {}

  MOCK_METHOD0_T(SendInitialMetadata, void());

  bool Write(const T& response,  // NOLINT(readability-identifier-naming)
             const grpc::WriteOptions /* options */) override {
    responses_.push_back(response);
    return true;
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  bool Write(const T& msg) { return Write(msg, grpc::WriteOptions()); }

  std::vector<T> get_responses() { return responses_; }

  void clear_responses() { responses_.clear(); }

 private:
  grpc::internal::Call* const call_ = nullptr;
  grpc::ServerContext* const ctx_ = nullptr;
  template <class ServiceType, class RequestType, class ResponseType>
  friend class grpc::internal::ServerStreamingHandler;

  std::vector<T> responses_;
};

}  // namespace modyn::storage
