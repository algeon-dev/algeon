#include "execution_context_internal.hpp"
#include "ast_handle.hpp"
#include "io.h"
#include "io_test_hooks.hpp"
#include "shim_common.h"

#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <vector>

#ifdef CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_ENABLED
#include <kvikio/defaults.hpp>
#include <kvikio/remote_handle.hpp>
#endif

namespace {
constexpr int32_t CUDF_IO_REMOTE_PARQUET_BACKEND_UNAVAILABLE = 0;
constexpr int32_t CUDF_IO_REMOTE_PARQUET_BACKEND_KVIKIO_REMOTE_DATASOURCE = 1;
constexpr int32_t CUDF_IO_REMOTE_PARQUET_FALLBACK_NONE = 0;
constexpr int32_t CUDF_IO_REMOTE_PARQUET_FALLBACK_BACKEND_UNAVAILABLE = 1;

#ifdef CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_ENABLED
constexpr unsigned int CUDF_IO_REMOTE_PARQUET_DEFAULT_KVIKIO_THREADS = 64;
constexpr size_t CUDF_IO_REMOTE_PARQUET_DEFAULT_KVIKIO_TASK_SIZE      = 64 * 1024 * 1024;
#endif

template <typename Builder>
cudf_status_t configure_parquet_reader_builder(Builder& builder,
                                               const char** columns,
                                               int32_t num_columns,
                                               int64_t skip_rows,
                                               int64_t num_rows,
                                               const int32_t* row_groups,
                                               int32_t num_row_groups)
{
  if (columns != nullptr && num_columns > 0) {
    cudf_status_t status = validate_c_string_array(columns, num_columns, "columns");
    if (status != CUDF_SUCCESS) { return status; }
    std::vector<std::string> col_names;
    col_names.reserve(num_columns);
    for (int32_t i = 0; i < num_columns; ++i) {
      col_names.emplace_back(columns[i]);
    }
    builder.column_names(col_names);
  }

  if (num_row_groups > 0) {
    std::vector<cudf::size_type> rg_indices(row_groups, row_groups + num_row_groups);
    builder.row_groups({std::move(rg_indices)});

  } else {
    if (skip_rows > 0) { builder.skip_rows(skip_rows); }
    if (num_rows >= 0) { builder.num_rows(num_rows); }
  }

  return CUDF_SUCCESS;
}

cudf_status_t validate_parquet_reader_args(const char** columns,
                                           int32_t num_columns,
                                           int64_t skip_rows,
                                           int64_t num_rows,
                                           const int32_t* row_groups,
                                           int32_t num_row_groups)
{
  if (num_columns < 0) {
    set_error("num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (skip_rows < 0) {
    set_error("skip_rows must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_rows < -1) {
    set_error("num_rows must be >= -1");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_groups < 0) {
    set_error("num_row_groups must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_groups > 0 && row_groups == nullptr) {
    set_error("row_groups is null but num_row_groups > 0");
    return CUDF_ERROR_NULL;
  }
  if (num_row_groups > 0 && (skip_rows > 0 || num_rows >= 0)) {
    set_error("row_groups cannot be combined with skip_rows or num_rows");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0) { CUDF_RUST_NULL_CHECK(columns, "columns is null"); }
  return CUDF_SUCCESS;
}

cudf_status_t validate_parquet_multi_reader_args(const char** filepaths,
                                                 int32_t num_files,
                                                 const char** columns,
                                                 int32_t num_columns,
                                                 const int32_t* row_group_offsets,
                                                 int32_t num_row_group_offsets,
                                                 const int32_t* row_groups,
                                                 int32_t num_row_groups)
{
  if (num_files <= 0) {
    set_error("num_files must be > 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_NULL_CHECK(filepaths, "filepaths is null");
  cudf_status_t const filepaths_status = validate_c_string_array(filepaths, num_files, "filepaths");
  if (filepaths_status != CUDF_SUCCESS) { return filepaths_status; }

  if (num_columns < 0) {
    set_error("num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0) { CUDF_RUST_NULL_CHECK(columns, "columns is null"); }
  if (num_columns > 0) {
    cudf_status_t const column_status = validate_c_string_array(columns, num_columns, "columns");
    if (column_status != CUDF_SUCCESS) { return column_status; }
  }

  if (num_row_group_offsets < 0) {
    set_error("num_row_group_offsets must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_groups < 0) {
    set_error("num_row_groups must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_group_offsets == 0) {
    if (row_group_offsets != nullptr) {
      set_error("row_group_offsets must be null when num_row_group_offsets is 0");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    if (num_row_groups != 0 || row_groups != nullptr) {
      set_error("row_groups must be null and num_row_groups must be 0 when row_group_offsets are absent");
      return CUDF_ERROR_INVALID_OPERATION;
    }
    return CUDF_SUCCESS;
  }

  CUDF_RUST_NULL_CHECK(row_group_offsets, "row_group_offsets is null");
  if (num_row_group_offsets != num_files + 1) {
    set_error("num_row_group_offsets must equal num_files + 1");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_row_groups > 0) {
    CUDF_RUST_NULL_CHECK(row_groups, "row_groups is null");
  } else if (row_groups != nullptr) {
    set_error("row_groups must be null when num_row_groups is 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (row_group_offsets[0] != 0) {
    set_error("row_group_offsets[0] must be 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  for (int32_t i = 0; i < num_files; ++i) {
    if (row_group_offsets[i] > row_group_offsets[i + 1]) {
      set_error("row_group_offsets must be monotonically non-decreasing");
      return CUDF_ERROR_INVALID_OPERATION;
    }
  }
  if (row_group_offsets[num_files] != num_row_groups) {
    set_error("last row_group_offsets entry must equal num_row_groups");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  for (int32_t i = 0; i < num_row_groups; ++i) {
    if (row_groups[i] < 0) {
      set_error("row group index must be >= 0");
      return CUDF_ERROR_INVALID_OPERATION;
    }
  }
  return CUDF_SUCCESS;
}

cudf_status_t validate_host_parquet_buffer(const void* host_buffer, size_t host_buffer_len)
{
  if (host_buffer_len == 0) {
    set_error("host_buffer_len must be > 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_NULL_CHECK(host_buffer, "host_buffer is null");
  return CUDF_SUCCESS;
}

char** copy_schema_names(cudf::io::table_metadata const& metadata, int32_t* out_num_names)
{
  auto const& schema_info = metadata.schema_info;
  auto const n            = static_cast<int32_t>(schema_info.size());
  *out_num_names          = n;

  char** names = nullptr;
  if (n > 0) {
    names = new (std::nothrow) char*[n];
    if (names == nullptr) {
      set_error("failed to allocate column name array");
      return nullptr;
    }
    for (int32_t i = 0; i < n; ++i) {
      names[i] = nullptr;
    }
    for (int32_t i = 0; i < n; ++i) {
      auto const& name = schema_info[i].name;
      auto const size  = name.size() + 1;
      names[i]         = new (std::nothrow) char[size];
      if (names[i] == nullptr) {
        for (int32_t j = 0; j < i; ++j) {
          delete[] names[j];
        }
        delete[] names;
        set_error("failed to allocate column name string");
        return nullptr;
      }
      std::memcpy(names[i], name.c_str(), size);
    }
  }

  return names;
}

std::optional<std::string> optional_c_string(const char* value)
{
  if (value == nullptr || value[0] == '\0') { return std::nullopt; }
  return std::string(value);
}

void init_remote_read_evidence(cudf_io_remote_parquet_read_evidence_t* out)
{
  if (out == nullptr) { return; }
  out->backend              = CUDF_IO_REMOTE_PARQUET_BACKEND_UNAVAILABLE;
  out->fallback_reason      = CUDF_IO_REMOTE_PARQUET_FALLBACK_BACKEND_UNAVAILABLE;
  out->supports_device_read = 0;
  out->device_read_used     = 0;
  out->remote_bytes         = -1;
  out->remote_range_count   = -1;
  out->host_read_bytes      = -1;
  out->host_read_count      = -1;
  out->device_read_bytes    = -1;
  out->device_read_count    = -1;
}

void init_parquet_pruning_evidence(cudf_io_parquet_pruning_evidence_t* out)
{
  if (out == nullptr) { return; }
  out->input_row_groups                   = -1;
  out->has_row_groups_after_stats_filter = 0;
  out->row_groups_after_stats_filter     = -1;
  out->has_row_groups_after_bloom_filter = 0;
  out->row_groups_after_bloom_filter     = -1;
}

void write_parquet_pruning_evidence(cudf_io_parquet_pruning_evidence_t* out,
                                    cudf::io::table_metadata const& metadata)
{
  if (out == nullptr) { return; }
  out->input_row_groups = static_cast<int64_t>(metadata.num_input_row_groups);
  if (metadata.num_row_groups_after_stats_filter.has_value()) {
    out->has_row_groups_after_stats_filter = 1;
    out->row_groups_after_stats_filter =
      static_cast<int64_t>(metadata.num_row_groups_after_stats_filter.value());
  }
  if (metadata.num_row_groups_after_bloom_filter.has_value()) {
    out->has_row_groups_after_bloom_filter = 1;
    out->row_groups_after_bloom_filter =
      static_cast<int64_t>(metadata.num_row_groups_after_bloom_filter.value());
  }
}

#ifdef CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_ENABLED
bool kvikio_thread_pool_env_configured()
{
  return std::getenv("KVIKIO_NTHREADS") != nullptr || std::getenv("KVIKIO_NUM_THREADS") != nullptr;
}

bool kvikio_task_size_env_configured() { return std::getenv("KVIKIO_TASK_SIZE") != nullptr; }

size_t remote_parquet_tail_cache_bytes()
{
  static auto const value = [] {
    constexpr size_t default_tail_cache_bytes = 512 * 1024;
    auto const* raw = std::getenv("CUDF_IO_REMOTE_PARQUET_TAIL_CACHE_BYTES");
    if (raw == nullptr || raw[0] == '\0') { return default_tail_cache_bytes; }
    char* end  = nullptr;
    auto bytes = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0') { return default_tail_cache_bytes; }
    return static_cast<size_t>(bytes);
  }();
  return value;
}

void configure_remote_parquet_kvikio_defaults_once()
{
  static std::once_flag once;
  std::call_once(once, [] {
    if (!kvikio_thread_pool_env_configured() &&
        kvikio::defaults::num_threads() < CUDF_IO_REMOTE_PARQUET_DEFAULT_KVIKIO_THREADS) {
      kvikio::defaults::set_num_threads(CUDF_IO_REMOTE_PARQUET_DEFAULT_KVIKIO_THREADS);
    }
    if (!kvikio_task_size_env_configured() &&
        kvikio::defaults::task_size() < CUDF_IO_REMOTE_PARQUET_DEFAULT_KVIKIO_TASK_SIZE) {
      kvikio::defaults::set_task_size(CUDF_IO_REMOTE_PARQUET_DEFAULT_KVIKIO_TASK_SIZE);
    }
  });
}

struct remote_read_counters {
  static constexpr std::size_t size_bucket_count = 7;

  std::atomic<int64_t> host_read_bytes{0};
  std::atomic<int64_t> host_read_count{0};
  std::atomic<int64_t> device_read_bytes{0};
  std::atomic<int64_t> device_read_count{0};
  std::array<std::atomic<int64_t>, size_bucket_count> size_bucket_counts{};

  void record_host(size_t bytes)
  {
    host_read_bytes.fetch_add(static_cast<int64_t>(bytes), std::memory_order_relaxed);
    host_read_count.fetch_add(1, std::memory_order_relaxed);
    record_size_bucket(bytes);
  }

  void record_device(size_t bytes)
  {
    device_read_bytes.fetch_add(static_cast<int64_t>(bytes), std::memory_order_relaxed);
    device_read_count.fetch_add(1, std::memory_order_relaxed);
    record_size_bucket(bytes);
  }

  void record_size_bucket(size_t bytes)
  {
    size_bucket_counts[size_bucket_index(bytes)].fetch_add(1, std::memory_order_relaxed);
  }

  [[nodiscard]] static std::size_t size_bucket_index(size_t bytes)
  {
    if (bytes <= 64 * 1024) { return 0; }
    if (bytes <= 1024 * 1024) { return 1; }
    if (bytes <= 4 * 1024 * 1024) { return 2; }
    if (bytes <= 8 * 1024 * 1024) { return 3; }
    if (bytes <= 16 * 1024 * 1024) { return 4; }
    if (bytes <= 32 * 1024 * 1024) { return 5; }
    return 6;
  }
};

class counted_remote_parquet_source final : public cudf::io::datasource {
 public:
  counted_remote_parquet_source(kvikio::RemoteHandle&& handle,
                                std::shared_ptr<cuda::mr::any_resource<cuda::mr::device_accessible>> mr,
                                std::shared_ptr<void> mr_lifetime)
    : handle_{std::move(handle)}, mr_{std::move(mr)}, mr_lifetime_{std::move(mr_lifetime)}
  {
    CUDF_EXPECTS(mr_ != nullptr, "remote datasource memory resource is null");
  }

  std::unique_ptr<buffer> host_read(size_t offset, size_t size) override
  {
    auto const read_size = clamped_read_size(offset, size);
    if (auto cached = cached_tail_read(offset, read_size)) { return cached; }
    std::vector<uint8_t> out(read_size);
    if (read_size > 0) {
      if (!fill_tail_cache_for_read(offset, read_size, out)) {
        handle_.pread(out.data(), read_size, offset, read_size).get();
        counters_.record_host(read_size);
      }
    }
    return buffer::create(std::move(out));
  }

  std::future<std::unique_ptr<cudf::io::datasource::buffer>> host_read_async(size_t offset,
                                                                             size_t size) override
  {
    return std::async(std::launch::deferred, [this, offset, size] {
      return this->host_read(offset, size);
    });
  }

  size_t host_read(size_t offset, size_t size, uint8_t* dst) override
  {
    return host_read_async(offset, size, dst).get();
  }

  std::future<size_t> host_read_async(size_t offset, size_t size, uint8_t* dst) override
  {
    auto const read_size = clamped_read_size(offset, size);
    counters_.record_host(read_size);
    if (read_size == 0) {
      return std::async(std::launch::deferred, [] { return size_t{0}; });
    }
    return handle_.pread(dst, read_size, offset, read_size);
  }

  [[nodiscard]] bool supports_device_read() const override { return true; }

  [[nodiscard]] bool is_device_read_preferred(size_t) const override { return true; }

  std::future<size_t> device_read_async(size_t offset,
                                        size_t size,
                                        uint8_t* dst,
                                        rmm::cuda_stream_view stream) override
  {
    auto const read_size = clamped_read_size(offset, size);
    counters_.record_device(read_size);
    if (read_size == 0) {
      return std::async(std::launch::deferred, [] { return size_t{0}; });
    }
    // KvikIO's device write is not ordered by libcudf's datasource stream
    // contract, so wait for prior stream work before writing into `dst`.
    stream.synchronize();
    cudf_record_synchronized_stream(stream);
    return handle_.pread(dst, read_size, offset, read_size);
  }

  size_t device_read(size_t offset,
                     size_t size,
                     uint8_t* dst,
                     rmm::cuda_stream_view stream) override
  {
    return device_read_async(offset, size, dst, stream).get();
  }

  std::unique_ptr<buffer> device_read(size_t offset,
                                      size_t size,
                                      rmm::cuda_stream_view stream) override
  {
    auto const read_size = clamped_read_size(offset, size);
    rmm::device_buffer out(read_size, stream, *mr_);
    if (read_size > 0) {
      device_read(offset, read_size, reinterpret_cast<uint8_t*>(out.data()), stream);
    } else {
      counters_.record_device(read_size);
    }
    return buffer::create(std::move(out));
  }

  [[nodiscard]] size_t size() const override { return handle_.nbytes(); }

  [[nodiscard]] remote_read_counters const& counters() const { return counters_; }

 private:
  [[nodiscard]] size_t clamped_read_size(size_t offset, size_t size) const
  {
    if (offset >= handle_.nbytes()) { return 0; }
    return std::min(size, handle_.nbytes() - offset);
  }

  std::unique_ptr<buffer> cached_tail_read(size_t offset, size_t size)
  {
    std::lock_guard const lock(tail_cache_mutex_);
    if (!tail_cache_.has_value()) { return nullptr; }
    auto const& cache = tail_cache_.value();
    if (offset < cache.offset || offset + size > cache.offset + cache.bytes.size()) {
      return nullptr;
    }
    auto const cache_offset = offset - cache.offset;
    std::vector<uint8_t> out(size);
    std::memcpy(out.data(), cache.bytes.data() + cache_offset, size);
    return buffer::create(std::move(out));
  }

  bool fill_tail_cache_for_read(size_t offset, size_t size, std::vector<uint8_t>& out)
  {
    auto const tail_cache_bytes = remote_parquet_tail_cache_bytes();
    if (tail_cache_bytes == 0 || size == 0 || offset + size != handle_.nbytes()) {
      return false;
    }
    auto const tail_size   = std::min(tail_cache_bytes, handle_.nbytes());
    auto const tail_offset = handle_.nbytes() - tail_size;
    if (offset < tail_offset) { return false; }
    std::vector<uint8_t> tail(tail_size);
    handle_.pread(tail.data(), tail_size, tail_offset, tail_size).get();
    counters_.record_host(tail_size);

    auto const cache_offset = offset - tail_offset;
    std::memcpy(out.data(), tail.data() + cache_offset, size);
    {
      std::lock_guard const lock(tail_cache_mutex_);
      tail_cache_ = tail_cache_entry{tail_offset, std::move(tail)};
    }
    return true;
  }

  struct tail_cache_entry {
    size_t offset;
    std::vector<uint8_t> bytes;
  };

  kvikio::RemoteHandle handle_;
  std::shared_ptr<cuda::mr::any_resource<cuda::mr::device_accessible>> mr_;
  std::shared_ptr<void> mr_lifetime_;
  remote_read_counters counters_;
  std::mutex tail_cache_mutex_;
  std::optional<tail_cache_entry> tail_cache_;
};

std::string s3_path_style_url(std::string const& bucket,
                              std::string const& object,
                              std::optional<std::string> const& region)
{
  auto const resolved_region = region.value_or("us-east-1");
  return "https://s3." + resolved_region + ".amazonaws.com/" + bucket + "/" + object;
}

void validate_remote_endpoint_url_no_secret(std::optional<std::string> const& endpoint_url)
{
  if (!endpoint_url.has_value()) { return; }
  auto const& value = endpoint_url.value();
  if (value.find('@') != std::string::npos || value.find('?') != std::string::npos ||
      value.find('#') != std::string::npos) {
    CUDF_FAIL("remote Parquet endpoint URL must not contain userinfo, query, or fragment components");
  }
}

std::unique_ptr<kvikio::RemoteEndpoint> make_remote_endpoint(
  const char* uri,
  const cudf_io_remote_parquet_profile_t* profile)
{
  std::string url(uri);
  if (url.size() < 5 || (url[0] != 's' && url[0] != 'S') || url[1] != '3' ||
      url[2] != ':' || url[3] != '/' || url[4] != '/') {
    return nullptr;
  }
  url[0] = 's';

  auto const [bucket, object] = kvikio::S3Endpoint::parse_s3_url(url);
  auto const region          = profile == nullptr ? std::nullopt : optional_c_string(profile->aws_region);
  auto const access_key      = profile == nullptr ? std::nullopt : optional_c_string(profile->aws_access_key_id);
  auto const secret_key      = profile == nullptr ? std::nullopt : optional_c_string(profile->aws_secret_access_key);
  auto const session_token   = profile == nullptr ? std::nullopt : optional_c_string(profile->aws_session_token);
  auto const endpoint_url    = profile == nullptr ? std::nullopt : optional_c_string(profile->aws_endpoint_url);
  auto const unsigned_request = profile != nullptr && profile->unsigned_request != 0;
  auto const path_style_access = profile == nullptr ? -1 : profile->path_style_access;
  validate_remote_endpoint_url_no_secret(endpoint_url);

  if (unsigned_request) {
    auto unsigned_url = endpoint_url.has_value()
                          ? endpoint_url.value() + "/" + bucket + "/" + object
                          : s3_path_style_url(bucket, object, region);
    return std::make_unique<kvikio::S3PublicEndpoint>(std::move(unsigned_url));
  }

  if (endpoint_url.has_value()) {
    if (path_style_access == 0) {
      CUDF_FAIL("virtual-hosted custom S3 endpoints are not supported by the scoped remote Parquet datasource");
    }
    return std::make_unique<kvikio::S3Endpoint>(
      std::pair{bucket, object}, region, access_key, secret_key, endpoint_url, session_token);
  }

  if (path_style_access == 1) {
    return std::make_unique<kvikio::S3Endpoint>(
      s3_path_style_url(bucket, object, region), region, access_key, secret_key, session_token);
  }

  return std::make_unique<kvikio::S3Endpoint>(
    std::pair{bucket, object}, region, access_key, secret_key, std::nullopt, session_token);
}

std::unique_ptr<counted_remote_parquet_source> make_counted_remote_source(
  const char* uri,
  const cudf_io_remote_parquet_profile_t* profile,
  std::shared_ptr<cuda::mr::any_resource<cuda::mr::device_accessible>> mr,
  std::shared_ptr<void> mr_lifetime)
{
  configure_remote_parquet_kvikio_defaults_once();
  auto endpoint = make_remote_endpoint(uri, profile);
  CUDF_EXPECTS(endpoint != nullptr, "only s3:// remote Parquet URIs are supported by the scoped datasource");
  if (profile != nullptr && profile->known_size_bytes >= 0) {
    return std::make_unique<counted_remote_parquet_source>(
      kvikio::RemoteHandle(std::move(endpoint), static_cast<size_t>(profile->known_size_bytes)),
      std::move(mr),
      std::move(mr_lifetime));
  }
  return std::make_unique<counted_remote_parquet_source>(
    kvikio::RemoteHandle(std::move(endpoint)), std::move(mr), std::move(mr_lifetime));
}

void write_remote_read_evidence(cudf_io_remote_parquet_read_evidence_t* out,
                                std::vector<counted_remote_parquet_source*> const& sources)
{
  if (out == nullptr) { return; }
  int64_t host_bytes   = 0;
  int64_t host_count   = 0;
  int64_t device_bytes = 0;
  int64_t device_count = 0;
  std::array<int64_t, remote_read_counters::size_bucket_count> size_bucket_counts{};
  for (auto const* source : sources) {
    auto const& counters = source->counters();
    host_bytes += counters.host_read_bytes.load(std::memory_order_relaxed);
    host_count += counters.host_read_count.load(std::memory_order_relaxed);
    device_bytes += counters.device_read_bytes.load(std::memory_order_relaxed);
    device_count += counters.device_read_count.load(std::memory_order_relaxed);
    for (std::size_t i = 0; i < size_bucket_counts.size(); ++i) {
      size_bucket_counts[i] += counters.size_bucket_counts[i].load(std::memory_order_relaxed);
    }
  }
  if (std::getenv("CUDF_IO_REMOTE_PARQUET_DEBUG_RANGES") != nullptr) {
    std::cerr << "[cudf_io_remote_parquet] event=read_range_summary"
              << " host_count=" << host_count << " host_bytes=" << host_bytes
              << " device_count=" << device_count << " device_bytes=" << device_bytes
              << " bucket_le_64k=" << size_bucket_counts[0]
              << " bucket_le_1m=" << size_bucket_counts[1]
              << " bucket_le_4m=" << size_bucket_counts[2]
              << " bucket_le_8m=" << size_bucket_counts[3]
              << " bucket_le_16m=" << size_bucket_counts[4]
              << " bucket_le_32m=" << size_bucket_counts[5]
              << " bucket_gt_32m=" << size_bucket_counts[6] << std::endl;
  }
  out->backend              = CUDF_IO_REMOTE_PARQUET_BACKEND_KVIKIO_REMOTE_DATASOURCE;
  out->fallback_reason      = CUDF_IO_REMOTE_PARQUET_FALLBACK_NONE;
  out->supports_device_read = 1;
  out->device_read_used     = device_count > 0 ? 1 : 0;
  out->host_read_bytes      = host_bytes;
  out->host_read_count      = host_count;
  out->device_read_bytes    = device_bytes;
  out->device_read_count    = device_count;
  out->remote_bytes         = host_bytes + device_bytes;
  out->remote_range_count   = host_count + device_count;
}
#endif
}  // namespace

extern "C" {
cudf_status_t cudf_io_read_parquet_ex(const char* filepath,
                                      const char** columns,
                                      int32_t num_columns,
                                      int64_t skip_rows,
                                      int64_t num_rows,
                                      const int32_t* row_groups,
                                      int32_t num_row_groups,
                                      cudf_table_handle* out,
                                      cudf_exec_options_t const* exec_options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  cudf_status_t const args_status =
    validate_parquet_reader_args(columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto source  = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::parquet_reader_options::builder(source);

    cudf_status_t const config_status = configure_parquet_reader_builder(
      builder, columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
    if (config_status != CUDF_SUCCESS) { return config_status; }

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options         = builder.build();
      auto [tbl, metadata] = cudf::io::read_parquet(options, exec->stream, *exec->mr);

      *out = cudf_release_table_handle(std::move(tbl), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_read_parquet_from_host_buffer_ex(const void* host_buffer,
                                                       size_t host_buffer_len,
                                                       const char** columns,
                                                       int32_t num_columns,
                                                       int64_t skip_rows,
                                                       int64_t num_rows,
                                                       const int32_t* row_groups,
                                                       int32_t num_row_groups,
                                                       cudf_table_handle* out,
                                                       cudf_exec_options_t const* exec_options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  cudf_status_t const buffer_status = validate_host_parquet_buffer(host_buffer, host_buffer_len);
  if (buffer_status != CUDF_SUCCESS) { return buffer_status; }
  cudf_status_t const args_status =
    validate_parquet_reader_args(columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto bytes = cudf::host_span<std::byte const>{
      static_cast<std::byte const*>(host_buffer), host_buffer_len};
    auto source  = cudf::io::source_info(bytes);
    auto builder = cudf::io::parquet_reader_options::builder(source);

    cudf_status_t const config_status = configure_parquet_reader_builder(
      builder, columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
    if (config_status != CUDF_SUCCESS) { return config_status; }

    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options         = builder.build();
      auto [tbl, metadata] = cudf::io::read_parquet(options, exec->stream, *exec->mr);

      *out = cudf_release_table_handle(std::move(tbl), cudf_capture_exec_provenance(*exec));
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_write_parquet_ex(cudf_table_view_handle table,
                                       const char* filepath,
                                       const char** column_names,
                                       int32_t num_columns,
                                       int32_t row_group_size_rows,
                                       int32_t dictionary_policy,
                                       cudf_exec_options_t const* exec_options)
{
  CUDF_RUST_NULL_CHECK(table, "table is null");
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  if (num_columns < 0) {
    set_error("num_columns must be >= 0");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  if (num_columns > 0) { CUDF_RUST_NULL_CHECK(column_names, "column_names is null"); }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* view = static_cast<cudf::table_view*>(table);
    auto sink  = cudf::io::sink_info(std::string(filepath));

    auto builder = cudf::io::parquet_writer_options::builder(sink, *view);

    if (column_names != nullptr && num_columns > 0) {
      cudf_status_t status = validate_c_string_array(column_names, num_columns, "column_names");
      if (status != CUDF_SUCCESS) { return status; }
      cudf::io::table_input_metadata table_meta(*view);
      auto const names_to_apply =
        std::min(static_cast<size_t>(num_columns), table_meta.column_metadata.size());
      for (size_t i = 0; i < names_to_apply; ++i) {
        table_meta.column_metadata[i].set_name(std::string(column_names[i]));
      }
      builder.metadata(std::move(table_meta));
    }

    auto options = builder.build();

    if (row_group_size_rows > 0) { options.set_row_group_size_rows(row_group_size_rows); }
    switch (dictionary_policy) {
      case -1: break;
      case 0: options.set_dictionary_policy(cudf::io::dictionary_policy::ADAPTIVE); break;
      case 1: options.set_dictionary_policy(cudf::io::dictionary_policy::NEVER); break;
      case 2: options.set_dictionary_policy(cudf::io::dictionary_policy::ALWAYS); break;
      default:
        set_error("dictionary_policy must be -1 (default), 0 (adaptive), 1 (never), or 2 (always)");
        return CUDF_ERROR_INVALID_OPERATION;
    }
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      cudf::io::write_parquet(options, exec->stream);
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_read_parquet_with_metadata_ex(const char* filepath,
                                                    const char** columns,
                                                    int32_t num_columns,
                                                    int64_t skip_rows,
                                                    int64_t num_rows,
                                                    const int32_t* row_groups,
                                                    int32_t num_row_groups,
                                                    cudf_table_handle* out_table,
                                                    char*** out_names,
                                                    int32_t* out_num_names,
                                                    cudf_exec_options_t const* exec_options)
{
  CUDF_RUST_NULL_CHECK(out_table, "out_table is null");
  *out_table = nullptr;
  CUDF_RUST_NULL_CHECK(out_names, "out_names is null");
  *out_names = nullptr;
  CUDF_RUST_NULL_CHECK(out_num_names, "out_num_names is null");
  *out_num_names = 0;
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  cudf_status_t const args_status =
    validate_parquet_reader_args(columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto source  = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::parquet_reader_options::builder(source);

    cudf_status_t const config_status = configure_parquet_reader_builder(
      builder, columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
    if (config_status != CUDF_SUCCESS) { return config_status; }

    cudf::io::table_with_metadata result;
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options = builder.build();
      result       = cudf::io::read_parquet(options, exec->stream, *exec->mr);
    }

    auto const fault = cudf_test_io_take_next_metadata_fault();
    if (fault != CUDF_TEST_IO_METADATA_FAULT_NONE) {
      cudf_status_t const fault_status =
        cudf_test_io_apply_metadata_fault(fault, out_names, out_num_names);
      if (fault_status != CUDF_SUCCESS) { return fault_status; }
      *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    char** names = copy_schema_names(result.metadata, out_num_names);
    if (*out_num_names > 0 && names == nullptr) { return CUDF_ERROR_OUT_OF_MEMORY; }

    *out_table     = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
    *out_names     = names;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_read_parquet_with_metadata_filter_ex(const char* filepath,
                                                           const char** columns,
                                                           int32_t num_columns,
                                                           int64_t skip_rows,
                                                           int64_t num_rows,
                                                           const int32_t* row_groups,
                                                           int32_t num_row_groups,
                                                           cudf_ast_expression_handle predicate,
                                                           cudf_table_handle* out_table,
                                                           char*** out_names,
                                                           int32_t* out_num_names,
                                                           cudf_io_parquet_pruning_evidence_t* out_pruning_evidence,
                                                           cudf_exec_options_t const* exec_options)
{
  CUDF_RUST_NULL_CHECK(out_table, "out_table is null");
  *out_table = nullptr;
  CUDF_RUST_NULL_CHECK(out_names, "out_names is null");
  *out_names = nullptr;
  CUDF_RUST_NULL_CHECK(out_num_names, "out_num_names is null");
  *out_num_names = 0;
  init_parquet_pruning_evidence(out_pruning_evidence);
  CUDF_RUST_NULL_CHECK(out_pruning_evidence, "out_pruning_evidence is null");
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  cudf_status_t const args_status =
    validate_parquet_reader_args(columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto source  = cudf::io::source_info(std::string(filepath));
    auto builder = cudf::io::parquet_reader_options::builder(source);

    cudf_status_t const config_status = configure_parquet_reader_builder(
      builder, columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
    if (config_status != CUDF_SUCCESS) { return config_status; }

    if (predicate != nullptr) { builder.filter(cudf_ast_expression_from_handle(predicate)); }

    cudf::io::table_with_metadata result;
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options = builder.build();
      result       = cudf::io::read_parquet(options, exec->stream, *exec->mr);
    }
    write_parquet_pruning_evidence(out_pruning_evidence, result.metadata);

    auto const fault = cudf_test_io_take_next_metadata_fault();
    if (fault != CUDF_TEST_IO_METADATA_FAULT_NONE) {
      cudf_status_t const fault_status =
        cudf_test_io_apply_metadata_fault(fault, out_names, out_num_names);
      if (fault_status != CUDF_SUCCESS) { return fault_status; }
      *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    char** names = copy_schema_names(result.metadata, out_num_names);
    if (*out_num_names > 0 && names == nullptr) { return CUDF_ERROR_OUT_OF_MEMORY; }

    *out_table     = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
    *out_names     = names;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_read_remote_parquet_with_metadata_filter_ex(
  const char* uri,
  const cudf_io_remote_parquet_profile_t* profile,
  const char** columns,
  int32_t num_columns,
  int64_t skip_rows,
  int64_t num_rows,
  const int32_t* row_groups,
  int32_t num_row_groups,
  cudf_ast_expression_handle predicate,
  cudf_table_handle* out_table,
  char*** out_names,
  int32_t* out_num_names,
  cudf_io_remote_parquet_read_evidence_t* out_evidence,
  cudf_io_parquet_pruning_evidence_t* out_pruning_evidence,
  cudf_exec_options_t const* exec_options)
{
  CUDF_RUST_NULL_CHECK(out_table, "out_table is null");
  *out_table = nullptr;
  CUDF_RUST_NULL_CHECK(out_names, "out_names is null");
  *out_names = nullptr;
  CUDF_RUST_NULL_CHECK(out_num_names, "out_num_names is null");
  *out_num_names = 0;
  init_remote_read_evidence(out_evidence);
  CUDF_RUST_NULL_CHECK(out_evidence, "out_evidence is null");
  init_parquet_pruning_evidence(out_pruning_evidence);
  CUDF_RUST_NULL_CHECK(out_pruning_evidence, "out_pruning_evidence is null");
  CUDF_RUST_NULL_CHECK(uri, "uri is null");
  cudf_status_t const args_status =
    validate_parquet_reader_args(columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }
#ifndef CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_ENABLED
  set_error("libcudf was not built with KvikIO remote IO");
  return CUDF_ERROR_UNSUPPORTED;
#else
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto remote_source = make_counted_remote_source(uri, profile, exec->mr, exec->mr_lifetime);
    std::vector<counted_remote_parquet_source*> evidence_sources{remote_source.get()};
    std::vector<std::unique_ptr<cudf::io::datasource>> sources;
    sources.push_back(cudf::io::datasource::create(remote_source.get()));

    auto builder = cudf::io::parquet_reader_options::builder();
    cudf_status_t const config_status = configure_parquet_reader_builder(
      builder, columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
    if (config_status != CUDF_SUCCESS) { return config_status; }

    if (predicate != nullptr) { builder.filter(cudf_ast_expression_from_handle(predicate)); }

    cudf::io::table_with_metadata result;
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options = builder.build();
      result       = cudf::io::read_parquet(std::move(sources), {}, options, exec->stream, *exec->mr);
    }
    write_remote_read_evidence(out_evidence, evidence_sources);
    write_parquet_pruning_evidence(out_pruning_evidence, result.metadata);

    auto const fault = cudf_test_io_take_next_metadata_fault();
    if (fault != CUDF_TEST_IO_METADATA_FAULT_NONE) {
      cudf_status_t const fault_status =
        cudf_test_io_apply_metadata_fault(fault, out_names, out_num_names);
      if (fault_status != CUDF_SUCCESS) { return fault_status; }
      *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    char** names = copy_schema_names(result.metadata, out_num_names);
    if (*out_num_names > 0 && names == nullptr) { return CUDF_ERROR_OUT_OF_MEMORY; }

    *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
    *out_names = names;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
#endif
}

cudf_status_t cudf_io_read_parquet_multi_with_metadata_filter_ex(
  const char** filepaths,
  int32_t num_files,
  const char** columns,
  int32_t num_columns,
  const int32_t* row_group_offsets,
  int32_t num_row_group_offsets,
  const int32_t* row_groups,
  int32_t num_row_groups,
  cudf_ast_expression_handle predicate,
  cudf_table_handle* out_table,
  char*** out_names,
  int32_t* out_num_names,
  cudf_io_parquet_pruning_evidence_t* out_pruning_evidence,
  cudf_exec_options_t const* exec_options)
{
  CUDF_RUST_NULL_CHECK(out_table, "out_table is null");
  *out_table = nullptr;
  CUDF_RUST_NULL_CHECK(out_names, "out_names is null");
  *out_names = nullptr;
  CUDF_RUST_NULL_CHECK(out_num_names, "out_num_names is null");
  *out_num_names = 0;
  init_parquet_pruning_evidence(out_pruning_evidence);
  CUDF_RUST_NULL_CHECK(out_pruning_evidence, "out_pruning_evidence is null");
  cudf_status_t const args_status = validate_parquet_multi_reader_args(
    filepaths,
    num_files,
    columns,
    num_columns,
    row_group_offsets,
    num_row_group_offsets,
    row_groups,
    num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    std::vector<std::string> paths;
    paths.reserve(num_files);
    for (int32_t i = 0; i < num_files; ++i) {
      paths.emplace_back(filepaths[i]);
    }
    auto source  = cudf::io::source_info(std::move(paths));
    auto builder = cudf::io::parquet_reader_options::builder(source);

    if (columns != nullptr && num_columns > 0) {
      std::vector<std::string> col_names;
      col_names.reserve(num_columns);
      for (int32_t i = 0; i < num_columns; ++i) {
        col_names.emplace_back(columns[i]);
      }
      builder.column_names(col_names);
    }

    if (num_row_group_offsets > 0) {
      std::vector<std::vector<cudf::size_type>> grouped_row_groups;
      grouped_row_groups.reserve(num_files);
      for (int32_t file_index = 0; file_index < num_files; ++file_index) {
        auto const begin = row_group_offsets[file_index];
        auto const end   = row_group_offsets[file_index + 1];
        std::vector<cudf::size_type> file_row_groups;
        file_row_groups.reserve(end - begin);
        for (int32_t row_group_index = begin; row_group_index < end; ++row_group_index) {
          file_row_groups.push_back(static_cast<cudf::size_type>(row_groups[row_group_index]));
        }
        grouped_row_groups.push_back(std::move(file_row_groups));
      }
      builder.row_groups(std::move(grouped_row_groups));
    }

    if (predicate != nullptr) { builder.filter(cudf_ast_expression_from_handle(predicate)); }

    cudf::io::table_with_metadata result;
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options = builder.build();
      result       = cudf::io::read_parquet(options, exec->stream, *exec->mr);
    }
    write_parquet_pruning_evidence(out_pruning_evidence, result.metadata);

    auto const fault = cudf_test_io_take_next_metadata_fault();
    if (fault != CUDF_TEST_IO_METADATA_FAULT_NONE) {
      cudf_status_t const fault_status =
        cudf_test_io_apply_metadata_fault(fault, out_names, out_num_names);
      if (fault_status != CUDF_SUCCESS) { return fault_status; }
      *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    char** names = copy_schema_names(result.metadata, out_num_names);
    if (*out_num_names > 0 && names == nullptr) { return CUDF_ERROR_OUT_OF_MEMORY; }

    *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
    *out_names = names;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_read_parquet_multi_mixed_with_metadata_filter_ex(
  const char** filepaths,
  const int32_t* remote_source_flags,
  const cudf_io_remote_parquet_profile_t* profiles,
  int32_t num_files,
  const char** columns,
  int32_t num_columns,
  const int32_t* row_group_offsets,
  int32_t num_row_group_offsets,
  const int32_t* row_groups,
  int32_t num_row_groups,
  cudf_ast_expression_handle predicate,
  cudf_table_handle* out_table,
  char*** out_names,
  int32_t* out_num_names,
  cudf_io_remote_parquet_read_evidence_t* out_evidence,
  cudf_io_parquet_pruning_evidence_t* out_pruning_evidence,
  cudf_exec_options_t const* exec_options)
{
  CUDF_RUST_NULL_CHECK(out_table, "out_table is null");
  *out_table = nullptr;
  CUDF_RUST_NULL_CHECK(out_names, "out_names is null");
  *out_names = nullptr;
  CUDF_RUST_NULL_CHECK(out_num_names, "out_num_names is null");
  *out_num_names = 0;
  init_remote_read_evidence(out_evidence);
  CUDF_RUST_NULL_CHECK(out_evidence, "out_evidence is null");
  init_parquet_pruning_evidence(out_pruning_evidence);
  CUDF_RUST_NULL_CHECK(out_pruning_evidence, "out_pruning_evidence is null");
  CUDF_RUST_NULL_CHECK(remote_source_flags, "remote_source_flags is null");
  cudf_status_t const args_status = validate_parquet_multi_reader_args(
    filepaths,
    num_files,
    columns,
    num_columns,
    row_group_offsets,
    num_row_group_offsets,
    row_groups,
    num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }

  bool has_remote_sources = false;
  for (int32_t i = 0; i < num_files; ++i) {
    if (remote_source_flags[i] != 0) {
      has_remote_sources = true;
      break;
    }
  }
  if (!has_remote_sources) {
    return cudf_io_read_parquet_multi_with_metadata_filter_ex(filepaths,
                                                             num_files,
                                                             columns,
                                                             num_columns,
                                                             row_group_offsets,
                                                             num_row_group_offsets,
                                                             row_groups,
                                                             num_row_groups,
                                                             predicate,
                                                             out_table,
                                                             out_names,
                                                             out_num_names,
                                                             out_pruning_evidence,
                                                             exec_options);
  }

  CUDF_RUST_NULL_CHECK(profiles, "profiles is null");
#ifndef CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_ENABLED
  set_error("libcudf was not built with KvikIO remote IO");
  return CUDF_ERROR_UNSUPPORTED;
#else
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    std::vector<std::unique_ptr<counted_remote_parquet_source>> remote_sources;
    remote_sources.reserve(num_files);
    std::vector<counted_remote_parquet_source*> evidence_sources;
    std::vector<std::unique_ptr<cudf::io::datasource>> sources;
    sources.reserve(num_files);
    for (int32_t i = 0; i < num_files; ++i) {
      if (remote_source_flags[i] != 0) {
        auto remote_source = make_counted_remote_source(filepaths[i], &profiles[i], exec->mr, exec->mr_lifetime);
        evidence_sources.push_back(remote_source.get());
        sources.push_back(cudf::io::datasource::create(remote_source.get()));
        remote_sources.push_back(std::move(remote_source));
      } else {
        sources.push_back(cudf::io::datasource::create(std::string(filepaths[i])));
      }
    }

    auto builder = cudf::io::parquet_reader_options::builder();

    if (columns != nullptr && num_columns > 0) {
      std::vector<std::string> col_names;
      col_names.reserve(num_columns);
      for (int32_t i = 0; i < num_columns; ++i) {
        col_names.emplace_back(columns[i]);
      }
      builder.column_names(col_names);
    }

    if (num_row_group_offsets > 0) {
      std::vector<std::vector<cudf::size_type>> grouped_row_groups;
      grouped_row_groups.reserve(num_files);
      for (int32_t file_index = 0; file_index < num_files; ++file_index) {
        auto const begin = row_group_offsets[file_index];
        auto const end   = row_group_offsets[file_index + 1];
        std::vector<cudf::size_type> file_row_groups;
        file_row_groups.reserve(end - begin);
        for (int32_t row_group_index = begin; row_group_index < end; ++row_group_index) {
          file_row_groups.push_back(static_cast<cudf::size_type>(row_groups[row_group_index]));
        }
        grouped_row_groups.push_back(std::move(file_row_groups));
      }
      builder.row_groups(std::move(grouped_row_groups));
    }

    if (predicate != nullptr) { builder.filter(cudf_ast_expression_from_handle(predicate)); }

    cudf::io::table_with_metadata result;
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options = builder.build();
      result       = cudf::io::read_parquet(std::move(sources), {}, options, exec->stream, *exec->mr);
    }
    write_remote_read_evidence(out_evidence, evidence_sources);
    write_parquet_pruning_evidence(out_pruning_evidence, result.metadata);

    auto const fault = cudf_test_io_take_next_metadata_fault();
    if (fault != CUDF_TEST_IO_METADATA_FAULT_NONE) {
      cudf_status_t const fault_status =
        cudf_test_io_apply_metadata_fault(fault, out_names, out_num_names);
      if (fault_status != CUDF_SUCCESS) { return fault_status; }
      *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    char** names = copy_schema_names(result.metadata, out_num_names);
    if (*out_num_names > 0 && names == nullptr) { return CUDF_ERROR_OUT_OF_MEMORY; }

    *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
    *out_names = names;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
#endif
}

cudf_status_t cudf_io_read_parquet_from_host_buffer_with_metadata_filter_ex(
  const void* host_buffer,
  size_t host_buffer_len,
  const char** columns,
  int32_t num_columns,
  int64_t skip_rows,
  int64_t num_rows,
  const int32_t* row_groups,
  int32_t num_row_groups,
  cudf_ast_expression_handle predicate,
  cudf_table_handle* out_table,
  char*** out_names,
  int32_t* out_num_names,
  cudf_io_parquet_pruning_evidence_t* out_pruning_evidence,
  cudf_exec_options_t const* exec_options)
{
  CUDF_RUST_NULL_CHECK(out_table, "out_table is null");
  *out_table = nullptr;
  CUDF_RUST_NULL_CHECK(out_names, "out_names is null");
  *out_names = nullptr;
  CUDF_RUST_NULL_CHECK(out_num_names, "out_num_names is null");
  *out_num_names = 0;
  init_parquet_pruning_evidence(out_pruning_evidence);
  CUDF_RUST_NULL_CHECK(out_pruning_evidence, "out_pruning_evidence is null");
  cudf_status_t const buffer_status = validate_host_parquet_buffer(host_buffer, host_buffer_len);
  if (buffer_status != CUDF_SUCCESS) { return buffer_status; }
  cudf_status_t const args_status =
    validate_parquet_reader_args(columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
  if (args_status != CUDF_SUCCESS) { return args_status; }
  CUDF_RUST_TRY
  {
    auto exec                       = std::unique_ptr<cudf_exec_context>{};
    cudf_status_t const exec_status = cudf_resolve_required_exec_options(exec_options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto bytes = cudf::host_span<std::byte const>{
      static_cast<std::byte const*>(host_buffer), host_buffer_len};
    auto source  = cudf::io::source_info(bytes);
    auto builder = cudf::io::parquet_reader_options::builder(source);

    cudf_status_t const config_status = configure_parquet_reader_builder(
      builder, columns, num_columns, skip_rows, num_rows, row_groups, num_row_groups);
    if (config_status != CUDF_SUCCESS) { return config_status; }

    if (predicate != nullptr) { builder.filter(cudf_ast_expression_from_handle(predicate)); }

    cudf::io::table_with_metadata result;
    {
      cudf_current_device_resource_guard current_mr{exec->mr_raw};
      auto options = builder.build();
      result       = cudf::io::read_parquet(options, exec->stream, *exec->mr);
    }
    write_parquet_pruning_evidence(out_pruning_evidence, result.metadata);

    auto const fault = cudf_test_io_take_next_metadata_fault();
    if (fault != CUDF_TEST_IO_METADATA_FAULT_NONE) {
      cudf_status_t const fault_status =
        cudf_test_io_apply_metadata_fault(fault, out_names, out_num_names);
      if (fault_status != CUDF_SUCCESS) { return fault_status; }
      *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
      return CUDF_SUCCESS;
    }

    char** names = copy_schema_names(result.metadata, out_num_names);
    if (*out_num_names > 0 && names == nullptr) { return CUDF_ERROR_OUT_OF_MEMORY; }

    *out_table = cudf_release_table_handle(std::move(result.tbl), cudf_capture_exec_provenance(*exec));
    *out_names = names;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

int32_t cudf_io_remote_parquet_kvikio_remote_io_enabled(void)
{
#ifdef CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_ENABLED
  return 1;
#elif defined(CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_DISABLED)
  return 0;
#else
  return -1;
#endif
}

cudf_status_t cudf_io_probe_path_datasource(const char* filepath,
                                            int64_t preferred_read_size,
                                            cudf_io_datasource_probe_result_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  out->can_open               = 0;
  out->supports_device_read   = 0;
  out->device_read_preferred  = 0;
  out->size_bytes             = -1;
  CUDF_RUST_NULL_CHECK(filepath, "filepath is null");
  if (preferred_read_size < -1) {
    set_error("preferred_read_size must be >= -1");
    return CUDF_ERROR_INVALID_OPERATION;
  }
  CUDF_RUST_TRY
  {
    auto source              = cudf::io::datasource::create(std::string(filepath));
    out->can_open            = 1;
    out->supports_device_read = source->supports_device_read() ? 1 : 0;
    if (preferred_read_size >= 0) {
      out->device_read_preferred =
        source->is_device_read_preferred(static_cast<size_t>(preferred_read_size)) ? 1 : 0;
    }
    auto const source_size = source->size();
    out->size_bytes =
      source_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())
        ? std::numeric_limits<int64_t>::max()
        : static_cast<int64_t>(source_size);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
}

cudf_status_t cudf_io_probe_remote_parquet_datasource(
  const char* uri,
  const cudf_io_remote_parquet_profile_t* profile,
  int64_t preferred_read_size,
  cudf_io_datasource_probe_result_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  out->can_open              = 0;
  out->supports_device_read  = 0;
  out->device_read_preferred = 0;
  out->size_bytes            = -1;
  CUDF_RUST_NULL_CHECK(uri, "uri is null");
  if (preferred_read_size < -1) {
    set_error("preferred_read_size must be >= -1");
    return CUDF_ERROR_INVALID_OPERATION;
  }
#ifndef CUDF_RUST_LIBCUDF_KVIKIO_REMOTE_IO_ENABLED
  set_error("libcudf was not built with KvikIO remote IO");
  return CUDF_ERROR_UNSUPPORTED;
#else
  CUDF_RUST_TRY
  {
    auto exec                 = cudf_make_exec_context();
    auto source               = make_counted_remote_source(uri, profile, exec.mr, exec.mr_lifetime);
    out->can_open             = 1;
    out->supports_device_read = source->supports_device_read() ? 1 : 0;
    if (preferred_read_size >= 0) {
      out->device_read_preferred =
        source->is_device_read_preferred(static_cast<size_t>(preferred_read_size)) ? 1 : 0;
    }
    auto const source_size = source->size();
    out->size_bytes =
      source_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())
        ? std::numeric_limits<int64_t>::max()
        : static_cast<int64_t>(source_size);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_IO_CATCH
#endif
}

void cudf_io_free_column_names(char** names, int32_t num_names)
{
  if (names == nullptr) return;
  for (int32_t i = 0; i < num_names; ++i) {
    delete[] names[i];
  }
  delete[] names;
}
}  // extern "C"
