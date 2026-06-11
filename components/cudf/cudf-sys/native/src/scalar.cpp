#include "scalar.h"

#include "execution_context_internal.hpp"
#include "shim_common.h"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

// Helper: construct a numeric scalar of the given type from a host value pointer.
template <typename T>
static std::unique_ptr<cudf::scalar> make_numeric_scalar(const void* value,
                                                         bool is_valid,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  T val{};
  if (value != nullptr) { std::memcpy(&val, value, sizeof(T)); }
  return std::make_unique<cudf::numeric_scalar<T>>(val, is_valid, stream, mr);
}

extern "C" {

cudf_status_t cudf_scalar_create_numeric(cudf_type_id_t type_id,
                                         const void* value,
                                         int32_t is_valid,
                                         cudf_scalar_handle* out)
{
  return cudf_scalar_create_numeric_ex(type_id, value, is_valid, nullptr, out);
}

cudf_status_t cudf_scalar_create_numeric_ex(cudf_type_id_t type_id,
                                            const void* value,
                                            int32_t is_valid,
                                            cudf_exec_options_t const* options,
                                            cudf_scalar_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    cudf_exec_context exec{
      cudf::get_default_stream(), cudf::get_current_device_resource_ref(), nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    std::unique_ptr<cudf::scalar> s;
    bool valid = is_valid != 0;
    if (valid) { CUDF_RUST_NULL_CHECK(value, "value is null"); }
    switch (to_cudf_type_id(type_id)) {
      case cudf::type_id::INT8:
        s = make_numeric_scalar<int8_t>(value, valid, exec.stream, exec.mr);
        break;
      case cudf::type_id::INT16:
        s = make_numeric_scalar<int16_t>(value, valid, exec.stream, exec.mr);
        break;
      case cudf::type_id::INT32:
        s = make_numeric_scalar<int32_t>(value, valid, exec.stream, exec.mr);
        break;
      case cudf::type_id::INT64:
        s = make_numeric_scalar<int64_t>(value, valid, exec.stream, exec.mr);
        break;
      case cudf::type_id::UINT8:
        s = make_numeric_scalar<uint8_t>(value, valid, exec.stream, exec.mr);
        break;
      case cudf::type_id::UINT16:
        s = make_numeric_scalar<uint16_t>(value, valid, exec.stream, exec.mr);
        break;
      case cudf::type_id::UINT32:
        s = make_numeric_scalar<uint32_t>(value, valid, exec.stream, exec.mr);
        break;
      case cudf::type_id::UINT64:
        s = make_numeric_scalar<uint64_t>(value, valid, exec.stream, exec.mr);
        break;
      case cudf::type_id::FLOAT32:
        s = make_numeric_scalar<float>(value, valid, exec.stream, exec.mr);
        break;
      case cudf::type_id::FLOAT64:
        s = make_numeric_scalar<double>(value, valid, exec.stream, exec.mr);
        break;
      case cudf::type_id::BOOL8:
        s = make_numeric_scalar<bool>(value, valid, exec.stream, exec.mr);
        break;
      default: set_error("Unsupported type for numeric scalar"); return CUDF_ERROR_TYPE;
    }
    *out = static_cast<cudf_scalar_handle>(s.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_scalar_create_string(const char* value,
                                        int32_t is_valid,
                                        cudf_scalar_handle* out)
{
  return cudf_scalar_create_string_ex(value, is_valid, nullptr, out);
}

cudf_status_t cudf_scalar_create_string_ex(const char* value,
                                           int32_t is_valid,
                                           cudf_exec_options_t const* options,
                                           cudf_scalar_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  if (is_valid != 0) { CUDF_RUST_NULL_CHECK(value, "value is null"); }
  CUDF_RUST_TRY
  {
    cudf_exec_context exec{
      cudf::get_default_stream(), cudf::get_current_device_resource_ref(), nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    std::string str_val = (value != nullptr) ? std::string(value) : std::string();
    auto s = std::make_unique<cudf::string_scalar>(str_val, is_valid != 0, exec.stream, exec.mr);
    exec.stream.synchronize();
    cudf_record_synchronized_stream(exec.stream);
    *out = static_cast<cudf_scalar_handle>(s.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_scalar_copy(cudf_scalar_handle src, cudf_scalar_handle* out)
{
  return cudf_scalar_copy_ex(src, nullptr, out);
}

cudf_status_t cudf_scalar_copy_ex(cudf_scalar_handle src,
                                  cudf_exec_options_t const* options,
                                  cudf_scalar_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(src, "src is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec{
      cudf::get_default_stream(), cudf::get_current_device_resource_ref(), nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* s = static_cast<cudf::scalar*>(src);
    std::unique_ptr<cudf::scalar> copy;

    switch (s->type().id()) {
      // Numeric types
#define CUDF_COPY_NUMERIC(ENUM, TYPE)                                            \
  case cudf::type_id::ENUM:                                                      \
    copy = std::make_unique<cudf::numeric_scalar<TYPE>>(                         \
      static_cast<cudf::numeric_scalar<TYPE> const&>(*s), exec.stream, exec.mr); \
    break;
      CUDF_COPY_NUMERIC(INT8, int8_t)
      CUDF_COPY_NUMERIC(INT16, int16_t)
      CUDF_COPY_NUMERIC(INT32, int32_t)
      CUDF_COPY_NUMERIC(INT64, int64_t)
      CUDF_COPY_NUMERIC(UINT8, uint8_t)
      CUDF_COPY_NUMERIC(UINT16, uint16_t)
      CUDF_COPY_NUMERIC(UINT32, uint32_t)
      CUDF_COPY_NUMERIC(UINT64, uint64_t)
      CUDF_COPY_NUMERIC(FLOAT32, float)
      CUDF_COPY_NUMERIC(FLOAT64, double)
      CUDF_COPY_NUMERIC(BOOL8, bool)
#undef CUDF_COPY_NUMERIC

      // Timestamp types
#define CUDF_COPY_TIMESTAMP(ENUM, TYPE)                                            \
  case cudf::type_id::ENUM:                                                        \
    copy = std::make_unique<cudf::timestamp_scalar<TYPE>>(                         \
      static_cast<cudf::timestamp_scalar<TYPE> const&>(*s), exec.stream, exec.mr); \
    break;
      CUDF_COPY_TIMESTAMP(TIMESTAMP_DAYS, cudf::timestamp_D)
      CUDF_COPY_TIMESTAMP(TIMESTAMP_SECONDS, cudf::timestamp_s)
      CUDF_COPY_TIMESTAMP(TIMESTAMP_MILLISECONDS, cudf::timestamp_ms)
      CUDF_COPY_TIMESTAMP(TIMESTAMP_MICROSECONDS, cudf::timestamp_us)
      CUDF_COPY_TIMESTAMP(TIMESTAMP_NANOSECONDS, cudf::timestamp_ns)
#undef CUDF_COPY_TIMESTAMP

      // Duration types
#define CUDF_COPY_DURATION(ENUM, TYPE)                                            \
  case cudf::type_id::ENUM:                                                       \
    copy = std::make_unique<cudf::duration_scalar<TYPE>>(                         \
      static_cast<cudf::duration_scalar<TYPE> const&>(*s), exec.stream, exec.mr); \
    break;
      CUDF_COPY_DURATION(DURATION_DAYS, cudf::duration_D)
      CUDF_COPY_DURATION(DURATION_SECONDS, cudf::duration_s)
      CUDF_COPY_DURATION(DURATION_MILLISECONDS, cudf::duration_ms)
      CUDF_COPY_DURATION(DURATION_MICROSECONDS, cudf::duration_us)
      CUDF_COPY_DURATION(DURATION_NANOSECONDS, cudf::duration_ns)
#undef CUDF_COPY_DURATION

      // String
      case cudf::type_id::STRING:
        copy = std::make_unique<cudf::string_scalar>(
          static_cast<cudf::string_scalar const&>(*s), exec.stream, exec.mr);
        break;

      // Fixed-point decimal
      case cudf::type_id::DECIMAL32:
        copy = std::make_unique<cudf::fixed_point_scalar<numeric::decimal32>>(
          static_cast<cudf::fixed_point_scalar<numeric::decimal32> const&>(*s),
          exec.stream,
          exec.mr);
        break;
      case cudf::type_id::DECIMAL64:
        copy = std::make_unique<cudf::fixed_point_scalar<numeric::decimal64>>(
          static_cast<cudf::fixed_point_scalar<numeric::decimal64> const&>(*s),
          exec.stream,
          exec.mr);
        break;
      case cudf::type_id::DECIMAL128:
        copy = std::make_unique<cudf::fixed_point_scalar<numeric::decimal128>>(
          static_cast<cudf::fixed_point_scalar<numeric::decimal128> const&>(*s),
          exec.stream,
          exec.mr);
        break;

      // List
      case cudf::type_id::LIST:
        copy = std::make_unique<cudf::list_scalar>(
          static_cast<cudf::list_scalar const&>(*s), exec.stream, exec.mr);
        break;

      // Struct
      case cudf::type_id::STRUCT:
        copy = std::make_unique<cudf::struct_scalar>(
          static_cast<cudf::struct_scalar const&>(*s), exec.stream, exec.mr);
        break;

      default: set_error("Unsupported type for scalar copy"); return CUDF_ERROR_TYPE;
    }

    *out = static_cast<cudf_scalar_handle>(copy.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_scalar_destroy(cudf_scalar_handle handle) { delete static_cast<cudf::scalar*>(handle); }

cudf_status_t cudf_scalar_type(cudf_scalar_handle handle, cudf_type_id_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = CUDF_TYPE_EMPTY;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* s = static_cast<cudf::scalar*>(handle);
    *out    = static_cast<cudf_type_id_t>(s->type().id());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_scalar_data_type(cudf_scalar_handle handle,
                                    cudf_type_id_t* out_type_id,
                                    int32_t* out_scale)
{
  CUDF_RUST_NULL_CHECK(out_type_id, "out_type_id is null");
  CUDF_RUST_NULL_CHECK(out_scale, "out_scale is null");
  *out_type_id = CUDF_TYPE_EMPTY;
  *out_scale   = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto* s      = static_cast<cudf::scalar*>(handle);
    auto dtype   = s->type();
    *out_type_id = static_cast<cudf_type_id_t>(dtype.id());
    *out_scale   = dtype.scale();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_scalar_is_valid(cudf_scalar_handle handle, int32_t* out)
{
  return cudf_scalar_is_valid_ex(handle, out, nullptr);
}

cudf_status_t cudf_scalar_is_valid_ex(cudf_scalar_handle handle,
                                      int32_t* out,
                                      cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec{
      cudf::get_default_stream(), cudf::get_current_device_resource_ref(), nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* s = static_cast<cudf::scalar*>(handle);
    *out    = s->is_valid(exec.stream) ? 1 : 0;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_scalar_value(cudf_scalar_handle handle, void* out)
{
  return cudf_scalar_value_ex(handle, out, nullptr);
}

cudf_status_t cudf_scalar_value_ex(cudf_scalar_handle handle,
                                   void* out,
                                   cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec{
      cudf::get_default_stream(), cudf::get_current_device_resource_ref(), nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* s  = static_cast<cudf::scalar*>(handle);
    auto tid = s->type().id();

    // Dispatch to the correct numeric_scalar<T>::value().
    switch (tid) {
#define CUDF_SCALAR_CASE(ENUM, TYPE)                                            \
  case cudf::type_id::ENUM: {                                                   \
    auto val = static_cast<cudf::numeric_scalar<TYPE>*>(s)->value(exec.stream); \
    std::memcpy(out, &val, sizeof(TYPE));                                       \
    break;                                                                      \
  }
      CUDF_SCALAR_CASE(INT8, int8_t)
      CUDF_SCALAR_CASE(INT16, int16_t)
      CUDF_SCALAR_CASE(INT32, int32_t)
      CUDF_SCALAR_CASE(INT64, int64_t)
      CUDF_SCALAR_CASE(UINT8, uint8_t)
      CUDF_SCALAR_CASE(UINT16, uint16_t)
      CUDF_SCALAR_CASE(UINT32, uint32_t)
      CUDF_SCALAR_CASE(UINT64, uint64_t)
      CUDF_SCALAR_CASE(FLOAT32, float)
      CUDF_SCALAR_CASE(FLOAT64, double)
      CUDF_SCALAR_CASE(BOOL8, bool)
#undef CUDF_SCALAR_CASE
      default: set_error("Unsupported type for scalar value extraction"); return CUDF_ERROR_TYPE;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_scalar_string_value(cudf_scalar_handle handle, char** out, int32_t* len_out)
{
  return cudf_scalar_string_value_ex(handle, out, len_out, nullptr);
}

cudf_status_t cudf_scalar_string_value_ex(cudf_scalar_handle handle,
                                          char** out,
                                          int32_t* len_out,
                                          cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(len_out, "len_out is null");
  *len_out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    cudf_exec_context exec{
      cudf::get_default_stream(), cudf::get_current_device_resource_ref(), nullptr};
    cudf_status_t const exec_status = cudf_resolve_exec_options(options, &exec);
    if (exec_status != CUDF_SUCCESS) { return exec_status; }

    auto* s = static_cast<cudf::scalar*>(handle);
    if (s->type().id() != cudf::type_id::STRING) {
      set_error("Scalar is not a string type");
      return CUDF_ERROR_TYPE;
    }
    auto* str_scalar = static_cast<cudf::string_scalar*>(s);
    auto host_value  = str_scalar->to_string(exec.stream);
    if (host_value.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      set_error("String scalar value exceeds supported host length");
      return CUDF_ERROR_INVALID_OPERATION;
    }

    auto const length = static_cast<int32_t>(host_value.size());
    auto* copied      = static_cast<char*>(std::malloc(static_cast<size_t>(length) + 1));
    if (copied == nullptr) {
      set_error("Out of memory allocating string scalar host copy");
      return CUDF_ERROR_OUT_OF_MEMORY;
    }
    if (length > 0) { std::memcpy(copied, host_value.data(), static_cast<size_t>(length)); }
    copied[length] = '\0';

    *out     = copied;
    *len_out = length;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_string_free(char* ptr) { std::free(ptr); }

}  // extern "C"
