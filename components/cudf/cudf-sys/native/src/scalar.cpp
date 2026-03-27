#include "scalar.h"
#include "shim_common.h"

#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <cstring>
#include <memory>
#include <string>

static cudf::type_id to_cudf_type_id(cudf_type_id_t id) {
  return static_cast<cudf::type_id>(id);
}

// Helper: construct a numeric scalar of the given type from a host value pointer.
template <typename T>
static std::unique_ptr<cudf::scalar> make_numeric_scalar(const void* value,
                                                         bool is_valid) {
  auto stream = cudf::get_default_stream();
  auto mr = cudf::get_current_device_resource_ref();
  T val{};
  if (value != nullptr) {
    std::memcpy(&val, value, sizeof(T));
  }
  auto s = cudf::make_numeric_scalar(cudf::data_type(cudf::type_to_id<T>()),
                                     stream, mr);
  static_cast<cudf::numeric_scalar<T>*>(s.get())->set_value(val, stream);
  s->set_valid_async(is_valid, stream);
  stream.synchronize();
  return s;
}

extern "C" {

cudf_status_t cudf_scalar_create_numeric(cudf_type_id_t type_id,
                                         const void* value, int32_t is_valid,
                                         cudf_scalar_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    std::unique_ptr<cudf::scalar> s;
    bool valid = is_valid != 0;
    if (valid) {
      CUDF_RUST_NULL_CHECK(value, "value is null");
    }
    switch (to_cudf_type_id(type_id)) {
      case cudf::type_id::INT8:
        s = make_numeric_scalar<int8_t>(value, valid);
        break;
      case cudf::type_id::INT16:
        s = make_numeric_scalar<int16_t>(value, valid);
        break;
      case cudf::type_id::INT32:
        s = make_numeric_scalar<int32_t>(value, valid);
        break;
      case cudf::type_id::INT64:
        s = make_numeric_scalar<int64_t>(value, valid);
        break;
      case cudf::type_id::UINT8:
        s = make_numeric_scalar<uint8_t>(value, valid);
        break;
      case cudf::type_id::UINT16:
        s = make_numeric_scalar<uint16_t>(value, valid);
        break;
      case cudf::type_id::UINT32:
        s = make_numeric_scalar<uint32_t>(value, valid);
        break;
      case cudf::type_id::UINT64:
        s = make_numeric_scalar<uint64_t>(value, valid);
        break;
      case cudf::type_id::FLOAT32:
        s = make_numeric_scalar<float>(value, valid);
        break;
      case cudf::type_id::FLOAT64:
        s = make_numeric_scalar<double>(value, valid);
        break;
      case cudf::type_id::BOOL8:
        s = make_numeric_scalar<bool>(value, valid);
        break;
      default:
        set_error("Unsupported type for numeric scalar");
        return CUDF_ERROR_TYPE;
    }
    *out = static_cast<cudf_scalar_handle>(s.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_scalar_create_string(const char* value, int32_t is_valid,
                                        cudf_scalar_handle* out) {
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY {
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();
    std::string str_val = (value != nullptr) ? std::string(value) : std::string();
    auto s = cudf::make_string_scalar(str_val, stream, mr);
    s->set_valid_async(is_valid != 0, stream);
    stream.synchronize();
    *out = static_cast<cudf_scalar_handle>(s.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

void cudf_scalar_destroy(cudf_scalar_handle handle) {
  delete static_cast<cudf::scalar*>(handle);
}

cudf_status_t cudf_scalar_type(cudf_scalar_handle handle,
                               cudf_type_id_t* out) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* s = static_cast<cudf::scalar*>(handle);
    *out = static_cast<cudf_type_id_t>(s->type().id());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_scalar_is_valid(cudf_scalar_handle handle, int32_t* out) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* s = static_cast<cudf::scalar*>(handle);
    auto stream = cudf::get_default_stream();
    *out = s->is_valid(stream) ? 1 : 0;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_scalar_value(cudf_scalar_handle handle, void* out) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_TRY {
    auto* s = static_cast<cudf::scalar*>(handle);
    auto stream = cudf::get_default_stream();
    auto tid = s->type().id();

    // Dispatch to the correct numeric_scalar<T>::value().
    switch (tid) {
#define CUDF_SCALAR_CASE(ENUM, TYPE)                                       \
  case cudf::type_id::ENUM: {                                              \
    auto val = static_cast<cudf::numeric_scalar<TYPE>*>(s)->value(stream); \
    std::memcpy(out, &val, sizeof(TYPE));                                  \
    break;                                                                 \
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
      default:
        set_error("Unsupported type for scalar value extraction");
        return CUDF_ERROR_TYPE;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_scalar_string_value(cudf_scalar_handle handle,
                                       const char** out, int32_t* len_out) {
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_NULL_CHECK(out, "out is null");
  CUDF_RUST_NULL_CHECK(len_out, "len_out is null");
  *out = nullptr;
  *len_out = 0;
  CUDF_RUST_TRY {
    auto* s = static_cast<cudf::scalar*>(handle);
    if (s->type().id() != cudf::type_id::STRING) {
      set_error("Scalar is not a string type");
      return CUDF_ERROR_TYPE;
    }
    auto* str_scalar = static_cast<cudf::string_scalar*>(s);
    auto stream = cudf::get_default_stream();
    // to_string() copies from device to host.
    // We store the result in a thread-local buffer.
    // Returned pointer is valid until the next shim call on this thread.
    static thread_local std::string tl_string;
    tl_string = str_scalar->to_string(stream);
    *out = tl_string.c_str();
    *len_out = static_cast<int32_t>(tl_string.size());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

}  // extern "C"
