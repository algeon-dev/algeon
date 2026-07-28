#include "cudf_rust/error.h"

#include <cudf/utilities/error.hpp>

#include <ios>
#include <new>
#include <stdexcept>
#include <string>

// Thread-local storage for the last error message.
static thread_local std::string g_last_error_message;
static thread_local bool g_has_error = false;

void set_error(const std::string& msg)
{
  g_last_error_message = msg;
  g_has_error          = true;
}

void set_error(const char* msg)
{
  g_last_error_message = msg;
  g_has_error          = true;
}

extern "C" {
const char* cudf_last_error_message(void)
{
  if (g_has_error) { return g_last_error_message.c_str(); }
  return nullptr;
}

void cudf_clear_error(void)
{
  g_has_error = false;
  g_last_error_message.clear();
}
}  // extern "C"

namespace {
cudf_status_t translate_exception_impl(cudf_status_t runtime_error_status)
{
  try {
    throw;  // re-throw current exception
  } catch (cudf::fatal_cuda_error const& e) {
    set_error(e.what());
    return CUDF_ERROR_FATAL_CUDA;
  } catch (cudf::cuda_error const& e) {
    set_error(e.what());
    return CUDF_ERROR_CUDA;
  } catch (cudf::data_type_error const& e) {
    set_error(e.what());
    return CUDF_ERROR_TYPE;
  } catch (cudf::logic_error const& e) {
    set_error(e.what());
    return CUDF_ERROR_INVALID_OPERATION;
  } catch (std::bad_alloc const& e) {
    set_error(e.what());
    return CUDF_ERROR_OUT_OF_MEMORY;
  } catch (std::out_of_range const& e) {
    set_error(e.what());
    return CUDF_ERROR_INDEX;
  } catch (std::invalid_argument const& e) {
    set_error(e.what());
    return CUDF_ERROR_INVALID_ARGUMENT;
  } catch (std::overflow_error const& e) {
    set_error(e.what());
    return CUDF_ERROR_SIZE_OVERFLOW;
  } catch (std::ios_base::failure const& e) {
    set_error(e.what());
    return CUDF_ERROR_IO;
  } catch (std::runtime_error const& e) {
    set_error(e.what());
    return runtime_error_status;
  } catch (std::exception const& e) {
    set_error(e.what());
    return CUDF_ERROR_UNKNOWN;
  } catch (...) {
    set_error("Unknown error");
    return CUDF_ERROR_UNKNOWN;
  }
}
}  // namespace

// Exception-catching helper used by all C shim functions.
// Returns the appropriate cudf_status_t for the caught exception.
cudf_status_t translate_exception() { return translate_exception_impl(CUDF_ERROR_UNKNOWN); }

cudf_status_t translate_io_exception() { return translate_exception_impl(CUDF_ERROR_IO); }

// Macro used throughout the C shim to wrap function bodies.
// Usage:
//   CUDF_RUST_TRY {
//     ... C++ code ...
//     return CUDF_SUCCESS;
//   } CUDF_RUST_CATCH
//
// Defined in the internal header below.
