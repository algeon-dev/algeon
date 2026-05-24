#include "gds.h"
#include "shim_common.h"

#include <dlfcn.h>

#include <cstdlib>
#include <string>

namespace {
struct cufile_error_abi {
  int err;
  int cu_err;
};

using cu_file_driver_open_fn  = cufile_error_abi (*)();
using cu_file_driver_close_fn = cufile_error_abi (*)();

constexpr int CU_FILE_SUCCESS_ABI = 0;

cudf_status_t cufile_status_to_cudf(cufile_error_abi status, char const* operation)
{
  if (status.err == CU_FILE_SUCCESS_ABI) { return CUDF_SUCCESS; }
  set_error(std::string(operation) + " failed: cuFile err=" + std::to_string(status.err) +
            " cu_err=" + std::to_string(status.cu_err));
  return CUDF_ERROR_IO;
}
}  // namespace

extern "C" {
cudf_status_t cudf_gds_driver_available(void)
{
  CUDF_RUST_TRY
  {
    void* library = dlopen("libcufile.so", RTLD_LAZY | RTLD_LOCAL);
    if (library == nullptr) {
      char const* error = dlerror();
      set_error(std::string("failed to load libcufile.so: ") + (error == nullptr ? "unknown error" : error));
      return CUDF_ERROR_IO;
    }

    auto close_library = [&]() { dlclose(library); };
    dlerror();
    auto* open_symbol = dlsym(library, "cuFileDriverOpen");
    char const* open_error = dlerror();
    if (open_error != nullptr || open_symbol == nullptr) {
      set_error(std::string("failed to resolve cuFileDriverOpen: ") +
                (open_error == nullptr ? "unknown error" : open_error));
      close_library();
      return CUDF_ERROR_IO;
    }

    dlerror();
    auto* close_symbol = dlsym(library, "cuFileDriverClose");
    char const* close_error = dlerror();
    if (close_error != nullptr || close_symbol == nullptr) {
      set_error(std::string("failed to resolve cuFileDriverClose: ") +
                (close_error == nullptr ? "unknown error" : close_error));
      close_library();
      return CUDF_ERROR_IO;
    }

    auto* driver_open  = reinterpret_cast<cu_file_driver_open_fn>(open_symbol);
    auto* driver_close = reinterpret_cast<cu_file_driver_close_fn>(close_symbol);

    cudf_status_t const open_status = cufile_status_to_cudf(driver_open(), "cuFileDriverOpen");
    if (open_status != CUDF_SUCCESS) {
      close_library();
      return open_status;
    }

    cudf_status_t const close_status = cufile_status_to_cudf(driver_close(), "cuFileDriverClose");
    close_library();
    return close_status;
  }
  CUDF_RUST_CATCH
}
}  // extern "C"
