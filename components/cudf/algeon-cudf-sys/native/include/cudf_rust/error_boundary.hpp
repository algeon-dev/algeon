#ifndef CUDF_RUST_MEMORY_ERROR_BOUNDARY_HPP
#define CUDF_RUST_MEMORY_ERROR_BOUNDARY_HPP

#include "cudf_rust/memory_resource.h"

#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace cudf::rust {

/// A shim ownership or lifetime invariant failed. The exception boundary
/// returns `CUDF_ERROR_INVARIANT_VIOLATION`; caller-input preconditions keep
/// their own exception types.
class invariant_error final : public std::logic_error {
 public:
  using std::logic_error::logic_error;
};

}  // namespace cudf::rust

namespace cudf::rust::memory {

/// Stable identity copied into allocation-failure diagnostics.
struct memory_failure_domain_identity {
  cudf_memory_failure_domain_t kind = CUDF_MEMORY_FAILURE_DOMAIN_NONE;
  std::uint64_t primary_id          = 0;
  std::uint64_t secondary_id        = 0;
  /// Monotonically minted by the native query-domain owner. It is never a
  /// pointer identity and is required before the engine may correlate an
  /// event to an attempt.
  std::uint64_t generation = 0;
};

/// RMM allocation failure carrying the owner attribution known at the throw
/// site. The defaults keep existing native throw sites source-compatible while
/// allowing allocation owners to provide request facts when available.
class attributed_out_of_memory final : public rmm::out_of_memory {
 public:
  attributed_out_of_memory(char const* message,
                           cudf_memory_failure_source_t source,
                           memory_failure_domain_identity identity,
                           cudaMemPool_t pool = nullptr,
                           std::size_t requested_bytes = 0,
                           std::size_t alignment       = 0,
                           std::int32_t cuda_error     = -1)
    : rmm::out_of_memory{message},
      source_{source},
      identity_{identity},
      pool_{pool},
      requested_bytes_{requested_bytes},
      alignment_{alignment},
      cuda_error_{cuda_error}
  {
  }

  [[nodiscard]] cudf_memory_failure_source_t source() const noexcept { return source_; }
  [[nodiscard]] memory_failure_domain_identity identity() const noexcept { return identity_; }
  [[nodiscard]] cudaMemPool_t pool() const noexcept { return pool_; }
  [[nodiscard]] std::size_t requested_bytes() const noexcept { return requested_bytes_; }
  [[nodiscard]] std::size_t alignment() const noexcept { return alignment_; }
  [[nodiscard]] std::int32_t cuda_error() const noexcept { return cuda_error_; }

 private:
  cudf_memory_failure_source_t source_;
  memory_failure_domain_identity identity_;
  cudaMemPool_t pool_;
  std::size_t requested_bytes_;
  std::size_t alignment_;
  std::int32_t cuda_error_;
};

}  // namespace cudf::rust::memory

#endif  // CUDF_RUST_MEMORY_ERROR_BOUNDARY_HPP
