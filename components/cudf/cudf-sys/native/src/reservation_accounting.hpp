#ifndef CUDF_RUST_RESERVATION_ACCOUNTING_HPP
#define CUDF_RUST_RESERVATION_ACCOUNTING_HPP

#include "memory_resource_compat.hpp"

#include <rmm/aligned.hpp>

#include <atomic>
#include <cstddef>
#include <memory>

namespace cudf::rust::reservation {

/// The single allocation-charge rule used by reservation admission and by
/// boundary observers. `alignment` must be the value received by the shim's
/// device_memory_resource callback; measuring before this compatibility
/// boundary can observe a different default alignment.
inline std::size_t allocation_charge(std::size_t bytes, std::size_t alignment) noexcept
{
  return rmm::align_up(bytes, alignment);
}

class statistics_resource_adaptor final : public rmm::mr::device_memory_resource {
 public:
  explicit statistics_resource_adaptor(rmm::mr::device_memory_resource* upstream)
    : upstream_{rmm::mr::any_device_resource_from_raw(upstream)}
  {
  }

  [[nodiscard]] std::size_t current_bytes() const noexcept
  {
    return current_bytes_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::size_t peak_bytes() const noexcept
  {
    return peak_bytes_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::size_t current_aligned_bytes() const noexcept
  {
    return current_aligned_bytes_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] std::size_t peak_aligned_bytes() const noexcept
  {
    return peak_aligned_bytes_.load(std::memory_order_relaxed);
  }

 private:
  static void update_peak(std::atomic<std::size_t>& peak, std::size_t value) noexcept
  {
    auto observed = peak.load(std::memory_order_relaxed);
    while (observed < value &&
           !peak.compare_exchange_weak(observed, value, std::memory_order_relaxed)) {}
  }

  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream, std::size_t alignment) override
  {
    auto* ptr                = upstream_.allocate(stream, bytes, alignment);
    auto const aligned       = allocation_charge(bytes, alignment);
    auto const requested_now = current_bytes_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    auto const aligned_now =
      current_aligned_bytes_.fetch_add(aligned, std::memory_order_relaxed) + aligned;
    update_peak(peak_bytes_, requested_now);
    update_peak(peak_aligned_bytes_, aligned_now);
    return ptr;
  }

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override
  {
    upstream_.deallocate(stream, ptr, bytes, alignment);
    current_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
    current_aligned_bytes_.fetch_sub(allocation_charge(bytes, alignment),
                                     std::memory_order_relaxed);
  }

  [[nodiscard]] bool do_is_equal(
    rmm::mr::device_memory_resource const& other) const noexcept override
  {
    return this == std::addressof(other);
  }

  cuda::mr::any_resource<cuda::mr::device_accessible> upstream_;
  std::atomic<std::size_t> current_bytes_{0};
  std::atomic<std::size_t> peak_bytes_{0};
  std::atomic<std::size_t> current_aligned_bytes_{0};
  std::atomic<std::size_t> peak_aligned_bytes_{0};
};

}  // namespace cudf::rust::reservation

#endif  // CUDF_RUST_RESERVATION_ACCOUNTING_HPP
