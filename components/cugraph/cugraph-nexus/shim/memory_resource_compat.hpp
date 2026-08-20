#ifndef CUDF_RUST_MEMORY_RESOURCE_COMPAT_HPP
#define CUDF_RUST_MEMORY_RESOURCE_COMPAT_HPP

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/export.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <utility>

RMM_NAMESPACE_BEGIN
namespace mr {

/// Project-local compatibility base preserving the RMM resource pointer and
/// vtable identity shared by cuDF and cuGraph interop.
class device_memory_resource {
 public:
  device_memory_resource()                                         = default;
  virtual ~device_memory_resource()                                = default;
  device_memory_resource(device_memory_resource const&)            = delete;
  device_memory_resource& operator=(device_memory_resource const&) = delete;

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return do_allocate(bytes, rmm::cuda_stream_view{stream}, alignment);
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    do_deallocate(ptr, bytes, rmm::cuda_stream_view{stream}, alignment);
  }

  [[nodiscard]] bool operator==(device_memory_resource const& other) const noexcept
  {
    return do_is_equal(other);
  }

  [[nodiscard]] bool operator!=(device_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }

  friend void get_property(device_memory_resource const&, cuda::mr::device_accessible) noexcept {}

 protected:
  virtual void* do_allocate(std::size_t bytes,
                            rmm::cuda_stream_view stream,
                            std::size_t alignment) = 0;

  virtual void do_deallocate(void* ptr,
                             std::size_t bytes,
                             rmm::cuda_stream_view stream,
                             std::size_t alignment) noexcept = 0;

  [[nodiscard]] virtual bool do_is_equal(device_memory_resource const& other) const noexcept = 0;
};

class device_memory_resource_pointer {
 public:
  explicit device_memory_resource_pointer(device_memory_resource* resource) : resource_{resource} {}

  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return resource_->allocate(stream, bytes, alignment);
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto* ptr = resource_->allocate(cuda::stream_ref{cudaStream_t{nullptr}}, bytes, alignment);
    cudaStreamSynchronize(cudaStream_t{nullptr});
    return ptr;
  }

  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    resource_->deallocate(stream, ptr, bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    resource_->deallocate(cuda::stream_ref{cudaStream_t{nullptr}}, ptr, bytes, alignment);
  }

  [[nodiscard]] bool operator==(device_memory_resource_pointer const& other) const noexcept
  {
    return resource_ == other.resource_;
  }

  [[nodiscard]] bool operator!=(device_memory_resource_pointer const& other) const noexcept
  {
    return !(*this == other);
  }

  friend void get_property(device_memory_resource_pointer const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

 private:
  device_memory_resource* resource_;
};

inline cuda::mr::any_resource<cuda::mr::device_accessible> any_device_resource_from_raw(
  device_memory_resource* resource)
{
  return cuda::mr::any_resource<cuda::mr::device_accessible>{
    device_memory_resource_pointer{resource}};
}

template <typename Resource>
class device_memory_resource_box final : public device_memory_resource {
 public:
  template <typename... Args>
  explicit device_memory_resource_box(Args&&... args) : resource_{std::forward<Args>(args)...}
  {
  }

  [[nodiscard]] Resource& resource() noexcept { return resource_; }
  [[nodiscard]] Resource const& resource() const noexcept { return resource_; }

 private:
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream, std::size_t alignment) override
  {
    return resource_.allocate(stream, bytes, alignment);
  }

  void do_deallocate(void* ptr,
                     std::size_t bytes,
                     rmm::cuda_stream_view stream,
                     std::size_t alignment) noexcept override
  {
    resource_.deallocate(stream, ptr, bytes, alignment);
  }

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return this == std::addressof(other);
  }

  Resource resource_;
};

}  // namespace mr
RMM_NAMESPACE_END

#endif  // CUDF_RUST_MEMORY_RESOURCE_COMPAT_HPP
