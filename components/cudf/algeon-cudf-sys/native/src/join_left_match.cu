#include "join_common.hpp"

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/scatter.h>

using namespace cudf_rust::join;

namespace {
// cudf::hash_join exposes no per-build-row match marking, so the state probes
// with inner_join and scatters the build-side indices into a byte per left row.
struct hash_left_match_state_owner {
  std::shared_ptr<void> upstream_mr_lifetime;
  std::shared_ptr<void> stream_lifetime;
  std::unique_ptr<statistics_resource_box> observed_mr;
  // The resource object handed to libcudf must outlive every retained device
  // allocation; members destroy in reverse declaration order.
  std::shared_ptr<cuda::mr::any_resource<cuda::mr::device_accessible>> observed_mr_ref;
  std::unique_ptr<cudf::hash_join> join;
  std::unique_ptr<rmm::device_uvector<bool>> matched;
  rmm::mr::device_memory_resource* mr_raw = nullptr;
  cudaStream_t stream                     = nullptr;
  std::optional<bool> sealed_anti;
  std::size_t output_size = 0;
  int32_t device          = 0;
};

struct equals_flag {
  bool selected;
  __device__ bool operator()(bool matched) const { return matched == selected; }
};

void validate_hash_left_match_context(hash_left_match_state_owner const& owner,
                                      cudf_exec_context const& exec)
{
  CUDF_EXPECTS(owner.device == exec.device && owner.stream == exec.stream.value() &&
                 owner.mr_raw == exec.mr_raw,
               "hash left match state requires its original device, stream and memory resource");
}
}  // namespace

extern "C" {
cudf_status_t cudf_hash_left_match_state_create_ex(cudf_table_view_handle left_keys,
                                                   cudf_null_equality_t null_equality,
                                                   cudf_hash_left_match_state_handle* out,
                                                   cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(left_keys, "left keys is null");
  cudf::null_equality nulls;
  auto status = to_null_equality(null_equality, &nulls);
  if (status != CUDF_SUCCESS) { return status; }
  CUDF_RUST_TRY
  {
    std::unique_ptr<cudf_exec_context> exec;
    status = cudf_resolve_required_exec_options(options, &exec);
    if (status != CUDF_SUCCESS) { return status; }
    auto const& left            = *reinterpret_cast<cudf::table_view*>(left_keys);
    auto owner                  = std::make_unique<hash_left_match_state_owner>();
    owner->device               = exec->device;
    owner->stream               = exec->stream.value();
    owner->mr_raw               = exec->mr_raw;
    owner->upstream_mr_lifetime = exec->mr_lifetime;
    owner->stream_lifetime      = exec->stream_lifetime;
    owner->observed_mr          = std::make_unique<statistics_resource_box>(
      cuda::mr::any_resource<cuda::mr::device_accessible>{*exec->mr});
    owner->observed_mr_ref = std::make_shared<cuda::mr::any_resource<cuda::mr::device_accessible>>(
      rmm::mr::any_device_resource_from_raw(owner->observed_mr.get()));
    owner->join =
      std::make_unique<cudf::hash_join>(left, nulls, exec->stream, *owner->observed_mr_ref);
    owner->matched = std::make_unique<rmm::device_uvector<bool>>(
      static_cast<std::size_t>(left.num_rows()), exec->stream, *owner->observed_mr_ref);
    if (!owner->matched->is_empty()) {
      CUDF_CUDA_TRY(cudaMemsetAsync(
        owner->matched->data(), 0, owner->matched->size() * sizeof(bool), exec->stream.value()));
    }
    *out = reinterpret_cast<cudf_hash_left_match_state_handle>(owner.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_left_match_state_consume_ex(cudf_hash_left_match_state_handle state,
                                                    cudf_table_view_handle right_keys,
                                                    cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(state, "state is null");
  CUDF_RUST_NULL_CHECK(right_keys, "right keys is null");
  CUDF_RUST_TRY
  {
    std::unique_ptr<cudf_exec_context> exec;
    auto status = cudf_resolve_required_exec_options(options, &exec);
    if (status != CUDF_SUCCESS) { return status; }
    auto& owner = *reinterpret_cast<hash_left_match_state_owner*>(state);
    validate_hash_left_match_context(owner, *exec);
    CUDF_EXPECTS(!owner.sealed_anti.has_value(), "hash left match state is already sealed");
    auto const& right = *reinterpret_cast<cudf::table_view*>(right_keys);
    if (right.num_rows() == 0 || owner.matched->is_empty()) { return CUDF_SUCCESS; }
    // The pair's second map holds build (left) row indices; both maps are
    // freed on this stream after the scatter is ordered behind them.
    auto const indices = owner.join->inner_join(right, std::nullopt, exec->stream, *exec->mr);
    auto const& left_indices = *indices.second;
    thrust::scatter(rmm::exec_policy_nosync(exec->stream, *exec->mr),
                    thrust::make_constant_iterator(true),
                    thrust::make_constant_iterator(true) + left_indices.size(),
                    left_indices.begin(),
                    owner.matched->begin());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_left_match_state_seal_ex(cudf_hash_left_match_state_handle state,
                                                 cudf_join_kind_t kind,
                                                 cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(state, "state is null");
  CUDF_RUST_TRY
  {
    CUDF_EXPECTS(kind == CUDF_JOIN_KIND_LEFT_SEMI_JOIN || kind == CUDF_JOIN_KIND_LEFT_ANTI_JOIN,
                 "hash left match state only supports semi/anti output");
    std::unique_ptr<cudf_exec_context> exec;
    auto status = cudf_resolve_required_exec_options(options, &exec);
    if (status != CUDF_SUCCESS) { return status; }
    auto& owner = *reinterpret_cast<hash_left_match_state_owner*>(state);
    validate_hash_left_match_context(owner, *exec);
    CUDF_EXPECTS(!owner.sealed_anti.has_value(), "hash left match state is already sealed");
    auto const anti     = kind == CUDF_JOIN_KIND_LEFT_ANTI_JOIN;
    auto const selected = thrust::count(rmm::exec_policy(exec->stream, *exec->mr),
                                        owner.matched->begin(),
                                        owner.matched->end(),
                                        !anti);
    owner.output_size   = static_cast<std::size_t>(selected);
    owner.sealed_anti   = anti;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_left_match_state_output_size(cudf_hash_left_match_state_handle state,
                                                     size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(state, "state is null");
  CUDF_RUST_TRY
  {
    auto const& owner = *reinterpret_cast<hash_left_match_state_owner*>(state);
    CUDF_EXPECTS(owner.sealed_anti.has_value(), "hash left match output size requires seal");
    *out = owner.output_size;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_left_match_state_device_allocated_size(
  cudf_hash_left_match_state_handle state, size_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(state, "state is null");
  CUDF_RUST_TRY
  {
    auto const bytes = reinterpret_cast<hash_left_match_state_owner*>(state)
                         ->observed_mr->resource()
                         .get_bytes_counter()
                         .value;
    *out = bytes > 0 ? static_cast<std::size_t>(bytes) : 0;
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_left_match_state_materialize_indices_ex(
  cudf_hash_left_match_state_handle state,
  cudf_column_handle* out,
  cudf_exec_options_t const* options)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(state, "state is null");
  CUDF_RUST_TRY
  {
    std::unique_ptr<cudf_exec_context> exec;
    auto status = cudf_resolve_required_exec_options(options, &exec);
    if (status != CUDF_SUCCESS) { return status; }
    auto& owner = *reinterpret_cast<hash_left_match_state_owner*>(state);
    validate_hash_left_match_context(owner, *exec);
    CUDF_EXPECTS(owner.sealed_anti.has_value(), "hash left match materialization requires seal");
    auto indices = std::make_unique<rmm::device_uvector<cudf::size_type>>(
      owner.output_size, exec->stream, *exec->mr);
    if (owner.output_size > 0) {
      auto const rows = static_cast<cudf::size_type>(owner.matched->size());
      thrust::copy_if(rmm::exec_policy_nosync(exec->stream, *exec->mr),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator<cudf::size_type>{rows},
                      owner.matched->begin(),
                      indices->begin(),
                      equals_flag{!*owner.sealed_anti});
    }
    *out = cudf_release_column_handle(release_join_index_column(std::move(indices)),
                                      cudf_capture_exec_provenance(*exec));
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_hash_left_match_state_destroy(cudf_hash_left_match_state_handle state)
{
  if (state == nullptr) { return CUDF_SUCCESS; }
  auto* owner = reinterpret_cast<hash_left_match_state_owner*>(state);
  return cudf_destroy_under_provenance(cached_join_owner_provenance(*owner),
                                       [owner] { delete owner; });
}
}  // extern "C"
