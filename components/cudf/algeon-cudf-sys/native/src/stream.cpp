#include "cudf_rust/stream.h"

#include "cudf_rust/error_boundary.hpp"
#include "execution_context_internal.hpp"
#include "handle_access.hpp"
#include "shim_common.h"

#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
struct cudf_stream_owner {
  rmm::cuda_stream stream;
  int32_t device = -1;
};

struct stream_handle_state {
  std::shared_ptr<cudf_stream_owner> owner;
};

struct attempt_stream_route {
  // The last entry is the parent's reserved fork; the preceding entries are
  // the auxiliary forks aliases claim. A dynamically registered execution
  // scope registers the parent as its own single fork, so its reserved fork is
  // the parent itself and it has no auxiliary forks.
  std::vector<cudaStream_t> fork_streams;
  std::unordered_set<cudaStream_t> canonical_streams;
  // Live auxiliary-fork claims, keyed by the alias that owns them. This is the
  // only place a claim is recorded: fork streams' own registry entries are
  // never mutated, so two aliases can never race to set or clear one another's
  // confinement, and a detach releases exactly the claim it made.
  std::unordered_map<cudaStream_t, std::size_t> alias_fork_claims;
  std::size_t execution_scopes = 0;
  bool persistent_registration = true;

  // Index of the fork reserved for the parent stream. Aliases never claim it,
  // so the root's kernels never share a stream with a lane or a source worker.
  std::size_t reserved_fork_index() const { return fork_streams.size() - 1; }

  // Number of forks an alias may claim: every fork but the parent's reserved
  // one. This is the route's `S_d`.
  std::size_t auxiliary_fork_count() const { return fork_streams.size() - 1; }
};

// One registry entry per fork parent. A claiming alias additionally appears in
// its route's `alias_fork_claims`.
struct attempt_stream_route_entry {
  std::shared_ptr<attempt_stream_route> route;
};

std::mutex attempt_stream_routes_mutex;
std::unordered_map<cudaStream_t, attempt_stream_route_entry> attempt_stream_routes;

// The single fork of `route` that serves `stream`: a pure lookup, because
// every active caller owns a fixed fork for its lifetime.
//
// (a) `stream` is an alias: the fork it claimed at attach. (b) `stream` is a
// fork stream: the same fork, so work libcudf placed on a caller's fork keeps
// forking inside that caller. (c) `stream` is the parent: its reserved fork.
//
// There is no unclaimed-set computation and no whole-pool fallback: a caller
// that could be served a fork another live caller is launching kernels on
// would be exactly the defect this routing exists to prevent.
cudaStream_t resolve_fork_stream(attempt_stream_route const& route, cudaStream_t stream)
{
  auto const claimed = route.alias_fork_claims.find(stream);
  if (claimed != route.alias_fork_claims.end()) { return route.fork_streams[claimed->second]; }
  for (std::size_t index = 0; index < route.fork_streams.size(); ++index) {
    if (route.fork_streams[index] == stream) { return stream; }
  }
  return route.fork_streams[route.reserved_fork_index()];
}

std::vector<rmm::cuda_stream_view> attempt_stream_provider_fork(rmm::cuda_stream_view parent,
                                                                std::size_t count)
{
  cudaStream_t served = nullptr;
  {
    std::lock_guard<std::mutex> const lock(attempt_stream_routes_mutex);
    auto const it = attempt_stream_routes.find(parent.value());
    if (it == attempt_stream_routes.end()) {
      throw cudf::rust::invariant_error{
        "stream routing invariant: fork parent is not attached to an admitted attempt"};
    }
    served = resolve_fork_stream(*it->second.route, parent.value());
  }

  if (served == nullptr) {
    throw cudf::rust::invariant_error{
      "stream routing invariant: admitted attempt has no auxiliary streams"};
  }
  std::vector<rmm::cuda_stream_view> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    result.emplace_back(served);
  }
  return result;
}

void install_attempt_stream_provider()
{
  cudf::detail::install_stream_provider(&attempt_stream_provider_fork);
}

void register_attempt_stream_route(void* parent_stream,
                                   void* const* fork_streams,
                                   std::size_t fork_count)
{
  if (parent_stream == nullptr) { throw std::invalid_argument{"parent stream is null"}; }
  // One reserved fork for the parent plus at least one auxiliary fork: a
  // route with no auxiliary fork could admit no alias at all.
  if (fork_count < 2) { throw std::invalid_argument{"auxiliary stream set is empty"}; }
  if (fork_streams == nullptr) {
    throw std::invalid_argument{"auxiliary stream array is null"};
  }

  auto const parent = reinterpret_cast<cudaStream_t>(parent_stream);
  std::vector<cudaStream_t> keys;
  keys.reserve(fork_count + 1);
  keys.push_back(parent);
  std::vector<cudaStream_t> fork_streams_owned;
  fork_streams_owned.reserve(fork_count);
  std::unordered_set<cudaStream_t> unique_keys;
  unique_keys.reserve(fork_count + 1);
  unique_keys.insert(parent);
  for (std::size_t index = 0; index < fork_count; ++index) {
    auto const stream = reinterpret_cast<cudaStream_t>(fork_streams[index]);
    if (stream == nullptr || !unique_keys.insert(stream).second) {
      throw std::invalid_argument{"attempt stream route contains a null or duplicate stream"};
    }
    keys.push_back(stream);
    fork_streams_owned.push_back(stream);
  }

  auto route = std::make_shared<attempt_stream_route>();
  route->fork_streams = std::move(fork_streams_owned);
  route->canonical_streams.insert(keys.begin(), keys.end());

  std::lock_guard<std::mutex> const lock(attempt_stream_routes_mutex);
  for (auto const stream : keys) {
    if (attempt_stream_routes.find(stream) != attempt_stream_routes.end()) {
      throw cudf::logic_error{"attempt stream route is already registered"};
    }
  }
  std::vector<cudaStream_t> inserted;
  inserted.reserve(keys.size());
  try {
    for (auto const stream : keys) {
      auto const [it, inserted_new] = attempt_stream_routes.emplace(
        stream, attempt_stream_route_entry{route});
      if (!inserted_new) {
        for (auto const inserted_stream : inserted) { attempt_stream_routes.erase(inserted_stream); }
        throw cudf::logic_error{"attempt stream route is already registered"};
      }
      static_cast<void>(it);
      inserted.push_back(stream);
    }
  } catch (...) {
    for (auto const inserted_stream : inserted) { attempt_stream_routes.erase(inserted_stream); }
    throw;
  }
}

void enter_execution_stream_route(rmm::cuda_stream_view parent)
{
  install_attempt_stream_provider();
  std::lock_guard<std::mutex> const lock(attempt_stream_routes_mutex);
  auto const existing = attempt_stream_routes.find(parent.value());
  if (existing != attempt_stream_routes.end()) {
    existing->second.route->execution_scopes += 1;
    return;
  }

  auto route                     = std::make_shared<attempt_stream_route>();
  route->fork_streams            = {parent.value()};
  route->canonical_streams       = {parent.value()};
  route->execution_scopes        = 1;
  route->persistent_registration = false;
  attempt_stream_routes.emplace(parent.value(), attempt_stream_route_entry{std::move(route)});
}

void exit_execution_stream_route(rmm::cuda_stream_view parent) noexcept
{
  std::lock_guard<std::mutex> const lock(attempt_stream_routes_mutex);
  auto const found = attempt_stream_routes.find(parent.value());
  if (found == attempt_stream_routes.end()) { return; }
  auto const route = found->second.route;
  if (route->execution_scopes != 0) { route->execution_scopes -= 1; }
  if (route->execution_scopes != 0 || route->persistent_registration) { return; }
  for (auto it = attempt_stream_routes.begin(); it != attempt_stream_routes.end();) {
    if (it->second.route == route) {
      it = attempt_stream_routes.erase(it);
    } else {
      ++it;
    }
  }
}

// Lowest auxiliary fork index of `route` that no live alias claims, or
// `auxiliary_fork_count()` when every auxiliary fork is claimed.
//
// The route, not the caller, picks the index: `alias_fork_claims` is the only
// record of which forks are live, so a caller could only guess, and two callers
// guessing the same free index would race. Choosing here, under the routes
// mutex and in the same critical section as the attach, makes the claim atomic.
//
// A caller holding a freshly obtained attempt auxiliary permit always finds
// one. Every claim is backed by one such permit and nothing else claims, so
// live claims never exceed the attempt's permit holders, which admission caps
// at `S_d` -- the auxiliary fork count. The caller's own permit is one of
// those, so at most `S_d - 1` forks can be claimed when it asks.
std::size_t lowest_unclaimed_fork(attempt_stream_route const& route)
{
  for (std::size_t index = 0; index < route.auxiliary_fork_count(); ++index) {
    auto const is_claimed =
      std::any_of(route.alias_fork_claims.begin(),
                  route.alias_fork_claims.end(),
                  [index](auto const& entry) { return entry.second == index; });
    if (!is_claimed) { return index; }
  }
  return route.auxiliary_fork_count();
}

void attach_attempt_stream_route(void* parent_stream,
                                 void* alias_stream,
                                 std::size_t* out_claimed_fork_index)
{
  if (parent_stream == nullptr || alias_stream == nullptr) {
    throw std::invalid_argument{"attempt stream route parent and alias must be non-null"};
  }
  auto const parent = reinterpret_cast<cudaStream_t>(parent_stream);
  auto const alias  = reinterpret_cast<cudaStream_t>(alias_stream);
  if (parent == alias) { throw std::invalid_argument{"attempt stream route alias duplicates its parent"}; }

  std::lock_guard<std::mutex> const lock(attempt_stream_routes_mutex);
  auto const parent_it = attempt_stream_routes.find(parent);
  if (parent_it == attempt_stream_routes.end()) {
    throw cudf::logic_error{"attempt stream route parent is not registered"};
  }
  if (attempt_stream_routes.find(alias) != attempt_stream_routes.end()) {
    throw cudf::logic_error{"attempt stream route alias is already registered"};
  }
  auto route                      = parent_it->second.route;
  std::size_t const claimed_index = lowest_unclaimed_fork(*route);
  if (claimed_index == route->auxiliary_fork_count()) {
    throw cudf::logic_error{"every attempt auxiliary fork is already claimed by a live alias"};
  }
  try {
    attempt_stream_routes.emplace(alias, attempt_stream_route_entry{route});
    route->alias_fork_claims.emplace(alias, claimed_index);
  } catch (...) {
    route->alias_fork_claims.erase(alias);
    attempt_stream_routes.erase(alias);
    throw;
  }
  if (out_claimed_fork_index != nullptr) { *out_claimed_fork_index = claimed_index; }
}

void detach_attempt_stream_route(void* alias_stream)
{
  if (alias_stream == nullptr) { throw std::invalid_argument{"attempt stream route alias is null"}; }
  auto const alias = reinterpret_cast<cudaStream_t>(alias_stream);
  std::lock_guard<std::mutex> const lock(attempt_stream_routes_mutex);
  auto const it = attempt_stream_routes.find(alias);
  if (it == attempt_stream_routes.end()) {
    throw cudf::logic_error{"attempt stream route alias is not registered"};
  }
  auto const route = it->second.route;
  if (route->canonical_streams.contains(alias)) {
    throw cudf::logic_error{"attempt stream route canonical stream cannot be detached as an alias"};
  }
  route->alias_fork_claims.erase(alias);
  attempt_stream_routes.erase(it);
}

void unregister_attempt_stream_route(void* parent_stream)
{
  if (parent_stream == nullptr) { throw std::invalid_argument{"parent stream is null"}; }
  auto const parent = reinterpret_cast<cudaStream_t>(parent_stream);
  std::lock_guard<std::mutex> const lock(attempt_stream_routes_mutex);
  auto const parent_it = attempt_stream_routes.find(parent);
  if (parent_it == attempt_stream_routes.end()) {
    throw cudf::logic_error{"attempt stream route is not registered"};
  }
  auto const route = parent_it->second.route;
  route->persistent_registration = false;
  if (route->execution_scopes != 0) { return; }
  for (auto it = attempt_stream_routes.begin(); it != attempt_stream_routes.end();) {
    if (it->second.route == route) {
      it = attempt_stream_routes.erase(it);
    } else {
      ++it;
    }
  }
}

stream_handle_state& cudf_stream_state_from_handle(cudf_stream_handle handle)
{
  return *reinterpret_cast<stream_handle_state*>(handle);
}
}  // namespace

void cudf_stream_provider_enter_execution(rmm::cuda_stream_view parent)
{
  enter_execution_stream_route(parent);
}

void cudf_stream_provider_exit_execution(rmm::cuda_stream_view parent) noexcept
{
  exit_execution_stream_route(parent);
}

rmm::cuda_stream_view cudf_stream_view_from_handle(cudf_stream_handle handle)
{
  return cudf_stream_state_from_handle(handle).owner->stream.view();
}

int32_t cudf_stream_device_from_handle(cudf_stream_handle handle)
{
  return cudf_stream_state_from_handle(handle).owner->device;
}

std::shared_ptr<void> cudf_stream_lifetime_from_handle(cudf_stream_handle handle)
{
  return cudf_stream_state_from_handle(handle).owner;
}

extern "C" {
/// Create a new CUDA stream.
cudf_status_t cudf_stream_create(cudf_stream_handle* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_TRY
  {
    auto state     = std::make_unique<stream_handle_state>();
    state->owner   = std::make_shared<cudf_stream_owner>();
    int32_t device = 0;
    CUDF_CUDA_TRY(cudaGetDevice(&device));
    state->owner->device = device;
    *out                 = reinterpret_cast<cudf_stream_handle>(state.release());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Destroy a CUDA stream handle.
cudf_status_t cudf_stream_destroy(cudf_stream_handle handle)
{
  try {
    // Ownership is consumed on call. Shared execution contexts may keep the
    // underlying CUDA stream owner alive after this handle is reclaimed.
    delete reinterpret_cast<stream_handle_state*>(handle);
    return CUDF_SUCCESS;
  } catch (...) {
    return cudf_translate_cleanup_exception_noexcept();
  }
}

/// Synchronize a CUDA stream (block until all operations complete).
cudf_status_t cudf_stream_synchronize(cudf_stream_handle handle)
{
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto stream = cudf_stream_view_from_handle(handle);
    cudf::rust::test_support::record_stream_synchronize();
    if (cudf::rust::test_support::stream_synchronize_should_fail()) {
      set_error("forced stream synchronization failure for tests");
      return CUDF_ERROR_CUDA;
    }
    cudf::rust::test_support::wait_for_stream_synchronize_release();
    stream.synchronize();
    cudf_record_synchronized_stream(stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Synchronize a raw borrowed CUDA stream pointer.
cudf_status_t cudf_raw_cuda_stream_synchronize(void* raw_stream)
{
  CUDF_RUST_NULL_CHECK(raw_stream, "raw_stream is null");
  CUDF_RUST_TRY
  {
    auto stream = rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(raw_stream)};
    stream.synchronize();
    cudf_record_synchronized_stream(stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Check if all operations on the stream have completed (non-blocking).
/// Sets *out to 1 if ready, 0 if not.
cudf_status_t cudf_stream_is_ready(cudf_stream_handle handle, int32_t* out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = 0;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    auto stream     = cudf_stream_view_from_handle(handle);
    cudaError_t err = cudaStreamQuery(stream.value());
    if (err == cudaSuccess) {
      *out = 1;

    } else if (err == cudaErrorNotReady) {
      *out = 0;

    } else {
      auto msg = std::string("CUDA stream query failed: ") + cudaGetErrorString(err);
      set_error(msg);
      return CUDF_ERROR_CUDA;
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

cudf_status_t cudf_stream_raw(cudf_stream_handle handle, void** out)
{
  CUDF_RUST_NULL_CHECK(out, "out is null");
  *out = nullptr;
  CUDF_RUST_NULL_CHECK(handle, "handle is null");
  CUDF_RUST_TRY
  {
    *out = reinterpret_cast<void*>(cudf_stream_view_from_handle(handle).value());
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Install the Algeon attempt-owned fork-stream provider.
cudf_status_t cudf_stream_provider_install(void)
{
  CUDF_RUST_TRY
  {
    install_attempt_stream_provider();
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Register one attempt's parent stream and bounded fork stream set.
cudf_status_t cudf_stream_provider_register_attempt(void* parent_stream,
                                                    void* const* fork_streams,
                                                    size_t fork_count)
{
  CUDF_RUST_TRY
  {
    install_attempt_stream_provider();
    register_attempt_stream_route(parent_stream, fork_streams, fork_count);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Remove one attempt's stream route after all work has drained.
cudf_status_t cudf_stream_provider_unregister_attempt(void* parent_stream)
{
  CUDF_RUST_TRY
  {
    unregister_attempt_stream_route(parent_stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Attach one attempt-member stream as another valid fork parent.
cudf_status_t cudf_stream_provider_attach_attempt_stream(void* parent_stream,
                                                         void* alias_stream,
                                                         size_t* out_claimed_fork_index)
{
  CUDF_RUST_TRY
  {
    attach_attempt_stream_route(parent_stream, alias_stream, out_claimed_fork_index);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

/// Remove one attempt-member stream alias.
cudf_status_t cudf_stream_provider_detach_attempt_stream(void* alias_stream)
{
  CUDF_RUST_TRY
  {
    detach_attempt_stream_route(alias_stream);
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}

#ifdef CUDF_RUST_ENABLE_TEST_HOOKS
/// Serve `count` forks for `stream` exactly as the installed provider does.
///
/// The route registry is the only state read; the returned raw values are
/// never dereferenced, so this observes routing without CUDA work.
cudf_status_t cudf_rust_test_support_stream_provider_fork(void* stream,
                                                          size_t count,
                                                          void** out_forks)
{
  CUDF_RUST_NULL_CHECK(out_forks, "out_forks is null");
  CUDF_RUST_NULL_CHECK(stream, "stream is null");
  CUDF_RUST_TRY
  {
    auto const forks = attempt_stream_provider_fork(
      rmm::cuda_stream_view{reinterpret_cast<cudaStream_t>(stream)}, count);
    for (std::size_t index = 0; index < count; ++index) {
      out_forks[index] = reinterpret_cast<void*>(forks[index].value());
    }
    return CUDF_SUCCESS;
  }
  CUDF_RUST_CATCH
}
#endif  // CUDF_RUST_ENABLE_TEST_HOOKS
}  // extern "C"
