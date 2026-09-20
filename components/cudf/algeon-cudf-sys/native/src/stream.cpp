#include "cudf_rust/stream.h"

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

// A contiguous half-open claim on the route's fork pool, owned by one alias.
struct alias_fork_subset {
  std::size_t begin = 0;
  std::size_t end   = 0;

  bool claims(std::size_t index) const { return index >= begin && index < end; }
  bool overlaps(alias_fork_subset const& other) const
  {
    return begin < other.end && other.begin < end;
  }
};

struct attempt_stream_route {
  std::vector<cudaStream_t> fork_streams;
  std::unordered_set<cudaStream_t> canonical_streams;
  std::unordered_set<cudaStream_t> aliases;
  // Live subset claims, keyed by the alias that owns them. This is the only
  // place a subset is recorded: fork streams' own registry entries are never
  // mutated, so two aliases can never race to set or clear one another's
  // confinement, and a detach releases exactly the claim it made.
  std::unordered_map<cudaStream_t, alias_fork_subset> alias_fork_subsets;
  std::size_t execution_scopes = 0;
  bool persistent_registration = true;
};

// One registry entry per fork parent. A subset-carrying alias additionally
// appears in its route's `alias_fork_subsets`.
struct attempt_stream_route_entry {
  std::shared_ptr<attempt_stream_route> route;
};

std::mutex attempt_stream_routes_mutex;
std::unordered_map<cudaStream_t, attempt_stream_route_entry> attempt_stream_routes;

// Which forks of `route` serve `stream`, resolved from the live alias claims
// alone.
//
// (a) `stream` is an alias holding a subset: that subset. (b) `stream` is a
// fork stream: the subset of the live alias that claims its index, so work
// libcudf placed on a lane's fork keeps forking inside that lane. (c) the
// parent or a subset-less alias: every fork no live alias claims. A lane's
// forks belong to the permit holder that leased them, so a whole-pool caller
// must not be served them. Only when every fork is claimed does (c) fall back
// to the whole pool: a lane owning every fork means its attempt has no
// auxiliary permit left for another holder, so no one else can be served.
std::vector<cudaStream_t> resolve_fork_streams(attempt_stream_route const& route, cudaStream_t stream)
{
  auto const claimed = route.alias_fork_subsets.find(stream);
  if (claimed != route.alias_fork_subsets.end()) {
    return {route.fork_streams.begin() + static_cast<std::ptrdiff_t>(claimed->second.begin),
            route.fork_streams.begin() + static_cast<std::ptrdiff_t>(claimed->second.end)};
  }
  for (std::size_t index = 0; index < route.fork_streams.size(); ++index) {
    if (route.fork_streams[index] != stream) { continue; }
    for (auto const& [alias, subset] : route.alias_fork_subsets) {
      static_cast<void>(alias);
      if (subset.claims(index)) {
        return {route.fork_streams.begin() + static_cast<std::ptrdiff_t>(subset.begin),
                route.fork_streams.begin() + static_cast<std::ptrdiff_t>(subset.end)};
      }
    }
    break;
  }
  std::vector<cudaStream_t> unclaimed;
  unclaimed.reserve(route.fork_streams.size());
  for (std::size_t index = 0; index < route.fork_streams.size(); ++index) {
    auto const is_claimed =
      std::any_of(route.alias_fork_subsets.begin(),
                  route.alias_fork_subsets.end(),
                  [index](auto const& entry) { return entry.second.claims(index); });
    if (!is_claimed) { unclaimed.push_back(route.fork_streams[index]); }
  }
  if (unclaimed.empty()) { return route.fork_streams; }
  return unclaimed;
}

std::vector<rmm::cuda_stream_view> attempt_stream_provider_fork(rmm::cuda_stream_view parent,
                                                                std::size_t count)
{
  std::vector<cudaStream_t> served;
  {
    std::lock_guard<std::mutex> const lock(attempt_stream_routes_mutex);
    auto const it = attempt_stream_routes.find(parent.value());
    if (it == attempt_stream_routes.end()) {
      throw cudf::logic_error{
        "stream routing invariant: fork parent is not attached to an admitted attempt"};
    }
    served = resolve_fork_streams(*it->second.route, parent.value());
  }

  CUDF_EXPECTS(count == 0 || !served.empty(),
               "stream routing invariant: admitted attempt has no auxiliary streams");
  std::vector<rmm::cuda_stream_view> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    result.emplace_back(served[index % served.size()]);
  }
  return result;
}

void install_attempt_stream_provider()
{
  cudf::detail::install_stream_provider(&attempt_stream_provider_fork);
}

void register_attempt_stream_route(void* parent_stream,
                                   void* const* fork_streams,
                                   std::size_t auxiliary_count)
{
  if (parent_stream == nullptr) { throw std::invalid_argument{"parent stream is null"}; }
  if (auxiliary_count == 0) { throw std::invalid_argument{"auxiliary stream set is empty"}; }
  if (fork_streams == nullptr) {
    throw std::invalid_argument{"auxiliary stream array is null"};
  }

  auto const parent = reinterpret_cast<cudaStream_t>(parent_stream);
  std::vector<cudaStream_t> keys;
  keys.reserve(auxiliary_count + 1);
  keys.push_back(parent);
  std::vector<cudaStream_t> fork_streams_owned;
  fork_streams_owned.reserve(auxiliary_count);
  std::unordered_set<cudaStream_t> unique_keys;
  unique_keys.reserve(auxiliary_count + 1);
  unique_keys.insert(parent);
  for (std::size_t index = 0; index < auxiliary_count; ++index) {
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

void attach_attempt_stream_route(void* parent_stream,
                                 void* alias_stream,
                                 std::size_t fork_begin,
                                 std::size_t fork_end)
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
  auto route = parent_it->second.route;
  // `0, 0` is the whole-pool alias; any other range must name real forks.
  auto const claims_subset = fork_begin != 0 || fork_end != 0;
  if (claims_subset) {
    if (fork_begin >= fork_end || fork_end > route->fork_streams.size()) {
      throw std::invalid_argument{"alias fork subset is empty or outside the attempt fork pool"};
    }
    // A fork belongs to exactly one live claim, so a lane cannot be served a
    // stream another lane is already using. Equal ranges overlap too: two
    // aliases sharing one range would each release it on detach.
    alias_fork_subset const claim{fork_begin, fork_end};
    for (auto const& [claimed_alias, subset] : route->alias_fork_subsets) {
      static_cast<void>(claimed_alias);
      if (claim.overlaps(subset)) {
        throw cudf::logic_error{"attempt fork stream is already claimed by another alias fork subset"};
      }
    }
  }
  route->aliases.insert(alias);
  try {
    attempt_stream_routes.emplace(alias, attempt_stream_route_entry{route});
    if (claims_subset) { route->alias_fork_subsets.emplace(alias, alias_fork_subset{fork_begin, fork_end}); }
  } catch (...) {
    route->alias_fork_subsets.erase(alias);
    attempt_stream_routes.erase(alias);
    route->aliases.erase(alias);
    throw;
  }
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
  route->alias_fork_subsets.erase(alias);
  route->aliases.erase(alias);
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

/// Register one attempt's parent stream and bounded auxiliary stream set.
cudf_status_t cudf_stream_provider_register_attempt(void* parent_stream,
                                                    void* const* fork_streams,
                                                    size_t auxiliary_count)
{
  CUDF_RUST_TRY
  {
    install_attempt_stream_provider();
    register_attempt_stream_route(parent_stream, fork_streams, auxiliary_count);
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
                                                         size_t fork_begin,
                                                         size_t fork_end)
{
  CUDF_RUST_TRY
  {
    attach_attempt_stream_route(parent_stream, alias_stream, fork_begin, fork_end);
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
