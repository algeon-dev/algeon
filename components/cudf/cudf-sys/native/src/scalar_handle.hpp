#ifndef CUDF_RUST_SCALAR_HANDLE_HPP
#define CUDF_RUST_SCALAR_HANDLE_HPP

#include "handle_provenance.hpp"
#include "scalar.h"

#include <cudf/scalar/scalar.hpp>

#include <memory>
#include <utility>

struct cudf_scalar_owner {
  cudf_handle_provenance provenance;
  std::unique_ptr<cudf::scalar> scalar;
};

inline cudf_scalar_handle cudf_release_scalar_handle(std::unique_ptr<cudf::scalar> scalar,
                                                     cudf_handle_provenance provenance)
{
  auto owner        = std::make_unique<cudf_scalar_owner>();
  owner->scalar     = std::move(scalar);
  owner->provenance = std::move(provenance);
  return static_cast<cudf_scalar_handle>(owner.release());
}

inline cudf_scalar_owner& cudf_scalar_owner_from_handle(cudf_scalar_handle handle)
{
  return *static_cast<cudf_scalar_owner*>(handle);
}

inline cudf::scalar& cudf_scalar_from_handle(cudf_scalar_handle handle)
{
  return *cudf_scalar_owner_from_handle(handle).scalar;
}

inline cudf::scalar const& cudf_scalar_from_handle_const(cudf_scalar_handle handle)
{
  return *cudf_scalar_owner_from_handle(handle).scalar;
}

#endif  // CUDF_RUST_SCALAR_HANDLE_HPP
