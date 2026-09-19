//! Vertex labeling algorithms.
//!
//! Connected-components algorithms return per-vertex integer labels. Each
//! label is the minimum vertex id of its component, so labels are run-stable
//! and expressed in the caller-visible vertex space rather than cuGraph's
//! internal hook-race assignment.

pub mod connected_components;
mod result;

pub use connected_components::ConnectedComponentsParams;
pub use result::{StronglyConnectedComponentsResult, WeaklyConnectedComponentsResult};
