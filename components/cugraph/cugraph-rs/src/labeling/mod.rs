//! Vertex labeling algorithms.
//!
//! Connected-components algorithms return per-vertex integer labels. Component
//! IDs are native cuGraph labels: they are not guaranteed to be consecutive or
//! ordered by component size.

pub mod connected_components;
mod result;

pub use connected_components::ConnectedComponentsParams;
pub use result::{StronglyConnectedComponentsResult, WeaklyConnectedComponentsResult};
