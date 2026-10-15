//! Vertex labeling algorithms.
//!
//! Connected-components algorithms return per-vertex integer labels. Each
//! label is canonicalized to the minimum vertex ID of its component, ensuring
//! deterministic output in the caller-visible vertex space.

pub mod connected_components;
mod result;

pub use connected_components::ConnectedComponentsParams;
pub use result::LabelingResult;
