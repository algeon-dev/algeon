//! Graph structure queries (degrees, etc.).

pub mod degrees;
mod selection;
pub mod typed_results;
pub use degrees::{DegreeValues, DegreesResult, InDegreesResult, OutDegreesResult};
pub(crate) use selection::ResolvedVertexSelection;
pub use selection::VertexSelection;
pub use typed_results::{Distances, ScoreValues, VertexIds};
