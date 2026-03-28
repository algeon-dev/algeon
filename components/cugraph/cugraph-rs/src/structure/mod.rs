//! Graph structure queries (degrees, etc.).

pub mod degrees;
pub mod typed_results;
pub use degrees::{DegreeValues, DegreeVertices, DegreesResult, InDegreesResult, OutDegreesResult};
pub use typed_results::{Distances, ScoreValues, VertexIds};
