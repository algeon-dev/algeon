//! Graph structure queries (degrees, core numbers, etc.).

pub mod core;
pub mod degrees;
mod selection;
pub(crate) mod subgraph;
pub mod typed_results;
pub use core::{CoreNumberParams, CoreNumberResult, KCoreDegreeType, KCoreParams, KCoreResult};

pub use degrees::{DegreeValues, DegreesResult, InDegreesResult, OutDegreesResult};
pub(crate) use selection::ResolvedVertexSelection;
pub use selection::VertexSelection;
pub use typed_results::{Distances, IntegerValues, ScoreValues, SizeValues, VertexIds};
