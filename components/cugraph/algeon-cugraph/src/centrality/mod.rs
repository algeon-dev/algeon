//! Centrality algorithms (PageRank, betweenness centrality, etc.).

pub mod betweenness;
pub mod edge_betweenness;
pub mod hits;
pub mod pagerank;
mod result;
pub mod vertex;

pub use betweenness::BetweennessCentralityParams;
pub use edge_betweenness::EdgeBetweennessCentralityParams;
pub use hits::{HitsInitialHubs, HitsParams};
pub use pagerank::{
    InitialGuess, PageRankParams, PersonalizationVector, PersonalizedPageRankParams, PrecomputedWeights,
};
pub use result::{BetweennessCentralityResult, CentralityResult, EdgeBetweennessCentralityResult, HitsResult};
pub use vertex::{EigenvectorCentralityParams, KatzCentralityParams};

use crate::{
    error::{Error, ErrorKind},
    params::{IterationCount, NonNegativeF64},
};

/// Typed failure of an iterative centrality that stopped at `max_iterations`.
///
/// The `algorithm`, `iterations`, `max_iterations`, `epsilon`, and `remedy`
/// contexts are the facts a caller needs to change the call: `remedy` names
/// the parameters that make this algorithm converge.
fn non_convergence_error(
    algorithm: &'static str,
    iterations: usize,
    max_iterations: IterationCount,
    epsilon: NonNegativeF64,
    remedy: String,
) -> Error {
    Error::new(ErrorKind::NonConvergence, algorithm)
        .with_message(format!("{algorithm} did not converge within {iterations} iterations; {remedy}"))
        .with_context("algorithm", algorithm)
        .with_context("iterations", iterations.to_string())
        .with_context("max_iterations", usize::from(max_iterations).to_string())
        .with_context("epsilon", epsilon.get().to_string())
        .with_context("remedy", remedy)
}
