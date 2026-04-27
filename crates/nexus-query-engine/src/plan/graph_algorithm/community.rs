//! Community detection and component-labeling parameter contracts.

/// Louvain community detection parameters.
///
/// Mirrors `cugraph::community::LouvainParams` without the `cugraph` crate's
/// newtype wrappers and without `do_expensive_check` (always false in the
/// adapter).
#[derive(Debug, Clone, PartialEq)]
pub struct GraphLouvainParams {
    /// Maximum hierarchy depth / iteration count. Must be >= 1.
    pub max_level: u32,
    /// Convergence threshold for modularity gain at each level. Must be >= 0.
    pub threshold: f64,
    /// Resolution parameter (`gamma`) in the modularity formula. Must be > 0.
    pub resolution: f64,
}

impl Default for GraphLouvainParams {
    fn default() -> Self {
        // Matches cugraph::community::LouvainParams::default().
        Self { max_level: 100, threshold: 1.0e-7, resolution: 1.0 }
    }
}

/// Zero-argument weakly connected components.
///
/// Component labels follow cuGraph's stable result schema for the algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphWccParams;

/// Zero-argument strongly connected components.
///
/// Component labels follow cuGraph's stable result schema for the algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GraphSccParams;

/// Parameters for Leiden community detection.
///
/// Mirrors `cugraph::community::LeidenParams` without the `cugraph` crate's
/// newtype wrappers and without `do_expensive_check` (always false in the adapter).
#[derive(Debug, Clone, PartialEq)]
pub struct GraphLeidenParams {
    /// Maximum hierarchy depth / iteration count. Must be >= 1.
    pub max_level: u32,
    /// Resolution parameter (`gamma`) in the modularity formula. Must be > 0.
    pub resolution: f64,
    /// Modularity-gain scaling for the refinement stage. Must be > 0.
    pub theta: f64,
    /// RNG seed for the Leiden algorithm. Default 0 gives a deterministic run.
    pub seed: u64,
}

impl Default for GraphLeidenParams {
    fn default() -> Self {
        // Matches cugraph::community::LeidenParams::default().
        Self { max_level: 100, resolution: 1.0, theta: 1.0, seed: 0 }
    }
}

/// Parameters for ECG community detection.
///
/// Mirrors `cugraph::community::EcgParams` without the `cugraph` crate's
/// newtype wrappers and without `do_expensive_check` (always false in the adapter).
#[derive(Debug, Clone, PartialEq)]
pub struct GraphEcgParams {
    /// Minimum edge weight in the final Louvain graph. Must be >= 0.
    pub min_weight: f64,
    /// Number of ensemble Louvain iterations. Must be >= 1.
    pub ensemble_size: u32,
    /// Maximum hierarchy depth for final Louvain. Must be >= 1.
    pub max_level: u32,
    /// Convergence threshold for final Louvain. Must be >= 0.
    pub threshold: f64,
    /// Resolution parameter in the modularity formula. Must be > 0.
    pub resolution: f64,
    /// RNG seed. Default 0 gives a deterministic run.
    pub seed: u64,
}

impl Default for GraphEcgParams {
    fn default() -> Self {
        // Matches cugraph::community::EcgParams::default().
        Self { min_weight: 0.001, ensemble_size: 10, max_level: 100, threshold: 1.0e-7, resolution: 1.0, seed: 0 }
    }
}

/// Parameters for spectral modularity maximization.
///
/// Mirrors `cugraph::community::SpectralClusteringParams` without the `cugraph` crate's
/// newtype wrappers and without `do_expensive_check` (always false in the
/// adapter).
#[derive(Debug, Clone, PartialEq)]
pub struct GraphSpectralClusteringParams {
    /// Desired number of clusters. Must be >= 1.
    pub n_clusters: u32,
    /// Number of eigenvectors to use. Must be >= 1.
    pub n_eigenvectors: u32,
    /// Eigenvalue-solver tolerance. Must be >= 0.
    pub evs_tolerance: f64,
    /// Eigenvalue-solver maximum iteration count. Must be in `[1, i32::MAX]`.
    pub evs_max_iterations: u32,
    /// k-means tolerance. Must be >= 0.
    pub k_means_tolerance: f64,
    /// k-means maximum iteration count. Must be in `[1, i32::MAX]`.
    pub k_means_max_iterations: u32,
    /// RNG seed. Default 0 gives a deterministic run.
    pub seed: u64,
}

impl Default for GraphSpectralClusteringParams {
    fn default() -> Self {
        // Matches cugraph::community::SpectralClusteringParams::default().
        Self {
            n_clusters: 2,
            n_eigenvectors: 2,
            evs_tolerance: 0.01,
            evs_max_iterations: 500,
            k_means_tolerance: 0.001,
            k_means_max_iterations: 100,
            seed: 0,
        }
    }
}
