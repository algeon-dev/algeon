//! Community detection and component-labeling parameter contracts.

/// Louvain community detection parameters.
///
/// Mirrors `cugraph::community::LouvainParams` without the `cugraph` crate's
/// new type wrappers and without `do_expensive_check` (always false in the
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
/// new type wrappers and without `do_expensive_check` (always false in the adapter).
#[derive(Debug, Clone, PartialEq)]
pub struct GraphLeidenParams {
    /// Maximum hierarchy depth / iteration count. Must be >= 1.
    pub max_level: u32,
    /// Resolution parameter (`gamma`) in the modularity formula. Must be > 0.
    pub resolution: f64,
    /// RNG seed for the Leiden algorithm. Default 0 gives a deterministic run.
    pub seed: u64,
}

impl Default for GraphLeidenParams {
    fn default() -> Self {
        // Matches cugraph::community::LeidenParams::default().
        Self { max_level: 100, resolution: 1.0, seed: 0 }
    }
}

/// Parameters for ECG community detection.
///
/// Mirrors `cugraph::community::EcgParams` without the `cugraph` crate's
/// new type wrappers and without `do_expensive_check` (always false in the adapter).
#[derive(Debug, Clone, PartialEq)]
pub struct GraphEcgParams {
    /// Minimum edge weight in the final Louvain graph. Must be >= 0.
    pub min_weight: f64,
    /// Number of ensemble Louvain iterations. Must be >= 1.
    pub ensemble_size: u32,
    /// Maximum hierarchy depth for final Louvain. Must be >= 1.
    pub max_level: u32,
    /// Convergence threshold for final Louvain. Must be in (0, 1].
    pub threshold: f64,
    /// Resolution parameter in the modularity formula. Must be in (0, 1].
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
/// new type wrappers and without `do_expensive_check` (always false in the
/// adapter). libcugraph ignores the eigensolver and k-means iteration limits
/// and the k-means tolerance, so they are not part of the contract.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphSpectralClusteringParams {
    /// Desired number of clusters. Must be >= 2.
    pub n_clusters: u32,
    /// Number of eigenvectors to use. Must be >= 1.
    pub n_eigenvectors: u32,
    /// Eigenvalue-solver tolerance. Must be in `[0, 1)`.
    pub evs_tolerance: f64,
    /// RNG seed. Default 0.
    ///
    /// The seed only fixes the random initialization (Lanczos start vector
    /// and k-means centroid seeding). It does not make the partition
    /// bit-identical across runs: GPU parallel reductions inside k-means
    /// accumulate in scheduling order, so same-seed replays may land in a
    /// different but equally valid community assignment.
    pub seed: u64,
}

impl Default for GraphSpectralClusteringParams {
    fn default() -> Self {
        // Matches cugraph::community::SpectralClusteringParams::default().
        Self { n_clusters: 2, n_eigenvectors: 2, evs_tolerance: 0.01, seed: 0 }
    }
}
