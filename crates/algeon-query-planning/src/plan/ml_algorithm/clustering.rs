use std::fmt;

use super::{common::MlRelationBinding, definition::CumlOperation};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Rule for selecting clusters from the HDBSCAN hierarchy.
pub enum MlClusterSelectionMethod {
    /// Selects clusters by excess of mass.
    Eom,
    /// Selects leaf clusters in the hierarchy.
    Leaf,
}

impl MlClusterSelectionMethod {
    /// Canonical lowercase option value accepted by the SQL operation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Eom => "eom",
            Self::Leaf => "leaf",
        }
    }
}

impl fmt::Display for MlClusterSelectionMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Initial coordinates used before UMAP embedding optimization.
pub enum MlUmapInit {
    /// Initializes coordinates from a spectral embedding.
    Spectral,
    /// Initializes coordinates randomly using the configured seed.
    Random,
}

impl MlUmapInit {
    /// Canonical lowercase option value accepted by the SQL operation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Spectral => "spectral",
            Self::Random => "random",
        }
    }
}

impl fmt::Display for MlUmapInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Query-local HDBSCAN bindings and cluster-selection parameters.
pub struct MlHdbscanParams {
    /// Dense rows fitted and clustered by this invocation.
    pub input: MlRelationBinding,
    /// Minimum core-distance neighborhood size; must be below nonempty input row count.
    pub min_samples: usize,
    /// Minimum selected cluster size, in rows.
    pub min_cluster_size: usize,
    /// Maximum selected cluster size, in rows; zero disables this bound.
    pub max_cluster_size: usize,
    /// Nonnegative distance threshold used when selecting clusters.
    pub cluster_selection_epsilon: f64,
    /// Positive scale applied to mutual-reachability distances.
    pub alpha: f64,
    /// Whether one selected cluster may cover all input rows.
    pub allow_single_cluster: bool,
    /// Rule for extracting clusters from the fitted hierarchy.
    pub cluster_selection_method: MlClusterSelectionMethod,
}

impl MlHdbscanParams {
    /// Uses the operation registry's defaults for the supplied input binding.
    ///
    /// The constructor does not validate the input schema or data-dependent limits.
    pub fn new(input: MlRelationBinding) -> Self {
        let descriptor = CumlOperation::Hdbscan.descriptor();
        let integer = |name| {
            descriptor.option(name).and_then(|entry| entry.kind.default_u64()).expect("HDBSCAN integer default")
                as usize
        };
        let number =
            |name| descriptor.option(name).and_then(|entry| entry.kind.default_f64()).expect("HDBSCAN number default");
        Self {
            input,
            min_samples: integer("min_samples"),
            min_cluster_size: integer("min_cluster_size"),
            max_cluster_size: integer("max_cluster_size"),
            cluster_selection_epsilon: number("cluster_selection_epsilon"),
            alpha: number("alpha"),
            allow_single_cluster: descriptor
                .option("allow_single_cluster")
                .and_then(|entry| entry.kind.default_bool())
                .expect("HDBSCAN boolean default"),
            cluster_selection_method: MlClusterSelectionMethod::Eom,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Query-local UMAP bindings and embedding-optimization parameters.
pub struct MlUmapParams {
    /// Dense rows fitted and embedded by this invocation.
    pub input: MlRelationBinding,
    /// Number of output embedding dimensions.
    pub n_components: usize,
    /// Neighborhood size used to construct the manifold.
    pub n_neighbors: usize,
    /// Optimization epoch count; zero selects the native default.
    pub n_epochs: usize,
    /// Positive rate used to update embedding coordinates.
    pub learning_rate: f64,
    /// Minimum embedded point distance; must not exceed `spread`.
    pub min_dist: f64,
    /// Positive scale for embedded point distances.
    pub spread: f64,
    /// Fuzzy-set intersection/union mix, in `[0, 1]`.
    pub set_op_mix_ratio: f64,
    /// Nonnegative adjustment to local neighborhood connectivity.
    pub local_connectivity: f64,
    /// Nonnegative weight of repulsive forces during optimization.
    pub repulsion_strength: f64,
    /// Number of negative samples per positive graph edge.
    pub negative_sample_rate: usize,
    /// Initial coordinate construction method.
    pub init: MlUmapInit,
    /// Seed passed to embedding initialization.
    pub random_state: u64,
}

impl MlUmapParams {
    /// Uses the operation registry's defaults for the supplied input binding.
    ///
    /// The constructor does not validate the input schema or data-dependent limits.
    pub fn new(input: MlRelationBinding) -> Self {
        let descriptor = CumlOperation::Umap.descriptor();
        let integer = |name| {
            descriptor.option(name).and_then(|entry| entry.kind.default_u64()).expect("UMAP integer default") as usize
        };
        let number =
            |name| descriptor.option(name).and_then(|entry| entry.kind.default_f64()).expect("UMAP number default");
        Self {
            input,
            n_components: integer("n_components"),
            n_neighbors: integer("n_neighbors"),
            n_epochs: integer("n_epochs"),
            learning_rate: number("learning_rate"),
            min_dist: number("min_dist"),
            spread: number("spread"),
            set_op_mix_ratio: number("set_op_mix_ratio"),
            local_connectivity: number("local_connectivity"),
            repulsion_strength: number("repulsion_strength"),
            negative_sample_rate: integer("negative_sample_rate"),
            init: MlUmapInit::Spectral,
            random_state: descriptor
                .option("random_state")
                .and_then(|entry| entry.kind.default_u64())
                .expect("UMAP seed default"),
        }
    }
}
