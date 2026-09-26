use std::fmt;

use super::{common::MlRelationBinding, definition::CumlOperation};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MlClusterSelectionMethod {
    Eom,
    Leaf,
}

impl MlClusterSelectionMethod {
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
pub enum MlUmapInit {
    Spectral,
    Random,
}

impl MlUmapInit {
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
pub struct MlHdbscanParams {
    pub input: MlRelationBinding,
    pub min_samples: usize,
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub cluster_selection_epsilon: f64,
    pub alpha: f64,
    pub allow_single_cluster: bool,
    pub cluster_selection_method: MlClusterSelectionMethod,
}

impl MlHdbscanParams {
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
pub struct MlUmapParams {
    pub input: MlRelationBinding,
    pub n_components: usize,
    pub n_neighbors: usize,
    pub n_epochs: usize,
    pub learning_rate: f64,
    pub min_dist: f64,
    pub spread: f64,
    pub set_op_mix_ratio: f64,
    pub local_connectivity: f64,
    pub repulsion_strength: f64,
    pub negative_sample_rate: usize,
    pub init: MlUmapInit,
    pub random_state: u64,
}

impl MlUmapParams {
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
