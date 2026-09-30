use super::common::MlRelationBinding;

#[derive(Debug, Clone, PartialEq)]
pub struct MlLogisticRegressionParams {
    pub training: MlRelationBinding,
    pub predict: MlRelationBinding,
    pub penalty_l1: f64,
    pub penalty_l2: f64,
    pub max_iter: usize,
    pub linesearch_max_iter: usize,
    pub lbfgs_memory: usize,
    pub fit_intercept: bool,
    pub penalty_normalized: bool,
    pub grad_tol: f64,
    pub change_tol: f64,
}

impl MlLogisticRegressionParams {
    pub fn new(training: MlRelationBinding, predict: MlRelationBinding) -> Self {
        Self {
            training,
            predict,
            penalty_l1: 0.0,
            penalty_l2: 0.0,
            max_iter: 1000,
            linesearch_max_iter: 50,
            lbfgs_memory: 5,
            fit_intercept: true,
            penalty_normalized: true,
            grad_tol: 1e-4,
            change_tol: 1e-5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MlSplitCriterion {
    Gini,
    Entropy,
}

impl MlSplitCriterion {}

#[derive(Debug, Clone, PartialEq)]
pub struct MlRandomForestParams {
    pub training: MlRelationBinding,
    pub predict: MlRelationBinding,
    pub n_trees: usize,
    pub max_depth: i32,
    pub max_leaves: Option<u32>,
    pub max_features: f64,
    pub max_samples: f64,
    pub bootstrap: bool,
    pub max_n_bins: usize,
    pub min_samples_leaf: usize,
    pub min_samples_split: usize,
    pub min_impurity_decrease: f64,
    pub max_batch_size: usize,
    pub seed: u64,
    pub criterion: Option<MlSplitCriterion>,
}

impl MlRandomForestParams {
    pub fn new(training: MlRelationBinding, predict: MlRelationBinding) -> Self {
        Self {
            training,
            predict,
            n_trees: 100,
            max_depth: 16,
            max_leaves: None,
            max_features: 1.0,
            max_samples: 1.0,
            bootstrap: true,
            max_n_bins: 128,
            min_samples_leaf: 1,
            min_samples_split: 2,
            min_impurity_decrease: 0.0,
            max_batch_size: 4096,
            seed: 0,
            criterion: Some(MlSplitCriterion::Gini),
        }
    }
}
