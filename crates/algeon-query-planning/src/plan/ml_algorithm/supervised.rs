use super::common::MlRelationBinding;

#[derive(Debug, Clone, PartialEq)]
/// Bindings and quasi-Newton fit settings for a query-local classifier.
pub struct MlLogisticRegressionParams {
    /// Feature rows with the Int32 class label used to fit the model.
    pub training: MlRelationBinding,
    /// Independent feature rows to score; must omit a label binding.
    pub predict: MlRelationBinding,
    /// Nonnegative L1 regularization penalty.
    pub penalty_l1: f64,
    /// Nonnegative L2 regularization penalty.
    pub penalty_l2: f64,
    /// Maximum quasi-Newton optimization iterations.
    pub max_iter: usize,
    /// Maximum iterations for each line search.
    pub linesearch_max_iter: usize,
    /// Number of updates retained in the L-BFGS history.
    pub lbfgs_memory: usize,
    /// Whether fitting includes an intercept term.
    pub fit_intercept: bool,
    /// Whether penalties are normalized by training row count.
    pub penalty_normalized: bool,
    /// Nonnegative convergence tolerance on the optimization gradient.
    pub grad_tol: f64,
    /// Nonnegative convergence tolerance on objective change.
    pub change_tol: f64,
}

impl MlLogisticRegressionParams {
    /// Creates parameters with the current operation defaults.
    ///
    /// Bindings and values are retained without schema or row-dependent validation.
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
/// Classification impurity measure used when selecting a tree split.
pub enum MlSplitCriterion {
    /// Selects splits by Gini impurity.
    Gini,
    /// Selects splits by information entropy.
    Entropy,
}

#[derive(Debug, Clone, PartialEq)]
/// Bindings and tree-fitting settings shared by classification and regression forests.
pub struct MlRandomForestParams {
    /// Feature rows and the classification or regression target used to fit the forest.
    pub training: MlRelationBinding,
    /// Independent feature rows to score; must omit a label binding.
    pub predict: MlRelationBinding,
    /// Number of trees fitted for this invocation.
    pub n_trees: usize,
    /// Maximum tree depth; zero selects unlimited native depth.
    pub max_depth: i32,
    /// Maximum leaves per tree; `None` selects the native unlimited setting.
    pub max_leaves: Option<u32>,
    /// Fraction of features sampled for each split, in `(0, 1]`.
    pub max_features: f64,
    /// Fraction of training rows sampled for each tree, in `(0, 1]`.
    pub max_samples: f64,
    /// Whether training rows are sampled with replacement.
    pub bootstrap: bool,
    /// Maximum histogram bins used while choosing splits.
    pub max_n_bins: usize,
    /// Minimum training rows in a leaf.
    pub min_samples_leaf: usize,
    /// Minimum training rows required to split a node.
    pub min_samples_split: usize,
    /// Nonnegative minimum impurity reduction required for a split.
    pub min_impurity_decrease: f64,
    /// Maximum tree nodes processed together during fitting.
    pub max_batch_size: usize,
    /// Random seed for fitting the forest.
    pub seed: u64,
    /// Classification split rule; `None` for regression.
    pub criterion: Option<MlSplitCriterion>,
}

impl MlRandomForestParams {
    /// Creates parameters with the current operation defaults.
    ///
    /// Bindings and values are retained without schema or row-dependent validation.
    /// For a regression forest, set `criterion` to `None`.
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
