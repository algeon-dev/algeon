//! Native cuML algorithm IR and operation contracts.

use super::{NodeId, SchemaRef};

mod clustering;
mod common;
mod decomposition;
mod definition;
mod forest;
mod output;
mod supervised;

pub use clustering::{MlClusterSelectionMethod, MlHdbscanParams, MlUmapInit, MlUmapParams};
pub use common::MlRelationBinding;
pub use decomposition::{MlDecompositionSolver, MlTsvdParams};
pub use definition::{
    CUML_FOREST_PREDICT_FUNCTION, CUML_HDBSCAN_FUNCTION, CUML_LOGISTIC_REGRESSION_FUNCTION, CUML_OPERATION_REGISTRY,
    CUML_RANDOM_FOREST_CLASSIFIER_FUNCTION, CUML_RANDOM_FOREST_REGRESSOR_FUNCTION, CUML_TSVD_FUNCTION,
    CUML_UMAP_FUNCTION, CumlOperation, CumlOperationDescriptor, CumlOutputContract, MlAlgorithm, MlRelationRole,
    canonicalize_cuml_options,
};
pub use forest::MlForestPredictParams;
pub use output::{
    ml_classification_output_schema, ml_embedding_output_schema, ml_hdbscan_output_schema, ml_regression_output_schema,
};
pub use supervised::{MlLogisticRegressionParams, MlRandomForestParams, MlSplitCriterion};

#[derive(Debug, Clone, PartialEq)]
/// Operation-specific bindings and tuning values for one native ML node.
pub enum MlAlgorithmParams {
    /// Fits truncated SVD and transforms the input relation.
    Tsvd(MlTsvdParams),
    /// Clusters the input relation with HDBSCAN.
    Hdbscan(MlHdbscanParams),
    /// Fits UMAP and embeds the input relation.
    Umap(MlUmapParams),
    /// Fits a classifier on training rows and scores prediction rows.
    LogisticRegression(MlLogisticRegressionParams),
    /// Fits a classification forest and scores prediction rows.
    RandomForestClassifier(MlRandomForestParams),
    /// Fits a regression forest and scores prediction rows.
    RandomForestRegressor(MlRandomForestParams),
    /// Scores input rows with the immutable pretrained model snapshot.
    ForestPredict(MlForestPredictParams),
}

impl MlAlgorithmParams {
    /// Whether this parameter variant belongs to `algorithm`; this does not validate its values.
    pub const fn supports_algorithm(&self, algorithm: MlAlgorithm) -> bool {
        matches!(
            (self, algorithm),
            (Self::Tsvd(_), MlAlgorithm::Tsvd)
                | (Self::Hdbscan(_), MlAlgorithm::Hdbscan)
                | (Self::Umap(_), MlAlgorithm::Umap)
                | (Self::LogisticRegression(_), MlAlgorithm::LogisticRegression)
                | (Self::RandomForestClassifier(_), MlAlgorithm::RandomForestClassifier)
                | (Self::RandomForestRegressor(_), MlAlgorithm::RandomForestRegressor)
                | (Self::ForestPredict(_), MlAlgorithm::ForestPredict)
        )
    }

    /// Borrows bindings in input-role order: one input, or training followed by prediction.
    pub fn relation_bindings(&self) -> impl Iterator<Item = &MlRelationBinding> {
        let (first, second) = match self {
            Self::Tsvd(params) => (&params.input, None),
            Self::Hdbscan(params) => (&params.input, None),
            Self::Umap(params) => (&params.input, None),
            Self::ForestPredict(params) => (&params.input, None),
            Self::LogisticRegression(params) => (&params.training, Some(&params.predict)),
            Self::RandomForestClassifier(params) | Self::RandomForestRegressor(params) => {
                (&params.training, Some(&params.predict))
            },
        };
        std::iter::once(first).chain(second)
    }

    /// Role of the relation at `index` among the node's inputs.
    pub const fn relation_role(&self, index: usize) -> MlRelationRole {
        match self {
            Self::LogisticRegression(_) | Self::RandomForestClassifier(_) | Self::RandomForestRegressor(_)
                if index == 0 =>
            {
                MlRelationRole::Training
            },
            Self::LogisticRegression(_) | Self::RandomForestClassifier(_) | Self::RandomForestRegressor(_) => {
                MlRelationRole::Predict
            },
            _ => MlRelationRole::Input,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Algorithm, parameters, and result schema bound into the native plan.
pub struct MlAlgorithmSpec {
    /// Identity used to select the native operation.
    pub algorithm: MlAlgorithm,
    /// Bindings and tuning values whose variant must match `algorithm`.
    pub params: MlAlgorithmParams,
    pub(crate) output_schema: SchemaRef,
}

impl MlAlgorithmSpec {
    /// Binds an algorithm and its parameters to the planned result schema.
    ///
    /// # Panics
    ///
    /// Panics when the parameter variant does not match `algorithm`. Values and schema
    /// compatibility are checked by planning and execution.
    pub fn new(algorithm: MlAlgorithm, params: MlAlgorithmParams, output_schema: SchemaRef) -> Self {
        assert!(params.supports_algorithm(algorithm), "ML algorithm and parameters must match");
        Self { algorithm, params, output_schema }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Native ML operation with the relation nodes named by its bindings.
pub struct MlAlgorithmNode {
    /// Relation inputs; each relation binding names its relation by its
    /// position here.
    pub inputs: Vec<NodeId>,
    /// Algorithm contract and result schema for these inputs.
    pub spec: MlAlgorithmSpec,
}

impl MlAlgorithmNode {
    /// Borrows the result schema fixed when this node was planned.
    pub fn output_schema(&self) -> &SchemaRef {
        &self.spec.output_schema
    }
}
