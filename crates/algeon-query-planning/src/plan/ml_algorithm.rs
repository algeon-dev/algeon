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
pub use output::{ml_embedding_output_schema, ml_hdbscan_output_schema, ml_predictions_output_schema};
pub use supervised::{MlLogisticRegressionParams, MlRandomForestParams, MlSplitCriterion};

#[derive(Debug, Clone, PartialEq)]
pub enum MlAlgorithmParams {
    Tsvd(MlTsvdParams),
    Hdbscan(MlHdbscanParams),
    Umap(MlUmapParams),
    LogisticRegression(MlLogisticRegressionParams),
    RandomForestClassifier(MlRandomForestParams),
    RandomForestRegressor(MlRandomForestParams),
    ForestPredict(MlForestPredictParams),
}

impl MlAlgorithmParams {
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

    pub(crate) const fn relation_role(&self, index: usize) -> MlRelationRole {
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

    pub(crate) fn remap_relation_inputs(&mut self, mut remap: impl FnMut(NodeId) -> NodeId) {
        match self {
            Self::Tsvd(params) => params.input.input = remap(params.input.input),
            Self::Hdbscan(params) => params.input.input = remap(params.input.input),
            Self::Umap(params) => params.input.input = remap(params.input.input),
            Self::ForestPredict(params) => params.input.input = remap(params.input.input),
            Self::LogisticRegression(params) => {
                params.training.input = remap(params.training.input);
                params.predict.input = remap(params.predict.input);
            },
            Self::RandomForestClassifier(params) | Self::RandomForestRegressor(params) => {
                params.training.input = remap(params.training.input);
                params.predict.input = remap(params.predict.input);
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlAlgorithmSpec {
    pub algorithm: MlAlgorithm,
    pub params: MlAlgorithmParams,
    pub(crate) output_schema: SchemaRef,
}

impl MlAlgorithmSpec {
    pub fn new(algorithm: MlAlgorithm, params: MlAlgorithmParams, output_schema: SchemaRef) -> Self {
        assert!(params.supports_algorithm(algorithm), "ML algorithm and parameters must match");
        Self { algorithm, params, output_schema }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlAlgorithmNode {
    pub inputs: Vec<NodeId>,
    pub spec: MlAlgorithmSpec,
}

impl MlAlgorithmNode {
    pub fn output_schema(&self) -> &SchemaRef {
        &self.spec.output_schema
    }
}
