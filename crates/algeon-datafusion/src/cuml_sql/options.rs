use std::{
    cmp::Ordering,
    hash::{Hash, Hasher},
};

use query_planning::plan::{
    MlAlgorithmParams, MlClusterSelectionMethod, MlDecompositionSolver, MlForestModelFormat, MlHdbscanParams,
    MlLogisticRegressionParams, MlRandomForestParams, MlRelationBinding, MlSplitCriterion, MlTsvdParams, MlUmapInit,
    MlUmapParams, VectorRowIdBinding,
};
use serde_json::{Map, Value};

use super::{
    CumlOperation, DenseVectorBinding, MlRelationRole, canonicalize_cuml_options,
    error::{ERRORS, invalid_option},
    model_store::{ForestModelReference, ForestOutputRequest},
};
use crate::{
    error::Result,
    gpu_functions::dense_relation::{CanonicalOptions, DenseRelationArgument, DenseRelationFunction, LabelColumn},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Validated columns bound to one named relation role in a cuML SQL call.
pub struct CumlRelationBinding {
    relation_role: MlRelationRole,
    id: String,
    features: DenseVectorBinding,
    label: Option<String>,
}

impl CumlRelationBinding {
    /// Returns the semantic input role associated with this relation.
    pub fn relation_role(&self) -> MlRelationRole {
        self.relation_role
    }
    /// Returns the bound row-id column used to correlate results with input rows.
    pub fn id(&self) -> &str {
        &self.id
    }
    /// Returns the validated dense-vector representation and column binding.
    pub fn features(&self) -> &DenseVectorBinding {
        &self.features
    }
    /// Returns the supervised-learning label column, or `None` for a relation without labels.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

/// Options of one cuML call, validated and parsed once. The canonical JSON
/// is the call's identity and serialized form; validation and lowering read
/// the typed parameters.
#[derive(Debug, Clone)]
pub struct CumlValidatedOptions {
    canonical_json: String,
    value_options: Map<String, Value>,
    bindings: Vec<CumlRelationBinding>,
    params: CumlParams,
}

/// The typed parameters of a cuML call.
#[derive(Debug, Clone)]
pub(crate) enum CumlParams {
    /// An algorithm that fits and predicts within the query.
    Algorithm(MlAlgorithmParams),
    /// A prediction with the forest model the call names; planning loads its
    /// snapshot, which lowering binds.
    ForestPredict { input: MlRelationBinding, model: ForestModelReference },
}

impl CumlValidatedOptions {
    /// Returns normalized option JSON used as this call's identity and serialized representation.
    pub fn canonical_json(&self) -> &str {
        &self.canonical_json
    }

    /// Returns validated relation bindings in the operation descriptor's role order.
    pub fn bindings(&self) -> &[CumlRelationBinding] {
        &self.bindings
    }

    pub(crate) fn params(&self) -> &CumlParams {
        &self.params
    }

    /// The value options by name, without relation bindings.
    pub(super) fn value_options(&self) -> &Map<String, Value> {
        &self.value_options
    }

    /// The forest model a `cuml_forest_predict` call names; `None` for other operations.
    pub(super) fn forest_model_reference(&self) -> Option<&ForestModelReference> {
        match &self.params {
            CumlParams::ForestPredict { model, .. } => Some(model),
            CumlParams::Algorithm(_) => None,
        }
    }
}

// The canonical JSON determines every other field, so it is the identity.
impl PartialEq for CumlValidatedOptions {
    fn eq(&self, other: &Self) -> bool {
        self.canonical_json == other.canonical_json
    }
}

impl Eq for CumlValidatedOptions {}

impl Hash for CumlValidatedOptions {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.canonical_json.hash(state);
    }
}

impl PartialOrd for CumlValidatedOptions {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CumlValidatedOptions {
    fn cmp(&self, other: &Self) -> Ordering {
        self.canonical_json.cmp(&other.canonical_json)
    }
}

pub(super) fn validate_options(operation: CumlOperation, raw: &str) -> Result<CumlValidatedOptions> {
    let canonical = canonicalize_cuml_options(operation, raw)
        .map_err(|error| invalid_option(operation, error.code(), error.argument(), error.message()))?;
    let options = dense_relation_function(operation).canonical_options(&canonical);
    let bindings = operation
        .descriptor()
        .relation_roles
        .iter()
        .map(|&relation_role| {
            let binding = options.relation_binding(relation_role.as_str())?;
            Ok(CumlRelationBinding { relation_role, id: binding.id, features: binding.vector, label: binding.label })
        })
        .collect::<Result<Vec<_>>>()?;
    let params = algorithm_params(operation, &options, &bindings)?;
    let value_options = options.value_options(operation.descriptor().options);
    Ok(CumlValidatedOptions { canonical_json: Value::Object(canonical).to_string(), value_options, bindings, params })
}

fn algorithm_params(
    operation: CumlOperation,
    options: &CanonicalOptions<'_>,
    bindings: &[CumlRelationBinding],
) -> Result<CumlParams> {
    let relation = |ordinal: usize| {
        let binding = &bindings[ordinal];
        let row_id = VectorRowIdBinding::column(binding.id());
        let mut relation = match binding.features() {
            DenseVectorBinding::Columns(columns) => MlRelationBinding::new(ordinal, row_id, columns.iter().cloned()),
            DenseVectorBinding::VectorColumn(column) => {
                MlRelationBinding::new_vector_column(ordinal, row_id, column.clone())
            },
        };
        relation.label = binding.label().map(str::to_owned);
        relation
    };
    let params = match operation {
        CumlOperation::Tsvd => {
            let mut params = MlTsvdParams::new(relation(0), options.usize("n_components")?);
            params.solver = options.choice(
                "solver",
                &[
                    ("cov_eig_dq", MlDecompositionSolver::CovEigDq),
                    ("cov_eig_jacobi", MlDecompositionSolver::CovEigJacobi),
                ],
            )?;
            params.tolerance = options.f64("tol")?;
            params.n_iterations = options.u32("n_iterations")?;
            MlAlgorithmParams::Tsvd(params)
        },
        CumlOperation::Hdbscan => {
            let mut params = MlHdbscanParams::new(relation(0));
            params.min_samples = options.usize("min_samples")?;
            params.min_cluster_size = options.usize("min_cluster_size")?;
            params.max_cluster_size = options.usize("max_cluster_size")?;
            params.cluster_selection_epsilon = options.f64("cluster_selection_epsilon")?;
            params.alpha = options.f64("alpha")?;
            params.allow_single_cluster = options.bool("allow_single_cluster")?;
            params.cluster_selection_method = options.choice(
                "cluster_selection_method",
                &[("eom", MlClusterSelectionMethod::Eom), ("leaf", MlClusterSelectionMethod::Leaf)],
            )?;
            MlAlgorithmParams::Hdbscan(params)
        },
        CumlOperation::LogisticRegression => {
            let mut params = MlLogisticRegressionParams::new(relation(0), relation(1));
            params.penalty_l1 = options.f64("penalty_l1")?;
            params.penalty_l2 = options.f64("penalty_l2")?;
            params.max_iter = options.usize("max_iter")?;
            params.linesearch_max_iter = options.usize("linesearch_max_iter")?;
            params.lbfgs_memory = options.usize("lbfgs_memory")?;
            params.fit_intercept = options.bool("fit_intercept")?;
            params.penalty_normalized = options.bool("penalty_normalized")?;
            params.grad_tol = options.f64("grad_tol")?;
            params.change_tol = options.f64("change_tol")?;
            MlAlgorithmParams::LogisticRegression(params)
        },
        CumlOperation::RandomForestClassifier | CumlOperation::RandomForestRegressor => {
            let mut params = MlRandomForestParams::new(relation(0), relation(1));
            params.n_trees = options.usize("n_trees")?;
            params.max_depth = options.i32("max_depth")?;
            params.max_leaves = options.optional_u32("max_leaves")?;
            params.max_features = options.f64("max_features")?;
            params.max_samples = options.f64("max_samples")?;
            params.bootstrap = options.bool("bootstrap")?;
            params.max_n_bins = options.usize("max_n_bins")?;
            params.min_samples_leaf = options.usize("min_samples_leaf")?;
            params.min_samples_split = options.usize("min_samples_split")?;
            params.min_impurity_decrease = options.f64("min_impurity_decrease")?;
            params.max_batch_size = options.usize("max_batch_size")?;
            params.seed = options.u64("seed")?;
            if operation == CumlOperation::RandomForestRegressor {
                params.criterion = None;
                MlAlgorithmParams::RandomForestRegressor(params)
            } else {
                params.criterion = Some(options.choice(
                    "split_criterion",
                    &[("gini", MlSplitCriterion::Gini), ("entropy", MlSplitCriterion::Entropy)],
                )?);
                MlAlgorithmParams::RandomForestClassifier(params)
            }
        },
        CumlOperation::ForestPredict => {
            let model = ForestModelReference {
                path: options.str("model")?.to_owned(),
                format: options.parsed("model_format", MlForestModelFormat::from_option_value)?,
                output: options.choice(
                    "output",
                    &[("scores", ForestOutputRequest::Scores), ("class_index", ForestOutputRequest::ClassIndex)],
                )?,
            };
            return Ok(CumlParams::ForestPredict { input: relation(0), model });
        },
        CumlOperation::Umap => {
            let mut params = MlUmapParams::new(relation(0));
            params.n_components = options.usize("n_components")?;
            params.n_neighbors = options.usize("n_neighbors")?;
            params.n_epochs = options.usize("n_epochs")?;
            params.learning_rate = options.f64("learning_rate")?;
            params.min_dist = options.f64("min_dist")?;
            params.spread = options.f64("spread")?;
            params.set_op_mix_ratio = options.f64("set_op_mix_ratio")?;
            params.local_connectivity = options.f64("local_connectivity")?;
            params.repulsion_strength = options.f64("repulsion_strength")?;
            params.negative_sample_rate = options.usize("negative_sample_rate")?;
            params.init =
                options.choice("init", &[("spectral", MlUmapInit::Spectral), ("random", MlUmapInit::Random)])?;
            params.random_state = options.u64("random_state")?;
            MlAlgorithmParams::Umap(params)
        },
    };
    Ok(CumlParams::Algorithm(params))
}

/// The cuML operation as a dense-relation function: a training relation
/// carries the label it fits, and a predict relation omits it.
pub(super) fn dense_relation_function(operation: CumlOperation) -> DenseRelationFunction {
    let descriptor = operation.descriptor();
    DenseRelationFunction {
        errors: &ERRORS,
        sql_function_name: descriptor.sql_function_name,
        relations: descriptor
            .relation_roles
            .iter()
            .map(|role| DenseRelationArgument {
                role: role.as_str(),
                label: match role {
                    MlRelationRole::Training => LabelColumn::Required,
                    MlRelationRole::Predict => LabelColumn::Forbidden,
                    MlRelationRole::Input => LabelColumn::Feature,
                },
            })
            .collect(),
        options: descriptor.options,
    }
}
