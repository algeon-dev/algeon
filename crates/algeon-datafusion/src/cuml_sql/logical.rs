use std::{cmp::Ordering, fmt, hash::Hash, sync::Arc};

use arrow::datatypes::{DataType, FieldRef, SchemaRef};
#[cfg(feature = "server")]
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::{
    common::{DFSchema, DFSchemaRef, DataFusionError, Result as DataFusionResult},
    logical_expr::{Expr, LogicalPlan, SortExpr, UserDefinedLogicalNodeCore},
};
#[cfg(feature = "server")]
use query_planning::plan::MlForestModelDigest;
use query_planning::plan::{
    MlAlgorithmParams, MlForestModelSnapshot, RelationIdType, ml_classification_output_schema,
    ml_embedding_output_schema, ml_forest_output_schema, ml_hdbscan_output_schema, ml_regression_output_schema,
    vector_pca_output_schema,
};

use super::{
    CumlOperation, CumlValidatedOptions, DenseVectorBinding,
    error::{field_not_found, invalid_call, unsupported_type},
    options::{CumlParams, dense_relation_function, validate_options},
};
use crate::{
    error::Result,
    gpu_functions::{
        dense_relation::known_row_count, fmt_input_orderings, input_ordering_expressions,
        input_orderings_with_expressions,
    },
};

#[derive(Debug, Clone)]
pub struct CumlLogicalNode {
    operation: CumlOperation,
    inputs: Vec<LogicalPlan>,
    /// Top-level `ORDER BY` of each relation argument, in relation-role order;
    /// empty when the input declares no order.
    input_orderings: Vec<Vec<SortExpr>>,
    options: CumlValidatedOptions,
    /// Model bound at planning; present exactly for `cuml_forest_predict`.
    model: Option<Arc<MlForestModelSnapshot>>,
    schema: DFSchemaRef,
}

impl CumlLogicalNode {
    /// Builds a node for an operation without a bound model.
    /// `cuml_forest_predict` needs one and is rejected with `model_unbound`;
    /// use [`Self::try_new_with_model`] for it.
    pub fn try_new(operation: CumlOperation, inputs: Vec<LogicalPlan>, options_json: &str) -> DataFusionResult<Self> {
        Self::try_new_with_model(operation, inputs, options_json, None)
    }

    /// `model` must be the snapshot bound for a `cuml_forest_predict` call and
    /// `None` for every other operation.
    pub fn try_new_with_model(
        operation: CumlOperation,
        inputs: Vec<LogicalPlan>,
        options_json: &str,
        model: Option<Arc<MlForestModelSnapshot>>,
    ) -> DataFusionResult<Self> {
        let (options, output) = Self::validate_call(operation, &inputs, options_json, model.as_deref())?;
        let schema = Arc::new(DFSchema::try_from(output.as_ref().clone())?);
        let input_orderings = vec![Vec::new(); inputs.len()];
        Ok(Self { operation, inputs, input_orderings, options, model, schema })
    }

    fn validate_call(
        operation: CumlOperation,
        inputs: &[LogicalPlan],
        options_json: &str,
        model: Option<&MlForestModelSnapshot>,
    ) -> Result<(CumlValidatedOptions, SchemaRef)> {
        if inputs.len() != operation.descriptor().relation_roles.len() {
            return Err(invalid_call(
                operation,
                "invalid_relation_count",
                format!(
                    "{} requires {} relation argument(s), received {}",
                    operation.descriptor().sql_function_name,
                    operation.descriptor().relation_roles.len(),
                    inputs.len()
                ),
            ));
        }
        let options = validate_options(operation, options_json)?;
        if model.is_some() && !matches!(options.params(), CumlParams::ForestPredict { .. }) {
            return Err(invalid_call(
                operation,
                "unexpected_model_binding",
                format!("{} does not accept a bound model", operation.descriptor().sql_function_name),
            ));
        }
        let mut ids = Vec::new();
        let mut dimensions = Vec::new();
        for (input, binding) in inputs.iter().zip(options.bindings()) {
            let role = binding.relation_role().as_str();
            let id = input
                .schema()
                .field_with_unqualified_name(binding.id())
                .map_err(|_| field_not_found(operation, role, binding.id()))?;
            if RelationIdType::from_data_type(id.data_type()).is_none() {
                return Err(unsupported_type(
                    operation,
                    role,
                    binding.id(),
                    id.data_type(),
                    "Int32, Int64, Utf8, LargeUtf8, or Utf8View",
                ));
            }
            if role == "training" {
                let label = binding
                    .label()
                    .ok_or_else(|| invalid_call(operation, "missing_label", "cuML training requires label"))?;
                let field = input
                    .schema()
                    .field_with_unqualified_name(label)
                    .map_err(|_| field_not_found(operation, role, label))?;
                let target_type =
                    if operation == CumlOperation::RandomForestRegressor { DataType::Float32 } else { DataType::Int32 };
                if field.data_type() != &target_type {
                    return Err(unsupported_type(
                        operation,
                        role,
                        label,
                        field.data_type(),
                        if target_type == DataType::Float32 { "Float32" } else { "Int32" },
                    ));
                }
            }
            if role == "predict"
                && (binding.label().is_some() || input.schema().field_with_unqualified_name("label").is_ok())
            {
                return Err(invalid_call(
                    operation,
                    "predict_relation_has_label",
                    "cuML predict relation must omit label",
                ));
            }
            let dimension = match binding.features() {
                DenseVectorBinding::Columns(columns) => {
                    for column in columns {
                        let field = resolve_feature(operation, input, role, column)?;
                        require_float32(operation, role, column, field)?;
                    }
                    Some(columns.len())
                },
                DenseVectorBinding::VectorColumn(column) => {
                    let field = resolve_feature(operation, input, role, column)?;
                    match field.data_type() {
                        DataType::FixedSizeList(element, dimension) if *dimension > 0 => {
                            require_float32(operation, role, column, element)?;
                            Some(*dimension as usize)
                        },
                        DataType::List(element) | DataType::LargeList(element) => {
                            require_float32(operation, role, column, element)?;
                            None
                        },
                        actual => {
                            return Err(unsupported_type(
                                operation,
                                role,
                                column,
                                actual,
                                "FixedSizeList, List, or LargeList with Float32 elements",
                            ));
                        },
                    }
                },
            };
            if let Some(dimension) = dimension {
                if dimension < 2 && operation == CumlOperation::Tsvd {
                    return Err(invalid_call(
                        operation,
                        "insufficient_feature_dimension",
                        "cuML dimensionality reduction requires at least two input feature columns",
                    ));
                }
                if let CumlParams::Algorithm(MlAlgorithmParams::Tsvd(params)) = options.params()
                    && params.n_components > dimension
                {
                    return Err(invalid_call(
                        operation,
                        "component_count_exceeds_dimension",
                        format!(
                            "n_components ({}) exceeds the statically known feature dimension ({dimension})",
                            params.n_components
                        ),
                    ));
                }
            }
            if let Some(rows) = known_row_count(input)
                && rows == 1
                && role != "predict"
                && operation != CumlOperation::ForestPredict
            {
                return Err(invalid_call(
                    operation,
                    "insufficient_input_rows",
                    format!("{} requires at least two input rows", operation.descriptor().sql_function_name),
                ));
            }
            if let Some(rows) = known_row_count(input) {
                let violated = match options.params() {
                    CumlParams::Algorithm(MlAlgorithmParams::Hdbscan(params)) => [
                        ("min_samples_exceeds_input_rows", params.min_samples >= rows),
                        ("min_cluster_size_exceeds_input_rows", params.min_cluster_size > rows),
                        ("max_cluster_size_exceeds_input_rows", params.max_cluster_size > rows),
                    ]
                    .into_iter()
                    .find(|(_, violated)| *violated && rows > 0)
                    .map(|(reason, _)| reason),
                    CumlParams::Algorithm(MlAlgorithmParams::Umap(params))
                        if rows > 0 && params.n_neighbors >= rows =>
                    {
                        Some("n_neighbors_exceeds_input_rows")
                    },
                    _ => None,
                };
                if let Some(reason) = violated {
                    return Err(invalid_call(operation, reason, format!("cuML {reason}: input has {rows} rows")));
                }
            }
            ids.push(id.data_type().clone());
            dimensions.push(dimension);
        }
        if dimensions.len() == 2
            && let (Some(training), Some(predict)) = (dimensions[0], dimensions[1])
            && training != predict
        {
            return Err(invalid_call(
                operation,
                "feature_dimension_mismatch",
                format!("cuML training dimension {training} differs from predict dimension {predict}"),
            ));
        }
        if let (Some(model), Some(Some(dimension))) = (model, dimensions.first())
            && *dimension != model.num_features()
        {
            return Err(invalid_call(
                operation,
                "forest_feature_count_mismatch",
                format!("input has {dimension} features; the model expects {}", model.num_features()),
            )
            .with_log_safe_fact("expected", model.num_features())
            .with_log_safe_fact("actual", dimension));
        }
        let id_type = ids.last().expect("cuML requires a relation").clone();
        let output = match (options.params(), model) {
            (CumlParams::Algorithm(MlAlgorithmParams::Tsvd(params)), _) => {
                vector_pca_output_schema(id_type, params.n_components)
            },
            (CumlParams::Algorithm(MlAlgorithmParams::Hdbscan(_)), _) => ml_hdbscan_output_schema(id_type),
            (CumlParams::Algorithm(MlAlgorithmParams::Umap(params)), _) => {
                ml_embedding_output_schema(id_type, params.n_components)
            },
            (
                CumlParams::Algorithm(
                    MlAlgorithmParams::LogisticRegression(_) | MlAlgorithmParams::RandomForestClassifier(_),
                ),
                _,
            ) => ml_classification_output_schema(id_type),
            (CumlParams::Algorithm(MlAlgorithmParams::RandomForestRegressor(_)), _) => {
                ml_regression_output_schema(id_type)
            },
            (CumlParams::Algorithm(MlAlgorithmParams::ForestPredict(params)), _) => {
                ml_forest_output_schema(id_type, params.model.output())
            },
            (CumlParams::ForestPredict { .. }, Some(model)) => ml_forest_output_schema(id_type, model.output()),
            (CumlParams::ForestPredict { .. }, None) => {
                return Err(invalid_call(
                    operation,
                    "model_unbound",
                    "cuml_forest_predict requires a model bound during planning",
                ));
            },
        };
        Ok((options, output))
    }

    /// Declares each relation argument's order, which fixes its `row_ordinal`
    /// numbering; `input_orderings` has one entry per relation role.
    pub(crate) fn with_input_orderings(mut self, input_orderings: Vec<Vec<SortExpr>>) -> Result<Self> {
        if input_orderings.len() != self.inputs.len() {
            return Err(invalid_call(
                self.operation,
                "input_ordering_count_mismatch",
                "cuML input orderings do not match the relation arguments",
            ));
        }
        self.input_orderings = input_orderings;
        Ok(self)
    }

    pub(crate) fn input_orderings(&self) -> &[Vec<SortExpr>] {
        &self.input_orderings
    }

    pub fn operation(&self) -> CumlOperation {
        self.operation
    }
    pub fn validated_options(&self) -> &CumlValidatedOptions {
        &self.options
    }
    pub fn model(&self) -> Option<&Arc<MlForestModelSnapshot>> {
        self.model.as_ref()
    }
    pub fn output_schema(&self) -> SchemaRef {
        Arc::clone(self.schema.inner())
    }
}

#[cfg(feature = "server")]
/// Digests of every forest model bound in `plan`, including subqueries, in
/// pre-order. Two plans of one statement bind the same model bytes exactly
/// when these sequences are equal.
pub(crate) fn forest_model_digests(plan: &LogicalPlan) -> Vec<MlForestModelDigest> {
    let mut digests = Vec::new();
    plan.apply_with_subqueries(|node| {
        if let LogicalPlan::Extension(extension) = node
            && let Some(model) =
                extension.node.as_any().downcast_ref::<CumlLogicalNode>().and_then(CumlLogicalNode::model)
        {
            digests.push(model.digest());
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .expect("forest model digest collection is infallible");
    digests
}

fn resolve_feature<'a>(
    operation: CumlOperation,
    input: &'a LogicalPlan,
    role: &'static str,
    name: &str,
) -> Result<&'a FieldRef> {
    input.schema().field_with_unqualified_name(name).map_err(|_| field_not_found(operation, role, name))
}

fn require_float32(operation: CumlOperation, role: &'static str, name: &str, field: &FieldRef) -> Result<()> {
    if field.data_type() != &DataType::Float32 {
        return Err(unsupported_type(operation, role, name, field.data_type(), "Float32"));
    }
    Ok(())
}

impl PartialEq for CumlLogicalNode {
    fn eq(&self, other: &Self) -> bool {
        self.operation == other.operation
            && self.inputs == other.inputs
            && self.input_orderings == other.input_orderings
            && self.options == other.options
            && self.model == other.model
    }
}
impl Eq for CumlLogicalNode {}
impl Hash for CumlLogicalNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.operation.hash(state);
        self.inputs.hash(state);
        self.input_orderings.hash(state);
        self.options.hash(state);
        self.model.hash(state);
    }
}
impl PartialOrd for CumlLogicalNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (self.operation.descriptor().sql_function_name, &self.inputs, &self.input_orderings, &self.options, &self.model)
            .partial_cmp(&(
                other.operation.descriptor().sql_function_name,
                &other.inputs,
                &other.input_orderings,
                &other.options,
                &other.model,
            ))
    }
}

impl UserDefinedLogicalNodeCore for CumlLogicalNode {
    fn name(&self) -> &str {
        "CumlLogicalNode"
    }
    fn inputs(&self) -> Vec<&LogicalPlan> {
        self.inputs.iter().collect()
    }
    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }
    /// The input sort keys, flattened in relation-role order.
    fn expressions(&self) -> Vec<Expr> {
        input_ordering_expressions(&self.input_orderings)
    }
    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "CumlLogicalNode: operation={}, relation_count={}, options={}",
            self.operation.descriptor().sql_function_name,
            self.inputs.len(),
            self.options.canonical_json()
        )?;
        if let Some(model) = &self.model {
            write!(f, ", model=sha256:{}", model.digest().short_hex())?;
        }
        fmt_input_orderings(
            f,
            self.operation.descriptor().relation_roles.iter().map(|role| role.as_str()),
            &self.input_orderings,
        )
    }
    fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> DataFusionResult<Self> {
        let Some(input_orderings) = input_orderings_with_expressions(&self.input_orderings, exprs) else {
            return Err(invalid_call(
                self.operation,
                "unexpected_logical_expressions",
                "cuML logical node expressions must be its input sort keys",
            )
            .into());
        };
        let options_json =
            dense_relation_function(self.operation).options_json(&inputs, self.options.value_options().clone())?;
        Self::try_new_with_model(self.operation, inputs, &options_json, self.model.clone())?
            .with_input_orderings(input_orderings)
            .map_err(DataFusionError::from)
    }
    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}
