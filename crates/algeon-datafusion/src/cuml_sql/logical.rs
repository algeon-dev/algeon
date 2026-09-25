use std::{cmp::Ordering, fmt, hash::Hash, sync::Arc};

use arrow::datatypes::{DataType, FieldRef, SchemaRef};
#[cfg(feature = "server")]
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::{
    common::{DFSchema, DFSchemaRef, Result},
    logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore},
};
#[cfg(feature = "server")]
use query_engine::plan::MlForestModelDigest;
use query_engine::plan::{
    MlForestModelSnapshot, RelationIdType, ml_embedding_output_schema, ml_forest_output_schema,
    ml_hdbscan_output_schema, ml_predictions_output_schema, vector_pca_output_schema,
};

use super::{
    CumlOperation, CumlValidatedOptions, DenseVectorBinding,
    error::{field_not_found, invalid_call, invalid_call_error, unsupported_type},
    options::{options_from_named_values, validate_options},
};
use crate::gpu_functions::dense_relation::known_row_count;

#[derive(Debug, Clone)]
pub struct CumlLogicalNode {
    operation: CumlOperation,
    inputs: Vec<LogicalPlan>,
    options: CumlValidatedOptions,
    /// Model bound at planning; present exactly for `cuml_forest_predict`.
    model: Option<Arc<MlForestModelSnapshot>>,
    schema: DFSchemaRef,
}

impl CumlLogicalNode {
    /// Builds a node for an operation without a bound model.
    /// `cuml_forest_predict` needs one and is rejected with `model_unbound`;
    /// use [`Self::try_new_with_model`] for it.
    pub fn try_new(operation: CumlOperation, inputs: Vec<LogicalPlan>, options_json: &str) -> Result<Self> {
        Self::try_new_with_model(operation, inputs, options_json, None)
    }

    /// `model` must be the snapshot bound for a `cuml_forest_predict` call and
    /// `None` for every other operation.
    pub fn try_new_with_model(
        operation: CumlOperation,
        inputs: Vec<LogicalPlan>,
        options_json: &str,
        model: Option<Arc<MlForestModelSnapshot>>,
    ) -> Result<Self> {
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
        match (operation, &model) {
            (CumlOperation::ForestPredict, None) => {
                return Err(invalid_call(
                    operation,
                    "model_unbound",
                    "cuml_forest_predict requires a model bound during planning",
                ));
            },
            (CumlOperation::ForestPredict, Some(_)) | (_, None) => {},
            (_, Some(_)) => {
                return Err(invalid_call(
                    operation,
                    "unexpected_model_binding",
                    format!("{} does not accept a bound model", operation.descriptor().sql_function_name),
                ));
            },
        }
        let mut ids = Vec::new();
        let mut dimensions = Vec::new();
        for (input, binding) in inputs.iter().zip(options.bindings()) {
            let role = binding.relation_role().as_str();
            let id = input
                .schema()
                .field_with_unqualified_name(binding.id())
                .map_err(|_| field_not_found(operation, role, binding.id()))?;
            if id.is_nullable() {
                return Err(invalid_call(
                    operation,
                    "nullable_id_field",
                    format!("cuML {role} ID field '{}' must be non-nullable", binding.id()),
                ));
            }
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
                if field.is_nullable() {
                    return Err(invalid_call(operation, "nullable_label", "cuML training label must not be nullable"));
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
                    if field.is_nullable() {
                        return Err(invalid_call(
                            operation,
                            "nullable_vector_field",
                            format!("cuML {role} vector field '{column}' must be non-nullable"),
                        ));
                    }
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
                                "FixedSizeList, List, or LargeList with non-null Float32 elements",
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
                if options.n_components().is_some_and(|count| count > dimension) && operation == CumlOperation::Tsvd {
                    return Err(invalid_call(
                        operation,
                        "component_count_exceeds_dimension",
                        format!(
                            "n_components ({}) exceeds the statically known feature dimension ({dimension})",
                            options.n_components().expect("decomposition components")
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
                let rows = rows as u64;
                let canonical: serde_json::Value =
                    serde_json::from_str(options.canonical_json()).expect("validated options");
                let violated = match operation {
                    CumlOperation::Hdbscan => [
                        ("min_samples_exceeds_input_rows", canonical["min_samples"].as_u64().unwrap() >= rows),
                        ("min_cluster_size_exceeds_input_rows", canonical["min_cluster_size"].as_u64().unwrap() > rows),
                        ("max_cluster_size_exceeds_input_rows", canonical["max_cluster_size"].as_u64().unwrap() > rows),
                    ]
                    .into_iter()
                    .find(|(_, violated)| *violated && rows > 0)
                    .map(|(reason, _)| reason),
                    CumlOperation::Umap if rows > 0 && canonical["n_neighbors"].as_u64().unwrap() >= rows => {
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
        if let (Some(model), Some(Some(dimension))) = (&model, dimensions.first())
            && *dimension != model.num_features()
        {
            return Err(invalid_call_error(
                operation,
                "forest_feature_count_mismatch",
                format!("input has {dimension} features; the model expects {}", model.num_features()),
            )
            .with_log_safe_fact("expected", model.num_features())
            .with_log_safe_fact("actual", dimension)
            .into());
        }
        let id_type = ids.last().expect("cuML requires a relation").clone();
        let output = match operation {
            CumlOperation::Tsvd => {
                vector_pca_output_schema(id_type, options.n_components().expect("decomposition components"))
            },
            CumlOperation::Hdbscan => ml_hdbscan_output_schema(id_type),
            CumlOperation::Umap => {
                ml_embedding_output_schema(id_type, options.n_components().expect("UMAP components"))
            },
            CumlOperation::LogisticRegression | CumlOperation::RandomForestClassifier => {
                ml_predictions_output_schema(id_type, DataType::Int32)
            },
            CumlOperation::RandomForestRegressor => ml_predictions_output_schema(id_type, DataType::Float32),
            CumlOperation::ForestPredict => {
                ml_forest_output_schema(id_type, model.as_ref().expect("forest model checked above").output())
            },
        };
        let schema = Arc::new(DFSchema::try_from(output.as_ref().clone())?);
        Ok(Self { operation, inputs, options, model, schema })
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
    if field.is_nullable() {
        return Err(invalid_call(
            operation,
            "nullable_feature_field",
            format!("cuML {role} feature field '{name}' must be non-nullable"),
        ));
    }
    if field.data_type() != &DataType::Float32 {
        return Err(unsupported_type(operation, role, name, field.data_type(), "Float32"));
    }
    Ok(())
}

impl PartialEq for CumlLogicalNode {
    fn eq(&self, other: &Self) -> bool {
        self.operation == other.operation
            && self.inputs == other.inputs
            && self.options == other.options
            && self.model == other.model
    }
}
impl Eq for CumlLogicalNode {}
impl Hash for CumlLogicalNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.operation.hash(state);
        self.inputs.hash(state);
        self.options.hash(state);
        self.model.hash(state);
    }
}
impl PartialOrd for CumlLogicalNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (self.operation.descriptor().sql_function_name, &self.inputs, &self.options, &self.model).partial_cmp(&(
            other.operation.descriptor().sql_function_name,
            &other.inputs,
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
    fn expressions(&self) -> Vec<Expr> {
        Vec::new()
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
        Ok(())
    }
    fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        if !exprs.is_empty() {
            return Err(invalid_call(
                self.operation,
                "unexpected_logical_expressions",
                "cuML logical nodes do not expose optimizer expressions",
            ));
        }
        let options_json =
            options_from_named_values(self.operation, &inputs, self.options.named_values(self.operation))?;
        Self::try_new_with_model(self.operation, inputs, &options_json, self.model.clone())
    }
    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}
