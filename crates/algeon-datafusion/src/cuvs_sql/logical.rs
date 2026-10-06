use std::{cmp::Ordering, fmt, hash::Hash, sync::Arc};

use arrow::datatypes::{DataType, Field, FieldRef, SchemaRef};
use datafusion::{
    common::{DFSchema, DFSchemaRef, DataFusionError, Result as DataFusionResult},
    logical_expr::{Expr, LogicalPlan, SortExpr, UserDefinedLogicalNodeCore},
};
use query_planning::plan::{
    RelationIdType, VectorAlgorithmParams, VectorDistanceMetric, VectorElementType, VectorIvfFlatParams,
    VectorIvfPqParams, vector_kmeans_output_schema, vector_knn_output_schema, vector_pairwise_distance_output_schema,
    vector_pca_output_schema, vector_top_k_output_schema,
};
use serde_json::Value;

use super::{
    CuvsOperation, CuvsValidatedOptions, CuvsVectorRelationBinding, DenseVectorBinding,
    error::{field_not_found, invalid_call, unsupported_type},
    options::{dense_relation_function, validate_options},
};
use crate::{
    error::{Error, Result},
    gpu_functions::{
        dense_relation::known_row_count, fmt_input_orderings, input_ordering_expressions,
        input_orderings_with_expressions,
    },
};

#[derive(Debug, Clone)]
pub struct CuvsLogicalNode {
    operation: CuvsOperation,
    inputs: Vec<LogicalPlan>,
    /// Top-level `ORDER BY` of each relation argument, in relation-role order;
    /// empty when the input declares no order.
    input_orderings: Vec<Vec<SortExpr>>,
    options: CuvsValidatedOptions,
    schema: DFSchemaRef,
}

impl CuvsLogicalNode {
    pub fn try_new(operation: CuvsOperation, inputs: Vec<LogicalPlan>, options_json: &str) -> DataFusionResult<Self> {
        let descriptor = operation.descriptor();
        if inputs.len() != descriptor.relation_roles.len() {
            return Err(invalid_call(
                operation,
                "invalid_relation_count",
                format!(
                    "{} requires {} relation argument(s), received {}",
                    descriptor.sql_function_name,
                    descriptor.relation_roles.len(),
                    inputs.len()
                ),
            )
            .into());
        }
        let options = validate_options(operation, options_json)?;
        let (dimensions, element) = validate_input_schemas(operation, &inputs, &options)?;
        validate_known_cardinalities(operation, &inputs, &options)?;
        let schema = derive_output_schema(operation, &inputs, &options, &dimensions, element)?;
        let input_orderings = vec![Vec::new(); inputs.len()];
        Ok(Self { operation, inputs, input_orderings, options, schema })
    }

    /// Declares each relation argument's order, which fixes its `row_ordinal`
    /// numbering; `input_orderings` has one entry per relation role.
    pub(crate) fn with_input_orderings(mut self, input_orderings: Vec<Vec<SortExpr>>) -> Result<Self> {
        if input_orderings.len() != self.inputs.len() {
            return Err(invalid_call(
                self.operation,
                "input_ordering_count_mismatch",
                "cuVS input orderings do not match the relation arguments",
            ));
        }
        self.input_orderings = input_orderings;
        Ok(self)
    }

    pub(crate) fn input_orderings(&self) -> &[Vec<SortExpr>] {
        &self.input_orderings
    }

    pub fn operation(&self) -> CuvsOperation {
        self.operation
    }

    pub fn validated_options(&self) -> &CuvsValidatedOptions {
        &self.options
    }

    pub fn output_schema(&self) -> SchemaRef {
        Arc::clone(self.schema.inner())
    }
}

impl PartialEq for CuvsLogicalNode {
    fn eq(&self, other: &Self) -> bool {
        self.operation == other.operation
            && self.inputs == other.inputs
            && self.input_orderings == other.input_orderings
            && self.options == other.options
    }
}

impl Eq for CuvsLogicalNode {}

impl Hash for CuvsLogicalNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.operation.hash(state);
        self.inputs.hash(state);
        self.input_orderings.hash(state);
        self.options.hash(state);
    }
}

impl PartialOrd for CuvsLogicalNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.operation, &self.inputs, &self.input_orderings, &self.options).partial_cmp(&(
            &other.operation,
            &other.inputs,
            &other.input_orderings,
            &other.options,
        ))
    }
}

impl UserDefinedLogicalNodeCore for CuvsLogicalNode {
    fn name(&self) -> &str {
        "CuvsLogicalNode"
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
            "CuvsLogicalNode: operation={}, relation_count={}, options={}",
            self.operation.descriptor().sql_function_name,
            self.inputs.len(),
            self.options.canonical_json()
        )?;
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
                "cuVS logical node expressions must be its input sort keys",
            )
            .into());
        };
        let options_json =
            dense_relation_function(self.operation).options_json(&inputs, self.options.value_options().clone())?;
        Self::try_new(self.operation, inputs, &options_json)?
            .with_input_orderings(input_orderings)
            .map_err(DataFusionError::from)
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

fn validate_input_schemas(
    operation: CuvsOperation,
    inputs: &[LogicalPlan],
    options: &CuvsValidatedOptions,
) -> Result<(Vec<Option<usize>>, VectorElementType)> {
    let mut dimensions = Vec::with_capacity(inputs.len());
    let mut elements = Vec::with_capacity(inputs.len());
    for ((input, role), binding) in
        inputs.iter().zip(operation.descriptor().relation_roles.iter().copied()).zip(options.bindings())
    {
        if binding.relation_role() != role {
            return Err(invalid_call(
                operation,
                "binding_relation_mismatch",
                "cuVS options binding order does not match the relation arguments",
            ));
        }
        let (dimension, element) = validate_relation_binding(operation, input.schema(), binding)?;
        dimensions.push(dimension);
        elements.push(element);
    }
    if elements.len() == 2 && elements[0] != elements[1] {
        return Err(invalid_call(
            operation,
            "vector_element_type_mismatch",
            format!(
                "{} element {} does not match {} element {}",
                operation.descriptor().relation_roles[0].as_str(),
                elements[0].as_str(),
                operation.descriptor().relation_roles[1].as_str(),
                elements[1].as_str()
            ),
        ));
    }
    let element = elements[0];
    if let Some(metric) = options.value_options().get("metric").and_then(Value::as_str)
        && let Some(supported) = operation.descriptor().element_support(element)
        && !supported.metrics.contains(&metric)
    {
        return Err(invalid_call(
            operation,
            "unsupported_vector_element_metric",
            format!(
                "{} {} does not support metric '{metric}'; valid metrics: {}",
                operation.descriptor().sql_function_name,
                element.as_str(),
                supported.metrics.join(", ")
            ),
        ));
    }
    if matches!(
        operation,
        CuvsOperation::BruteForceKnn
            | CuvsOperation::IvfFlat
            | CuvsOperation::IvfPq
            | CuvsOperation::Cagra
            | CuvsOperation::PairwiseDistance
    ) && let [Some(first), Some(second)] = dimensions.as_slice()
        && first != second
    {
        return Err(invalid_call(
            operation,
            "vector_dimension_mismatch",
            format!(
                "{} dimension {first} does not match {} dimension {second}",
                operation.descriptor().relation_roles[0].as_str(),
                operation.descriptor().relation_roles[1].as_str()
            ),
        ));
    }
    let ivf_metric = match options.params() {
        VectorAlgorithmParams::IvfFlat(params) => Some(params.metric),
        VectorAlgorithmParams::IvfPq(params) => Some(params.metric),
        _ => None,
    };
    if ivf_metric == Some(VectorDistanceMetric::CosineExpanded)
        && let Some(Some(dimension)) = dimensions.first()
        && *dimension < 2
    {
        return Err(invalid_call(
            operation,
            "insufficient_feature_dimension",
            "cosine IVF search requires at least two dimensions",
        ));
    }
    if let VectorAlgorithmParams::TopK(params) = options.params()
        && let Some(Some(dimension)) = dimensions.first()
        && params.k > *dimension
    {
        return Err(invalid_call(operation, "k_exceeds_dimension", "Top-K k exceeds the vector dimension"));
    }
    if operation == CuvsOperation::Pca
        && let Some(Some(dimension)) = dimensions.first()
        && *dimension < 2
    {
        return Err(invalid_call(
            operation,
            "insufficient_feature_dimension",
            "cuVS PCA requires at least two input feature columns",
        ));
    }
    Ok((dimensions, element))
}

fn validate_relation_binding(
    operation: CuvsOperation,
    schema: &DFSchema,
    binding: &CuvsVectorRelationBinding,
) -> Result<(Option<usize>, VectorElementType)> {
    let role = binding.relation_role().as_str();
    let id = resolve_field(operation, schema, role, binding.id())?;
    if RelationIdType::from_data_type(id.data_type()).is_none() {
        return Err(unsupported_type(
            operation,
            role,
            binding.id(),
            id.data_type(),
            "Int32, Int64, Utf8, LargeUtf8, or Utf8View",
        ));
    }

    match binding.vector() {
        DenseVectorBinding::Columns(columns) => {
            let mut element = None;
            for column in columns {
                let field = resolve_field(operation, schema, role, column)?;
                let current = require_supported_vector_element(operation, role, column, field)?;
                if element.is_some_and(|element| element != current) {
                    return Err(invalid_call(
                        operation,
                        "vector_element_type_mismatch",
                        format!(
                            "cuVS {role} feature field '{column}' must have the same element type as the other feature fields"
                        ),
                    ));
                }
                element = Some(current);
            }
            Ok((Some(columns.len()), element.expect("validated non-empty columns")))
        },
        DenseVectorBinding::VectorColumn(column) => {
            let field = resolve_field(operation, schema, role, column)?;
            match field.data_type() {
                DataType::FixedSizeList(element, dimension) if *dimension > 0 => {
                    let element = require_supported_vector_element(operation, role, column, element)?;
                    Ok((Some(*dimension as usize), element))
                },
                DataType::List(element) | DataType::LargeList(element) => {
                    let element = require_supported_vector_element(operation, role, column, element)?;
                    Ok((None, element))
                },
                actual => Err(unsupported_type(
                    operation,
                    role,
                    column,
                    actual,
                    format!("FixedSizeList, List, or LargeList with {} elements", supported_vector_elements(operation)),
                )),
            }
        },
    }
}

fn resolve_field<'a>(
    operation: CuvsOperation,
    schema: &'a DFSchema,
    role: &'static str,
    name: &str,
) -> Result<&'a FieldRef> {
    schema.field_with_unqualified_name(name).map_err(|_| field_not_found(operation, role, name))
}

fn supported_vector_elements(operation: CuvsOperation) -> String {
    operation
        .descriptor()
        .vector_element_support
        .iter()
        .map(|support| support.element.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn require_supported_vector_element(
    operation: CuvsOperation,
    role: &'static str,
    name: &str,
    field: &FieldRef,
) -> Result<VectorElementType> {
    let element = VectorElementType::from_data_type(field.data_type())
        .filter(|element| operation.descriptor().element_support(*element).is_some())
        .ok_or_else(|| {
            unsupported_type(operation, role, name, field.data_type(), supported_vector_elements(operation))
        })?;
    Ok(element)
}

fn validate_known_cardinalities(
    operation: CuvsOperation,
    inputs: &[LogicalPlan],
    options: &CuvsValidatedOptions,
) -> Result<()> {
    let Some(input_rows) = known_row_count(&inputs[0]) else {
        return Ok(());
    };
    match options.params() {
        VectorAlgorithmParams::BruteForceKnn(params) => {
            if input_rows == 0 {
                return Err(invalid_call(
                    operation,
                    "empty_dataset",
                    "cuvs_brute_force_knn requires a non-empty dataset relation",
                ));
            }
            if params.k > input_rows {
                return Err(invalid_call(
                    operation,
                    "k_exceeds_dataset_rows",
                    format!("k ({}) exceeds the statically known dataset row count ({input_rows})", params.k),
                ));
            }
        },
        VectorAlgorithmParams::IvfFlat(VectorIvfFlatParams { k, n_lists, .. })
        | VectorAlgorithmParams::IvfPq(VectorIvfPqParams { k, n_lists, .. }) => {
            if input_rows == 0 || *k > input_rows || *n_lists as usize > input_rows {
                return Err(invalid_ann_dataset_cardinality(operation));
            }
        },
        VectorAlgorithmParams::Cagra(params) => {
            if input_rows == 0 || params.k > input_rows || params.intermediate_graph_degree as usize >= input_rows {
                return Err(invalid_ann_dataset_cardinality(operation));
            }
        },
        VectorAlgorithmParams::PairwiseDistance(_) | VectorAlgorithmParams::TopK(_) => {},
        VectorAlgorithmParams::KMeans(params) => {
            if input_rows == 0 {
                return Err(invalid_call(operation, "empty_input", "cuvs_kmeans requires a non-empty input relation"));
            }
            if params.n_clusters > input_rows {
                return Err(invalid_call(
                    operation,
                    "cluster_count_exceeds_input_rows",
                    format!(
                        "n_clusters ({}) exceeds the statically known input row count ({input_rows})",
                        params.n_clusters
                    ),
                ));
            }
        },
        VectorAlgorithmParams::Pca(_) => {
            if input_rows < 2 {
                return Err(invalid_call(
                    operation,
                    "insufficient_input_rows",
                    "cuvs_pca requires at least two input rows",
                ));
            }
        },
    }
    Ok(())
}

fn invalid_ann_dataset_cardinality(operation: CuvsOperation) -> Error {
    invalid_call(
        operation,
        "invalid_dataset_cardinality",
        "ANN dataset must be non-empty; k must not exceed rows and graph degrees must be less than rows",
    )
}

fn derive_output_schema(
    operation: CuvsOperation,
    inputs: &[LogicalPlan],
    options: &CuvsValidatedOptions,
    dimensions: &[Option<usize>],
    element: VectorElementType,
) -> DataFusionResult<DFSchemaRef> {
    let id_field = |ordinal: usize, output_name| {
        output_id_field(inputs[ordinal].schema(), options.bindings()[ordinal].id(), output_name)
            .map(|field| field.data_type().clone())
    };
    let schema = match options.params() {
        VectorAlgorithmParams::BruteForceKnn(_)
        | VectorAlgorithmParams::IvfFlat(_)
        | VectorAlgorithmParams::IvfPq(_)
        | VectorAlgorithmParams::Cagra(_) => {
            vector_knn_output_schema(id_field(1, "query_id")?, id_field(0, "neighbor_id")?)
        },
        VectorAlgorithmParams::PairwiseDistance(_) => vector_pairwise_distance_output_schema(
            id_field(0, "x_id")?,
            id_field(1, "y_id")?,
            match element {
                VectorElementType::Float64 => DataType::Float64,
                _ => DataType::Float32,
            },
        ),
        VectorAlgorithmParams::TopK(_) => vector_top_k_output_schema(id_field(0, "id")?),
        VectorAlgorithmParams::KMeans(_) => vector_kmeans_output_schema(id_field(0, "id")?),
        VectorAlgorithmParams::Pca(params) => {
            if let Some(dimension) = dimensions[0]
                && params.n_components > dimension
            {
                return Err(invalid_call(
                    operation,
                    "component_count_exceeds_dimension",
                    format!(
                        "n_components ({}) exceeds the statically known feature dimension ({dimension})",
                        params.n_components
                    ),
                )
                .into());
            }
            vector_pca_output_schema(id_field(0, "id")?, params.n_components)
        },
    };
    Ok(Arc::new(DFSchema::try_from(schema.as_ref().clone())?))
}

fn output_id_field(schema: &DFSchema, input_name: &str, output_name: &str) -> DataFusionResult<Field> {
    let input = schema.field_with_unqualified_name(input_name)?;
    Ok(Field::new(output_name, input.data_type().clone(), false))
}
