use std::{cmp::Ordering, fmt, hash::Hash, sync::Arc};

use arrow::datatypes::{DataType, Field, FieldRef, SchemaRef};
use datafusion::{
    common::{DFSchema, DFSchemaRef, Result},
    logical_expr::{Expr, FetchType, LogicalPlan, SkipType, UserDefinedLogicalNodeCore},
};
use query_engine::plan::{
    CuvsIdType, CuvsVectorElementType, vector_kmeans_output_schema, vector_knn_output_schema, vector_pca_output_schema,
};

use super::{
    CuvsDenseVectorBinding, CuvsOperation, CuvsOutputContract, CuvsValidatedOptions, CuvsVectorRelationBinding,
    error::{field_not_found, invalid_call, unsupported_type},
    options::{options_from_named_values, validate_options},
};

#[derive(Debug, Clone)]
pub struct CuvsLogicalNode {
    operation: CuvsOperation,
    inputs: Vec<LogicalPlan>,
    options: CuvsValidatedOptions,
    schema: DFSchemaRef,
}

impl CuvsLogicalNode {
    pub fn try_new(operation: CuvsOperation, inputs: Vec<LogicalPlan>, options_json: &str) -> Result<Self> {
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
            ));
        }
        let options = validate_options(operation, options_json)?;
        let dimensions = validate_input_schemas(operation, &inputs, &options)?;
        validate_known_cardinalities(operation, &inputs, &options)?;
        let schema = derive_output_schema(operation, &inputs, &options, &dimensions)?;
        Ok(Self { operation, inputs, options, schema })
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
        self.operation == other.operation && self.inputs == other.inputs && self.options == other.options
    }
}

impl Eq for CuvsLogicalNode {}

impl Hash for CuvsLogicalNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.operation.hash(state);
        self.inputs.hash(state);
        self.options.hash(state);
    }
}

impl PartialOrd for CuvsLogicalNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.operation, &self.inputs, &self.options).partial_cmp(&(&other.operation, &other.inputs, &other.options))
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

    fn expressions(&self) -> Vec<Expr> {
        Vec::new()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "CuvsLogicalNode: operation={}, relation_count={}, options={}",
            self.operation.descriptor().sql_function_name,
            self.inputs.len(),
            self.options.canonical_json()
        )
    }

    fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        if !exprs.is_empty() {
            return Err(invalid_call(
                self.operation,
                "unexpected_logical_expressions",
                "cuVS logical nodes do not expose optimizer expressions",
            ));
        }
        let options_json =
            options_from_named_values(self.operation, &inputs, self.options.named_values(self.operation))?;
        Self::try_new(self.operation, inputs, &options_json)
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

fn validate_input_schemas(
    operation: CuvsOperation,
    inputs: &[LogicalPlan],
    options: &CuvsValidatedOptions,
) -> Result<Vec<Option<usize>>> {
    let mut dimensions = Vec::with_capacity(inputs.len());
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
        dimensions.push(validate_relation_binding(operation, input.schema(), binding)?);
    }
    if operation == CuvsOperation::BruteForceKnn
        && let [Some(dataset), Some(queries)] = dimensions.as_slice()
        && dataset != queries
    {
        return Err(invalid_call(
            operation,
            "vector_dimension_mismatch",
            format!("dataset dimension {dataset} does not match query dimension {queries}"),
        ));
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
    Ok(dimensions)
}

fn validate_relation_binding(
    operation: CuvsOperation,
    schema: &DFSchema,
    binding: &CuvsVectorRelationBinding,
) -> Result<Option<usize>> {
    let role = binding.relation_role().as_str();
    let id = resolve_field(operation, schema, role, binding.id())?;
    if id.is_nullable() {
        return Err(invalid_call(
            operation,
            "nullable_id_field",
            format!("cuVS {role} ID field '{}' must be non-nullable", binding.id()),
        ));
    }
    if CuvsIdType::from_data_type(id.data_type()).is_none() {
        return Err(unsupported_type(
            operation,
            role,
            binding.id(),
            id.data_type(),
            "Int32, Int64, Utf8, LargeUtf8, or Utf8View",
        ));
    }

    match binding.vector() {
        CuvsDenseVectorBinding::Columns(columns) => {
            for column in columns {
                let field = resolve_field(operation, schema, role, column)?;
                require_supported_vector_element(operation, role, column, field)?;
            }
            Ok(Some(columns.len()))
        },
        CuvsDenseVectorBinding::VectorColumn(column) => {
            let field = resolve_field(operation, schema, role, column)?;
            if field.is_nullable() {
                return Err(invalid_call(
                    operation,
                    "nullable_vector_field",
                    format!("cuVS {role} vector field '{column}' must be non-nullable"),
                ));
            }
            match field.data_type() {
                DataType::FixedSizeList(element, dimension) if *dimension > 0 => {
                    require_supported_vector_element(operation, role, column, element)?;
                    Ok(Some(*dimension as usize))
                },
                DataType::List(element) | DataType::LargeList(element) => {
                    require_supported_vector_element(operation, role, column, element)?;
                    Ok(None)
                },
                actual => Err(unsupported_type(
                    operation,
                    role,
                    column,
                    actual,
                    format!(
                        "FixedSizeList, List, or LargeList with non-null {} elements",
                        supported_vector_elements(operation)
                    ),
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
    operation.descriptor().vector_element_support.iter().map(|element| element.as_str()).collect::<Vec<_>>().join(", ")
}

fn require_supported_vector_element(
    operation: CuvsOperation,
    role: &'static str,
    name: &str,
    field: &FieldRef,
) -> Result<()> {
    if field.is_nullable() {
        return Err(invalid_call(
            operation,
            "nullable_feature_field",
            format!("cuVS {role} feature field '{name}' must be non-nullable"),
        ));
    }
    if CuvsVectorElementType::from_data_type(field.data_type())
        .is_none_or(|element| !operation.descriptor().vector_element_support.contains(&element))
    {
        return Err(unsupported_type(operation, role, name, field.data_type(), supported_vector_elements(operation)));
    }
    Ok(())
}

fn validate_known_cardinalities(
    operation: CuvsOperation,
    inputs: &[LogicalPlan],
    options: &CuvsValidatedOptions,
) -> Result<()> {
    match operation {
        CuvsOperation::BruteForceKnn => {
            if let Some(dataset_rows) = known_row_count(&inputs[0]) {
                let k = options.knn_k().expect("validated kNN options contain k") as usize;
                if dataset_rows == 0 {
                    return Err(invalid_call(
                        operation,
                        "empty_dataset",
                        "cuvs_brute_force_knn requires a non-empty dataset relation",
                    ));
                }
                if k > dataset_rows {
                    return Err(invalid_call(
                        operation,
                        "k_exceeds_dataset_rows",
                        format!("k ({k}) exceeds the statically known dataset row count ({dataset_rows})"),
                    ));
                }
            }
        },
        CuvsOperation::KMeans => {
            if let Some(input_rows) = known_row_count(&inputs[0]) {
                let n_clusters =
                    options.kmeans_clusters().expect("validated KMeans options contain n_clusters") as usize;
                if input_rows == 0 {
                    return Err(invalid_call(
                        operation,
                        "empty_input",
                        "cuvs_kmeans requires a non-empty input relation",
                    ));
                }
                if n_clusters > input_rows {
                    return Err(invalid_call(
                        operation,
                        "cluster_count_exceeds_input_rows",
                        format!(
                            "n_clusters ({n_clusters}) exceeds the statically known input row count ({input_rows})"
                        ),
                    ));
                }
            }
        },
        CuvsOperation::Pca => {
            if let Some(input_rows) = known_row_count(&inputs[0])
                && input_rows < 2
            {
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

fn known_row_count(plan: &LogicalPlan) -> Option<usize> {
    match plan {
        LogicalPlan::EmptyRelation(empty) => Some(usize::from(empty.produce_one_row)),
        LogicalPlan::Values(values) => Some(values.values.len()),
        LogicalPlan::Projection(projection) => known_row_count(&projection.input),
        LogicalPlan::SubqueryAlias(alias) => known_row_count(&alias.input),
        LogicalPlan::Limit(limit) => {
            let input_rows = match limit.get_skip_type().ok()? {
                SkipType::Literal(skip) => known_row_count(&limit.input)?.saturating_sub(skip),
                SkipType::UnsupportedExpr => return None,
            };
            match limit.get_fetch_type().ok()? {
                FetchType::Literal(fetch) => Some(input_rows.min(fetch.unwrap_or(input_rows))),
                FetchType::UnsupportedExpr => None,
            }
        },
        _ => None,
    }
}

fn derive_output_schema(
    operation: CuvsOperation,
    inputs: &[LogicalPlan],
    options: &CuvsValidatedOptions,
    dimensions: &[Option<usize>],
) -> Result<DFSchemaRef> {
    let schema = match operation.descriptor().output_contract {
        CuvsOutputContract::KnnNeighbors => {
            let dataset_id = output_id_field(inputs[0].schema(), options.bindings()[0].id(), "neighbor_id")?;
            let query_id = output_id_field(inputs[1].schema(), options.bindings()[1].id(), "query_id")?;
            vector_knn_output_schema(query_id.data_type().clone(), dataset_id.data_type().clone())
        },
        CuvsOutputContract::KMeansAssignments => vector_kmeans_output_schema(
            output_id_field(inputs[0].schema(), options.bindings()[0].id(), "id")?.data_type().clone(),
        ),
        CuvsOutputContract::PcaComponents => {
            let n_components = options.pca_components().expect("validated PCA options contain n_components") as usize;
            if let Some(dimension) = dimensions[0]
                && n_components > dimension
            {
                return Err(invalid_call(
                    operation,
                    "component_count_exceeds_dimension",
                    format!(
                        "n_components ({n_components}) exceeds the statically known feature dimension ({dimension})"
                    ),
                ));
            }
            vector_pca_output_schema(
                output_id_field(inputs[0].schema(), options.bindings()[0].id(), "id")?.data_type().clone(),
                n_components,
            )
        },
    };
    Ok(Arc::new(DFSchema::try_from(schema.as_ref().clone())?))
}

fn output_id_field(schema: &DFSchema, input_name: &str, output_name: &str) -> Result<Field> {
    let input = schema.field_with_unqualified_name(input_name)?;
    Ok(Field::new(output_name, input.data_type().clone(), false))
}
