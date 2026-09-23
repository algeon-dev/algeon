//! Shared vector-algorithm contract validation.
//!
//! Capability analysis and execution preflight both consume this module so
//! neither layer delegates its contract ownership to the other.

use std::collections::{BTreeMap, BTreeSet};

use arrow::datatypes::{DataType, SchemaRef};

use crate::{
    backend::AnalysisContext,
    error::{QueryError, QueryErrorCode, Result},
    expr::Expr,
    plan::{
        CuvsDenseVectorBinding, CuvsIdType, CuvsOperation, CuvsOptionKind, CuvsVectorElementType, NodeId, PlanNode,
        ProjectionOutput, QueryNode, QueryPlan, VectorAlgorithmNode, VectorAlgorithmParams, VectorKMeansInit,
        VectorRelationBinding, VectorRowIdBinding, vector_kmeans_output_schema, vector_knn_output_schema,
        vector_pairwise_distance_output_schema, vector_pca_output_schema, vector_top_k_output_schema,
    },
};

pub fn vector_ingress_columns_by_input(plan: &QueryPlan) -> BTreeMap<NodeId, BTreeSet<String>> {
    let mut columns_by_input = BTreeMap::<NodeId, BTreeSet<String>>::new();
    for node in plan.nodes() {
        let PlanNode::VectorAlgorithm(vector) = &node.kind else {
            continue;
        };
        for binding in vector.spec.params.relation_bindings() {
            let CuvsDenseVectorBinding::VectorColumn(column) = &binding.vector else {
                continue;
            };
            mark_vector_ingress_lineage(plan, binding.input, column, &mut columns_by_input, &mut BTreeSet::new());
        }
    }
    columns_by_input
}

fn mark_vector_ingress_lineage(
    plan: &QueryPlan,
    node_id: NodeId,
    column: &str,
    columns_by_input: &mut BTreeMap<NodeId, BTreeSet<String>>,
    visited: &mut BTreeSet<(NodeId, String)>,
) {
    if !visited.insert((node_id, column.to_string())) {
        return;
    }
    columns_by_input.entry(node_id).or_default().insert(column.to_string());
    let Some(node) = plan.node(node_id) else {
        return;
    };
    match &node.kind {
        PlanNode::Projection(projection) => {
            let Some(output) = projection.outputs.iter().find(|output| output.field.name() == column) else {
                return;
            };
            let Some(input_column) = direct_column_name(&output.expr) else {
                return;
            };
            mark_vector_ingress_lineage(plan, projection.input, input_column, columns_by_input, visited);
        },
        PlanNode::Filter(filter) => {
            mark_vector_ingress_lineage(plan, filter.input, column, columns_by_input, visited);
        },
        PlanNode::Sort(sort) => {
            mark_vector_ingress_lineage(plan, sort.input, column, columns_by_input, visited);
        },
        PlanNode::Limit(limit) => {
            mark_vector_ingress_lineage(plan, limit.input, column, columns_by_input, visited);
        },
        _ => {},
    }
}

fn direct_column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(column) => Some(column.name()),
        Expr::Alias { expr, .. } => direct_column_name(expr),
        _ => None,
    }
}

pub fn validate_vector_ingress_projection_output(
    output: &ProjectionOutput,
    input_schema: &SchemaRef,
    available_columns: &BTreeSet<String>,
    input_vector_ingress_columns: Option<&BTreeSet<String>>,
) -> Result<()> {
    let Some(input_column) = direct_column_name(&output.expr) else {
        return Err(vector_algorithm_error("vector-column ingress projections must be direct column passthroughs")
            .with_fact("reason", "unsupported_vector_ingress_projection")
            .with_fact("output_column", output.field.name()));
    };
    if !available_columns.contains(input_column) {
        return Err(vector_algorithm_error("vector-column ingress projection input is not produced by its relation")
            .with_fact("reason", "missing_feature_column")
            .with_fact("column", input_column)
            .with_fact("output_column", output.field.name()));
    }
    if input_vector_ingress_columns.is_none_or(|columns| !columns.contains(input_column)) {
        return Err(vector_algorithm_error(
            "vector-column ingress projection is not connected to a typed vector input",
        )
        .with_fact("reason", "unsupported_vector_ingress_projection")
        .with_fact("column", input_column)
        .with_fact("output_column", output.field.name()));
    }
    let input_field = input_schema.field_with_name(input_column).map_err(|_| {
        vector_algorithm_error("vector-column ingress projection input is missing from its schema")
            .with_fact("reason", "missing_feature_column")
            .with_fact("column", input_column)
            .with_fact("output_column", output.field.name())
    })?;
    if !matches!(input_field.data_type(), DataType::FixedSizeList(_, _) | DataType::List(_) | DataType::LargeList(_))
        || input_field.data_type() != output.field.data_type()
        || input_field.is_nullable() != output.field.is_nullable()
    {
        return Err(vector_algorithm_error(
            "vector-column ingress projection must preserve its list type and nullability exactly",
        )
        .with_fact("reason", "unsupported_vector_ingress_projection")
        .with_fact("column", input_column)
        .with_fact("input_type", format!("{:?}", input_field.data_type()))
        .with_fact("output_type", format!("{:?}", output.field.data_type()))
        .with_fact("input_nullable", input_field.is_nullable())
        .with_fact("output_nullable", output.field.is_nullable()));
    }
    Ok(())
}

pub fn validate_schema_supported_with_vector_ingress(
    schema: &SchemaRef,
    operation: &'static str,
    vector_ingress_columns: Option<&BTreeSet<String>>,
    ctx: AnalysisContext<'_>,
) -> Result<()> {
    for field in schema.fields() {
        if ctx.backend().supports_type(field.data_type()) {
            continue;
        }
        let is_scoped_vector_ingress = vector_ingress_columns.is_some_and(|columns| columns.contains(field.name()))
            && matches!(field.data_type(), DataType::FixedSizeList(_, _) | DataType::List(_) | DataType::LargeList(_));
        if is_scoped_vector_ingress {
            continue;
        }
        return Err(QueryError::new(QueryErrorCode::UnsupportedType, operation, "schema field type is not supported")
            .with_fact("field", field.name())
            .with_fact("data_type", format!("{:?}", field.data_type())));
    }
    Ok(())
}

pub fn validate_vector_algorithm_node(
    vector: &VectorAlgorithmNode,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<()> {
    if !vector.spec.params.supports_algorithm(vector.spec.algorithm) {
        return Err(vector_algorithm_error("vector algorithm and parameter variant must match")
            .with_fact("node_id", node_id)
            .with_fact("algorithm", vector.spec.algorithm.as_str())
            .with_fact("reason", "algorithm_parameter_mismatch"));
    }
    let expected_inputs = vector.spec.params.relation_bindings().map(|binding| binding.input).collect::<Vec<_>>();
    if vector.inputs != expected_inputs {
        return Err(vector_algorithm_error("vector algorithm input list must match its relation bindings")
            .with_fact("node_id", node_id)
            .with_fact("algorithm", vector.spec.algorithm.as_str())
            .with_fact("actual_inputs", comma_separated_node_ids(&vector.inputs))
            .with_fact("expected_inputs", comma_separated_node_ids(&expected_inputs)));
    }

    let expected_output_schema = match &vector.spec.params {
        VectorAlgorithmParams::BruteForceKnn(params) => {
            if params.k == 0 {
                return Err(vector_algorithm_error("brute-force kNN requires k greater than zero")
                    .with_fact("node_id", node_id)
                    .with_fact("algorithm", vector.spec.algorithm.as_str())
                    .with_fact("reason", "zero_k"));
            }
            validate_registry_maximum(CuvsOperation::BruteForceKnn, "k", params.k as u64, node_id)?;
            let dataset = validate_vector_relation_binding(
                CuvsOperation::BruteForceKnn,
                "dataset",
                &params.dataset,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            let queries = validate_vector_relation_binding(
                CuvsOperation::BruteForceKnn,
                "queries",
                &params.queries,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            if let (Some(dataset_dimension), Some(query_dimension)) = (dataset.dimension, queries.dimension)
                && dataset_dimension != query_dimension
            {
                return Err(vector_algorithm_error("brute-force kNN dataset and query feature dimensions must match")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "feature_dimension_mismatch")
                    .with_fact("dataset_features", dataset_dimension)
                    .with_fact("query_features", query_dimension));
            }
            validate_matching_element_types("dataset", &dataset, "queries", &queries, node_id)?;
            vector_knn_output_schema(queries.id_type, dataset.id_type)
        },
        VectorAlgorithmParams::PairwiseDistance(params) => {
            let x = validate_vector_relation_binding(
                CuvsOperation::PairwiseDistance,
                "x",
                &params.x,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            let y = validate_vector_relation_binding(
                CuvsOperation::PairwiseDistance,
                "y",
                &params.y,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            validate_matching_element_types("x", &x, "y", &y, node_id)?;
            if let (Some(x_dimension), Some(y_dimension)) = (x.dimension, y.dimension)
                && x_dimension != y_dimension
            {
                return Err(vector_algorithm_error("pairwise vector dimensions must match")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "feature_dimension_mismatch")
                    .with_fact("x_dimension", x_dimension)
                    .with_fact("y_dimension", y_dimension));
            }
            vector_pairwise_distance_output_schema(x.id_type, y.id_type)
        },
        VectorAlgorithmParams::TopK(params) => {
            if params.k == 0 {
                return Err(vector_algorithm_error("Top-K requires k greater than zero")
                    .with_fact("node_id", node_id)
                    .with_fact("algorithm", vector.spec.algorithm.as_str())
                    .with_fact("reason", "zero_k"));
            }
            validate_registry_maximum(CuvsOperation::TopK, "k", params.k as u64, node_id)?;
            let input = validate_vector_relation_binding(
                CuvsOperation::TopK,
                "input",
                &params.input,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            if input.dimension.is_some_and(|dimension| params.k > dimension) {
                return Err(vector_algorithm_error("Top-K k must be within the input dimension")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "k_exceeds_dimension")
                    .with_fact("k", params.k));
            }
            vector_top_k_output_schema(input.id_type)
        },
        VectorAlgorithmParams::KMeans(params) => {
            if params.n_clusters == 0 || params.max_iterations == 0 || params.n_init == 0 {
                return Err(vector_algorithm_error("K-means positive-count parameters must be greater than zero")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "zero_positive_count_parameter")
                    .with_fact("n_clusters", params.n_clusters)
                    .with_fact("max_iterations", params.max_iterations)
                    .with_fact("n_init", params.n_init));
            }
            validate_registry_maximum(CuvsOperation::KMeans, "n_clusters", params.n_clusters as u64, node_id)?;
            validate_registry_maximum(CuvsOperation::KMeans, "max_iter", u64::from(params.max_iterations), node_id)?;
            validate_registry_maximum(CuvsOperation::KMeans, "n_init", u64::from(params.n_init), node_id)?;
            if !params.tolerance.is_finite() || params.tolerance < 0.0 {
                return Err(vector_algorithm_error("K-means tolerance must be finite and nonnegative")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "invalid_tolerance")
                    .with_fact("tolerance", params.tolerance));
            }
            let input = validate_vector_relation_binding(
                CuvsOperation::KMeans,
                "input",
                &params.input,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            match (params.init, &params.initial_centroids) {
                (VectorKMeansInit::Relation, Some(centroids)) => {
                    let centroids = validate_vector_relation_binding(
                        CuvsOperation::KMeans,
                        "initial_centroids",
                        centroids,
                        plan,
                        available_columns_by_node,
                        node_id,
                    )?;
                    validate_matching_element_types("input", &input, "initial_centroids", &centroids, node_id)?;
                    if let (Some(input_dimension), Some(centroid_dimension)) = (input.dimension, centroids.dimension)
                        && input_dimension != centroid_dimension
                    {
                        return Err(vector_algorithm_error(
                            "K-means input and initial-centroid feature dimensions must match",
                        )
                        .with_fact("node_id", node_id)
                        .with_fact("reason", "feature_dimension_mismatch")
                        .with_fact("input_features", input_dimension)
                        .with_fact("centroid_features", centroid_dimension));
                    }
                },
                (VectorKMeansInit::Relation, None) => {
                    return Err(vector_algorithm_error(
                        "relation-initialized K-means requires an initial-centroids relation",
                    )
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "missing_initial_centroids"));
                },
                (_, Some(_)) => {
                    return Err(vector_algorithm_error(
                        "K-means initial-centroids relation requires relation initialization",
                    )
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "unexpected_initial_centroids"));
                },
                (_, None) => {},
            }
            vector_kmeans_output_schema(input.id_type)
        },
        VectorAlgorithmParams::Pca(params) => {
            let input = validate_vector_relation_binding(
                CuvsOperation::Pca,
                "input",
                &params.input,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            if params.n_iterations == 0 {
                return Err(vector_algorithm_error("PCA iteration count must be greater than zero")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "zero_n_iterations"));
            }
            validate_registry_maximum(CuvsOperation::Pca, "n_components", params.n_components as u64, node_id)?;
            validate_registry_maximum(CuvsOperation::Pca, "n_iterations", u64::from(params.n_iterations), node_id)?;
            if !params.tolerance.is_finite() || params.tolerance < 0.0 {
                return Err(vector_algorithm_error("PCA tolerance must be finite and nonnegative")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "invalid_tolerance")
                    .with_fact("tolerance", params.tolerance));
            }
            if let Some(feature_count) = input.dimension.filter(|dimension| *dimension < 2) {
                return Err(vector_algorithm_error("PCA requires at least two input feature columns")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "insufficient_feature_dimension")
                    .with_fact("feature_count", feature_count)
                    .with_fact("minimum_feature_count", 2));
            }
            if params.n_components == 0 || input.dimension.is_some_and(|dimension| params.n_components > dimension) {
                return Err(vector_algorithm_error("PCA component count must be within the input feature dimension")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "invalid_component_count")
                    .with_fact("n_components", params.n_components)
                    .with_fact("feature_count", input.dimension.unwrap_or(0)));
            }
            vector_pca_output_schema(input.id_type, params.n_components)
        },
    };
    if vector.output_schema().as_ref() != expected_output_schema.as_ref() {
        return Err(vector_algorithm_error("vector algorithm output schema must match its relational result contract")
            .with_fact("node_id", node_id)
            .with_fact("algorithm", vector.spec.algorithm.as_str())
            .with_fact("reason", "unexpected_output_schema")
            .with_fact("output_fields", vector.output_schema().fields().len())
            .with_fact("expected_output_fields", expected_output_schema.fields().len()));
    }
    Ok(())
}

fn validate_registry_maximum(
    operation: CuvsOperation,
    option_name: &'static str,
    value: u64,
    node_id: NodeId,
) -> Result<()> {
    let maximum = match operation.descriptor().option(option_name).map(|option| option.kind) {
        Some(CuvsOptionKind::PositiveInteger { maximum, .. }) => maximum,
        _ => {
            return Err(QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.vector_algorithm.validate",
                "canonical cuVS registry is missing a positive-integer option contract",
            )
            .with_fact("node_id", node_id)
            .with_fact("operation", operation.descriptor().sql_function_name)
            .with_fact("option", option_name));
        },
    };
    if value > maximum {
        return Err(vector_algorithm_error("vector algorithm parameter exceeds its canonical registry maximum")
            .with_fact("node_id", node_id)
            .with_fact("reason", "parameter_exceeds_registry_maximum")
            .with_fact("operation", operation.descriptor().sql_function_name)
            .with_fact("option", option_name)
            .with_fact("value", value)
            .with_fact("maximum", maximum));
    }
    Ok(())
}

pub fn validate_vector_algorithm_node_contract(
    vector: &VectorAlgorithmNode,
    plan: &QueryPlan,
    node_id: NodeId,
) -> Result<()> {
    let available_columns_by_node = plan
        .nodes()
        .iter()
        .take(node_id.index())
        .map(|node| schema_column_names(node.output_schema()))
        .collect::<Vec<_>>();
    validate_vector_algorithm_node(vector, plan, &available_columns_by_node, node_id)
}

struct VectorRelationValidation {
    id_type: DataType,
    dimension: Option<usize>,
    element_type: CuvsVectorElementType,
}

fn validate_matching_element_types(
    left_role: &'static str,
    left: &VectorRelationValidation,
    right_role: &'static str,
    right: &VectorRelationValidation,
    node_id: NodeId,
) -> Result<()> {
    if left.element_type != right.element_type {
        return Err(vector_algorithm_error("vector relation element types must match")
            .with_fact("node_id", node_id)
            .with_fact("reason", "vector_element_type_mismatch")
            .with_fact("left_role", left_role)
            .with_fact("left_type", left.element_type.as_str())
            .with_fact("right_role", right_role)
            .with_fact("right_type", right.element_type.as_str()));
    }
    Ok(())
}

fn validate_vector_element_support(
    operation: CuvsOperation,
    role: &'static str,
    relation: &VectorRelationValidation,
    node_id: NodeId,
) -> Result<()> {
    if !operation.descriptor().vector_element_support.contains(&relation.element_type) {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedType,
            "query_engine.vector_algorithm.validate",
            "vector element type is not supported by this operation",
        )
        .with_fact("node_id", node_id)
        .with_fact("relation_role", role)
        .with_fact("reason", "unsupported_feature_type")
        .with_fact("data_type", relation.element_type.as_str())
        .with_fact(
            "supported_types",
            operation
                .descriptor()
                .vector_element_support
                .iter()
                .map(|element| element.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        ));
    }
    Ok(())
}

fn validate_vector_relation_binding(
    operation: CuvsOperation,
    role: &'static str,
    binding: &VectorRelationBinding,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    node_id: NodeId,
) -> Result<VectorRelationValidation> {
    let input = input_node(plan, binding.input, node_id)?;
    let available = available_columns(available_columns_by_node, binding.input)?;
    let (dimension, element_type) = match &binding.vector {
        CuvsDenseVectorBinding::Columns(columns) => {
            if columns.is_empty() {
                return Err(vector_algorithm_error("vector relation must bind at least one feature column")
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
                    .with_fact("reason", "empty_feature_columns"));
            }
            let mut feature_names = BTreeSet::new();
            let mut element_type = None;
            for column in columns {
                if !feature_names.insert(column.as_str()) {
                    return Err(vector_algorithm_error("vector relation feature columns must be unique")
                        .with_fact("node_id", node_id)
                        .with_fact("relation_role", role)
                        .with_fact("reason", "duplicate_feature_column")
                        .with_fact("column", column));
                }
                if !available.contains(column) {
                    return Err(vector_algorithm_error("vector feature column is not produced by its input relation")
                        .with_fact("node_id", node_id)
                        .with_fact("relation_role", role)
                        .with_fact("reason", "missing_feature_column")
                        .with_fact("column", column));
                }
                let field = input.output_schema().field_with_name(column).map_err(|_| {
                    vector_algorithm_error("vector feature column is not present in its input schema")
                        .with_fact("node_id", node_id)
                        .with_fact("relation_role", role)
                        .with_fact("reason", "missing_feature_column")
                        .with_fact("column", column)
                })?;
                let current_type = CuvsVectorElementType::from_data_type(field.data_type());
                if current_type.is_none() || field.is_nullable() {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "query_engine.vector_algorithm.validate",
                        "vector feature columns must have a non-null supported element type",
                    )
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
                    .with_fact("reason", "unsupported_feature_type")
                    .with_fact("column", column)
                    .with_fact("data_type", format!("{:?}", field.data_type()))
                    .with_fact("nullable", field.is_nullable()));
                }
                let current_type = current_type.expect("validated vector element type");
                if element_type.is_some_and(|element| element != current_type) {
                    return Err(vector_algorithm_error("vector feature columns must have the same element type")
                        .with_fact("node_id", node_id)
                        .with_fact("relation_role", role)
                        .with_fact("reason", "vector_element_type_mismatch")
                        .with_fact("column", column));
                }
                element_type = Some(current_type);
            }
            (Some(columns.len()), element_type.expect("non-empty feature columns"))
        },
        CuvsDenseVectorBinding::VectorColumn(column) => {
            if !available.contains(column) {
                return Err(vector_algorithm_error("vector column is not produced by its input relation")
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
                    .with_fact("reason", "missing_feature_column")
                    .with_fact("column", column));
            }
            let field = input.output_schema().field_with_name(column).map_err(|_| {
                vector_algorithm_error("vector column is not present in its input schema")
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
                    .with_fact("reason", "missing_feature_column")
                    .with_fact("column", column)
            })?;
            if field.is_nullable() {
                return Err(vector_algorithm_error("vector column must be non-nullable")
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
                    .with_fact("reason", "unsupported_feature_type")
                    .with_fact("column", column));
            }
            let (element, dimension) = match field.data_type() {
                DataType::FixedSizeList(element, dimension) if *dimension > 0 => (element, Some(*dimension as usize)),
                DataType::List(element) | DataType::LargeList(element) => (element, None),
                actual => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedType,
                        "query_engine.vector_algorithm.validate",
                        "vector column must be a fixed-size or variable-size list",
                    )
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
                    .with_fact("reason", "unsupported_feature_type")
                    .with_fact("column", column)
                    .with_fact("data_type", format!("{actual:?}")));
                },
            };
            if element.is_nullable() {
                return Err(vector_algorithm_error("vector column elements must be non-null")
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
                    .with_fact("reason", "unsupported_feature_type")
                    .with_fact("column", column));
            }
            let element_type = CuvsVectorElementType::from_data_type(element.data_type()).ok_or_else(|| {
                vector_algorithm_error("vector column element type is not supported")
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
                    .with_fact("reason", "unsupported_feature_type")
                    .with_fact("column", column)
                    .with_fact("data_type", format!("{:?}", element.data_type()))
            })?;
            (dimension, element_type)
        },
    };
    let relation = VectorRelationValidation { id_type: DataType::Int64, dimension, element_type };
    validate_vector_element_support(operation, role, &relation, node_id)?;
    match &binding.row_id {
        VectorRowIdBinding::RowIndex if role == "initial_centroids" => Ok(relation),
        VectorRowIdBinding::RowIndex => {
            Err(vector_algorithm_error("vector result relations require an explicit logical row-id column")
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("reason", "missing_logical_row_id_column"))
        },
        VectorRowIdBinding::Column(column) => {
            if !available.contains(column) {
                return Err(vector_algorithm_error("vector row-id column is not produced by its input relation")
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
                    .with_fact("reason", "missing_row_id_column")
                    .with_fact("column", column));
            }
            let field = input.output_schema().field_with_name(column).map_err(|_| {
                vector_algorithm_error("vector row-id column is not present in its input schema")
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
                    .with_fact("reason", "missing_row_id_column")
                    .with_fact("column", column)
            })?;
            if CuvsIdType::from_data_type(field.data_type()).is_none() || field.is_nullable() {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "query_engine.vector_algorithm.validate",
                    "vector row-id columns must be non-null Int32, Int64, Utf8, LargeUtf8, or Utf8View values",
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("reason", "unsupported_row_id_type")
                .with_fact("column", column)
                .with_fact("data_type", format!("{:?}", field.data_type()))
                .with_fact("nullable", field.is_nullable()));
            }
            Ok(VectorRelationValidation { id_type: field.data_type().clone(), ..relation })
        },
    }
}

fn input_node(plan: &QueryPlan, input: NodeId, consumer: NodeId) -> Result<&QueryNode> {
    plan.node(input).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.plan.validate",
            "query plan node references a missing input",
        )
        .with_fact("consumer", consumer)
        .with_fact("input", input)
    })
}

fn available_columns(columns_by_node: &[BTreeSet<String>], input: NodeId) -> Result<&BTreeSet<String>> {
    columns_by_node.get(input.index()).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            "query_engine.plan.validate",
            "query plan node references an input that has not been validated",
        )
        .with_fact("input", input)
    })
}

fn schema_column_names(schema: &SchemaRef) -> BTreeSet<String> {
    schema.fields().iter().map(|field| field.name().clone()).collect()
}

fn comma_separated_node_ids(ids: &[NodeId]) -> String {
    ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
}

fn vector_algorithm_error(message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::InvalidPlan, "query_engine.vector_algorithm.validate", message)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{Field, Schema};

    use super::*;
    use crate::{
        plan::{
            QueryPlanBuilder, VectorAlgorithm, VectorAlgorithmSpec, VectorBruteForceKnnParams, VectorDistanceMetric,
            VectorKMeansParams, VectorPcaParams, VectorTopKParams,
        },
        source::{GpuResidentSource, SourceNode},
    };

    #[test]
    fn unsupported_element_types_are_rejected_per_binding_before_cross_binding_checks() {
        let float32_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Float32, false),
        ]));
        let float64_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("embedding", DataType::List(Arc::new(Field::new("item", DataType::Float64, false))), false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let dataset =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("dataset", Arc::clone(&float32_schema))));
        let queries =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("queries", Arc::clone(&float64_schema))));
        let params = VectorBruteForceKnnParams::new(
            VectorRelationBinding::new(dataset, VectorRowIdBinding::column("id"), ["x"]),
            VectorRelationBinding::new_vector_column(queries, VectorRowIdBinding::column("id"), "embedding"),
            1,
            VectorDistanceMetric::L2Expanded,
        );
        let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
            VectorAlgorithm::BruteForceKnn,
            VectorAlgorithmParams::BruteForceKnn(params),
            vector_knn_output_schema(DataType::Int64, DataType::Int64),
        ));
        let plan = builder.finish_with_host_sink(root);
        let PlanNode::VectorAlgorithm(vector) = &plan.node(root).expect("vector root").kind else {
            panic!("vector root")
        };
        let error = validate_vector_algorithm_node_contract(vector, &plan, root).expect_err("mixed element types");
        assert_eq!(error.fact_value("reason"), Some("unsupported_feature_type"));
        assert_eq!(error.fact_value("data_type"), Some("Float64"));

        let mut builder = QueryPlanBuilder::new();
        let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", float32_schema)));
        let centroids =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("centroids", float64_schema)));
        let params =
            VectorKMeansParams::new(VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x"]), 1)
                .with_initial_centroids(VectorRelationBinding::new_vector_column(
                    centroids,
                    VectorRowIdBinding::RowIndex,
                    "embedding",
                ));
        let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
            VectorAlgorithm::KMeans,
            VectorAlgorithmParams::KMeans(params),
            vector_kmeans_output_schema(DataType::Int64),
        ));
        let plan = builder.finish_with_host_sink(root);
        let PlanNode::VectorAlgorithm(vector) = &plan.node(root).expect("vector root").kind else {
            panic!("vector root")
        };
        let error =
            validate_vector_algorithm_node_contract(vector, &plan, root).expect_err("mixed centroid element type");
        assert_eq!(error.fact_value("reason"), Some("unsupported_feature_type"));
        assert_eq!(error.fact_value("relation_role"), Some("initial_centroids"));
    }

    #[test]
    fn top_k_zero_and_oversized_k_have_distinct_reasons() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Float32, false),
        ]));
        for (k, reason) in [(0, "zero_k"), (2, "k_exceeds_dimension")] {
            let mut builder = QueryPlanBuilder::new();
            let input =
                builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", Arc::clone(&schema))));
            let params = VectorTopKParams {
                input: VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x"]),
                k,
                select_min: true,
            };
            let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
                VectorAlgorithm::TopK,
                VectorAlgorithmParams::TopK(params),
                vector_top_k_output_schema(DataType::Int64),
            ));
            let plan = builder.finish_with_host_sink(root);
            let PlanNode::VectorAlgorithm(vector) = &plan.node(root).expect("vector root").kind else {
                panic!("vector root")
            };
            let error = validate_vector_algorithm_node_contract(vector, &plan, root).expect_err("invalid k");
            assert_eq!(error.fact_value("reason"), Some(reason));
        }
    }

    #[test]
    fn relation_features_must_be_present_in_the_execution_available_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Float32, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("input", schema)));
        let params =
            VectorPcaParams::new(VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x"]), 1);
        let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
            VectorAlgorithm::Pca,
            VectorAlgorithmParams::Pca(params),
            vector_pca_output_schema(DataType::Int64, 1),
        ));
        let plan = builder.finish_with_host_sink(root);
        let mut available_columns_by_node = plan
            .nodes()
            .iter()
            .take(root.index())
            .map(|node| schema_column_names(node.output_schema()))
            .collect::<Vec<_>>();
        assert!(available_columns_by_node[input.index()].remove("x"));
        let PlanNode::VectorAlgorithm(vector) = &plan.node(root).expect("vector root").kind else {
            panic!("root must be a vector algorithm");
        };

        let error = validate_vector_algorithm_node(vector, &plan, &available_columns_by_node, root)
            .expect_err("schema-only feature columns must not pass execution-frame validation");
        assert_eq!(error.fact_value("reason"), Some("missing_feature_column"));
        assert_eq!(error.fact_value("column"), Some("x"));
    }
}
