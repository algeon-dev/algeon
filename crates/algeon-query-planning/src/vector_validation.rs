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
        CuvsOperation, DenseVectorBinding, GpuOptionKind, NodeId, PlanNode, ProjectionOutput, QueryNode, QueryPlan,
        RelationIdType, VectorAlgorithmNode, VectorAlgorithmParams, VectorCagraBuildAlgo, VectorCagraSearchAlgo,
        VectorDistanceMetric, VectorElementType, VectorKMeansInit, VectorRelationBinding, VectorRowIdBinding,
        vector_kmeans_output_schema, vector_knn_output_schema, vector_pairwise_distance_output_schema,
        vector_pca_output_schema, vector_top_k_output_schema,
    },
    validation::algorithm_input,
};

pub(crate) fn vector_ingress_columns_by_input(plan: &QueryPlan) -> BTreeMap<NodeId, BTreeSet<String>> {
    let mut columns_by_input = BTreeMap::<NodeId, BTreeSet<String>>::new();
    for node in plan.nodes() {
        let bindings = match &node.kind {
            PlanNode::VectorAlgorithm(vector) => vector
                .spec
                .params
                .relation_bindings()
                .map(|binding| (vector.inputs[binding.input], &binding.vector, &binding.row_id))
                .collect::<Vec<_>>(),
            PlanNode::MlAlgorithm(ml) => ml
                .spec
                .params
                .relation_bindings()
                .map(|binding| (ml.inputs[binding.input], &binding.features, &binding.row_id))
                .collect::<Vec<_>>(),
            _ => continue,
        };
        for (input, vector, _row_id) in bindings {
            let DenseVectorBinding::VectorColumn(column) = vector else {
                continue;
            };
            mark_vector_ingress_lineage(plan, input, column, &mut columns_by_input, &mut BTreeSet::new());
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

pub(crate) fn validate_vector_ingress_projection_output(
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

pub(crate) fn validate_schema_supported_with_vector_ingress(
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

pub(crate) fn validate_vector_algorithm_node(
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
            validate_element_metric(
                CuvsOperation::BruteForceKnn,
                dataset.element_type,
                metric_name(params.metric),
                node_id,
            )?;
            vector_knn_output_schema(queries.id_type, dataset.id_type)
        },
        VectorAlgorithmParams::IvfFlat(params) => {
            if params.k == 0
                || params.n_lists == 0
                || params.n_probes == 0
                || params.kmeans_n_iters == 0
                || params.n_probes > params.n_lists
                || !params.kmeans_trainset_fraction.is_finite()
                || params.kmeans_trainset_fraction <= 0.0
                || params.kmeans_trainset_fraction > 1.0
            {
                return Err(vector_algorithm_error("invalid IVF-Flat build or search parameters")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "invalid_ivf_flat_parameters"));
            }
            validate_registry_maximum(CuvsOperation::IvfFlat, "k", params.k as u64, node_id)?;
            let dataset = validate_vector_relation_binding(
                CuvsOperation::IvfFlat,
                "dataset",
                &params.dataset,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            let queries = validate_vector_relation_binding(
                CuvsOperation::IvfFlat,
                "queries",
                &params.queries,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            validate_matching_element_types("dataset", &dataset, "queries", &queries, node_id)?;
            if let (Some(dataset_dimension), Some(query_dimension)) = (dataset.dimension, queries.dimension)
                && dataset_dimension != query_dimension
            {
                return Err(vector_algorithm_error("IVF-Flat dataset and query dimensions must match")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "feature_dimension_mismatch"));
            }
            if params.metric == VectorDistanceMetric::CosineExpanded
                && dataset.dimension.is_some_and(|dimension| dimension < 2)
            {
                return Err(vector_algorithm_error("cosine IVF-Flat requires at least two dimensions")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "insufficient_feature_dimension"));
            }
            validate_element_metric(CuvsOperation::IvfFlat, dataset.element_type, metric_name(params.metric), node_id)?;
            vector_knn_output_schema(queries.id_type, dataset.id_type)
        },
        VectorAlgorithmParams::IvfPq(params) => {
            if params.k == 0
                || params.n_lists == 0
                || params.n_probes == 0
                || params.n_probes > params.n_lists
                || params.kmeans_n_iters == 0
                || !params.kmeans_trainset_fraction.is_finite()
                || params.kmeans_trainset_fraction <= 0.0
                || params.kmeans_trainset_fraction > 1.0
                || !(4..=8).contains(&params.pq_bits)
                || (params.pq_dim != 0 && (u64::from(params.pq_dim) * u64::from(params.pq_bits)) % 8 != 0)
            {
                return Err(vector_algorithm_error("invalid IVF-PQ build or search parameters")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "invalid_ivf_pq_parameters"));
            }
            validate_registry_maximum(CuvsOperation::IvfPq, "k", params.k as u64, node_id)?;
            let dataset = validate_vector_relation_binding(
                CuvsOperation::IvfPq,
                "dataset",
                &params.dataset,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            let queries = validate_vector_relation_binding(
                CuvsOperation::IvfPq,
                "queries",
                &params.queries,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            validate_matching_element_types("dataset", &dataset, "queries", &queries, node_id)?;
            if let (Some(dataset_dimension), Some(query_dimension)) = (dataset.dimension, queries.dimension)
                && dataset_dimension != query_dimension
            {
                return Err(vector_algorithm_error("IVF-PQ dataset and query dimensions must match")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "feature_dimension_mismatch"));
            }
            if params.metric == VectorDistanceMetric::CosineExpanded
                && dataset.dimension.is_some_and(|dimension| dimension < 2)
            {
                return Err(vector_algorithm_error("cosine IVF-PQ requires at least two dimensions")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "insufficient_feature_dimension"));
            }
            validate_element_metric(CuvsOperation::IvfPq, dataset.element_type, metric_name(params.metric), node_id)?;
            vector_knn_output_schema(queries.id_type, dataset.id_type)
        },
        VectorAlgorithmParams::Cagra(params) => {
            if params.k == 0
                || params.graph_degree == 0
                || params.intermediate_graph_degree <= params.graph_degree
                || u64::from(params.itopk_size) < params.k as u64
                || params.search_width == 0
                || params.metric == VectorDistanceMetric::L2SqrtExpanded
                || (params.metric == VectorDistanceMetric::CosineExpanded
                    && params.build_algo == VectorCagraBuildAlgo::NnDescent)
                || (params.search_algo == VectorCagraSearchAlgo::SingleCta && params.itopk_size > 1024)
            {
                return Err(vector_algorithm_error("invalid CAGRA build or search parameters")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "invalid_cagra_parameters"));
            }
            validate_registry_maximum(CuvsOperation::Cagra, "k", params.k as u64, node_id)?;
            let dataset = validate_vector_relation_binding(
                CuvsOperation::Cagra,
                "dataset",
                &params.dataset,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            let queries = validate_vector_relation_binding(
                CuvsOperation::Cagra,
                "queries",
                &params.queries,
                plan,
                available_columns_by_node,
                node_id,
            )?;
            validate_matching_element_types("dataset", &dataset, "queries", &queries, node_id)?;
            if let (Some(first), Some(second)) = (dataset.dimension, queries.dimension)
                && first != second
            {
                return Err(vector_algorithm_error("CAGRA dataset and query dimensions must match")
                    .with_fact("node_id", node_id)
                    .with_fact("reason", "feature_dimension_mismatch"));
            }
            validate_element_metric(CuvsOperation::Cagra, dataset.element_type, metric_name(params.metric), node_id)?;
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
            validate_element_metric(
                CuvsOperation::PairwiseDistance,
                x.element_type,
                metric_name(params.metric),
                node_id,
            )?;
            vector_pairwise_distance_output_schema(
                x.id_type,
                y.id_type,
                match x.element_type {
                    VectorElementType::Float64 => DataType::Float64,
                    _ => DataType::Float32,
                },
            )
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
            validate_element_metric(CuvsOperation::KMeans, input.element_type, params.metric.as_str(), node_id)?;
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
        Some(GpuOptionKind::Integer { maximum, .. }) => maximum,
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

pub(crate) struct DenseRelationValidation {
    pub(crate) id_type: DataType,
    pub(crate) dimension: Option<usize>,
    pub(crate) element_type: VectorElementType,
}

fn validate_matching_element_types(
    left_role: &'static str,
    left: &DenseRelationValidation,
    right_role: &'static str,
    right: &DenseRelationValidation,
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

fn metric_name(metric: VectorDistanceMetric) -> &'static str {
    match metric {
        VectorDistanceMetric::CosineExpanded => "cosine",
        _ => metric.as_str(),
    }
}

fn validate_element_metric(
    operation: CuvsOperation,
    element: VectorElementType,
    metric: &str,
    node_id: NodeId,
) -> Result<()> {
    let support = operation.descriptor().element_support(element).expect("validated element");
    if !support.metrics.contains(&metric) {
        return Err(vector_algorithm_error("metric is not supported for this vector element type")
            .with_fact("node_id", node_id)
            .with_fact("reason", "unsupported_vector_element_metric")
            .with_fact("element", element.as_str())
            .with_fact("metric", metric)
            .with_fact("supported_metrics", support.metrics.join(", ")));
    }
    Ok(())
}

fn validate_vector_element_support(
    operation: CuvsOperation,
    role: &'static str,
    relation: &DenseRelationValidation,
    node_id: NodeId,
) -> Result<()> {
    if operation.descriptor().element_support(relation.element_type).is_none() {
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
                .map(|support| support.element.as_str())
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
) -> Result<DenseRelationValidation> {
    validate_dense_relation_binding(
        (&binding.vector, &binding.row_id),
        algorithm_input(plan, node_id, binding.input),
        node_id,
        role,
        plan,
        available_columns_by_node,
        None,
        |relation| validate_vector_element_support(operation, role, relation, node_id),
    )
}

fn with_supported_types(error: QueryError, supported_types: Option<&'static str>) -> QueryError {
    match supported_types {
        Some(types) => error.with_fact("supported_types", types),
        None => error,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the validated relation needs its binding, plan, element contract, and consumer identity"
)]
pub(crate) fn validate_dense_relation_binding(
    binding: (&DenseVectorBinding, &VectorRowIdBinding),
    input_id: NodeId,
    consumer_id: NodeId,
    role: &'static str,
    plan: &QueryPlan,
    available_columns_by_node: &[BTreeSet<String>],
    supported_types: Option<&'static str>,
    validate_element: impl FnOnce(&DenseRelationValidation) -> Result<()>,
) -> Result<DenseRelationValidation> {
    let node_id = consumer_id;
    let (vector, row_id) = binding;
    let input = input_node(plan, input_id, consumer_id)?;
    let available = available_columns(available_columns_by_node, input_id)?;
    let (dimension, element_type) = match vector {
        DenseVectorBinding::Columns(columns) => {
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
                let current_type = VectorElementType::from_data_type(field.data_type());
                if current_type.is_none() {
                    return Err(with_supported_types(
                        QueryError::new(
                            QueryErrorCode::UnsupportedType,
                            "query_engine.vector_algorithm.validate",
                            "vector feature columns must have a supported element type",
                        )
                        .with_fact("node_id", node_id)
                        .with_fact("relation_role", role)
                        .with_fact("reason", "unsupported_feature_type")
                        .with_fact("column", column)
                        .with_fact("data_type", format!("{:?}", field.data_type())),
                        supported_types,
                    ));
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
        DenseVectorBinding::VectorColumn(column) => {
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
            let (element, dimension) = match field.data_type() {
                DataType::FixedSizeList(element, dimension) if *dimension > 0 => (element, Some(*dimension as usize)),
                DataType::List(element) | DataType::LargeList(element) => (element, None),
                actual => {
                    return Err(with_supported_types(
                        QueryError::new(
                            QueryErrorCode::UnsupportedType,
                            "query_engine.vector_algorithm.validate",
                            "vector column must be a fixed-size or variable-size list",
                        )
                        .with_fact("node_id", node_id)
                        .with_fact("relation_role", role)
                        .with_fact("reason", "unsupported_feature_type")
                        .with_fact("column", column)
                        .with_fact("data_type", format!("{actual:?}")),
                        supported_types,
                    ));
                },
            };
            let element_type = VectorElementType::from_data_type(element.data_type()).ok_or_else(|| {
                with_supported_types(
                    vector_algorithm_error("vector column element type is not supported")
                        .with_fact("node_id", node_id)
                        .with_fact("relation_role", role)
                        .with_fact("reason", "unsupported_feature_type")
                        .with_fact("column", column)
                        .with_fact("data_type", format!("{:?}", element.data_type())),
                    supported_types,
                )
            })?;
            (dimension, element_type)
        },
    };
    let relation = DenseRelationValidation { id_type: DataType::Int64, dimension, element_type };
    validate_element(&relation)?;
    match row_id {
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
            if RelationIdType::from_data_type(field.data_type()).is_none() {
                return Err(QueryError::new(
                    QueryErrorCode::UnsupportedType,
                    "query_engine.vector_algorithm.validate",
                    "vector row-id columns must be Int32, Int64, Utf8, LargeUtf8, or Utf8View values",
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("reason", "unsupported_row_id_type")
                .with_fact("column", column)
                .with_fact("data_type", format!("{:?}", field.data_type())));
            }
            Ok(DenseRelationValidation { id_type: field.data_type().clone(), ..relation })
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
            VectorIvfPqParams, VectorPairwiseDistanceParams, VectorPcaParams, VectorTopKParams,
        },
        source::{GpuResidentSource, SourceNode},
        validation::PlanColumns,
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
            VectorRelationBinding::new(0, VectorRowIdBinding::column("id"), ["x"]),
            VectorRelationBinding::new_vector_column(1, VectorRowIdBinding::column("id"), "embedding"),
            1,
            VectorDistanceMetric::L2Expanded,
        );
        let root = builder
            .add_vector_algorithm(
                vec![dataset, queries],
                VectorAlgorithmSpec::new(
                    VectorAlgorithm::BruteForceKnn,
                    VectorAlgorithmParams::BruteForceKnn(params),
                    vector_knn_output_schema(DataType::Int64, DataType::Int64),
                ),
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(root).unwrap();
        let PlanNode::VectorAlgorithm(vector) = &plan.node(root).expect("vector root").kind else {
            panic!("vector root")
        };
        let error =
            validate_vector_algorithm_node(vector, &plan, PlanColumns::for_plan(&plan).available_by_node(), root)
                .expect_err("mixed element types");
        assert_eq!(error.fact_value("reason"), Some("unsupported_feature_type"));
        assert_eq!(error.fact_value("data_type"), Some("Float64"));
    }

    #[test]
    fn pairwise_cross_binding_element_mismatch_is_rejected() {
        let float32_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Float32, false),
        ]));
        let float64_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Float64, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let x = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("x", float32_schema)));
        let y = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("y", float64_schema)));
        let params = VectorPairwiseDistanceParams {
            x: VectorRelationBinding::new(0, VectorRowIdBinding::column("id"), ["x"]),
            y: VectorRelationBinding::new(1, VectorRowIdBinding::column("id"), ["x"]),
            metric: VectorDistanceMetric::L2Expanded,
        };
        let root = builder
            .add_vector_algorithm(
                vec![x, y],
                VectorAlgorithmSpec::new(
                    VectorAlgorithm::PairwiseDistance,
                    VectorAlgorithmParams::PairwiseDistance(params),
                    vector_pairwise_distance_output_schema(DataType::Int64, DataType::Int64, DataType::Float32),
                ),
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(root).unwrap();
        let PlanNode::VectorAlgorithm(vector) = &plan.node(root).expect("vector root").kind else {
            panic!("vector root")
        };
        let error =
            validate_vector_algorithm_node(vector, &plan, PlanColumns::for_plan(&plan).available_by_node(), root)
                .expect_err("mixed pairwise elements");
        assert_eq!(error.fact_value("reason"), Some("vector_element_type_mismatch"));
    }

    #[test]
    fn integer_ivf_pq_cosine_is_rejected_by_ir_validation() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Int8, false),
            Field::new("y", DataType::Int8, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let dataset =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("dataset", Arc::clone(&schema))));
        let queries = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("queries", schema)));
        let params = VectorIvfPqParams::new(
            VectorRelationBinding::new(0, VectorRowIdBinding::column("id"), ["x", "y"]),
            VectorRelationBinding::new(1, VectorRowIdBinding::column("id"), ["x", "y"]),
            1,
            VectorDistanceMetric::CosineExpanded,
            20,
        );
        let root = builder
            .add_vector_algorithm(
                vec![dataset, queries],
                VectorAlgorithmSpec::new(
                    VectorAlgorithm::IvfPq,
                    VectorAlgorithmParams::IvfPq(params),
                    vector_knn_output_schema(DataType::Int64, DataType::Int64),
                ),
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(root).unwrap();
        let PlanNode::VectorAlgorithm(vector) = &plan.node(root).expect("vector root").kind else {
            panic!("vector root")
        };
        let error =
            validate_vector_algorithm_node(vector, &plan, PlanColumns::for_plan(&plan).available_by_node(), root)
                .expect_err("integer cosine");
        assert_eq!(error.fact_value("reason"), Some("unsupported_vector_element_metric"));
        assert_eq!(error.fact_value("supported_metrics"), Some("l2_expanded, l2_sqrt_expanded, inner_product"));
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
                input: VectorRelationBinding::new(0, VectorRowIdBinding::column("id"), ["x"]),
                k,
                select_min: true,
            };
            let root = builder
                .add_vector_algorithm(
                    vec![input],
                    VectorAlgorithmSpec::new(
                        VectorAlgorithm::TopK,
                        VectorAlgorithmParams::TopK(params),
                        vector_top_k_output_schema(DataType::Int64),
                    ),
                )
                .unwrap();
            let plan = builder.finish_with_host_sink(root).unwrap();
            let PlanNode::VectorAlgorithm(vector) = &plan.node(root).expect("vector root").kind else {
                panic!("vector root")
            };
            let error =
                validate_vector_algorithm_node(vector, &plan, PlanColumns::for_plan(&plan).available_by_node(), root)
                    .expect_err("invalid k");
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
        let params = VectorPcaParams::new(VectorRelationBinding::new(0, VectorRowIdBinding::column("id"), ["x"]), 1);
        let root = builder
            .add_vector_algorithm(
                vec![input],
                VectorAlgorithmSpec::new(
                    VectorAlgorithm::Pca,
                    VectorAlgorithmParams::Pca(params),
                    vector_pca_output_schema(DataType::Int64, 1),
                ),
            )
            .unwrap();
        let plan = builder.finish_with_host_sink(root).unwrap();
        let mut available_columns_by_node = PlanColumns::for_plan(&plan).available_by_node().to_vec();
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
