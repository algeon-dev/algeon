use std::collections::BTreeMap;

use cudf::{Series, frame::dataframe::GpuRecordBatch};
use cuvs::{
    BruteForceIndex, CuvsResources, DeviceBuffer, DistanceMetric, KMeansInit, KMeansMetric, KMeansModel, KMeansParams,
    MatrixLayout, OwnedFloat32Matrix, PcaOutputBuffers, PcaParams, PcaSolver, ValidatedDeviceTensor,
    fit_transform_owned,
};

use super::{
    super::{GpuDataFrame, *},
    assembly::{assemble_brute_force_knn_output, assemble_kmeans_output, assemble_pca_output},
};
use crate::{
    observability::{
        RuntimeOperatorStageGuard, RuntimeOperatorStageIdentity, RuntimeOperatorStageKind, stage_context_from_execution,
    },
    plan::{
        CuvsDenseVectorBinding, QueryPlan, VectorAlgorithm, VectorAlgorithmNode, VectorAlgorithmParams,
        VectorBruteForceKnnParams, VectorDistanceMetric, VectorKMeansInit, VectorKMeansMetric, VectorKMeansParams,
        VectorPcaParams, VectorPcaSolver, VectorRelationBinding,
    },
};

const EXECUTE_OPERATION: &str = "query_engine.vector_algorithm.execute";
const PACK_OPERATION: &str = "query_engine.vector_algorithm.pack";
const NATIVE_OPERATION: &str = "query_engine.vector_algorithm.native";

pub(in crate::exec) fn execute_vector_algorithm(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    node: &VectorAlgorithmNode,
    inputs: &BTreeMap<NodeId, GpuDataFrame>,
) -> Result<GpuDataFrame> {
    context.check_cancellation(EXECUTE_OPERATION)?;
    // Vector operators do not have an independent workspace estimate. Use
    // the immutable context grant for the boundary: the live member makes
    // cuVS's device-current temporaries domain-accounted before resources are
    // constructed, and remains attached through all cuVS calls below.
    let _domain_token = context
        .attempt_allocation_domain_id()
        .map(|_| {
            let budget_bytes = context.bounded_device_budget_bytes().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::CudfInvariantViolation,
                    EXECUTE_OPERATION,
                    "admitted vector execution is missing its bounded device budget",
                )
            })?;
            let budget = usize::try_from(budget_bytes).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    EXECUTE_OPERATION,
                    "vector execution budget exceeds platform usize range",
                )
                .with_fact("budget_bytes", budget_bytes)
            })?;
            context.execution_budget_token(cudf::ExecutionBudgetRequest::new(budget).with_boundary("vector_algorithm"))
        })
        .transpose()?;
    let runtime = context.shared_runtime()?;
    let resources = CuvsResources::for_runtime(runtime).map_err(|error| {
        cuvs_query_error(EXECUTE_OPERATION, "failed to create query-local cuVS resources", error)
            .with_fact("node_id", node_id)
            .with_fact("algorithm", node.spec.algorithm.as_str())
    })?;
    let native_version = cuvs_sys::validate_runtime_version().map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            EXECUTE_OPERATION,
            "failed to validate the native cuVS runtime version",
        )
        .with_fact("node_id", node_id)
        .with_fact("algorithm", node.spec.algorithm.as_str())
        .with_source(error)
    })?;

    match &node.spec.params {
        VectorAlgorithmParams::BruteForceKnn(params) => {
            execute_brute_force_knn(context, plan, node_id, params, inputs, &resources, native_version)
        },
        VectorAlgorithmParams::KMeans(params) => {
            execute_kmeans(context, plan, node_id, params, inputs, &resources, native_version)
        },
        VectorAlgorithmParams::Pca(params) => {
            execute_pca(context, plan, node_id, params, inputs, &resources, native_version)
        },
    }
}

pub(in crate::exec) fn validate_vector_algorithm_execution(node_id: NodeId, node: &VectorAlgorithmNode) -> Result<()> {
    if node.spec.params.supports_algorithm(node.spec.algorithm) {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::InvalidPlan,
        EXECUTE_OPERATION,
        "vector execution received an unknown typed algorithm payload",
    )
    .with_fact("node_id", node_id)
    .with_fact("operator", "vector_algorithm")
    .with_fact("algorithm", node.spec.algorithm.as_str())
    .with_fact("reason", "unknown_vector_algorithm_payload"))
}

fn execute_brute_force_knn(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    params: &VectorBruteForceKnnParams,
    inputs: &BTreeMap<NodeId, GpuDataFrame>,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
) -> Result<GpuDataFrame> {
    let dataset = relation_input(inputs, &params.dataset, node_id, "dataset")?;
    let queries = relation_input(inputs, &params.queries, node_id, "queries")?;
    let dataset_rows = dataset.num_rows();
    if dataset_rows == 0 {
        return Err(knn_input_error(node_id, "empty_dataset", "kNN dataset must contain at least one row")
            .with_fact("k", params.k));
    }
    if params.k > dataset_rows {
        return Err(knn_input_error(
            node_id,
            "k_exceeds_dataset_rows",
            "kNN k must not exceed the evaluated dataset row count",
        )
        .with_fact("k", params.k)
        .with_fact("dataset_rows", dataset_rows));
    }
    let output_rows = checked_output_rows(queries.num_rows(), params.k, node_id)?;

    with_relation_tensor(
        context,
        plan,
        node_id,
        VectorAlgorithm::BruteForceKnn,
        "dataset",
        &params.dataset,
        dataset,
        resources,
        native_version,
        |dataset_tensor| {
            let packed_dataset_rows = dataset_tensor.rows().map_err(|error| {
                cuvs_query_error(EXECUTE_OPERATION, "failed to inspect the validated kNN dataset shape", error)
            })?;
            let dimension = dataset_tensor.columns().map_err(|error| {
                cuvs_query_error(EXECUTE_OPERATION, "failed to inspect the validated kNN dataset dimension", error)
            })?;
            if packed_dataset_rows != dataset_rows {
                return Err(QueryError::new(
                    QueryErrorCode::CudfInvariantViolation,
                    EXECUTE_OPERATION,
                    "packed kNN dataset row count does not match the evaluated relation",
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_rows", dataset_rows)
                .with_fact("packed_rows", packed_dataset_rows)
                .with_fact("reason", "packed_row_count_mismatch"));
            }

            if queries.num_rows() == 0 {
                let query_schema = plan
                    .node(params.queries.input)
                    .ok_or_else(|| {
                        invalid_node_reference(params.queries.input, "vector query relation references a missing input")
                    })?
                    .output_schema();
                let query_dimension = relation_dimension(&params.queries, query_schema, node_id, "queries")?;
                if query_dimension != dimension {
                    return Err(knn_input_error(
                        node_id,
                        "feature_dimension_mismatch",
                        "kNN dataset and query dimensions must match at execution",
                    )
                    .with_fact("dataset_dimension", dimension)
                    .with_fact("query_dimension", query_dimension));
                }
                let neighbors = resources.runtime().series(&[] as &[i64]).map_err(|error| {
                    cudf_query_error(EXECUTE_OPERATION, "failed to construct empty kNN neighbor output", error)
                })?;
                let distances = resources.runtime().series(&[] as &[f32]).map_err(|error| {
                    cudf_query_error(EXECUTE_OPERATION, "failed to construct empty kNN distance output", error)
                })?;
                return observe_assembly(
                    context,
                    node_id,
                    VectorAlgorithm::BruteForceKnn,
                    resources,
                    native_version,
                    output_rows,
                    || assemble_brute_force_knn_output(context, params, dataset, queries, neighbors, distances),
                );
            }

            with_relation_tensor(
                context,
                plan,
                node_id,
                VectorAlgorithm::BruteForceKnn,
                "queries",
                &params.queries,
                queries,
                resources,
                native_version,
                |query_tensor| {
                    let query_rows = query_tensor.rows().map_err(|error| {
                        cuvs_query_error(EXECUTE_OPERATION, "failed to inspect the validated kNN query shape", error)
                    })?;
                    if query_rows != queries.num_rows() {
                        return Err(QueryError::new(
                            QueryErrorCode::CudfInvariantViolation,
                            EXECUTE_OPERATION,
                            "packed kNN query row count does not match the evaluated relation",
                        )
                        .with_fact("node_id", node_id)
                        .with_fact("relation_rows", queries.num_rows())
                        .with_fact("packed_rows", query_rows)
                        .with_fact("reason", "packed_row_count_mismatch"));
                    }
                    let query_dimension = query_tensor.columns().map_err(|error| {
                        cuvs_query_error(
                            EXECUTE_OPERATION,
                            "failed to inspect the validated kNN query dimension",
                            error,
                        )
                    })?;
                    if query_dimension != dimension {
                        return Err(knn_input_error(
                            node_id,
                            "feature_dimension_mismatch",
                            "kNN dataset and query dimensions must match at execution",
                        )
                        .with_fact("dataset_dimension", dimension)
                        .with_fact("query_dimension", query_dimension));
                    }
                    let metric = distance_metric(params.metric);
                    let dataset_bytes = matrix_bytes(dataset_rows, dimension, node_id, "dataset")?;
                    let index = observe_native(
                        context,
                        node_id,
                        VectorAlgorithm::BruteForceKnn,
                        resources,
                        native_version,
                        "build_index",
                        dataset_rows,
                        dataset_bytes,
                        None,
                        None,
                        || {
                            context.check_cancellation("query_engine.vector_algorithm.before_index_build")?;
                            let index = BruteForceIndex::build(resources, dataset_tensor, metric).map_err(|error| {
                                cuvs_query_error(NATIVE_OPERATION, "failed to build the query-local kNN index", error)
                            })?;
                            context.check_cancellation("query_engine.vector_algorithm.after_index_build")?;
                            Ok(index)
                        },
                    )?;

                    let mut neighbors = DeviceBuffer::<i64>::new(resources, output_rows).map_err(|error| {
                        cuvs_query_error(NATIVE_OPERATION, "failed to allocate kNN neighbor output", error)
                    })?;
                    let mut distances = DeviceBuffer::<f32>::new(resources, output_rows).map_err(|error| {
                        cuvs_query_error(NATIVE_OPERATION, "failed to allocate kNN distance output", error)
                    })?;
                    let query_bytes = matrix_bytes(query_rows, query_dimension, node_id, "queries")?;
                    let native_output_bytes = output_rows
                        .checked_mul(std::mem::size_of::<i64>() + std::mem::size_of::<f32>())
                        .and_then(|bytes| u64::try_from(bytes).ok())
                        .ok_or_else(|| size_overflow(node_id, "native_output_bytes"))?;
                    observe_native(
                        context,
                        node_id,
                        VectorAlgorithm::BruteForceKnn,
                        resources,
                        native_version,
                        "search",
                        query_rows,
                        query_bytes,
                        Some(output_rows),
                        Some(native_output_bytes),
                        || {
                            context.check_cancellation("query_engine.vector_algorithm.before_search")?;
                            index.search_into(query_tensor, params.k, &mut neighbors, &mut distances).map_err(
                                |error| cuvs_query_error(NATIVE_OPERATION, "failed to execute exact kNN search", error),
                            )?;
                            context.check_cancellation("query_engine.vector_algorithm.after_search")?;
                            Ok(())
                        },
                    )?;

                    let neighbors = Series::from_column(neighbors.into_column().map_err(|error| {
                        cuvs_query_error(EXECUTE_OPERATION, "failed to adopt kNN neighbor output", error)
                    })?)
                    .map_err(|error| {
                        cudf_query_error(EXECUTE_OPERATION, "failed to wrap kNN neighbor output", error)
                    })?;
                    let distances = Series::from_column(distances.into_column().map_err(|error| {
                        cuvs_query_error(EXECUTE_OPERATION, "failed to adopt kNN distance output", error)
                    })?)
                    .map_err(|error| {
                        cudf_query_error(EXECUTE_OPERATION, "failed to wrap kNN distance output", error)
                    })?;

                    observe_assembly(
                        context,
                        node_id,
                        VectorAlgorithm::BruteForceKnn,
                        resources,
                        native_version,
                        output_rows,
                        || {
                            context.check_cancellation("query_engine.vector_algorithm.before_assembly")?;
                            let output = assemble_brute_force_knn_output(
                                context, params, dataset, queries, neighbors, distances,
                            )?;
                            context.check_cancellation("query_engine.vector_algorithm.after_assembly")?;
                            Ok(output)
                        },
                    )
                },
            )
        },
    )
}

fn execute_kmeans(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    params: &VectorKMeansParams,
    inputs: &BTreeMap<NodeId, GpuDataFrame>,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
) -> Result<GpuDataFrame> {
    let input = relation_input(inputs, &params.input, node_id, "input")?;
    let input_rows = input.num_rows();
    if input_rows == 0 {
        return Err(kmeans_input_error(node_id, "empty_input", "K-means input must contain at least one row"));
    }
    if input_rows > i32::MAX as usize {
        return Err(kmeans_input_error(
            node_id,
            "row_count_exceeds_cuvs_i32",
            "K-means input row count exceeds the cuVS i32 range",
        )
        .with_fact("input_rows", input_rows)
        .with_fact("maximum_rows", i32::MAX));
    }
    if params.n_clusters > input_rows {
        return Err(kmeans_input_error(
            node_id,
            "cluster_count_exceeds_input_rows",
            "K-means n_clusters must not exceed the evaluated input row count",
        )
        .with_fact("n_clusters", params.n_clusters)
        .with_fact("input_rows", input_rows));
    }
    let native_params = native_kmeans_params(params, node_id)?;

    with_relation_tensor(
        context,
        plan,
        node_id,
        VectorAlgorithm::KMeans,
        "input",
        &params.input,
        input,
        resources,
        native_version,
        |input_tensor| {
            let dimension = tensor_dimension(input_tensor, node_id, "input", "K-means")?;
            let input_bytes = matrix_bytes(input_rows, dimension, node_id, "input")?;
            let fit_model = |initial_centroids| {
                observe_native(
                    context,
                    node_id,
                    VectorAlgorithm::KMeans,
                    resources,
                    native_version,
                    "fit",
                    input_rows,
                    input_bytes,
                    None,
                    None,
                    || {
                        context.check_cancellation("query_engine.vector_algorithm.before_kmeans_fit")?;
                        let model = KMeansModel::fit_with_initial_centroids(
                            resources,
                            input_tensor,
                            native_params.clone(),
                            initial_centroids,
                        )
                        .map_err(|error| {
                            cuvs_query_error(NATIVE_OPERATION, "failed to fit query-local K-means", error)
                        })?;
                        context.check_cancellation("query_engine.vector_algorithm.after_kmeans_fit")?;
                        Ok(model)
                    },
                )
            };

            let model = match (&params.init, &params.initial_centroids) {
                (VectorKMeansInit::Relation, Some(centroid_binding)) => with_owned_relation_matrix(
                    context,
                    plan,
                    node_id,
                    VectorAlgorithm::KMeans,
                    "initial_centroids",
                    centroid_binding,
                    relation_input(inputs, centroid_binding, node_id, "initial_centroids")?,
                    resources,
                    native_version,
                    MatrixLayout::RowMajor,
                    |centroids, centroid_dimension| {
                        if centroid_dimension != dimension {
                            return Err(kmeans_input_error(
                                node_id,
                                "feature_dimension_mismatch",
                                "K-means input and initial-centroid dimensions must match at execution",
                            )
                            .with_fact("input_dimension", dimension)
                            .with_fact("centroid_dimension", centroid_dimension));
                        }
                        fit_model(Some(centroids))
                    },
                )?,
                (VectorKMeansInit::Relation, None) => {
                    return Err(kmeans_input_error(
                        node_id,
                        "missing_initial_centroids",
                        "relation-initialized K-means requires an initial-centroids input",
                    ));
                },
                (_, Some(_)) => {
                    return Err(kmeans_input_error(
                        node_id,
                        "unexpected_initial_centroids",
                        "K-means initial-centroids input requires relation initialization",
                    ));
                },
                (_, None) => fit_model(None)?,
            };

            let mut labels = DeviceBuffer::<i32>::new(resources, input_rows).map_err(|error| {
                cuvs_query_error(NATIVE_OPERATION, "failed to allocate K-means assignment output", error)
            })?;
            observe_native(
                context,
                node_id,
                VectorAlgorithm::KMeans,
                resources,
                native_version,
                "predict",
                input_rows,
                input_bytes,
                Some(input_rows),
                Some(
                    input_rows
                        .checked_mul(std::mem::size_of::<i32>())
                        .and_then(|bytes| u64::try_from(bytes).ok())
                        .ok_or_else(|| size_overflow(node_id, "kmeans_assignment_output_bytes"))?,
                ),
                || {
                    context.check_cancellation("query_engine.vector_algorithm.before_kmeans_predict")?;
                    model.predict_into(input_tensor, &mut labels).map_err(|error| {
                        cuvs_query_error(NATIVE_OPERATION, "failed to predict K-means assignments", error)
                    })?;
                    context.check_cancellation("query_engine.vector_algorithm.after_kmeans_predict")?;
                    Ok(())
                },
            )?;
            let labels = Series::from_column(labels.into_column().map_err(|error| {
                cuvs_query_error(EXECUTE_OPERATION, "failed to adopt K-means assignment output", error)
            })?)
            .map_err(|error| cudf_query_error(EXECUTE_OPERATION, "failed to wrap K-means assignment output", error))?;
            observe_assembly(context, node_id, VectorAlgorithm::KMeans, resources, native_version, input_rows, || {
                assemble_kmeans_output(context, input, &params.input, labels)
            })
        },
    )
}

fn execute_pca(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    params: &VectorPcaParams,
    inputs: &BTreeMap<NodeId, GpuDataFrame>,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
) -> Result<GpuDataFrame> {
    let input = relation_input(inputs, &params.input, node_id, "input")?;
    let input_rows = input.num_rows();
    if input_rows < 2 {
        return Err(pca_input_error(node_id, "insufficient_input_rows", "PCA requires at least two input rows")
            .with_fact("input_rows", input_rows));
    }
    let native_params = native_pca_params(params, node_id)?;
    with_owned_relation_matrix(
        context,
        plan,
        node_id,
        VectorAlgorithm::Pca,
        "input",
        &params.input,
        input,
        resources,
        native_version,
        MatrixLayout::ColumnMajor,
        |mut packed_input, dimension| {
            if dimension < 2 {
                return Err(pca_input_error(
                    node_id,
                    "insufficient_feature_dimension",
                    "PCA requires at least two evaluated input feature columns",
                )
                .with_fact("feature_dimension", dimension)
                .with_fact("minimum_feature_dimension", 2));
            }
            if params.n_components > dimension {
                return Err(pca_input_error(
                    node_id,
                    "component_count_exceeds_feature_dimension",
                    "PCA n_components must not exceed the evaluated feature dimension",
                )
                .with_fact("n_components", params.n_components)
                .with_fact("feature_dimension", dimension));
            }
            let input_bytes = matrix_bytes(input_rows, dimension, node_id, "input")?;
            let mut outputs = PcaOutputBuffers::new(resources, input_rows, dimension, params.n_components)
                .map_err(|error| cuvs_query_error(NATIVE_OPERATION, "failed to allocate PCA device outputs", error))?;
            let transformed_bytes = matrix_bytes(input_rows, params.n_components, node_id, "pca_transformed_output")?;
            let fit = observe_native(
                context,
                node_id,
                VectorAlgorithm::Pca,
                resources,
                native_version,
                "fit_transform",
                input_rows,
                input_bytes,
                Some(input_rows),
                Some(transformed_bytes),
                || {
                    context.check_cancellation("query_engine.vector_algorithm.before_pca_fit_transform")?;
                    let fit = fit_transform_owned(resources, &mut packed_input, &native_params, &mut outputs).map_err(
                        |error| cuvs_query_error(NATIVE_OPERATION, "failed to fit and transform PCA", error),
                    )?;
                    context.check_cancellation("query_engine.vector_algorithm.after_pca_fit_transform")?;
                    Ok(fit)
                },
            )?;
            let components = outputs.into_transformed_columns(fit).map_err(|error| {
                cuvs_query_error(EXECUTE_OPERATION, "failed to adopt PCA transformed device output", error)
            })?;
            observe_assembly(context, node_id, VectorAlgorithm::Pca, resources, native_version, input_rows, || {
                assemble_pca_output(context, input, &params.input, &components)
            })
        },
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "native cuVS call parameters are fixed by the FFI signature and stay explicit"
)]
fn with_relation_tensor<T>(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    role: &'static str,
    binding: &VectorRelationBinding,
    input: &GpuDataFrame,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
    body: impl FnOnce(&ValidatedDeviceTensor<'_>) -> Result<T>,
) -> Result<T> {
    let input_schema = plan
        .node(binding.input)
        .ok_or_else(|| invalid_node_reference(binding.input, "vector relation references a missing input"))?
        .output_schema();
    if matches!(
        &binding.vector,
        CuvsDenseVectorBinding::VectorColumn(column)
            if matches!(
                input_schema.field_with_name(column).map(|field| field.data_type()),
                Ok(arrow::datatypes::DataType::List(_) | arrow::datatypes::DataType::LargeList(_))
            )
    ) {
        return with_uniform_list_relation_tensor(
            context,
            node_id,
            algorithm,
            role,
            binding,
            input,
            input_schema,
            resources,
            native_version,
            body,
        );
    }
    let dimension = relation_dimension(binding, input_schema, node_id, role)?;
    let input_rows = input.num_rows();
    let input_bytes = input.device_allocated_size_bytes()?;
    let packed_bytes = matrix_bytes(input_rows, dimension, node_id, role)?;
    let identity = vector_identity(node_id, algorithm, resources, native_version)
        .with_vector_relation_role(role)
        .with_vector_dimension(dimension)
        .with_matrix_layout("row_major")
        .with_input_rows(input_rows as u64)
        .with_input_bytes(input_bytes);
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        RuntimeOperatorStageKind::VectorPack,
        identity,
    );
    if let Err(error) = context.check_cancellation("query_engine.vector_algorithm.before_pack") {
        guard.fail(&error);
        return Err(error);
    }

    match &binding.vector {
        CuvsDenseVectorBinding::Columns(columns) => {
            let packed = (|| {
                let selected = input.frame().select(columns.iter().map(String::as_str)).map_err(|error| {
                    cudf_query_error(PACK_OPERATION, "failed to select the kNN feature columns", error)
                        .with_fact("node_id", node_id)
                        .with_fact("relation_role", role)
                })?;
                let export = selected.try_prepare_device_table_export().map_err(|error| {
                    cudf_interop_query_error(
                        PACK_OPERATION,
                        "failed to prepare one atomic kNN feature-table export",
                        format!("{:?}", error.kind()),
                        error,
                    )
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
                })?;
                resources.pack_prepared_float32_table(&export, MatrixLayout::RowMajor).map_err(|error| {
                    cuvs_query_error(PACK_OPERATION, "failed to pack the prepared kNN feature table", error)
                        .with_fact("node_id", node_id)
                        .with_fact("relation_role", role)
                })
            })();
            let packed = match packed {
                Ok(packed) => packed,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            };
            let tensor = match packed.as_validated_tensor(resources).map_err(|error| {
                cuvs_query_error(PACK_OPERATION, "failed to bind the validated packed kNN matrix", error)
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
            }) {
                Ok(tensor) => tensor,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            };
            guard.finish(Some(input_rows as u64), Some(packed_bytes));
            body(&tensor)
        },
        CuvsDenseVectorBinding::VectorColumn(column) => {
            let batch = match GpuRecordBatch::new(input.frame().clone(), input_schema.clone()).map_err(|error| {
                cudf_query_error(
                    PACK_OPERATION,
                    "failed to bind the vector relation to its FixedSizeList schema",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("column", column)
            }) {
                Ok(batch) => batch,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            };
            let field_index = input_schema.index_of(column).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    PACK_OPERATION,
                    "vector column is absent from the evaluated relation schema",
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("column", column)
                .with_fact("reason", "missing_feature_column")
            });
            let field_index = match field_index {
                Ok(field_index) => field_index,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            };
            let validated = match batch.try_prepare_fixed_size_list_float32_export(field_index).map_err(|error| {
                cudf_query_error(
                    PACK_OPERATION,
                    "failed to prove the FixedSizeList vector child range and finite values",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("column", column)
            }) {
                Ok(validated) => validated,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            };
            let lease = match resources.activate_validated_fixed_size_list_tensor(&validated).map_err(|error| {
                cuvs_query_error(PACK_OPERATION, "failed to activate the validated FixedSizeList tensor", error)
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
                    .with_fact("column", column)
            }) {
                Ok(lease) => lease,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            };
            guard.finish(Some(input_rows as u64), Some(packed_bytes));
            let result = body(lease.tensor());
            let completion = lease.complete().and_then(|completion| completion.release()).map_err(|error| {
                cuvs_query_error(PACK_OPERATION, "failed to complete the FixedSizeList producer read lifetime", error)
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
            });
            match (result, completion) {
                (Err(error), _) => Err(error),
                (Ok(value), Ok(())) => Ok(value),
                (Ok(_), Err(error)) => Err(error),
            }
        },
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "native cuVS call parameters are fixed by the FFI signature and stay explicit"
)]
fn with_owned_relation_matrix<T>(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    role: &'static str,
    binding: &VectorRelationBinding,
    input: &GpuDataFrame,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
    layout: MatrixLayout,
    body: impl FnOnce(OwnedFloat32Matrix, usize) -> Result<T>,
) -> Result<T> {
    let input_schema = plan
        .node(binding.input)
        .ok_or_else(|| invalid_node_reference(binding.input, "vector relation references a missing input"))?
        .output_schema();
    if matches!(
        &binding.vector,
        CuvsDenseVectorBinding::VectorColumn(column)
            if matches!(
                input_schema.field_with_name(column).map(|field| field.data_type()),
                Ok(arrow::datatypes::DataType::List(_) | arrow::datatypes::DataType::LargeList(_))
            )
    ) {
        return with_owned_uniform_list_relation_matrix(
            context,
            node_id,
            algorithm,
            role,
            binding,
            input,
            input_schema,
            resources,
            native_version,
            layout,
            body,
        );
    }
    let dimension = relation_dimension(binding, input_schema, node_id, role)?;
    let input_rows = input.num_rows();
    let input_bytes = input.device_allocated_size_bytes()?;
    let packed_bytes = matrix_bytes(input_rows, dimension, node_id, role)?;
    let identity = vector_identity(node_id, algorithm, resources, native_version)
        .with_vector_relation_role(role)
        .with_vector_dimension(dimension)
        .with_matrix_layout(matrix_layout_name(layout))
        .with_input_rows(input_rows as u64)
        .with_input_bytes(input_bytes);
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        RuntimeOperatorStageKind::VectorPack,
        identity,
    );
    if let Err(error) = context.check_cancellation("query_engine.vector_algorithm.before_pack") {
        guard.fail(&error);
        return Err(error);
    }

    let packed = match &binding.vector {
        CuvsDenseVectorBinding::Columns(columns) => {
            let packed = (|| {
                let selected = input.frame().select(columns.iter().map(String::as_str)).map_err(|error| {
                    cudf_query_error(PACK_OPERATION, "failed to select vector feature columns", error)
                        .with_fact("node_id", node_id)
                        .with_fact("relation_role", role)
                })?;
                let export = selected.try_prepare_device_table_export().map_err(|error| {
                    cudf_interop_query_error(
                        PACK_OPERATION,
                        "failed to prepare one atomic vector feature-table export",
                        format!("{:?}", error.kind()),
                        error,
                    )
                    .with_fact("node_id", node_id)
                    .with_fact("relation_role", role)
                })?;
                resources.pack_prepared_float32_table(&export, layout).map_err(|error| {
                    cuvs_query_error(PACK_OPERATION, "failed to pack the prepared vector feature table", error)
                        .with_fact("node_id", node_id)
                        .with_fact("relation_role", role)
                })
            })();
            match packed {
                Ok(packed) => packed,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            }
        },
        CuvsDenseVectorBinding::VectorColumn(column) => {
            let batch = match GpuRecordBatch::new(input.frame().clone(), input_schema.clone()).map_err(|error| {
                cudf_query_error(
                    PACK_OPERATION,
                    "failed to bind the vector relation to its FixedSizeList schema",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("column", column)
            }) {
                Ok(batch) => batch,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            };
            let field_index = match input_schema.index_of(column).map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    PACK_OPERATION,
                    "vector column is absent from the evaluated relation schema",
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("column", column)
                .with_fact("reason", "missing_feature_column")
            }) {
                Ok(field_index) => field_index,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            };
            let validated = match batch.try_prepare_fixed_size_list_float32_export(field_index).map_err(|error| {
                cudf_query_error(
                    PACK_OPERATION,
                    "failed to prove the FixedSizeList vector child range and finite values",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("column", column)
            }) {
                Ok(validated) => validated,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            };
            match resources.pack_validated_fixed_size_list_float32(&validated, layout).map_err(|error| {
                cuvs_query_error(
                    PACK_OPERATION,
                    "failed to pack the validated FixedSizeList vector child into an owned matrix",
                    error,
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("column", column)
            }) {
                Ok(packed) => packed,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            }
        },
    };
    guard.finish(Some(input_rows as u64), Some(packed_bytes));
    body(packed, dimension)
}

#[expect(
    clippy::too_many_arguments,
    reason = "native cuVS call parameters are fixed by the FFI signature and stay explicit"
)]
fn with_uniform_list_relation_tensor<T>(
    context: &ExecutionContext,
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    role: &'static str,
    binding: &VectorRelationBinding,
    input: &GpuDataFrame,
    input_schema: &arrow::datatypes::SchemaRef,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
    body: impl FnOnce(&ValidatedDeviceTensor<'_>) -> Result<T>,
) -> Result<T> {
    let column = uniform_list_vector_column(binding, node_id, role)?;
    context.check_cancellation("query_engine.vector_algorithm.before_pack")?;
    let batch = GpuRecordBatch::new(input.frame().clone(), input_schema.clone()).map_err(|error| {
        cudf_query_error(PACK_OPERATION, "failed to bind the vector relation to its variable-width List schema", error)
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
    })?;
    let field_index = uniform_list_field_index(input_schema, column, node_id, role)?;
    let validated = batch.try_prepare_uniform_list_float32_export(field_index).map_err(|error| {
        cudf_query_error(PACK_OPERATION, "failed to prove uniform List vector offsets and finite values", error)
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
    })?;
    let rows = validated.rows();
    let dimension = validated.dimension();
    let input_bytes = input.device_allocated_size_bytes()?;
    let packed_bytes = matrix_bytes(rows, dimension, node_id, role)?;
    let identity = vector_identity(node_id, algorithm, resources, native_version)
        .with_vector_relation_role(role)
        .with_vector_dimension(dimension)
        .with_matrix_layout("row_major")
        .with_input_rows(rows as u64)
        .with_input_bytes(input_bytes);
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        RuntimeOperatorStageKind::VectorPack,
        identity,
    );
    if let Err(error) = context.check_cancellation("query_engine.vector_algorithm.before_uniform_list_activation") {
        guard.fail(&error);
        return Err(error);
    }
    let lease = match resources.activate_validated_uniform_list_tensor(&validated).map_err(|error| {
        cuvs_query_error(PACK_OPERATION, "failed to activate the validated uniform List tensor", error)
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
    }) {
        Ok(lease) => lease,
        Err(error) => {
            guard.fail(&error);
            return Err(error);
        },
    };
    guard.finish(Some(rows as u64), Some(packed_bytes));
    let result = body(lease.tensor());
    let completion = lease.complete().and_then(|completion| completion.release()).map_err(|error| {
        cuvs_query_error(PACK_OPERATION, "failed to complete the uniform List producer read lifetime", error)
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
    });
    match (result, completion) {
        (Err(error), _) => Err(error),
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(error)) => Err(error),
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "native cuVS call parameters are fixed by the FFI signature and stay explicit"
)]
fn with_owned_uniform_list_relation_matrix<T>(
    context: &ExecutionContext,
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    role: &'static str,
    binding: &VectorRelationBinding,
    input: &GpuDataFrame,
    input_schema: &arrow::datatypes::SchemaRef,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
    layout: MatrixLayout,
    body: impl FnOnce(OwnedFloat32Matrix, usize) -> Result<T>,
) -> Result<T> {
    let column = uniform_list_vector_column(binding, node_id, role)?;
    context.check_cancellation("query_engine.vector_algorithm.before_pack")?;
    let batch = GpuRecordBatch::new(input.frame().clone(), input_schema.clone()).map_err(|error| {
        cudf_query_error(PACK_OPERATION, "failed to bind the vector relation to its variable-width List schema", error)
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
    })?;
    let field_index = uniform_list_field_index(input_schema, column, node_id, role)?;
    let validated = batch.try_prepare_uniform_list_float32_export(field_index).map_err(|error| {
        cudf_query_error(PACK_OPERATION, "failed to prove uniform List vector offsets and finite values", error)
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
    })?;
    let rows = validated.rows();
    let dimension = validated.dimension();
    let input_bytes = input.device_allocated_size_bytes()?;
    let packed_bytes = matrix_bytes(rows, dimension, node_id, role)?;
    let identity = vector_identity(node_id, algorithm, resources, native_version)
        .with_vector_relation_role(role)
        .with_vector_dimension(dimension)
        .with_matrix_layout(matrix_layout_name(layout))
        .with_input_rows(rows as u64)
        .with_input_bytes(input_bytes);
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        RuntimeOperatorStageKind::VectorPack,
        identity,
    );
    if let Err(error) = context.check_cancellation("query_engine.vector_algorithm.before_uniform_list_pack") {
        guard.fail(&error);
        return Err(error);
    }
    let packed = match resources.pack_validated_uniform_list_float32(&validated, layout).map_err(|error| {
        cuvs_query_error(
            PACK_OPERATION,
            "failed to pack the validated uniform List vector child into an owned matrix",
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("relation_role", role)
        .with_fact("column", column)
    }) {
        Ok(packed) => packed,
        Err(error) => {
            guard.fail(&error);
            return Err(error);
        },
    };
    guard.finish(Some(rows as u64), Some(packed_bytes));
    body(packed, dimension)
}

fn uniform_list_vector_column<'a>(
    binding: &'a VectorRelationBinding,
    node_id: NodeId,
    role: &'static str,
) -> Result<&'a str> {
    match &binding.vector {
        CuvsDenseVectorBinding::VectorColumn(column) => Ok(column),
        CuvsDenseVectorBinding::Columns(_) => Err(QueryError::new(
            QueryErrorCode::CudfInvariantViolation,
            PACK_OPERATION,
            "uniform List vector packing received a scalar-column binding",
        )
        .with_fact("node_id", node_id)
        .with_fact("relation_role", role)
        .with_fact("reason", "uniform_list_binding_kind_mismatch")),
    }
}

fn uniform_list_field_index(
    input_schema: &arrow::datatypes::SchemaRef,
    column: &str,
    node_id: NodeId,
    role: &'static str,
) -> Result<usize> {
    input_schema.index_of(column).map_err(|_| {
        QueryError::new(
            QueryErrorCode::InvalidPlan,
            PACK_OPERATION,
            "vector column is absent from the evaluated relation schema",
        )
        .with_fact("node_id", node_id)
        .with_fact("relation_role", role)
        .with_fact("column", column)
        .with_fact("reason", "missing_feature_column")
    })
}

fn tensor_dimension(
    tensor: &ValidatedDeviceTensor<'_>,
    node_id: NodeId,
    role: &'static str,
    algorithm: &'static str,
) -> Result<usize> {
    tensor.columns().map_err(|error| {
        cuvs_query_error(
            EXECUTE_OPERATION,
            format!("failed to inspect the validated {algorithm} {role} feature dimension"),
            error,
        )
        .with_fact("node_id", node_id)
        .with_fact("relation_role", role)
    })
}

fn native_kmeans_params(params: &VectorKMeansParams, node_id: NodeId) -> Result<KMeansParams> {
    let n_clusters = i32::try_from(params.n_clusters).map_err(|_| {
        kmeans_input_error(node_id, "cluster_count_exceeds_cuvs_i32", "K-means n_clusters exceeds the cuVS i32 range")
            .with_fact("n_clusters", params.n_clusters)
            .with_fact("maximum_clusters", i32::MAX)
    })?;
    let max_iter = i32::try_from(params.max_iterations).map_err(|_| {
        kmeans_input_error(node_id, "max_iterations_exceeds_cuvs_i32", "K-means max_iter exceeds the cuVS i32 range")
            .with_fact("max_iterations", params.max_iterations)
            .with_fact("maximum_iterations", i32::MAX)
    })?;
    let n_init = i32::try_from(params.n_init).map_err(|_| {
        kmeans_input_error(node_id, "n_init_exceeds_cuvs_i32", "K-means n_init exceeds the cuVS i32 range")
            .with_fact("n_init", params.n_init)
            .with_fact("maximum_n_init", i32::MAX)
    })?;
    let init = match params.init {
        VectorKMeansInit::KMeansPlusPlus => KMeansInit::KMeansPlusPlus,
        VectorKMeansInit::Random => KMeansInit::Random,
        VectorKMeansInit::Relation => KMeansInit::Array,
    };
    let metric = match params.metric {
        VectorKMeansMetric::L2Expanded => KMeansMetric::L2Expanded,
        VectorKMeansMetric::L2SqrtExpanded => KMeansMetric::L2SqrtExpanded,
    };
    Ok(KMeansParams {
        n_clusters: usize::try_from(n_clusters).expect("checked nonnegative K-means clusters"),
        max_iter,
        tol: params.tolerance,
        n_init,
        init,
        metric,
    })
}

fn native_pca_params(params: &VectorPcaParams, node_id: NodeId) -> Result<PcaParams> {
    let n_components = i32::try_from(params.n_components).map_err(|_| {
        pca_input_error(node_id, "component_count_exceeds_cuvs_i32", "PCA n_components exceeds the cuVS i32 range")
            .with_fact("n_components", params.n_components)
            .with_fact("maximum_components", i32::MAX)
    })?;
    let iterations = i32::try_from(params.n_iterations).map_err(|_| {
        pca_input_error(node_id, "iterations_exceed_cuvs_i32", "PCA n_iterations exceeds the cuVS i32 range")
            .with_fact("n_iterations", params.n_iterations)
            .with_fact("maximum_iterations", i32::MAX)
    })?;
    let tolerance = params.tolerance as f32;
    if !tolerance.is_finite() {
        return Err(pca_input_error(
            node_id,
            "tolerance_out_of_float32_range",
            "PCA tolerance cannot be represented by cuVS Float32",
        )
        .with_fact("tolerance", params.tolerance));
    }
    let solver = match params.solver {
        VectorPcaSolver::CovEigDq => PcaSolver::CovarianceDivideAndConquer,
        VectorPcaSolver::CovEigJacobi => PcaSolver::CovarianceJacobi,
    };
    Ok(PcaParams {
        n_components: usize::try_from(n_components).expect("checked nonnegative PCA components"),
        // SQL must preserve the input relation even when cuVS is allowed to
        // mutate a private packing allocation internally.
        copy: true,
        whiten: params.whiten,
        solver,
        tolerance,
        iterations,
        flip_signs_based_on_u: params.flip_signs_based_on_u,
    })
}

fn matrix_layout_name(layout: MatrixLayout) -> &'static str {
    match layout {
        MatrixLayout::RowMajor => "row_major",
        MatrixLayout::ColumnMajor => "column_major",
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "native cuVS call parameters are fixed by the FFI signature and stay explicit"
)]
fn observe_native<T>(
    context: &ExecutionContext,
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
    strategy: &'static str,
    input_rows: usize,
    input_bytes: u64,
    output_rows: Option<usize>,
    output_bytes: Option<u64>,
    body: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let identity = vector_identity(node_id, algorithm, resources, native_version)
        .with_strategy_label(strategy)
        .with_input_rows(input_rows as u64)
        .with_input_bytes(input_bytes);
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        RuntimeOperatorStageKind::VectorNative,
        identity,
    );
    match body() {
        Ok(value) => {
            guard.finish(output_rows.map(|rows| rows as u64), output_bytes);
            Ok(value)
        },
        Err(error) => {
            guard.fail(&error);
            Err(error)
        },
    }
}

fn observe_assembly(
    context: &ExecutionContext,
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
    output_rows: usize,
    body: impl FnOnce() -> Result<GpuDataFrame>,
) -> Result<GpuDataFrame> {
    let identity = vector_identity(node_id, algorithm, resources, native_version);
    let (observation, plan_execution_id, parent_span_id) = stage_context_from_execution(context);
    let guard = RuntimeOperatorStageGuard::start(
        observation,
        plan_execution_id,
        parent_span_id,
        RuntimeOperatorStageKind::VectorAssemble,
        identity,
    );
    match body() {
        Ok(output) => {
            let output_bytes = match output.device_allocated_size_bytes() {
                Ok(output_bytes) => output_bytes,
                Err(error) => {
                    guard.fail(&error);
                    return Err(error);
                },
            };
            guard.finish(Some(output_rows as u64), Some(output_bytes));
            Ok(output)
        },
        Err(error) => {
            guard.fail(&error);
            Err(error)
        },
    }
}

fn vector_identity(
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
) -> RuntimeOperatorStageIdentity {
    RuntimeOperatorStageIdentity::new()
        .with_node_id(node_id.to_string())
        .with_vector_algorithm(algorithm.as_str())
        .with_execution_binding_id(resources.binding_id().get())
        .with_device_ordinal(resources.device().ordinal())
        .with_native_version(native_version.0, native_version.1, native_version.2)
}

fn relation_input<'a>(
    inputs: &'a BTreeMap<NodeId, GpuDataFrame>,
    binding: &VectorRelationBinding,
    node_id: NodeId,
    role: &'static str,
) -> Result<&'a GpuDataFrame> {
    inputs.get(&binding.input).ok_or_else(|| {
        QueryError::new(QueryErrorCode::InvalidPlan, EXECUTE_OPERATION, "evaluated vector relation input is missing")
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("input_node_id", binding.input)
            .with_fact("reason", "missing_evaluated_relation")
    })
}

fn relation_dimension(
    binding: &VectorRelationBinding,
    input_schema: &arrow::datatypes::SchemaRef,
    node_id: NodeId,
    role: &'static str,
) -> Result<usize> {
    match &binding.vector {
        CuvsDenseVectorBinding::Columns(columns) => Ok(columns.len()),
        CuvsDenseVectorBinding::VectorColumn(column) => match input_schema
            .field_with_name(column)
            .map_err(|_| {
                QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    PACK_OPERATION,
                    "vector column is absent from its input schema",
                )
                .with_fact("node_id", node_id)
                .with_fact("relation_role", role)
                .with_fact("column", column)
                .with_fact("reason", "missing_feature_column")
            })?
            .data_type()
        {
            arrow::datatypes::DataType::FixedSizeList(_, dimension) if *dimension > 0 => Ok(*dimension as usize),
            arrow::datatypes::DataType::List(_) | arrow::datatypes::DataType::LargeList(_) => Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                PACK_OPERATION,
                "variable List<Float32> vector columns require a uniform-dimension device witness",
            )
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
            .with_fact("reason", "uniform_list_dimension_witness_unavailable")
            .with_fact("host_fallback", "disabled")),
            actual => Err(QueryError::new(
                QueryErrorCode::UnsupportedType,
                PACK_OPERATION,
                "vector column is not a positive-width FixedSizeList",
            )
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
            .with_fact("data_type", format!("{actual:?}"))
            .with_fact("reason", "unsupported_feature_type")),
        },
    }
}

fn distance_metric(metric: VectorDistanceMetric) -> DistanceMetric {
    match metric {
        VectorDistanceMetric::L2Expanded => DistanceMetric::L2Expanded,
        VectorDistanceMetric::L2SqrtExpanded => DistanceMetric::L2SqrtExpanded,
        VectorDistanceMetric::CosineExpanded => DistanceMetric::CosineExpanded,
        VectorDistanceMetric::InnerProduct => DistanceMetric::InnerProduct,
    }
}

fn checked_output_rows(query_rows: usize, k: usize, node_id: NodeId) -> Result<usize> {
    let output_rows = query_rows.checked_mul(k).ok_or_else(|| size_overflow(node_id, "output_rows"))?;
    if output_rows > i32::MAX as usize {
        return Err(QueryError::new(
            QueryErrorCode::CudfInvalidInput,
            EXECUTE_OPERATION,
            "kNN output row count exceeds libcudf size_type range",
        )
        .with_fact("node_id", node_id)
        .with_fact("query_rows", query_rows)
        .with_fact("k", k)
        .with_fact("output_rows", output_rows)
        .with_fact("maximum_rows", i32::MAX)
        .with_fact("reason", "row_count_exceeds_libcudf_size_type"));
    }
    Ok(output_rows)
}

fn matrix_bytes(rows: usize, dimension: usize, node_id: NodeId, role: &'static str) -> Result<u64> {
    rows.checked_mul(dimension)
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| size_overflow(node_id, role))
}

fn size_overflow(node_id: NodeId, resource: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::SizeOverflow, EXECUTE_OPERATION, "vector execution size calculation overflowed")
        .with_fact("node_id", node_id)
        .with_fact("algorithm", "brute_force_knn")
        .with_fact("resource", resource)
        .with_fact("reason", "size_calculation_overflow")
}

fn knn_input_error(node_id: NodeId, reason: &'static str, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::CudfInvalidInput, EXECUTE_OPERATION, message)
        .with_fact("node_id", node_id)
        .with_fact("algorithm", "brute_force_knn")
        .with_fact("reason", reason)
}

fn kmeans_input_error(node_id: NodeId, reason: &'static str, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::CudfInvalidInput, EXECUTE_OPERATION, message)
        .with_fact("node_id", node_id)
        .with_fact("algorithm", "kmeans")
        .with_fact("reason", reason)
}

fn pca_input_error(node_id: NodeId, reason: &'static str, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::CudfInvalidInput, EXECUTE_OPERATION, message)
        .with_fact("node_id", node_id)
        .with_fact("algorithm", "pca")
        .with_fact("reason", reason)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        plan::{
            PlanNode, QueryPlanBuilder, VectorAlgorithm, VectorAlgorithmSpec, VectorPcaParams, VectorRowIdBinding,
            vector_pca_output_schema,
        },
        runtime::ExecutionCancellation,
        source::{GpuResidentSource, SourceNode},
    };

    #[test]
    fn direct_enabled_execution_observes_pre_cancel_before_resources_or_inputs() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Float32, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let input = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("missing", Arc::clone(&schema))));
        let params =
            VectorPcaParams::new(VectorRelationBinding::new(input, VectorRowIdBinding::column("id"), ["x"]), 1);
        let root = builder.add_vector_algorithm(VectorAlgorithmSpec::new(
            VectorAlgorithm::Pca,
            VectorAlgorithmParams::Pca(params),
            vector_pca_output_schema(DataType::Int64, 1),
        ));
        let plan = builder.finish_gpu_resident(root);
        let PlanNode::VectorAlgorithm(node) = &plan.node(root).expect("vector node").kind else {
            panic!("root should remain a vector node");
        };
        let cancellation = ExecutionCancellation::new();
        cancellation.cancel();
        let context =
            ExecutionContext::builder().with_cancellation(cancellation).build().expect("cancelled execution context");

        let error = match execute_vector_algorithm(&context, &plan, root, node, &BTreeMap::new()) {
            Ok(_) => panic!("pre-cancelled direct execution must stop before resource or input access"),
            Err(error) => error,
        };
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.operation(), EXECUTE_OPERATION);
    }
}
