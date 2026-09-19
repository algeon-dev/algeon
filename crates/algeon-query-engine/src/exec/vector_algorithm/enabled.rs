use std::{collections::BTreeMap, sync::Arc};

use arrow::datatypes::{DataType, SchemaRef};
use cudf::{CudfType, DenseMatrixLayout, PackedMatrix, Series, TypeId, frame::dataframe::GpuRecordBatch};
use cuvs::{
    BruteForceIndex, CagraBuildAlgo, CagraIndex, CagraIndexParams, CagraSearchAlgo, CagraSearchParams, CuvsResources,
    DeviceBuffer, DeviceElement, DistanceMetric, IvfFlatIndex, IvfFlatIndexParams, IvfFlatSearchParams,
    IvfPqCodebookKind, IvfPqIndex, IvfPqIndexParams, IvfPqSearchParams, KMeansElement, KMeansInit, KMeansMetric,
    KMeansModel, KMeansParams, MatrixLayout, OwnedMatrix, PairwiseElement, PcaOutputBuffers, PcaParams, PcaSolver,
    ValidatedDeviceTensor, fit_transform_owned, pairwise_distance_into, select_k_into,
};

use super::assembly::{
    KnnAssemblyParams, assemble_kmeans_output, assemble_knn_output, assemble_pairwise_distance_output,
    assemble_pca_output, assemble_top_k_output,
};
use crate::{
    exec::{
        GpuDataFrame,
        dense_relation::{
            VECTOR_OPERATIONS, checked_output_rows, matrix_bytes_for, pack_dense_relation_for, relation_dimension,
            relation_element_type_for, relation_input_for, size_overflow_for,
        },
        *,
    },
    observability::{
        RuntimeOperatorStageGuard, RuntimeOperatorStageIdentity, RuntimeOperatorStageKind, stage_context_from_execution,
    },
    plan::{
        DenseVectorBinding, QueryPlan, VectorAlgorithm, VectorAlgorithmNode, VectorAlgorithmParams,
        VectorCagraBuildAlgo, VectorCagraSearchAlgo, VectorDistanceMetric, VectorIvfPqCodebookKind, VectorKMeansInit,
        VectorKMeansMetric, VectorKMeansParams, VectorPairwiseDistanceParams, VectorPcaParams, VectorPcaSolver,
        VectorRelationBinding, VectorTopKParams,
    },
};

const EXECUTE_OPERATION: &str = "query_engine.vector_algorithm.execute";
const PACK_OPERATION: &str = "query_engine.vector_algorithm.pack";
const NATIVE_OPERATION: &str = "query_engine.vector_algorithm.native";

macro_rules! dispatch_element {
    ($element:expr, $node_id:expr, $algorithm:expr, [$($id:pat => $ty:ty),+ $(,)?], |$type_name:ident| $body:block) => {
        match $element {
            $($id => { type $type_name = $ty; $body },)+
            actual => Err(unsupported_vector_element($node_id, $algorithm, "input", Some(actual))),
        }
    };
}

fn unsupported_vector_element(
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    role: &'static str,
    element: Option<TypeId>,
) -> QueryError {
    let error = QueryError::new(
        QueryErrorCode::UnsupportedType,
        PACK_OPERATION,
        "vector element type is not supported by this operation",
    )
    .with_fact("node_id", node_id)
    .with_fact("algorithm", algorithm.as_str())
    .with_fact("relation_role", role)
    .with_fact("reason", "unsupported_vector_element_type");
    match element {
        Some(element) => error.with_fact("element", element.name()),
        None => error,
    }
}

fn relation_element(
    plan: &QueryPlan,
    inputs: &BTreeMap<NodeId, GpuDataFrame>,
    binding: &VectorRelationBinding,
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    role: &'static str,
) -> Result<TypeId> {
    relation_element_type_for(
        &VECTOR_OPERATIONS,
        plan,
        inputs,
        binding.input,
        &binding.vector,
        node_id,
        algorithm.as_str(),
        role,
    )
}

fn require_matching_elements(left: TypeId, right: TypeId, node_id: NodeId, algorithm: VectorAlgorithm) -> Result<()> {
    if left != right {
        return Err(vector_input_error(
            node_id,
            algorithm,
            "vector_element_type_mismatch",
            "vector relation element types must match at execution",
        )
        .with_fact("left_element", left.name())
        .with_fact("right_element", right.name()));
    }
    Ok(())
}

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
    let domain_token = context
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
    // The pool guard remains live through native index destruction and output assembly.
    let _pool_coverage = resources
        .stream_pool_coverage(domain_token.as_ref().map(cudf::ExecutionBudgetToken::stream_reservation))
        .map_err(|error| {
            cuvs_query_error(EXECUTE_OPERATION, "failed to cover cuVS stream pool allocations", error)
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
            require_vector_element::<f32>(
                relation_element(plan, inputs, &params.dataset, node_id, VectorAlgorithm::BruteForceKnn, "dataset")?,
                node_id,
                VectorAlgorithm::BruteForceKnn,
                "dataset",
            )?;
            execute_knn(
                context,
                plan,
                node_id,
                KnnExecutionOptions {
                    dataset: &params.dataset,
                    queries: &params.queries,
                    k: params.k,
                    metric: params.metric,
                    algorithm: VectorAlgorithm::BruteForceKnn,
                    n_lists: None,
                    missing_neighbor: None,
                    build_and_search: |dataset: &ValidatedDeviceTensor<'_, f32>,
                                       queries: &ValidatedDeviceTensor<'_, f32>,
                                       output_rows| {
                        run_knn_native(
                            context,
                            node_id,
                            VectorAlgorithm::BruteForceKnn,
                            &resources,
                            native_version,
                            dataset,
                            queries,
                            output_rows,
                            || BruteForceIndex::build(&resources, dataset, distance_metric(params.metric)),
                            |index, neighbors, distances| {
                                index.search_into(queries, params.k, neighbors, distances).map(|_| ())
                            },
                            "failed to execute exact kNN search",
                        )
                    },
                },
                inputs,
                &resources,
                native_version,
            )
        },
        VectorAlgorithmParams::IvfFlat(params) => dispatch_element!(
            relation_element(plan, inputs, &params.dataset, node_id, VectorAlgorithm::IvfFlat, "dataset")?,
            node_id, VectorAlgorithm::IvfFlat,
            [TypeId::Float32 => f32, TypeId::Int8 => i8, TypeId::UInt8 => u8], |Element| {
            execute_knn(
            context,
            plan,
            node_id,
            KnnExecutionOptions {
                dataset: &params.dataset,
                queries: &params.queries,
                k: params.k,
                metric: params.metric,
                algorithm: VectorAlgorithm::IvfFlat,
                n_lists: Some(params.n_lists),
                missing_neighbor: Some(i64::MAX),
                build_and_search: |dataset: &ValidatedDeviceTensor<'_, Element>,
                                   queries: &ValidatedDeviceTensor<'_, Element>,
                                   output_rows| {
                    let mut build = IvfFlatIndexParams::new(distance_metric(params.metric), params.n_lists);
                    build.kmeans_n_iters = params.kmeans_n_iters;
                    build.kmeans_trainset_fraction = params.kmeans_trainset_fraction;
                    run_knn_native(
                        context,
                        node_id,
                        VectorAlgorithm::IvfFlat,
                        &resources,
                        native_version,
                        dataset,
                        queries,
                        output_rows,
                        || IvfFlatIndex::build(&resources, dataset, build),
                        |index, neighbors, distances| {
                            index
                                .search_into(
                                    queries,
                                    params.k,
                                    IvfFlatSearchParams { n_probes: params.n_probes },
                                    neighbors,
                                    distances,
                                )
                                .map(|_| ())
                        },
                        "failed to execute IVF-Flat kNN search",
                    )
                },
            },
            inputs,
            &resources,
            native_version,
        )
        }),
        VectorAlgorithmParams::IvfPq(params) => dispatch_element!(
            relation_element(plan, inputs, &params.dataset, node_id, VectorAlgorithm::IvfPq, "dataset")?,
            node_id, VectorAlgorithm::IvfPq,
            [TypeId::Float32 => f32, TypeId::Int8 => i8, TypeId::UInt8 => u8], |Element| {
            execute_knn(
            context,
            plan,
            node_id,
            KnnExecutionOptions {
                dataset: &params.dataset,
                queries: &params.queries,
                k: params.k,
                metric: params.metric,
                algorithm: VectorAlgorithm::IvfPq,
                n_lists: Some(params.n_lists),
                missing_neighbor: Some(i64::MAX),
                build_and_search: |dataset: &ValidatedDeviceTensor<'_, Element>,
                                   queries: &ValidatedDeviceTensor<'_, Element>,
                                   output_rows| {
                    let mut build = IvfPqIndexParams::new(distance_metric(params.metric), params.n_lists);
                    build.kmeans_n_iters = params.kmeans_n_iters;
                    build.kmeans_trainset_fraction = params.kmeans_trainset_fraction;
                    build.pq_dim = params.pq_dim;
                    build.pq_bits = params.pq_bits;
                    build.codebook_kind = match params.codebook_kind {
                        VectorIvfPqCodebookKind::PerSubspace => IvfPqCodebookKind::PerSubspace,
                        VectorIvfPqCodebookKind::PerCluster => IvfPqCodebookKind::PerCluster,
                    };
                    run_knn_native(
                        context,
                        node_id,
                        VectorAlgorithm::IvfPq,
                        &resources,
                        native_version,
                        dataset,
                        queries,
                        output_rows,
                        || IvfPqIndex::build(&resources, dataset, build),
                        |index, neighbors, distances| {
                            index
                                .search_into(
                                    queries,
                                    params.k,
                                    IvfPqSearchParams { n_probes: params.n_probes },
                                    neighbors,
                                    distances,
                                )
                                .map(|_| ())
                        },
                        "failed to execute IVF-PQ kNN search",
                    )
                },
            },
            inputs,
            &resources,
            native_version,
        )
        }),
        VectorAlgorithmParams::Cagra(params) => dispatch_element!(
            relation_element(plan, inputs, &params.dataset, node_id, VectorAlgorithm::Cagra, "dataset")?,
            node_id, VectorAlgorithm::Cagra,
            [TypeId::Float32 => f32, TypeId::Int8 => i8, TypeId::UInt8 => u8], |Element| {
            execute_knn(
            context,
            plan,
            node_id,
            KnnExecutionOptions {
                dataset: &params.dataset,
                queries: &params.queries,
                k: params.k,
                metric: params.metric,
                algorithm: VectorAlgorithm::Cagra,
                n_lists: None,
                missing_neighbor: Some(i64::MAX),
                build_and_search: |dataset: &ValidatedDeviceTensor<'_, Element>,
                                   queries: &ValidatedDeviceTensor<'_, Element>,
                                   output_rows| {
                    let mut build = CagraIndexParams::new(distance_metric(params.metric));
                    build.graph_degree = params.graph_degree as usize;
                    build.intermediate_graph_degree = params.intermediate_graph_degree as usize;
                    build.build_algo = match params.build_algo {
                        VectorCagraBuildAlgo::IvfPq => CagraBuildAlgo::IvfPq,
                        VectorCagraBuildAlgo::NnDescent => CagraBuildAlgo::NnDescent,
                    };
                    let search = CagraSearchParams {
                        itopk_size: params.itopk_size as usize,
                        search_width: params.search_width as usize,
                        max_iterations: params.max_iterations as usize,
                        search_algo: match params.search_algo {
                            VectorCagraSearchAlgo::Auto => CagraSearchAlgo::Auto,
                            VectorCagraSearchAlgo::SingleCta => CagraSearchAlgo::SingleCta,
                            VectorCagraSearchAlgo::MultiCta => CagraSearchAlgo::MultiCta,
                            VectorCagraSearchAlgo::MultiKernel => CagraSearchAlgo::MultiKernel,
                        },
                    };
                    run_knn_native(
                        context,
                        node_id,
                        VectorAlgorithm::Cagra,
                        &resources,
                        native_version,
                        dataset,
                        queries,
                        output_rows,
                        || CagraIndex::build(&resources, dataset, build),
                        |index, neighbors, distances| {
                            index.search_into(queries, params.k, search, neighbors, distances).map(|_| ())
                        },
                        "failed to execute CAGRA kNN search",
                    )
                },
            },
            inputs,
            &resources,
            native_version,
        )
        }),
        VectorAlgorithmParams::PairwiseDistance(params) => dispatch_element!(
            relation_element(plan, inputs, &params.x, node_id, VectorAlgorithm::PairwiseDistance, "x")?,
            node_id, VectorAlgorithm::PairwiseDistance, [TypeId::Float32 => f32, TypeId::Float64 => f64], |Element| {
                execute_pairwise_distance::<Element>(context, plan, node_id, params, inputs, &resources, native_version)
            }
        ),
        VectorAlgorithmParams::TopK(params) => {
            require_vector_element::<f32>(
                relation_element(plan, inputs, &params.input, node_id, VectorAlgorithm::TopK, "input")?,
                node_id,
                VectorAlgorithm::TopK,
                "input",
            )?;
            execute_top_k(context, plan, node_id, params, inputs, &resources, native_version)
        },
        VectorAlgorithmParams::KMeans(params) => dispatch_element!(
            relation_element(plan, inputs, &params.input, node_id, VectorAlgorithm::KMeans, "input")?,
            node_id, VectorAlgorithm::KMeans, [TypeId::Float32 => f32, TypeId::Float64 => f64], |Element| {
                execute_kmeans::<Element>(context, plan, node_id, params, inputs, &resources, native_version)
            }
        ),
        VectorAlgorithmParams::Pca(params) => {
            require_vector_element::<f32>(
                relation_element(plan, inputs, &params.input, node_id, VectorAlgorithm::Pca, "input")?,
                node_id,
                VectorAlgorithm::Pca,
                "input",
            )?;
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

struct KnnExecutionOptions<'a, F> {
    dataset: &'a VectorRelationBinding,
    queries: &'a VectorRelationBinding,
    k: usize,
    metric: VectorDistanceMetric,
    algorithm: VectorAlgorithm,
    n_lists: Option<u32>,
    missing_neighbor: Option<i64>,
    build_and_search: F,
}

#[expect(
    clippy::too_many_arguments,
    reason = "the observer needs the native call's device shapes and execution identity"
)]
fn run_knn_native<I, T: DeviceElement>(
    context: &ExecutionContext,
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
    dataset: &ValidatedDeviceTensor<'_, T>,
    queries: &ValidatedDeviceTensor<'_, T>,
    output_rows: usize,
    build: impl FnOnce() -> cuvs::Result<I>,
    search: impl FnOnce(&I, &mut DeviceBuffer<i64>, &mut DeviceBuffer<f32>) -> cuvs::Result<()>,
    search_error: &'static str,
) -> Result<(DeviceBuffer<i64>, DeviceBuffer<f32>)> {
    let dataset_rows = dataset.rows().map_err(|error| {
        cuvs_query_error(EXECUTE_OPERATION, "failed to inspect the validated kNN dataset shape", error)
    })?;
    let dimension = dataset.columns().map_err(|error| {
        cuvs_query_error(EXECUTE_OPERATION, "failed to inspect the validated kNN dataset dimension", error)
    })?;
    let query_rows = queries.rows().map_err(|error| {
        cuvs_query_error(EXECUTE_OPERATION, "failed to inspect the validated kNN query shape", error)
    })?;
    let dataset_bytes = matrix_bytes_for::<T>(&VECTOR_OPERATIONS, dataset_rows, dimension, node_id, "dataset")?;
    let index = observe_native(
        context,
        node_id,
        algorithm,
        resources,
        native_version,
        "build_index",
        dataset_rows,
        dataset_bytes,
        None,
        None,
        || {
            context.check_cancellation("query_engine.vector_algorithm.before_index_build")?;
            let index = build().map_err(|error| {
                cuvs_query_error(NATIVE_OPERATION, "failed to build the query-local kNN index", error)
            })?;
            context.check_cancellation("query_engine.vector_algorithm.after_index_build")?;
            Ok(index)
        },
    )?;
    let mut neighbors = DeviceBuffer::<i64>::new(resources, output_rows)
        .map_err(|error| cuvs_query_error(NATIVE_OPERATION, "failed to allocate kNN neighbor output", error))?;
    let mut distances = DeviceBuffer::<f32>::new(resources, output_rows)
        .map_err(|error| cuvs_query_error(NATIVE_OPERATION, "failed to allocate kNN distance output", error))?;
    let query_bytes = matrix_bytes_for::<T>(&VECTOR_OPERATIONS, query_rows, dimension, node_id, "queries")?;
    let native_output_bytes = output_rows
        .checked_mul(std::mem::size_of::<i64>() + std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| size_overflow_for(&VECTOR_OPERATIONS, node_id, "native_output_bytes"))?;
    observe_native(
        context,
        node_id,
        algorithm,
        resources,
        native_version,
        "search",
        query_rows,
        query_bytes,
        Some(output_rows),
        Some(native_output_bytes),
        || {
            context.check_cancellation("query_engine.vector_algorithm.before_search")?;
            search(&index, &mut neighbors, &mut distances)
                .map_err(|error| cuvs_query_error(NATIVE_OPERATION, search_error, error))?;
            context.check_cancellation("query_engine.vector_algorithm.after_search")?;
            Ok(())
        },
    )?;
    Ok((neighbors, distances))
}

fn execute_knn<T: DeviceElement, F>(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    knn: KnnExecutionOptions<'_, F>,
    inputs: &BTreeMap<NodeId, GpuDataFrame>,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
) -> Result<GpuDataFrame>
where
    F: FnOnce(
        &ValidatedDeviceTensor<'_, T>,
        &ValidatedDeviceTensor<'_, T>,
        usize,
    ) -> Result<(DeviceBuffer<i64>, DeviceBuffer<f32>)>,
{
    let KnnExecutionOptions { dataset, queries, k, metric, algorithm, n_lists, missing_neighbor, build_and_search } =
        knn;
    require_matching_elements(
        relation_element(plan, inputs, dataset, node_id, algorithm, "dataset")?,
        relation_element(plan, inputs, queries, node_id, algorithm, "queries")?,
        node_id,
        algorithm,
    )?;
    let params = KnnAssemblyParams { dataset, queries, k, metric, missing_neighbor };
    let dataset = relation_input_for(&VECTOR_OPERATIONS, inputs, params.dataset.input, node_id, "dataset")?;
    let queries = relation_input_for(&VECTOR_OPERATIONS, inputs, params.queries.input, node_id, "queries")?;
    let dataset_rows = dataset.num_rows();
    if dataset_rows == 0 {
        return Err(vector_input_error(
            node_id,
            algorithm,
            "empty_dataset",
            "kNN dataset must contain at least one row",
        )
        .with_fact("k", params.k));
    }
    if params.k > dataset_rows {
        return Err(vector_input_error(
            node_id,
            algorithm,
            "k_exceeds_dataset_rows",
            "kNN k must not exceed the evaluated dataset row count",
        )
        .with_fact("k", params.k)
        .with_fact("dataset_rows", dataset_rows));
    }
    if let Some(n_lists) = n_lists
        && n_lists as usize > dataset_rows
    {
        return Err(vector_input_error(
            node_id,
            algorithm,
            "n_lists_exceeds_dataset_rows",
            "n_lists must not exceed the evaluated dataset row count",
        )
        .with_fact("n_lists", n_lists)
        .with_fact("dataset_rows", dataset_rows));
    }
    let output_rows = checked_output_rows(queries.num_rows(), params.k, node_id, algorithm.as_str())?;

    with_relation_tensor::<T, _>(
        context,
        plan,
        node_id,
        algorithm,
        "dataset",
        params.dataset,
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
                let query_dimension = relation_dimension(&params.queries.vector, query_schema, node_id, "queries")?;
                if query_dimension != dimension {
                    return Err(vector_input_error(
                        node_id,
                        algorithm,
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
                return observe_assembly(context, node_id, algorithm, resources, native_version, output_rows, || {
                    assemble_knn_output(
                        context,
                        KnnAssemblyParams {
                            dataset: params.dataset,
                            queries: params.queries,
                            k: params.k,
                            metric: params.metric,
                            missing_neighbor,
                        },
                        dataset,
                        queries,
                        neighbors,
                        distances,
                    )
                });
            }

            with_relation_tensor::<T, _>(
                context,
                plan,
                node_id,
                algorithm,
                "queries",
                params.queries,
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
                        return Err(vector_input_error(
                            node_id,
                            algorithm,
                            "feature_dimension_mismatch",
                            "kNN dataset and query dimensions must match at execution",
                        )
                        .with_fact("dataset_dimension", dimension)
                        .with_fact("query_dimension", query_dimension));
                    }
                    let (neighbors, distances) = build_and_search(dataset_tensor, query_tensor, output_rows)?;

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

                    observe_assembly(context, node_id, algorithm, resources, native_version, output_rows, || {
                        context.check_cancellation("query_engine.vector_algorithm.before_assembly")?;
                        let output = assemble_knn_output(
                            context,
                            KnnAssemblyParams {
                                dataset: params.dataset,
                                queries: params.queries,
                                k: params.k,
                                metric: params.metric,
                                missing_neighbor,
                            },
                            dataset,
                            queries,
                            neighbors,
                            distances,
                        )?;
                        context.check_cancellation("query_engine.vector_algorithm.after_assembly")?;
                        Ok(output)
                    })
                },
            )
        },
    )
}

fn execute_pairwise_distance<E: PairwiseElement>(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    params: &VectorPairwiseDistanceParams,
    inputs: &BTreeMap<NodeId, GpuDataFrame>,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
) -> Result<GpuDataFrame> {
    require_matching_elements(
        relation_element(plan, inputs, &params.x, node_id, VectorAlgorithm::PairwiseDistance, "x")?,
        relation_element(plan, inputs, &params.y, node_id, VectorAlgorithm::PairwiseDistance, "y")?,
        node_id,
        VectorAlgorithm::PairwiseDistance,
    )?;
    let x = relation_input_for(&VECTOR_OPERATIONS, inputs, params.x.input, node_id, "x")?;
    let y = relation_input_for(&VECTOR_OPERATIONS, inputs, params.y.input, node_id, "y")?;
    let rows = checked_output_rows(x.num_rows(), y.num_rows(), node_id, VectorAlgorithm::PairwiseDistance.as_str())?;
    let assemble = |distances| {
        observe_assembly(context, node_id, VectorAlgorithm::PairwiseDistance, resources, native_version, rows, || {
            context.check_cancellation("query_engine.vector_algorithm.before_assembly")?;
            let result = assemble_pairwise_distance_output(context, params, x, y, distances, E::Distance::TYPE_ID)?;
            context.check_cancellation("query_engine.vector_algorithm.after_assembly")?;
            Ok(result)
        })
    };
    if rows == 0 {
        let empty = resources
            .runtime()
            .series(&[] as &[E::Distance])
            .map_err(|error| cudf_query_error(EXECUTE_OPERATION, "failed to create empty pairwise output", error))?;
        return assemble(empty);
    }
    with_relation_tensor::<E, _>(
        context,
        plan,
        node_id,
        VectorAlgorithm::PairwiseDistance,
        "x",
        &params.x,
        x,
        resources,
        native_version,
        |x_tensor| {
            let dimension = tensor_dimension(x_tensor, node_id, "x", "pairwise")?;
            with_relation_tensor::<E, _>(
                context,
                plan,
                node_id,
                VectorAlgorithm::PairwiseDistance,
                "y",
                &params.y,
                y,
                resources,
                native_version,
                |y_tensor| {
                    if dimension != tensor_dimension(y_tensor, node_id, "y", "pairwise")? {
                        return Err(vector_input_error(
                            node_id,
                            VectorAlgorithm::PairwiseDistance,
                            "feature_dimension_mismatch",
                            "pairwise dimensions must match",
                        ));
                    }
                    let mut distances = DeviceBuffer::<E::Distance>::new(resources, rows).map_err(|error| {
                        cuvs_query_error(NATIVE_OPERATION, "failed to allocate pairwise output", error)
                    })?;
                    observe_native(
                        context,
                        node_id,
                        VectorAlgorithm::PairwiseDistance,
                        resources,
                        native_version,
                        "pairwise_distance",
                        x.num_rows(),
                        matrix_bytes_for::<E>(&VECTOR_OPERATIONS, x.num_rows(), dimension, node_id, "x")?,
                        Some(rows),
                        Some(matrix_bytes_for::<E::Distance>(
                            &VECTOR_OPERATIONS,
                            x.num_rows(),
                            y.num_rows(),
                            node_id,
                            "pairwise_output",
                        )?),
                        || {
                            context.check_cancellation("query_engine.vector_algorithm.before_pairwise")?;
                            pairwise_distance_into(
                                resources,
                                x_tensor,
                                y_tensor,
                                distance_metric(params.metric),
                                &mut distances,
                            )
                            .map_err(|error| {
                                cuvs_query_error(NATIVE_OPERATION, "failed to compute pairwise distances", error)
                            })?;
                            context.check_cancellation("query_engine.vector_algorithm.after_pairwise")
                        },
                    )?;
                    let distances = adopt_vector_output(distances, "pairwise distances")?;
                    assemble(distances)
                },
            )
        },
    )
}

fn execute_top_k(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    params: &VectorTopKParams,
    inputs: &BTreeMap<NodeId, GpuDataFrame>,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
) -> Result<GpuDataFrame> {
    let input = relation_input_for(&VECTOR_OPERATIONS, inputs, params.input.input, node_id, "input")?;
    let rows = checked_output_rows(input.num_rows(), params.k, node_id, VectorAlgorithm::TopK.as_str())?;
    let assemble = |positions, values| {
        observe_assembly(context, node_id, VectorAlgorithm::TopK, resources, native_version, rows, || {
            context.check_cancellation("query_engine.vector_algorithm.before_assembly")?;
            let result = assemble_top_k_output(context, params, input, positions, values)?;
            context.check_cancellation("query_engine.vector_algorithm.after_assembly")?;
            Ok(result)
        })
    };
    if input.num_rows() == 0 {
        let positions = resources
            .runtime()
            .series(&[] as &[i64])
            .map_err(|error| cudf_query_error(EXECUTE_OPERATION, "failed to create empty Top-K positions", error))?;
        let values = resources
            .runtime()
            .series(&[] as &[f32])
            .map_err(|error| cudf_query_error(EXECUTE_OPERATION, "failed to create empty Top-K values", error))?;
        return assemble(positions, values);
    }
    with_relation_tensor(
        context,
        plan,
        node_id,
        VectorAlgorithm::TopK,
        "input",
        &params.input,
        input,
        resources,
        native_version,
        |tensor| {
            let dimension = tensor_dimension(tensor, node_id, "input", "Top-K")?;
            if params.k > dimension {
                return Err(vector_input_error(
                    node_id,
                    VectorAlgorithm::TopK,
                    "k_exceeds_dimension",
                    "Top-K k must not exceed the evaluated vector dimension",
                )
                .with_fact("k", params.k)
                .with_fact("dimension", dimension));
            }
            let mut positions = DeviceBuffer::<i64>::new(resources, rows)
                .map_err(|error| cuvs_query_error(NATIVE_OPERATION, "failed to allocate Top-K positions", error))?;
            let mut values = DeviceBuffer::<f32>::new(resources, rows)
                .map_err(|error| cuvs_query_error(NATIVE_OPERATION, "failed to allocate Top-K values", error))?;
            let output_bytes = rows
                .checked_mul(std::mem::size_of::<i64>() + std::mem::size_of::<f32>())
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| size_overflow_for(&VECTOR_OPERATIONS, node_id, "top_k_output"))?;
            observe_native(
                context,
                node_id,
                VectorAlgorithm::TopK,
                resources,
                native_version,
                "select_k",
                input.num_rows(),
                matrix_bytes_for::<f32>(&VECTOR_OPERATIONS, input.num_rows(), dimension, node_id, "input")?,
                Some(rows),
                Some(output_bytes),
                || {
                    context.check_cancellation("query_engine.vector_algorithm.before_select_k")?;
                    select_k_into(resources, tensor, params.k, params.select_min, &mut values, &mut positions)
                        .map_err(|error| cuvs_query_error(NATIVE_OPERATION, "failed to select Top-K values", error))?;
                    context.check_cancellation("query_engine.vector_algorithm.after_select_k")
                },
            )?;
            assemble(adopt_vector_output(positions, "Top-K positions")?, adopt_vector_output(values, "Top-K values")?)
        },
    )
}

fn adopt_vector_output<T: DeviceElement>(buffer: DeviceBuffer<T>, name: &'static str) -> Result<Series> {
    Series::from_column(
        buffer
            .into_column()
            .map_err(|error| cuvs_query_error(EXECUTE_OPERATION, format!("failed to adopt {name}"), error))?,
    )
    .map_err(|error| cudf_query_error(EXECUTE_OPERATION, format!("failed to wrap {name}"), error))
}

fn vector_input_error(
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    reason: &'static str,
    message: &'static str,
) -> QueryError {
    QueryError::new(QueryErrorCode::CudfInvalidInput, EXECUTE_OPERATION, message)
        .with_fact("node_id", node_id)
        .with_fact("algorithm", algorithm.as_str())
        .with_fact("reason", reason)
}

fn execute_kmeans<E: KMeansElement>(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    params: &VectorKMeansParams,
    inputs: &BTreeMap<NodeId, GpuDataFrame>,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
) -> Result<GpuDataFrame> {
    let input = relation_input_for(&VECTOR_OPERATIONS, inputs, params.input.input, node_id, "input")?;
    let input_rows = input.num_rows();
    if input_rows == 0 {
        return Err(vector_input_error(
            node_id,
            VectorAlgorithm::KMeans,
            "empty_input",
            "K-means input must contain at least one row",
        ));
    }
    if input_rows > i32::MAX as usize {
        return Err(vector_input_error(
            node_id,
            VectorAlgorithm::KMeans,
            "row_count_exceeds_cuvs_i32",
            "K-means input row count exceeds the cuVS i32 range",
        )
        .with_fact("input_rows", input_rows)
        .with_fact("maximum_rows", i32::MAX));
    }
    if params.n_clusters > input_rows {
        return Err(vector_input_error(
            node_id,
            VectorAlgorithm::KMeans,
            "cluster_count_exceeds_input_rows",
            "K-means n_clusters must not exceed the evaluated input row count",
        )
        .with_fact("n_clusters", params.n_clusters)
        .with_fact("input_rows", input_rows));
    }
    let native_params = native_kmeans_params(params, node_id)?;
    if let Some(centroids) = &params.initial_centroids {
        require_matching_elements(
            relation_element(plan, inputs, &params.input, node_id, VectorAlgorithm::KMeans, "input")?,
            relation_element(plan, inputs, centroids, node_id, VectorAlgorithm::KMeans, "initial_centroids")?,
            node_id,
            VectorAlgorithm::KMeans,
        )?;
    }

    with_relation_tensor::<E, _>(
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
            let input_bytes = matrix_bytes_for::<E>(&VECTOR_OPERATIONS, input_rows, dimension, node_id, "input")?;
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
                (VectorKMeansInit::Relation, Some(centroid_binding)) => with_owned_relation_matrix::<E, _>(
                    context,
                    plan,
                    node_id,
                    VectorAlgorithm::KMeans,
                    "initial_centroids",
                    centroid_binding,
                    relation_input_for(
                        &VECTOR_OPERATIONS,
                        inputs,
                        centroid_binding.input,
                        node_id,
                        "initial_centroids",
                    )?,
                    resources,
                    native_version,
                    MatrixLayout::RowMajor,
                    |centroids, centroid_dimension| {
                        if centroid_dimension != dimension {
                            return Err(vector_input_error(
                                node_id,
                                VectorAlgorithm::KMeans,
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
                    return Err(vector_input_error(
                        node_id,
                        VectorAlgorithm::KMeans,
                        "missing_initial_centroids",
                        "relation-initialized K-means requires an initial-centroids input",
                    ));
                },
                (_, Some(_)) => {
                    return Err(vector_input_error(
                        node_id,
                        VectorAlgorithm::KMeans,
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
                        .ok_or_else(|| {
                            size_overflow_for(&VECTOR_OPERATIONS, node_id, "kmeans_assignment_output_bytes")
                        })?,
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
    let input = relation_input_for(&VECTOR_OPERATIONS, inputs, params.input.input, node_id, "input")?;
    let input_rows = input.num_rows();
    if input_rows < 2 {
        return Err(vector_input_error(
            node_id,
            VectorAlgorithm::Pca,
            "insufficient_input_rows",
            "PCA requires at least two input rows",
        )
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
                return Err(vector_input_error(
                    node_id,
                    VectorAlgorithm::Pca,
                    "insufficient_feature_dimension",
                    "PCA requires at least two evaluated input feature columns",
                )
                .with_fact("feature_dimension", dimension)
                .with_fact("minimum_feature_dimension", 2));
            }
            if params.n_components > dimension {
                return Err(vector_input_error(
                    node_id,
                    VectorAlgorithm::Pca,
                    "component_count_exceeds_feature_dimension",
                    "PCA n_components must not exceed the evaluated feature dimension",
                )
                .with_fact("n_components", params.n_components)
                .with_fact("feature_dimension", dimension));
            }
            let input_bytes = matrix_bytes_for::<f32>(&VECTOR_OPERATIONS, input_rows, dimension, node_id, "input")?;
            let mut outputs = PcaOutputBuffers::new(resources, input_rows, dimension, params.n_components)
                .map_err(|error| cuvs_query_error(NATIVE_OPERATION, "failed to allocate PCA device outputs", error))?;
            let transformed_bytes = matrix_bytes_for::<f32>(
                &VECTOR_OPERATIONS,
                input_rows,
                params.n_components,
                node_id,
                "pca_transformed_output",
            )?;
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
fn with_relation_tensor<E: DeviceElement, T>(
    context: &ExecutionContext,
    plan: &QueryPlan,
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    role: &'static str,
    binding: &VectorRelationBinding,
    input: &GpuDataFrame,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
    body: impl FnOnce(&ValidatedDeviceTensor<'_, E>) -> Result<T>,
) -> Result<T> {
    let input_schema = plan
        .node(binding.input)
        .ok_or_else(|| invalid_node_reference(binding.input, "vector relation references a missing input"))?
        .output_schema();
    if matches!(
        &binding.vector,
        DenseVectorBinding::VectorColumn(column)
            if matches!(
                input_schema.field_with_name(column).map(|field| field.data_type()),
                Ok(DataType::List(_) | DataType::LargeList(_))
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
    let dimension = relation_dimension(&binding.vector, input_schema, node_id, role)?;
    let input_rows = input.num_rows();
    let input_bytes = input.device_allocated_size_bytes()?;
    let packed_bytes = matrix_bytes_for::<E>(&VECTOR_OPERATIONS, input_rows, dimension, node_id, role)?;
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
        DenseVectorBinding::Columns(columns) => {
            let packed = (|| {
                let selected = input.frame().select(columns.iter().map(String::as_str)).map_err(|error| {
                    cudf_query_error(PACK_OPERATION, "failed to select the vector feature columns", error)
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
                let element = selected.series(&columns[0]).and_then(|series| series.dtype()).map_err(|error| {
                    cudf_query_error(PACK_OPERATION, "failed to inspect vector feature element type", error)
                })?;
                require_vector_element::<E>(element.id(), node_id, algorithm, role)?;
                resources.pack_prepared_table::<E>(&export, MatrixLayout::RowMajor).map_err(|error| {
                    cuvs_query_error(PACK_OPERATION, "failed to pack the prepared vector feature table", error)
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
        DenseVectorBinding::VectorColumn(column) => {
            let batch = match GpuRecordBatch::new(input.frame().clone(), Arc::clone(input_schema)).map_err(|error| {
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
            let validated = match batch.try_prepare_fixed_size_list_export(field_index).map_err(|error| {
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
            require_vector_element::<E>(validated.element(), node_id, algorithm, role)?;
            if validated.binding_id() != resources.binding_id() {
                let packed = validated.pack_active_child_into_runtime(resources.runtime(), DenseMatrixLayout::RowMajor);
                return with_copied_list_child_tensor(
                    resources,
                    packed,
                    node_id,
                    role,
                    column,
                    guard,
                    input_rows,
                    packed_bytes,
                    body,
                );
            }
            let lease = match resources.activate_validated_fixed_size_list_tensor::<E>(&validated).map_err(|error| {
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

#[expect(clippy::too_many_arguments, reason = "the native call owns the exact cuVS binding and matrix layout")]
fn with_owned_relation_matrix<E: DeviceElement, T>(
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
    body: impl FnOnce(OwnedMatrix<E>, usize) -> Result<T>,
) -> Result<T> {
    let (packed, dimension) = pack_dense_relation_for(
        &VECTOR_OPERATIONS,
        context,
        resources.runtime(),
        plan,
        node_id,
        role,
        algorithm.as_str(),
        binding.input,
        &binding.vector,
        input,
        E::TYPE_ID,
        layout.into(),
        vector_identity(node_id, algorithm, resources, native_version),
        RuntimeOperatorStageKind::VectorPack,
    )?;
    let owned = OwnedMatrix::from_packed(resources, packed).map_err(|error| {
        cuvs_query_error(PACK_OPERATION, "failed to pack the prepared vector feature table", error)
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
    })?;
    body(owned, dimension)
}

#[expect(
    clippy::too_many_arguments,
    reason = "native cuVS call parameters are fixed by the FFI signature and stay explicit"
)]
fn with_uniform_list_relation_tensor<E: DeviceElement, T>(
    context: &ExecutionContext,
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    role: &'static str,
    binding: &VectorRelationBinding,
    input: &GpuDataFrame,
    input_schema: &SchemaRef,
    resources: &CuvsResources,
    native_version: (u16, u16, u16),
    body: impl FnOnce(&ValidatedDeviceTensor<'_, E>) -> Result<T>,
) -> Result<T> {
    let column = uniform_list_vector_column(binding, node_id, role)?;
    context.check_cancellation("query_engine.vector_algorithm.before_pack")?;
    let batch = GpuRecordBatch::new(input.frame().clone(), Arc::clone(input_schema)).map_err(|error| {
        cudf_query_error(PACK_OPERATION, "failed to bind the vector relation to its variable-width List schema", error)
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
    })?;
    let field_index = uniform_list_field_index(input_schema, column, node_id, role)?;
    let validated = batch.try_prepare_uniform_list_export(field_index).map_err(|error| {
        cudf_query_error(PACK_OPERATION, "failed to prove uniform List vector offsets and finite values", error)
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
    })?;
    require_vector_element::<E>(validated.element(), node_id, algorithm, role)?;
    let rows = validated.rows();
    let dimension = validated.dimension();
    let input_bytes = input.device_allocated_size_bytes()?;
    let packed_bytes = matrix_bytes_for::<E>(&VECTOR_OPERATIONS, rows, dimension, node_id, role)?;
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
    if validated.binding_id() != resources.binding_id() {
        let packed = validated.pack_active_child_into_runtime(resources.runtime(), DenseMatrixLayout::RowMajor);
        return with_copied_list_child_tensor(
            resources,
            packed,
            node_id,
            role,
            column,
            guard,
            rows,
            packed_bytes,
            body,
        );
    }
    let lease = match resources.activate_validated_uniform_list_tensor::<E>(&validated).map_err(|error| {
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

/// Copies a validated list child into the execution binding of `resources`.
///
/// Zero-copy tensor activation requires the list to be owned by the cuVS
/// binding. A GPU-resident producer that ran on another execution lane owns
/// its output on that lane's binding, so its active child is packed into the
/// cuVS binding instead.
#[expect(clippy::too_many_arguments, reason = "the copy keeps the caller's stage guard and diagnostics identity")]
fn with_copied_list_child_tensor<E: DeviceElement, T>(
    resources: &CuvsResources,
    packed: cudf::Result<PackedMatrix>,
    node_id: NodeId,
    role: &'static str,
    column: &str,
    guard: RuntimeOperatorStageGuard,
    rows: usize,
    packed_bytes: u64,
    body: impl FnOnce(&ValidatedDeviceTensor<'_, E>) -> Result<T>,
) -> Result<T> {
    let owned = packed
        .map_err(|error| {
            cudf_query_error(
                PACK_OPERATION,
                "failed to copy the vector list child into the cuVS execution binding",
                error,
            )
        })
        .and_then(|packed| {
            OwnedMatrix::<E>::from_packed(resources, packed).map_err(|error| {
                cuvs_query_error(PACK_OPERATION, "failed to adopt the copied vector list child", error)
            })
        })
        .map_err(|error| {
            error.with_fact("node_id", node_id).with_fact("relation_role", role).with_fact("column", column)
        });
    let owned = match owned {
        Ok(owned) => owned,
        Err(error) => {
            guard.fail(&error);
            return Err(error);
        },
    };
    let tensor = match owned.as_validated_tensor(resources).map_err(|error| {
        cuvs_query_error(PACK_OPERATION, "failed to bind the copied vector list matrix", error)
            .with_fact("node_id", node_id)
            .with_fact("relation_role", role)
            .with_fact("column", column)
    }) {
        Ok(tensor) => tensor,
        Err(error) => {
            guard.fail(&error);
            return Err(error);
        },
    };
    guard.finish(Some(rows as u64), Some(packed_bytes));
    body(&tensor)
}

fn uniform_list_vector_column<'a>(
    binding: &'a VectorRelationBinding,
    node_id: NodeId,
    role: &'static str,
) -> Result<&'a str> {
    match &binding.vector {
        DenseVectorBinding::VectorColumn(column) => Ok(column),
        DenseVectorBinding::Columns(_) => Err(QueryError::new(
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
    input_schema: &SchemaRef,
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

fn require_vector_element<E: DeviceElement>(
    element: TypeId,
    node_id: NodeId,
    algorithm: VectorAlgorithm,
    role: &'static str,
) -> Result<()> {
    if element == E::TYPE_ID {
        return Ok(());
    }
    Err(unsupported_vector_element(node_id, algorithm, role, Some(element)))
}

fn tensor_dimension<E: DeviceElement>(
    tensor: &ValidatedDeviceTensor<'_, E>,
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
        vector_input_error(
            node_id,
            VectorAlgorithm::KMeans,
            "cluster_count_exceeds_cuvs_i32",
            "K-means n_clusters exceeds the cuVS i32 range",
        )
        .with_fact("n_clusters", params.n_clusters)
        .with_fact("maximum_clusters", i32::MAX)
    })?;
    let max_iter = i32::try_from(params.max_iterations).map_err(|_| {
        vector_input_error(
            node_id,
            VectorAlgorithm::KMeans,
            "max_iterations_exceeds_cuvs_i32",
            "K-means max_iter exceeds the cuVS i32 range",
        )
        .with_fact("max_iterations", params.max_iterations)
        .with_fact("maximum_iterations", i32::MAX)
    })?;
    let n_init = i32::try_from(params.n_init).map_err(|_| {
        vector_input_error(
            node_id,
            VectorAlgorithm::KMeans,
            "n_init_exceeds_cuvs_i32",
            "K-means n_init exceeds the cuVS i32 range",
        )
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
        vector_input_error(
            node_id,
            VectorAlgorithm::Pca,
            "component_count_exceeds_cuvs_i32",
            "PCA n_components exceeds the cuVS i32 range",
        )
        .with_fact("n_components", params.n_components)
        .with_fact("maximum_components", i32::MAX)
    })?;
    let iterations = i32::try_from(params.n_iterations).map_err(|_| {
        vector_input_error(
            node_id,
            VectorAlgorithm::Pca,
            "iterations_exceed_cuvs_i32",
            "PCA n_iterations exceeds the cuVS i32 range",
        )
        .with_fact("n_iterations", params.n_iterations)
        .with_fact("maximum_iterations", i32::MAX)
    })?;
    let tolerance = params.tolerance as f32;
    if !tolerance.is_finite() {
        return Err(vector_input_error(
            node_id,
            VectorAlgorithm::Pca,
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

fn distance_metric(metric: VectorDistanceMetric) -> DistanceMetric {
    match metric {
        VectorDistanceMetric::L2Expanded => DistanceMetric::L2Expanded,
        VectorDistanceMetric::L2SqrtExpanded => DistanceMetric::L2SqrtExpanded,
        VectorDistanceMetric::CosineExpanded => DistanceMetric::CosineExpanded,
        VectorDistanceMetric::InnerProduct => DistanceMetric::InnerProduct,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };

    use arrow::datatypes::{DataType, Field, Schema};
    use cudf::{QueryMemoryDomainOptions, Runtime};

    use super::*;
    use crate::{
        plan::{
            PlanNode, QueryPlanBuilder, VectorAlgorithm, VectorAlgorithmSpec, VectorPcaParams, VectorRowIdBinding,
            vector_pca_output_schema,
        },
        runtime::{ExecutionCancellation, MemoryControlPolicy},
        source::{GpuResidentSource, SourceNode},
    };

    #[test]
    fn cancellation_after_real_ivf_flat_build_releases_index_allocations() {
        static NEXT_DOMAIN: AtomicU64 = AtomicU64::new(0xCA_0000);
        let runtime = Arc::new(Runtime::new().expect("GPU runtime"));
        let domain_id = NEXT_DOMAIN.fetch_add(1, Ordering::Relaxed);
        let limit = 256 * 1024 * 1024;
        let domain = runtime
            .device_memory_manager()
            .unwrap()
            .query_memory_domain(domain_id, QueryMemoryDomainOptions::new(limit))
            .expect("query domain");
        let reservation = domain.reserve_stream(runtime.stream(), domain_id, limit).expect("stream reservation");
        let resources = CuvsResources::for_runtime(Arc::clone(&runtime)).expect("cuVS resources");
        let coverage = resources.stream_pool_coverage(Some(&reservation)).expect("RAFT pool coverage");
        let points = (0..256).map(|value| (value % 37) as f32).collect::<Vec<_>>();
        let dataset = DeviceBuffer::from_slice(&resources, &points).expect("dataset allocation");
        let tensor = dataset.as_validated_matrix(&resources, 64, 4).expect("validated dataset");
        let cancellation = ExecutionCancellation::new();
        let context = ExecutionContext::builder()
            .with_shared_runtime(Arc::clone(&runtime))
            .with_memory_control_policy(MemoryControlPolicy::Unbounded)
            .with_cancellation(cancellation.clone())
            .build()
            .expect("execution context");
        let main_before = reservation.usage().unwrap().current_bytes;
        let pool_before = coverage.usage().unwrap().iter().map(|usage| usage.current_bytes).collect::<Vec<_>>();
        let mut params = IvfFlatIndexParams::new(DistanceMetric::L2Expanded, 2);
        params.kmeans_n_iters = 2;
        let mut builder = QueryPlanBuilder::new();
        let node_id = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
            "dataset",
            Arc::new(Schema::new(vec![Field::new("x", DataType::Float32, false)])),
        )));
        let error = run_knn_native(
            &context,
            node_id,
            VectorAlgorithm::IvfFlat,
            &resources,
            cuvs_sys::validate_runtime_version().unwrap(),
            &tensor,
            &tensor,
            1,
            || {
                let index = IvfFlatIndex::build(&resources, &tensor, params)?;
                assert!(reservation.usage().unwrap().current_bytes > main_before);
                cancellation.cancel();
                Ok(index)
            },
            |_, _, _| panic!("cancelled build must not search"),
            "IVF-Flat search failed",
        )
        .expect_err("after-build cancellation checkpoint");
        assert_eq!(error.code(), QueryErrorCode::Cancelled);
        assert_eq!(error.operation(), "query_engine.vector_algorithm.after_index_build");
        assert_eq!(reservation.usage().unwrap().current_bytes, main_before);
        assert_eq!(coverage.usage().unwrap().iter().map(|usage| usage.current_bytes).collect::<Vec<_>>(), pool_before,);
    }

    #[test]
    fn dispatch_rejects_an_element_outside_an_operation_sealed_set() {
        let mut builder = QueryPlanBuilder::new();
        let node_id = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
            "input",
            Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)])),
        )));
        let error = require_vector_element::<f32>(TypeId::Float64, node_id, VectorAlgorithm::Pca, "input")
            .expect_err("PCA has no Float64 kernel");
        assert_eq!(error.code(), QueryErrorCode::UnsupportedType);
        assert_eq!(error.fact_value("reason"), Some("unsupported_vector_element_type"));
        assert_eq!(error.fact_value("element"), Some("float64"));
        let unsupported_arrow =
            unsupported_vector_element(node_id, VectorAlgorithm::Pca, "input", None).with_fact("arrow_type", "Float16");
        assert_eq!(unsupported_arrow.fact_value("element"), None);
        assert_eq!(unsupported_arrow.fact_value("arrow_type"), Some("Float16"));
    }

    #[test]
    fn dispatch_requires_matching_element_types_before_native_execution() {
        let mut builder = QueryPlanBuilder::new();
        let node_id = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(
            "input",
            Arc::new(Schema::new(vec![Field::new("x", DataType::Float32, false)])),
        )));
        let error =
            require_matching_elements(TypeId::Float32, TypeId::Float64, node_id, VectorAlgorithm::PairwiseDistance)
                .expect_err("mixed pairwise inputs");
        assert_eq!(error.fact_value("reason"), Some("vector_element_type_mismatch"));
    }

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
