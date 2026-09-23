use cudf::{Series, TypeId};

#[cfg(test)]
use crate::plan::VectorBruteForceKnnParams;
use crate::{
    exec::*,
    plan::{
        VectorDistanceMetric, VectorPairwiseDistanceParams, VectorRelationBinding, VectorRowIdBinding, VectorTopKParams,
    },
};

const ASSEMBLY_OPERATION: &str = "query_engine.vector_algorithm.assemble";

pub(in crate::exec) struct KnnAssemblyParams<'a> {
    pub dataset: &'a VectorRelationBinding,
    pub queries: &'a VectorRelationBinding,
    pub k: usize,
    pub metric: VectorDistanceMetric,
    pub missing_neighbor: Option<i64>,
}

#[cfg(test)]
pub(in crate::exec) fn assemble_brute_force_knn_output(
    context: &ExecutionContext,
    params: &VectorBruteForceKnnParams,
    dataset: &GpuDataFrame,
    queries: &GpuDataFrame,
    neighbor_ordinals_i64: Series,
    distances_f32: Series,
) -> Result<GpuDataFrame> {
    assemble_knn_output(
        context,
        KnnAssemblyParams {
            dataset: &params.dataset,
            queries: &params.queries,
            k: params.k,
            metric: params.metric,
            missing_neighbor: None,
        },
        dataset,
        queries,
        neighbor_ordinals_i64,
        distances_f32,
    )
}

pub(in crate::exec) fn assemble_knn_output(
    context: &ExecutionContext,
    params: KnnAssemblyParams<'_>,
    dataset: &GpuDataFrame,
    queries: &GpuDataFrame,
    neighbor_ordinals_i64: Series,
    distances_f32: Series,
) -> Result<GpuDataFrame> {
    let output_rows = queries.num_rows().checked_mul(params.k).ok_or_else(|| {
        QueryError::new(
            QueryErrorCode::SizeOverflow,
            ASSEMBLY_OPERATION,
            "brute-force kNN output row count overflowed usize",
        )
        .with_fact("query_rows", queries.num_rows())
        .with_fact("k", params.k)
        .with_fact("reason", "output_row_count_overflow")
    })?;
    require_libcudf_row_count(output_rows, "brute_force_knn")?;
    require_series_contract(&neighbor_ordinals_i64, TypeId::Int64, output_rows, "neighbor_ordinals")?;
    require_series_contract(&distances_f32, TypeId::Float32, output_rows, "distances")?;

    let runtime = context.runtime()?;
    let execution =
        ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    let k_u64 = u64::try_from(params.k).map_err(|_| {
        QueryError::new(
            QueryErrorCode::SizeOverflow,
            ASSEMBLY_OPERATION,
            "brute-force kNN k exceeds UInt64 output-assembly range",
        )
        .with_fact("k", params.k)
        .with_fact("reason", "k_exceeds_uint64")
    })?;
    let output_ordinal = runtime.sequence(0_u64, 1_u64, output_rows).map_err(|error| {
        cudf_query_error(ASSEMBLY_OPERATION, "failed to construct kNN output ordinals on device", error)
    })?;
    let query_ordinal = output_ordinal.div_scalar_in(k_u64, &execution).map_err(|error| {
        cudf_query_error(ASSEMBLY_OPERATION, "failed to derive kNN query ordinals on device", error)
    })?;
    let query_gather_map = query_ordinal.astype_in(TypeId::Int32, &execution).map_err(|error| {
        cudf_query_error(ASSEMBLY_OPERATION, "failed to construct kNN query gather map on device", error)
    })?;
    let query_id = relation_id_series(params.queries, queries, "queries")?
        .gather_in(&query_gather_map, &execution)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to gather kNN query ids on device", error))?;

    let is_sentinel = params
        .missing_neighbor
        .map(|missing| {
            neighbor_ordinals_i64.eq_scalar_in(missing, &execution).map_err(|error| {
                cudf_query_error(ASSEMBLY_OPERATION, "failed to identify missing kNN neighbors on device", error)
            })
        })
        .transpose()?;
    let mut unsorted = runtime
        .empty_dataframe()
        .assign([
            ("query_ordinal", query_ordinal),
            ("query_id", query_id),
            ("neighbor_ordinal", neighbor_ordinals_i64),
            ("distance", distances_f32),
        ])
        .map_err(|error| {
            cudf_query_error(ASSEMBLY_OPERATION, "failed to assemble kNN relational output on device", error)
        })?;
    let descending_distance = matches!(params.metric, VectorDistanceMetric::InnerProduct);
    let mut keys = vec!["query_ordinal"];
    let mut descending = vec![false];
    if let Some(is_sentinel) = is_sentinel {
        unsorted = unsorted
            .with_column("is_sentinel", is_sentinel)
            .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to attach missing-neighbor flags", error))?;
        keys.push("is_sentinel");
        descending.push(false);
    }
    keys.extend(["distance", "neighbor_ordinal"]);
    descending.extend([descending_distance, false]);
    let sorted = unsorted
        .with_execution(&execution)
        .sort_values(
            keys,
            CudfSortOptions::new()
                .with_descending(descending)
                .with_nulls_first(vec![false; if params.missing_neighbor.is_some() { 4 } else { 3 }])
                .with_stable(true),
        )
        .map_err(|error| {
            cudf_query_error(ASSEMBLY_OPERATION, "failed to deterministically order kNN neighbors on device", error)
        })?;
    let rank = output_ordinal
        .rem_scalar_in(k_u64, &execution)
        .and_then(|rank| rank.add_scalar_in(1_u64, &execution))
        .and_then(|rank| rank.astype_in(TypeId::UInt32, &execution))
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to derive kNN ranks on device", error))?;
    let ranked = sorted
        .with_column("rank", rank)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to rank kNN neighbors on device", error))?;
    let filtered = if params.missing_neighbor.is_some() {
        let valid = ranked
            .series("is_sentinel")
            .and_then(|sentinel| sentinel.eq_scalar_in(false, &execution))
            .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to mask missing kNN neighbors", error))?;
        ranked.filter(&valid).map_err(|error| {
            cudf_query_error(ASSEMBLY_OPERATION, "failed to omit missing kNN neighbors on device", error)
        })?
    } else {
        ranked
    };
    let neighbor_ordinals_i64 = filtered
        .series("neighbor_ordinal")
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to read valid kNN neighbor ordinals", error))?;
    let neighbor_gather_map = neighbor_ordinals_i64.astype_in(TypeId::Int32, &execution).map_err(|error| {
        cudf_query_error(ASSEMBLY_OPERATION, "failed to construct kNN neighbor gather map on device", error)
    })?;
    let neighbor_id = relation_id_series(params.dataset, dataset, "dataset")?
        .gather_in(&neighbor_gather_map, &execution)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to gather kNN neighbor ids on device", error))?;
    let neighbor_ordinal = neighbor_ordinals_i64.astype_in(TypeId::UInt64, &execution).map_err(|error| {
        cudf_query_error(ASSEMBLY_OPERATION, "failed to cast kNN neighbor ordinals on device", error)
    })?;
    let output = filtered
        .with_column("neighbor_ordinal", neighbor_ordinal)
        .and_then(|output| output.with_column("neighbor_id", neighbor_id))
        .and_then(|output| {
            output.select(["query_ordinal", "query_id", "neighbor_ordinal", "neighbor_id", "rank", "distance"])
        })
        .map_err(|error| {
            cudf_query_error(ASSEMBLY_OPERATION, "failed to finalize kNN relational output schema", error)
        })?;
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec) fn assemble_pairwise_distance_output(
    context: &ExecutionContext,
    params: &VectorPairwiseDistanceParams,
    x: &GpuDataFrame,
    y: &GpuDataFrame,
    distances: Series,
) -> Result<GpuDataFrame> {
    let rows = x.num_rows().checked_mul(y.num_rows()).ok_or_else(|| {
        QueryError::new(QueryErrorCode::SizeOverflow, ASSEMBLY_OPERATION, "pairwise output rows overflowed")
    })?;
    require_libcudf_row_count(rows, "pairwise_distance")?;
    require_series_contract(&distances, TypeId::Float32, rows, "distances")?;
    let runtime = context.runtime()?;
    let execution =
        ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    let sequence = runtime
        .sequence(0_u64, 1_u64, rows)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to create pairwise output ordinals", error))?;
    // An empty axis has no valid divisor; the output still carries the stable five-column schema.
    let divisor = y.num_rows().max(1) as u64;
    let x_ordinal = sequence
        .div_scalar_in(divisor, &execution)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to derive x ordinals", error))?;
    let y_ordinal = sequence
        .rem_scalar_in(divisor, &execution)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to derive y ordinals", error))?;
    let x_map = x_ordinal
        .astype_in(TypeId::Int32, &execution)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to cast x ordinals", error))?;
    let y_map = y_ordinal
        .astype_in(TypeId::Int32, &execution)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to cast y ordinals", error))?;
    let x_id = relation_id_series(&params.x, x, "x")?
        .gather_in(&x_map, &execution)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to gather x ids", error))?;
    let y_id = relation_id_series(&params.y, y, "y")?
        .gather_in(&y_map, &execution)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to gather y ids", error))?;
    let output = runtime
        .empty_dataframe()
        .assign([
            ("x_ordinal", x_ordinal),
            ("x_id", x_id),
            ("y_ordinal", y_ordinal),
            ("y_id", y_id),
            ("distance", distances),
        ])
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to assemble pairwise output", error))?;
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec) fn assemble_top_k_output(
    context: &ExecutionContext,
    params: &VectorTopKParams,
    input: &GpuDataFrame,
    positions: Series,
    values: Series,
) -> Result<GpuDataFrame> {
    let rows = input.num_rows().checked_mul(params.k).ok_or_else(|| {
        QueryError::new(QueryErrorCode::SizeOverflow, ASSEMBLY_OPERATION, "Top-K output rows overflowed")
    })?;
    require_libcudf_row_count(rows, "top_k")?;
    require_series_contract(&positions, TypeId::Int64, rows, "positions")?;
    require_series_contract(&values, TypeId::Float32, rows, "values")?;
    let runtime = context.runtime()?;
    let execution =
        ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    let sequence = runtime
        .sequence(0_u64, 1_u64, rows)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to create Top-K ordinals", error))?;
    let row_ordinal = sequence
        .div_scalar_in(params.k as u64, &execution)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to derive Top-K row ordinals", error))?;
    let gather_map = row_ordinal
        .astype_in(TypeId::Int32, &execution)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to cast Top-K ordinals", error))?;
    let id = relation_id_series(&params.input, input, "input")?
        .gather_in(&gather_map, &execution)
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to gather Top-K ids", error))?;
    let sorted = runtime
        .empty_dataframe()
        .assign([("row_ordinal", row_ordinal), ("id", id), ("position", positions), ("value", values)])
        .and_then(|frame| {
            frame.with_execution(&execution).sort_values(
                ["row_ordinal", "value", "position"],
                CudfSortOptions::new()
                    .with_descending(vec![false, !params.select_min, false])
                    .with_nulls_first(vec![false, false, false])
                    .with_stable(true),
            )
        })
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to order Top-K output", error))?;
    let rank = sequence
        .rem_scalar_in(params.k as u64, &execution)
        .and_then(|rank| rank.add_scalar_in(1_u64, &execution))
        .and_then(|rank| rank.astype_in(TypeId::UInt32, &execution))
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to derive Top-K ranks", error))?;
    let output = sorted
        .with_column("rank", rank)
        .and_then(|frame| frame.select(["row_ordinal", "id", "rank", "position", "value"]))
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to finalize Top-K output", error))?;
    Ok(GpuDataFrame::new(output))
}

pub(in crate::exec) fn assemble_kmeans_output(
    context: &ExecutionContext,
    input: &GpuDataFrame,
    binding: &VectorRelationBinding,
    labels_i32: Series,
) -> Result<GpuDataFrame> {
    require_libcudf_row_count(input.num_rows(), "kmeans")?;
    require_series_contract(&labels_i32, TypeId::Int32, input.num_rows(), "cluster_labels")?;

    let runtime = context.runtime()?;
    let row_ordinal = runtime.sequence(0_u64, 1_u64, input.num_rows()).map_err(|error| {
        cudf_query_error(ASSEMBLY_OPERATION, "failed to construct K-means row ordinals on device", error)
    })?;
    let id = relation_id_series(binding, input, "input")?.copy_into_runtime(&runtime).map_err(|error| {
        cudf_query_error(ASSEMBLY_OPERATION, "failed to place K-means logical ids in the query runtime", error)
    })?;
    let output = runtime
        .empty_dataframe()
        .assign([("row_ordinal", row_ordinal), ("id", id), ("cluster_id", labels_i32)])
        .map_err(|error| {
            cudf_query_error(ASSEMBLY_OPERATION, "failed to assemble K-means relational output on device", error)
        })?;
    GpuDataFrame::try_new_with_source_residency(output, input)
}

pub(in crate::exec) fn assemble_pca_output(
    context: &ExecutionContext,
    input: &GpuDataFrame,
    binding: &VectorRelationBinding,
    components_f32: &[Series],
) -> Result<GpuDataFrame> {
    require_libcudf_row_count(input.num_rows(), "pca")?;
    for (component_index, component) in components_f32.iter().enumerate() {
        require_series_contract(
            component,
            TypeId::Float32,
            input.num_rows(),
            &format!("pca_component_{component_index}"),
        )?;
    }

    let runtime = context.runtime()?;
    let row_ordinal = runtime.sequence(0_u64, 1_u64, input.num_rows()).map_err(|error| {
        cudf_query_error(ASSEMBLY_OPERATION, "failed to construct PCA row ordinals on device", error)
    })?;
    let id = relation_id_series(binding, input, "input")?.copy_into_runtime(&runtime).map_err(|error| {
        cudf_query_error(ASSEMBLY_OPERATION, "failed to place PCA logical ids in the query runtime", error)
    })?;
    let mut output = runtime
        .empty_dataframe()
        .assign([("row_ordinal", row_ordinal), ("id", id)])
        .map_err(|error| cudf_query_error(ASSEMBLY_OPERATION, "failed to initialize PCA relational output", error))?;
    for (component_index, component) in components_f32.iter().enumerate() {
        output = output.with_column(&format!("pc_{component_index}"), component.clone()).map_err(|error| {
            cudf_query_error(ASSEMBLY_OPERATION, "failed to append a PCA component on device", error)
                .with_fact("component_index", component_index)
        })?;
    }
    GpuDataFrame::try_new_with_source_residency(output, input)
}

fn relation_id_series(
    binding: &VectorRelationBinding,
    input: &GpuDataFrame,
    relation_role: &'static str,
) -> Result<Series> {
    let VectorRowIdBinding::Column(column) = &binding.row_id else {
        return Err(QueryError::new(
            QueryErrorCode::Internal,
            ASSEMBLY_OPERATION,
            "validated vector result relation is missing its logical row-id column",
        )
        .with_fact("relation_role", relation_role)
        .with_fact("reason", "missing_logical_row_id_column"));
    };
    input.frame().series(column).map_err(|error| {
        cudf_query_error(ASSEMBLY_OPERATION, "failed to access vector relation logical ids", error)
            .with_fact("relation_role", relation_role)
            .with_fact("column", column)
    })
}

fn require_series_contract(series: &Series, expected_type: TypeId, expected_rows: usize, output: &str) -> Result<()> {
    let actual_type = series.dtype().map_err(|error| {
        cudf_query_error(ASSEMBLY_OPERATION, "failed to inspect vector native output type", error)
            .with_fact("output", output)
    })?;
    if actual_type.id() != expected_type || series.len() != expected_rows || series.null_count() != 0 {
        return Err(QueryError::new(
            QueryErrorCode::CudfInvariantViolation,
            ASSEMBLY_OPERATION,
            "vector native output does not match its device assembly contract",
        )
        .with_fact("output", output)
        .with_fact("expected_type", expected_type.name())
        .with_fact("actual_type", actual_type.id().name())
        .with_fact("expected_rows", expected_rows)
        .with_fact("actual_rows", series.len())
        .with_fact("null_count", series.null_count())
        .with_fact("reason", "invalid_native_output_contract"));
    }
    Ok(())
}

fn require_libcudf_row_count(rows: usize, algorithm: &'static str) -> Result<()> {
    if rows <= i32::MAX as usize {
        return Ok(());
    }
    Err(QueryError::new(
        QueryErrorCode::CudfInvalidInput,
        ASSEMBLY_OPERATION,
        "vector result row count exceeds libcudf size_type range",
    )
    .with_fact("algorithm", algorithm)
    .with_fact("rows", rows)
    .with_fact("maximum_rows", i32::MAX)
    .with_fact("reason", "row_count_exceeds_libcudf_size_type"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cuvs")]
    use crate::plan::{NodeId, VectorDistanceMetric};

    #[test]
    fn libcudf_row_count_contract_is_structured_without_gpu_work() {
        require_libcudf_row_count(i32::MAX as usize, "pca").expect("libcudf maximum row count is admitted");
        let error = require_libcudf_row_count(i32::MAX as usize + 1, "pca")
            .expect_err("row count above libcudf size_type must reject");
        assert_eq!(error.code(), QueryErrorCode::CudfInvalidInput);
        assert_eq!(error.fact_value("reason"), Some("row_count_exceeds_libcudf_size_type"));
    }

    #[cfg(feature = "cuvs")]
    #[test]
    fn knn_assembly_sorts_ties_and_preserves_duplicate_logical_ids_on_device() {
        let context = ExecutionContext::builder().build().expect("execution context");
        let runtime = context.runtime().expect("runtime");
        let dataset = GpuDataFrame::new(
            runtime
                .empty_dataframe()
                .assign([("id", runtime.series(&[10_i64, 10, 20]).expect("dataset ids"))])
                .expect("dataset frame"),
        );
        let queries = GpuDataFrame::new(
            runtime
                .empty_dataframe()
                .assign([("query_id", runtime.series(&[7_i64, 7]).expect("query ids"))])
                .expect("query frame"),
        );
        let params = VectorBruteForceKnnParams::new(
            VectorRelationBinding::new(NodeId::from_index_for_internal_use(0), VectorRowIdBinding::column("id"), ["x"]),
            VectorRelationBinding::new(
                NodeId::from_index_for_internal_use(1),
                VectorRowIdBinding::column("query_id"),
                ["x"],
            ),
            2,
            VectorDistanceMetric::L2Expanded,
        );
        let neighbors = runtime.series(&[1_i64, 0, 2, 0]).expect("neighbor ordinals");
        let distances = runtime.series(&[1.0_f32, 1.0, 0.6, 0.4]).expect("distances");

        let output = assemble_brute_force_knn_output(&context, &params, &dataset, &queries, neighbors, distances)
            .expect("kNN assembly");

        assert_eq!(output.frame().series("query_ordinal").unwrap().to_vec_strict::<u64>().unwrap(), vec![0, 0, 1, 1]);
        assert_eq!(output.frame().series("query_id").unwrap().to_vec_strict::<i64>().unwrap(), vec![7, 7, 7, 7]);
        assert_eq!(
            output.frame().series("neighbor_ordinal").unwrap().to_vec_strict::<u64>().unwrap(),
            vec![0, 1, 0, 2]
        );
        assert_eq!(output.frame().series("neighbor_id").unwrap().to_vec_strict::<i64>().unwrap(), vec![10, 10, 10, 20]);
        assert_eq!(output.frame().series("rank").unwrap().to_vec_strict::<u32>().unwrap(), vec![1, 2, 1, 2]);
        assert_eq!(
            output.frame().series("distance").unwrap().to_vec_strict::<f32>().unwrap(),
            vec![1.0, 1.0, 0.4, 0.6]
        );

        let mut inner_product = params;
        inner_product.metric = VectorDistanceMetric::InnerProduct;
        let output = assemble_brute_force_knn_output(
            &context,
            &inner_product,
            &dataset,
            &queries,
            runtime.series(&[0_i64, 1, 0, 2]).expect("inner-product neighbor ordinals"),
            runtime.series(&[0.2_f32, 0.9, 0.4, 0.6]).expect("inner-product scores"),
        )
        .expect("inner-product kNN assembly");
        assert_eq!(
            output.frame().series("neighbor_ordinal").unwrap().to_vec_strict::<u64>().unwrap(),
            vec![1, 0, 2, 0]
        );
        assert_eq!(
            output.frame().series("distance").unwrap().to_vec_strict::<f32>().unwrap(),
            vec![0.9, 0.2, 0.6, 0.4]
        );
    }

    #[cfg(feature = "cuvs")]
    #[test]
    fn knn_assembly_omits_native_sentinels_and_ranks_valid_neighbors() {
        let context = ExecutionContext::builder().build().expect("execution context");
        let runtime = context.runtime().expect("runtime");
        let dataset = GpuDataFrame::new(
            runtime.empty_dataframe().assign([("id", runtime.series(&[10_i64, 20]).unwrap())]).unwrap(),
        );
        let queries = GpuDataFrame::new(
            runtime.empty_dataframe().assign([("id", runtime.series(&[7_i64, 8]).unwrap())]).unwrap(),
        );
        let dataset_binding =
            VectorRelationBinding::new(NodeId::from_index_for_internal_use(0), VectorRowIdBinding::column("id"), ["x"]);
        let queries_binding =
            VectorRelationBinding::new(NodeId::from_index_for_internal_use(1), VectorRowIdBinding::column("id"), ["x"]);
        let output = assemble_knn_output(
            &context,
            KnnAssemblyParams {
                dataset: &dataset_binding,
                queries: &queries_binding,
                k: 2,
                metric: VectorDistanceMetric::L2Expanded,
                missing_neighbor: Some(i64::MAX),
            },
            &dataset,
            &queries,
            runtime.series(&[i64::MAX, 1, 0, i64::MAX]).unwrap(),
            runtime.series(&[f32::MAX, 2.0, 1.0, f32::MAX]).unwrap(),
        )
        .expect("sentinel-aware kNN assembly");
        assert_eq!(output.frame().series("query_ordinal").unwrap().to_vec_strict::<u64>().unwrap(), vec![0, 1]);
        assert_eq!(output.frame().series("neighbor_id").unwrap().to_vec_strict::<i64>().unwrap(), vec![20, 10]);
        assert_eq!(output.frame().series("rank").unwrap().to_vec_strict::<u32>().unwrap(), vec![1, 1]);
    }

    #[cfg(feature = "cuvs")]
    #[test]
    fn knn_assembly_preserves_the_empty_query_contract() {
        let context = ExecutionContext::builder().build().expect("execution context");
        let runtime = context.runtime().expect("runtime");
        let dataset = GpuDataFrame::new(
            runtime
                .empty_dataframe()
                .assign([("id", runtime.series(&[10_i64]).expect("dataset ids"))])
                .expect("dataset frame"),
        );
        let queries = GpuDataFrame::new(
            runtime
                .empty_dataframe()
                .assign([("query_id", runtime.series(&[] as &[i64]).expect("empty query ids"))])
                .expect("query frame"),
        );
        let params = VectorBruteForceKnnParams::new(
            VectorRelationBinding::new(NodeId::from_index_for_internal_use(0), VectorRowIdBinding::column("id"), ["x"]),
            VectorRelationBinding::new(
                NodeId::from_index_for_internal_use(1),
                VectorRowIdBinding::column("query_id"),
                ["x"],
            ),
            1,
            VectorDistanceMetric::L2Expanded,
        );

        let output = assemble_brute_force_knn_output(
            &context,
            &params,
            &dataset,
            &queries,
            runtime.series(&[] as &[i64]).expect("empty neighbors"),
            runtime.series(&[] as &[f32]).expect("empty distances"),
        )
        .expect("empty kNN assembly");

        assert_eq!(output.num_rows(), 0);
        assert_eq!(output.num_columns(), 6);
    }

    #[cfg(feature = "cuvs")]
    #[test]
    fn kmeans_and_pca_assembly_emit_stable_ordinals_and_input_ids() {
        let context = ExecutionContext::builder().build().expect("execution context");
        let runtime = context.runtime().expect("runtime");
        let input = GpuDataFrame::new(
            runtime
                .empty_dataframe()
                .assign([("id", runtime.series(&[4_i64, 4, 9]).expect("input ids"))])
                .expect("input frame"),
        );
        let binding =
            VectorRelationBinding::new(NodeId::from_index_for_internal_use(0), VectorRowIdBinding::column("id"), ["x"]);

        let kmeans =
            assemble_kmeans_output(&context, &input, &binding, runtime.series(&[1_i32, 0, 1]).expect("labels"))
                .expect("K-means assembly");
        assert_eq!(kmeans.frame().series("row_ordinal").unwrap().to_vec_strict::<u64>().unwrap(), vec![0, 1, 2]);
        assert_eq!(kmeans.frame().series("id").unwrap().to_vec_strict::<i64>().unwrap(), vec![4, 4, 9]);
        assert_eq!(kmeans.frame().series("cluster_id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 0, 1]);

        let components =
            [runtime.series(&[0.5_f32, 1.5, 2.5]).expect("pc0"), runtime.series(&[-1.0_f32, 0.0, 1.0]).expect("pc1")];
        let pca = assemble_pca_output(&context, &input, &binding, &components).expect("PCA assembly");
        assert_eq!(pca.frame().series("row_ordinal").unwrap().to_vec_strict::<u64>().unwrap(), vec![0, 1, 2]);
        assert_eq!(pca.frame().series("id").unwrap().to_vec_strict::<i64>().unwrap(), vec![4, 4, 9]);
        assert_eq!(pca.frame().series("pc_0").unwrap().to_vec_strict::<f32>().unwrap(), vec![0.5, 1.5, 2.5]);
        assert_eq!(pca.frame().series("pc_1").unwrap().to_vec_strict::<f32>().unwrap(), vec![-1.0, 0.0, 1.0]);
    }
}
