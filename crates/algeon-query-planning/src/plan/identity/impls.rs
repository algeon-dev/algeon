//! `StableHash` for the node, expression, and source types of a plan.

use super::{PlanHasher, StableHash, stable_hash_fields, stable_hash_via_hash};
use crate::{
    dynamic_filter::DynamicFilterRoute,
    expr::{
        AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, CaseWhen, ColumnRef, DatePart, Expr,
        GroupingSetsSpec, LimitSpec, LiteralValue, SortExpr, SortSpec, UnaryMathOp,
    },
    plan::{
        AggregateNode, DenseVectorBinding, EdgeDedupStrategy, EdgeNormalizeNode, EdgeNormalizeSpec, EdgeWeightCombiner,
        FilterNode, GraphAlgorithm, GraphAlgorithmNode, GraphAlgorithmParams, GraphAlgorithmSpec,
        GraphBetweennessCentralityParams, GraphBetweennessSeeds, GraphBfsEndpointPolicy, GraphBfsParams,
        GraphBfsPathOutput, GraphBfsPredicateInputs, GraphConstructionOptions, GraphConstructionPolicy,
        GraphCoreNumberParams, GraphDegreesAllParams, GraphEcgParams, GraphEdgeBetweennessCentralityParams,
        GraphEdgeColumnBindings, GraphEdgePredicateInput, GraphEigenvectorCentralityParams,
        GraphForceAtlas2InitialPositionsInput, GraphForceAtlas2Params, GraphForceAtlas2VertexAttributesInput,
        GraphHitsParams, GraphInDegreesAllParams, GraphKCoreDegreeType, GraphKCoreParams, GraphKatzCentralityParams,
        GraphLeidenParams, GraphLouvainParams, GraphMstParams, GraphOutDegreesAllParams, GraphPageRankParams,
        GraphPageRankPersonalizationInput, GraphPersonalizedPageRankParams, GraphPredicateInput, GraphSccParams,
        GraphSimilarityParams, GraphSpectralClusteringParams, GraphSsspParams, GraphSsspSource,
        GraphTriangleCountAllParams, GraphVertexPairInput, GraphWccParams, HashJoinBuildArtifactId, JoinKey, JoinNode,
        JoinNullEquality, JoinOutputColumn, JoinSide, JoinSpec, JoinType, LimitNode, MlAlgorithm, MlAlgorithmNode,
        MlAlgorithmParams, MlAlgorithmSpec, MlClusterSelectionMethod, MlDecompositionSolver, MlForestModelSnapshot,
        MlForestPredictParams, MlHdbscanParams, MlLogisticRegressionParams, MlRandomForestParams, MlRelationBinding,
        MlSplitCriterion, MlTsvdParams, MlUmapInit, MlUmapParams, NodeId, PlanFacts, PlanNode, ProjectionNode,
        ProjectionOutput, QueryNode, ShiftDirection, SortNode, UnionNode, VectorAlgorithm, VectorAlgorithmNode,
        VectorAlgorithmParams, VectorAlgorithmSpec, VectorBruteForceKnnParams, VectorCagraBuildAlgo, VectorCagraParams,
        VectorCagraSearchAlgo, VectorDistanceMetric, VectorIvfFlatParams, VectorIvfPqCodebookKind, VectorIvfPqParams,
        VectorKMeansInit, VectorKMeansMetric, VectorKMeansParams, VectorPairwiseDistanceParams, VectorPcaParams,
        VectorPcaSolver, VectorRelationBinding, VectorRowIdBinding, VectorTopKParams, WindowFrame, WindowFunction,
        WindowNode, WindowSpec,
    },
    source::{
        GpuResidentSource, GraphSimpleDirectedEdgeTopology, LiteralBatchSource, LocalParquetFile, LocalParquetSource,
        MissingColumn, ParquetReaderDecimalType, ParquetSourceLocation, PartitionColumn, ResolvedParquetSource,
        SourceColumnRef, SourceNode,
    },
};

stable_hash_via_hash!(NodeId);

stable_hash_fields! {
    QueryNode { id, kind, facts }
    PlanFacts { output_schema, output_ordering }
}

impl StableHash for PlanNode {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            Self::Source(source) => {
                hasher.tag("source");
                source.stable_hash(hasher);
            },
            Self::Filter(node) => {
                hasher.tag("filter");
                node.stable_hash(hasher);
            },
            Self::Projection(node) => {
                hasher.tag("projection");
                node.stable_hash(hasher);
            },
            Self::Aggregate(node) => {
                hasher.tag("aggregate");
                node.stable_hash(hasher);
            },
            Self::Window(node) => {
                hasher.tag("window");
                node.stable_hash(hasher);
            },
            Self::Sort(node) => {
                hasher.tag("sort");
                node.stable_hash(hasher);
            },
            Self::Limit(node) => {
                hasher.tag("limit");
                node.stable_hash(hasher);
            },
            Self::Join(node) => {
                hasher.tag("join");
                node.stable_hash(hasher);
            },
            Self::Union(node) => {
                hasher.tag("union");
                node.stable_hash(hasher);
            },
            Self::EdgeNormalize(node) => {
                hasher.tag("edge_normalize");
                node.stable_hash(hasher);
            },
            Self::GraphAlgorithm(node) => {
                hasher.tag("graph_algorithm");
                node.stable_hash(hasher);
            },
            Self::VectorAlgorithm(node) => {
                hasher.tag("vector_algorithm");
                node.stable_hash(hasher);
            },
            Self::MlAlgorithm(node) => {
                hasher.tag("ml_algorithm");
                node.stable_hash(hasher);
            },
        }
    }
}

stable_hash_fields! {
    FilterNode { input, predicate }
    ProjectionOutput { expr, field }
    AggregateNode { input, spec }
    WindowNode { input, spec }
    WindowSpec { partition_by, order_by, functions }
    SortNode { input, spec }
    LimitNode { input, spec }
    UnionNode { inputs }
    EdgeNormalizeNode { input, spec }
}

impl StableHash for ProjectionNode {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        // The output schema is private to `plan/nodes.rs` and derived from `outputs`.
        let ProjectionNode { input, outputs, .. } = self;
        input.stable_hash(hasher);
        outputs.stable_hash(hasher);
    }
}

impl StableHash for WindowFunction {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            Self::Aggregate { function, expr, output, frame } => {
                hasher.tag("aggregate");
                function.stable_hash(hasher);
                expr.stable_hash(hasher);
                output.stable_hash(hasher);
                frame.stable_hash(hasher);
            },
            Self::Rank { output } => {
                hasher.tag("rank");
                output.stable_hash(hasher);
            },
            Self::RowNumber { output } => {
                hasher.tag("row_number");
                output.stable_hash(hasher);
            },
            Self::LagLead { expr, offset, default, output, direction } => {
                hasher.tag("lag_lead");
                expr.stable_hash(hasher);
                offset.stable_hash(hasher);
                default.stable_hash(hasher);
                output.stable_hash(hasher);
                direction.stable_hash(hasher);
            },
        }
    }
}

impl StableHash for ShiftDirection {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            Self::Lag => hasher.tag("lag"),
            Self::Lead => hasher.tag("lead"),
        }
    }
}

impl StableHash for WindowFrame {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            Self::WholePartition => hasher.tag("whole_partition"),
            Self::RowsUnboundedPrecedingToCurrent => hasher.tag("rows_unbounded_preceding_to_current"),
        }
    }
}

stable_hash_via_hash!(DatePart, UnaryMathOp, BinaryOp, AggregateFunction, LimitSpec, ColumnRef);

stable_hash_fields! {
    CaseWhen { when, then }
    AggregateSpec { group_exprs, aggr_exprs, grouping_sets }
    GroupingSetsSpec { groups, grouping_id_alias }
    AggregateExpr { function, expr, alias, distinct, filter }
    SortSpec { expressions }
    SortExpr { expr, descending, nulls_first }
}

impl StableHash for Expr {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            Self::Column(column) => {
                hasher.tag("column");
                column.stable_hash(hasher);
            },
            Self::Literal(literal) => {
                hasher.tag("literal");
                literal.stable_hash(hasher);
            },
            Self::Not(expr) => {
                hasher.tag("not");
                expr.stable_hash(hasher);
            },
            Self::IsNull(expr) => {
                hasher.tag("is_null");
                expr.stable_hash(hasher);
            },
            Self::IsNotNull(expr) => {
                hasher.tag("is_not_null");
                expr.stable_hash(hasher);
            },
            Self::Cast { expr, target_type } => {
                hasher.tag("cast");
                expr.stable_hash(hasher);
                target_type.stable_hash(hasher);
            },
            Self::TryCast { expr, target_type } => {
                hasher.tag("try_cast");
                expr.stable_hash(hasher);
                target_type.stable_hash(hasher);
            },
            Self::Binary { left, op, right, output_type } => {
                hasher.tag("binary");
                left.stable_hash(hasher);
                op.stable_hash(hasher);
                right.stable_hash(hasher);
                output_type.stable_hash(hasher);
            },
            Self::Like { expr, pattern, escape, negated } => {
                hasher.tag("like");
                expr.stable_hash(hasher);
                pattern.stable_hash(hasher);
                escape.stable_hash(hasher);
                negated.stable_hash(hasher);
            },
            Self::Substring { expr, start, length } => {
                hasher.tag("substring");
                expr.stable_hash(hasher);
                start.stable_hash(hasher);
                length.stable_hash(hasher);
            },
            Self::Lower(expr) => {
                hasher.tag("lower");
                expr.stable_hash(hasher);
            },
            Self::Upper(expr) => {
                hasher.tag("upper");
                expr.stable_hash(hasher);
            },
            Self::Trim { expr, trim_chars } => {
                hasher.tag("trim");
                expr.stable_hash(hasher);
                trim_chars.stable_hash(hasher);
            },
            Self::LTrim { expr, trim_chars } => {
                hasher.tag("l_trim");
                expr.stable_hash(hasher);
                trim_chars.stable_hash(hasher);
            },
            Self::RTrim { expr, trim_chars } => {
                hasher.tag("r_trim");
                expr.stable_hash(hasher);
                trim_chars.stable_hash(hasher);
            },
            Self::StringContains { expr, pattern } => {
                hasher.tag("string_contains");
                expr.stable_hash(hasher);
                pattern.stable_hash(hasher);
            },
            Self::StartsWith { expr, prefix } => {
                hasher.tag("starts_with");
                expr.stable_hash(hasher);
                prefix.stable_hash(hasher);
            },
            Self::EndsWith { expr, suffix } => {
                hasher.tag("ends_with");
                expr.stable_hash(hasher);
                suffix.stable_hash(hasher);
            },
            Self::StringLength { expr, output_type } => {
                hasher.tag("string_length");
                expr.stable_hash(hasher);
                output_type.stable_hash(hasher);
            },
            Self::RegexpReplace { expr, pattern, replacement, output_type } => {
                hasher.tag("regexp_replace");
                expr.stable_hash(hasher);
                pattern.stable_hash(hasher);
                replacement.stable_hash(hasher);
                output_type.stable_hash(hasher);
            },
            Self::Concat(exprs) => {
                hasher.tag("concat");
                exprs.stable_hash(hasher);
            },
            Self::Abs(expr) => {
                hasher.tag("abs");
                expr.stable_hash(hasher);
            },
            Self::UnaryMath { op, expr, output_type } => {
                hasher.tag("unary_math");
                op.stable_hash(hasher);
                expr.stable_hash(hasher);
                output_type.stable_hash(hasher);
            },
            Self::Round { expr, decimal_places, output_type } => {
                hasher.tag("round");
                expr.stable_hash(hasher);
                decimal_places.stable_hash(hasher);
                output_type.stable_hash(hasher);
            },
            Self::InList { expr, list, negated } => {
                hasher.tag("in_list");
                expr.stable_hash(hasher);
                list.stable_hash(hasher);
                negated.stable_hash(hasher);
            },
            Self::IsDistinctFrom { left, right } => {
                hasher.tag("is_distinct_from");
                left.stable_hash(hasher);
                right.stable_hash(hasher);
            },
            Self::IsNotDistinctFrom { left, right } => {
                hasher.tag("is_not_distinct_from");
                left.stable_hash(hasher);
                right.stable_hash(hasher);
            },
            Self::Case { branches, else_expr } => {
                hasher.tag("case");
                branches.stable_hash(hasher);
                else_expr.stable_hash(hasher);
            },
            Self::Extract { field, expr } => {
                hasher.tag("extract");
                field.stable_hash(hasher);
                expr.stable_hash(hasher);
            },
            Self::DateTrunc { field, expr, output_type } => {
                hasher.tag("date_trunc");
                field.stable_hash(hasher);
                expr.stable_hash(hasher);
                output_type.stable_hash(hasher);
            },
            Self::Alias { expr, alias } => {
                hasher.tag("alias");
                expr.stable_hash(hasher);
                alias.stable_hash(hasher);
            },
        }
    }
}

impl StableHash for LiteralValue {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            Self::Boolean(value) => {
                hasher.tag("boolean");
                value.stable_hash(hasher);
            },
            Self::Int8(value) => {
                hasher.tag("int8");
                value.stable_hash(hasher);
            },
            Self::Int16(value) => {
                hasher.tag("int16");
                value.stable_hash(hasher);
            },
            Self::Int32(value) => {
                hasher.tag("int32");
                value.stable_hash(hasher);
            },
            Self::Int64(value) => {
                hasher.tag("int64");
                value.stable_hash(hasher);
            },
            Self::UInt8(value) => {
                hasher.tag("uint8");
                value.stable_hash(hasher);
            },
            Self::UInt16(value) => {
                hasher.tag("uint16");
                value.stable_hash(hasher);
            },
            Self::UInt32(value) => {
                hasher.tag("uint32");
                value.stable_hash(hasher);
            },
            Self::UInt64(value) => {
                hasher.tag("uint64");
                value.stable_hash(hasher);
            },
            Self::Float32(value) => {
                hasher.tag("float32");
                value.stable_hash(hasher);
            },
            Self::Float64(value) => {
                hasher.tag("float64");
                value.stable_hash(hasher);
            },
            Self::Utf8(value) => {
                hasher.tag("utf8");
                value.stable_hash(hasher);
            },
            Self::LargeUtf8(value) => {
                hasher.tag("large_utf8");
                value.stable_hash(hasher);
            },
            Self::Decimal128 { raw, precision, scale } => {
                hasher.tag("decimal128");
                raw.stable_hash(hasher);
                precision.stable_hash(hasher);
                scale.stable_hash(hasher);
            },
            Self::Date32(value) => {
                hasher.tag("date32");
                value.stable_hash(hasher);
            },
            Self::Date64(value) => {
                hasher.tag("date64");
                value.stable_hash(hasher);
            },
            Self::Time32 { raw, unit } => {
                hasher.tag("time32");
                raw.stable_hash(hasher);
                unit.stable_hash(hasher);
            },
            Self::Time64 { raw, unit } => {
                hasher.tag("time64");
                raw.stable_hash(hasher);
                unit.stable_hash(hasher);
            },
            Self::Timestamp { raw, unit } => {
                hasher.tag("timestamp");
                raw.stable_hash(hasher);
                unit.stable_hash(hasher);
            },
            Self::Duration { raw, unit } => {
                hasher.tag("duration");
                raw.stable_hash(hasher);
                unit.stable_hash(hasher);
            },
            Self::IntervalMonthDayNano { months, days, nanoseconds } => {
                hasher.tag("interval_month_day_nano");
                months.stable_hash(hasher);
                days.stable_hash(hasher);
                nanoseconds.stable_hash(hasher);
            },
            Self::Null(value) => {
                hasher.tag("null");
                value.stable_hash(hasher);
            },
        }
    }
}

stable_hash_via_hash!(HashJoinBuildArtifactId, JoinType, JoinNullEquality, JoinSide);

stable_hash_fields! {
    JoinNode { inputs, spec }
    JoinKey { left, right }
    JoinOutputColumn { side, column, output_name }
}

impl StableHash for JoinSpec {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        // Advisory statistics never change what executes; the build side they informed is already `build_side`.
        let Self {
            join_type,
            keys,
            residual_predicate,
            output_projection,
            null_equality,
            build_side,
            datafusion_order_preserved_side,
            build_artifact_id,
            at_most_one_match_side,
            advisory_output_statistics: _,
        } = self;
        join_type.stable_hash(hasher);
        keys.stable_hash(hasher);
        residual_predicate.stable_hash(hasher);
        output_projection.stable_hash(hasher);
        null_equality.stable_hash(hasher);
        build_side.stable_hash(hasher);
        datafusion_order_preserved_side.stable_hash(hasher);
        build_artifact_id.stable_hash(hasher);
        at_most_one_match_side.stable_hash(hasher);
    }
}

stable_hash_via_hash!(ParquetReaderDecimalType, SourceColumnRef, ParquetSourceLocation, ResolvedParquetSource);

stable_hash_fields! {
    LiteralBatchSource { schema, batches }
    GpuResidentSource { input_name, schema, row_count_upper_bound, graph_vertex_upper_bound, unique_key_columns }
    LocalParquetFile {
        location,
        path,
        resolved_source,
        stable_id,
        file_size_bytes,
        row_groups,
        row_group_bytes,
        projected_row_group_compressed_bytes,
        projected_row_group_uncompressed_bytes,
        pure_filter_row_group_compressed_bytes,
        pure_filter_row_group_uncompressed_bytes,
        row_group_rows,
        graph_vertex_upper_bound,
        graph_simple_directed_edge_topology,
        row_group_null_counts,
        reader_decimal_types,
        partition_values,
    }
    PartitionColumn { name, data_type, nullable }
    MissingColumn { name, data_type, nullable }
    GraphSimpleDirectedEdgeTopology { source_column_index, destination_column_index }
}

impl StableHash for SourceNode {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            Self::LocalParquet(source) => {
                hasher.tag("local_parquet");
                source.stable_hash(hasher);
            },
            Self::GpuResident(source) => {
                hasher.tag("gpu_resident");
                source.stable_hash(hasher);
            },
            Self::LiteralBatch(source) => {
                hasher.tag("literal_batch");
                source.stable_hash(hasher);
            },
        }
    }
}

impl StableHash for LocalParquetSource {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        // `diagnostics` is a display-only report of the other fields.
        let Self {
            files,
            schema,
            output_schema_override,
            read_columns,
            read_column_file_indices,
            file_column_count,
            projected_output_columns,
            projected_output_column_source_refs,
            pure_filter_columns,
            pure_filter_column_source_refs,
            partition_columns,
            missing_columns,
            scan_limit,
            reader_predicate,
            max_rows,
            unique_key_columns,
            count_star_probe,
            diagnostics: _,
        } = self;
        files.stable_hash(hasher);
        schema.stable_hash(hasher);
        output_schema_override.stable_hash(hasher);
        read_columns.stable_hash(hasher);
        read_column_file_indices.stable_hash(hasher);
        file_column_count.stable_hash(hasher);
        projected_output_columns.stable_hash(hasher);
        projected_output_column_source_refs.stable_hash(hasher);
        pure_filter_columns.stable_hash(hasher);
        pure_filter_column_source_refs.stable_hash(hasher);
        partition_columns.stable_hash(hasher);
        missing_columns.stable_hash(hasher);
        scan_limit.stable_hash(hasher);
        reader_predicate.stable_hash(hasher);
        max_rows.stable_hash(hasher);
        unique_key_columns.stable_hash(hasher);
        count_star_probe.stable_hash(hasher);
    }
}

stable_hash_via_hash!(DynamicFilterRoute);

stable_hash_via_hash!(
    GraphAlgorithm,
    GraphConstructionPolicy,
    EdgeDedupStrategy,
    EdgeWeightCombiner,
    GraphKCoreDegreeType,
    GraphBfsPathOutput
);

stable_hash_fields! {
    GraphAlgorithmNode { inputs, spec }
    GraphAlgorithmSpec { algorithm, edge_columns, graph_options, edge_normalize_inserted, params, output_schema }
    GraphEdgeColumnBindings { source, destination, weight, edge_id }
    GraphConstructionOptions { directed, renumber, policy }
    GraphPageRankParams { alpha, epsilon, max_iterations }
    GraphPersonalizedPageRankParams { pagerank, personalization }
    GraphPageRankPersonalizationInput { input, vertex, value }
    GraphEigenvectorCentralityParams { epsilon, max_iterations }
    GraphKatzCentralityParams { alpha, beta, epsilon, max_iterations }
    GraphHitsParams { epsilon, max_iterations, normalize }
    GraphBetweennessCentralityParams { seeds, k, normalized, include_endpoints, exact_vertex_threshold }
    GraphEdgeBetweennessCentralityParams { seeds, k, normalized, exact_vertex_threshold }
    GraphLouvainParams { max_level, threshold, resolution }
    GraphWccParams {}
    GraphSccParams {}
    GraphLeidenParams { max_level, resolution, seed }
    GraphEcgParams { min_weight, ensemble_size, max_level, threshold, resolution, seed }
    GraphSpectralClusteringParams { n_clusters, n_eigenvectors, evs_tolerance, seed }
    GraphForceAtlas2Params {
        max_iter,
        outbound_attraction_distribution,
        lin_log_mode,
        prevent_overlapping,
        overlap_scaling_ratio,
        edge_weight_influence,
        jitter_tolerance,
        barnes_hut_optimize,
        barnes_hut_theta,
        scaling_ratio,
        strong_gravity_mode,
        gravity,
        verbose,
        seed,
        vertex_attributes,
        initial_positions,
    }
    GraphForceAtlas2VertexAttributesInput { input, vertex, radius, mobility, mass }
    GraphForceAtlas2InitialPositionsInput { input, vertex, x, y }
    GraphVertexPairInput { input, first_column, second_column }
    GraphSimilarityParams { vertex_pairs }
    EdgeNormalizeSpec { edge_columns, add_reverse, dedup, weight_combiner }
    GraphDegreesAllParams {}
    GraphInDegreesAllParams {}
    GraphOutDegreesAllParams {}
    GraphTriangleCountAllParams {}
    GraphCoreNumberParams { degree_type }
    GraphKCoreParams { k, degree_type }
    GraphMstParams {}
    GraphBfsParams {
        source_vertex,
        source_vertices,
        depth_limit,
        predicates,
        endpoint_policy,
        path_output,
        return_target_info,
    }
    GraphBfsPredicateInputs { include_vertices, exclude_vertices, target_vertices, include_edge_ids, include_edges }
    GraphPredicateInput { input, column }
    GraphEdgePredicateInput { input, source, destination, edge_id }
    GraphSsspParams { source_vertex, cutoff }
}

impl StableHash for GraphAlgorithmParams {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            Self::PageRank(params) => {
                hasher.tag("page_rank");
                params.stable_hash(hasher);
            },
            Self::PersonalizedPageRank(params) => {
                hasher.tag("personalized_page_rank");
                params.stable_hash(hasher);
            },
            Self::Bfs(params) => {
                hasher.tag("bfs");
                params.stable_hash(hasher);
            },
            Self::Louvain(params) => {
                hasher.tag("louvain");
                params.stable_hash(hasher);
            },
            Self::EigenvectorCentrality(params) => {
                hasher.tag("eigenvector_centrality");
                params.stable_hash(hasher);
            },
            Self::KatzCentrality(params) => {
                hasher.tag("katz_centrality");
                params.stable_hash(hasher);
            },
            Self::Hits(params) => {
                hasher.tag("hits");
                params.stable_hash(hasher);
            },
            Self::BetweennessCentrality(params) => {
                hasher.tag("betweenness_centrality");
                params.stable_hash(hasher);
            },
            Self::EdgeBetweennessCentrality(params) => {
                hasher.tag("edge_betweenness_centrality");
                params.stable_hash(hasher);
            },
            Self::Sssp(params) => {
                hasher.tag("sssp");
                params.stable_hash(hasher);
            },
            Self::DegreesAll(params) => {
                hasher.tag("degrees_all");
                params.stable_hash(hasher);
            },
            Self::InDegreesAll(params) => {
                hasher.tag("in_degrees_all");
                params.stable_hash(hasher);
            },
            Self::OutDegreesAll(params) => {
                hasher.tag("out_degrees_all");
                params.stable_hash(hasher);
            },
            Self::WeaklyConnectedComponents(params) => {
                hasher.tag("weakly_connected_components");
                params.stable_hash(hasher);
            },
            Self::StronglyConnectedComponents(params) => {
                hasher.tag("strongly_connected_components");
                params.stable_hash(hasher);
            },
            Self::TriangleCountAll(params) => {
                hasher.tag("triangle_count_all");
                params.stable_hash(hasher);
            },
            Self::CoreNumber(params) => {
                hasher.tag("core_number");
                params.stable_hash(hasher);
            },
            Self::KCore(params) => {
                hasher.tag("k_core");
                params.stable_hash(hasher);
            },
            Self::MinimumSpanningTree(params) => {
                hasher.tag("minimum_spanning_tree");
                params.stable_hash(hasher);
            },
            Self::Leiden(params) => {
                hasher.tag("leiden");
                params.stable_hash(hasher);
            },
            Self::Ecg(params) => {
                hasher.tag("ecg");
                params.stable_hash(hasher);
            },
            Self::SpectralModularityMaximization(params) => {
                hasher.tag("spectral_modularity_maximization");
                params.stable_hash(hasher);
            },
            Self::ForceAtlas2(params) => {
                hasher.tag("force_atlas2");
                params.stable_hash(hasher);
            },
            Self::Similarity(params) => {
                hasher.tag("similarity");
                params.stable_hash(hasher);
            },
        }
    }
}

impl StableHash for GraphBetweennessSeeds {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            Self::I64(values) => {
                hasher.tag("i64");
                values.stable_hash(hasher);
            },
            Self::Input { input, row_count_upper_bound } => {
                hasher.tag("input");
                input.stable_hash(hasher);
                row_count_upper_bound.stable_hash(hasher);
            },
        }
    }
}

impl StableHash for GraphBfsEndpointPolicy {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            Self::Strict => hasher.tag("strict"),
            Self::EmptyOnMissingPathEndpoint => hasher.tag("empty_on_missing_path_endpoint"),
        }
    }
}

impl StableHash for GraphSsspSource {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            Self::I64(vertex) => {
                hasher.tag("i64");
                vertex.stable_hash(hasher);
            },
            Self::Input(input) => {
                hasher.tag("input");
                input.stable_hash(hasher);
            },
        }
    }
}

stable_hash_via_hash!(
    VectorAlgorithm,
    VectorRowIdBinding,
    VectorDistanceMetric,
    DenseVectorBinding,
    VectorCagraBuildAlgo,
    VectorCagraSearchAlgo,
    VectorIvfPqCodebookKind,
    VectorKMeansInit,
    VectorKMeansMetric,
    VectorPcaSolver
);

stable_hash_fields! {
    VectorAlgorithmNode { inputs, spec }
    VectorAlgorithmSpec { algorithm, params, output_schema }
    VectorRelationBinding { input, row_id, vector }
    VectorBruteForceKnnParams { dataset, queries, k, metric }
    VectorCagraParams {
        dataset,
        queries,
        k,
        metric,
        graph_degree,
        intermediate_graph_degree,
        build_algo,
        itopk_size,
        search_width,
        max_iterations,
        search_algo,
    }
    VectorIvfFlatParams { dataset, queries, k, metric, n_lists, n_probes, kmeans_n_iters, kmeans_trainset_fraction }
    VectorIvfPqParams {
        dataset,
        queries,
        k,
        metric,
        n_lists,
        n_probes,
        kmeans_n_iters,
        kmeans_trainset_fraction,
        pq_dim,
        pq_bits,
        codebook_kind,
    }
    VectorKMeansParams { input, n_clusters, max_iterations, tolerance, n_init, init, initial_centroids, metric }
    VectorPairwiseDistanceParams { x, y, metric }
    VectorPcaParams { input, n_components, whiten, solver, tolerance, n_iterations, flip_signs_based_on_u }
    VectorTopKParams { input, k, select_min }
}

impl StableHash for VectorAlgorithmParams {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            Self::BruteForceKnn(params) => {
                hasher.tag("brute_force_knn");
                params.stable_hash(hasher);
            },
            Self::IvfFlat(params) => {
                hasher.tag("ivf_flat");
                params.stable_hash(hasher);
            },
            Self::IvfPq(params) => {
                hasher.tag("ivf_pq");
                params.stable_hash(hasher);
            },
            Self::Cagra(params) => {
                hasher.tag("cagra");
                params.stable_hash(hasher);
            },
            Self::PairwiseDistance(params) => {
                hasher.tag("pairwise_distance");
                params.stable_hash(hasher);
            },
            Self::TopK(params) => {
                hasher.tag("top_k");
                params.stable_hash(hasher);
            },
            Self::KMeans(params) => {
                hasher.tag("k_means");
                params.stable_hash(hasher);
            },
            Self::Pca(params) => {
                hasher.tag("pca");
                params.stable_hash(hasher);
            },
        }
    }
}

stable_hash_via_hash!(
    MlAlgorithm,
    MlClusterSelectionMethod,
    MlUmapInit,
    MlDecompositionSolver,
    MlSplitCriterion,
    MlForestModelSnapshot
);

stable_hash_fields! {
    MlAlgorithmNode { inputs, spec }
    MlAlgorithmSpec { algorithm, params, output_schema }
    MlRelationBinding { input, row_id, features, label }
    MlHdbscanParams {
        input,
        min_samples,
        min_cluster_size,
        max_cluster_size,
        cluster_selection_epsilon,
        alpha,
        allow_single_cluster,
        cluster_selection_method,
    }
    MlUmapParams {
        input,
        n_components,
        n_neighbors,
        n_epochs,
        learning_rate,
        min_dist,
        spread,
        set_op_mix_ratio,
        local_connectivity,
        repulsion_strength,
        negative_sample_rate,
        init,
        random_state,
    }
    MlTsvdParams { input, n_components, solver, tolerance, n_iterations }
    MlForestPredictParams { input, model }
    MlLogisticRegressionParams {
        training,
        predict,
        penalty_l1,
        penalty_l2,
        max_iter,
        linesearch_max_iter,
        lbfgs_memory,
        fit_intercept,
        penalty_normalized,
        grad_tol,
        change_tol,
    }
    MlRandomForestParams {
        training,
        predict,
        n_trees,
        max_depth,
        max_leaves,
        max_features,
        max_samples,
        bootstrap,
        max_n_bins,
        min_samples_leaf,
        min_samples_split,
        min_impurity_decrease,
        max_batch_size,
        seed,
        criterion,
    }
}

impl StableHash for MlAlgorithmParams {
    fn stable_hash(&self, hasher: &mut PlanHasher) {
        match self {
            Self::Tsvd(params) => {
                hasher.tag("tsvd");
                params.stable_hash(hasher);
            },
            Self::Hdbscan(params) => {
                hasher.tag("hdbscan");
                params.stable_hash(hasher);
            },
            Self::Umap(params) => {
                hasher.tag("umap");
                params.stable_hash(hasher);
            },
            Self::LogisticRegression(params) => {
                hasher.tag("logistic_regression");
                params.stable_hash(hasher);
            },
            Self::RandomForestClassifier(params) => {
                hasher.tag("random_forest_classifier");
                params.stable_hash(hasher);
            },
            Self::RandomForestRegressor(params) => {
                hasher.tag("random_forest_regressor");
                params.stable_hash(hasher);
            },
            Self::ForestPredict(params) => {
                hasher.tag("forest_predict");
                params.stable_hash(hasher);
            },
        }
    }
}
