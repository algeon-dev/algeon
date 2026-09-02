//! `Display` formatting for [`QueryPlan`] and its nodes.
//!
//! The format is human-oriented diagnostic text used by triage tools, not a
//! stable parser surface. Bump the relevant report TSV schema instead of
//! parsing display output.

use std::fmt;

use super::{HashJoinBuildArtifactId, JoinOutputColumn, PlanNode, QueryNode, QueryPlan, ResultSink, SortExpr};
use crate::{dynamic_filter::DynamicFilterRoute, expr::Expr, source::SourceNode};

impl fmt::Display for QueryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "QueryPlan root={} sink={} dynamic_filter_routes={}",
            self.root,
            self.sink.kind_name(),
            self.dynamic_filter_routes.len()
        )?;
        for node in &self.nodes {
            node.fmt_with_indent(f, 1)?;
        }
        for route in &self.dynamic_filter_routes {
            format_dynamic_filter_route(route, f, 1)?;
        }
        match &self.sink {
            ResultSink::GpuResident { input } => {
                writeln!(f, "  Sink id=sink input={input} mode=gpu_resident")
            },
            ResultSink::HostArrow { input, schema } => writeln!(
                f,
                "  Sink id=sink input={input} mode=host_arrow materialization=device_to_host schema_fields={}",
                schema.fields().len()
            ),
        }
    }
}

fn format_dynamic_filter_route(route: &DynamicFilterRoute, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
    let padding = "  ".repeat(indent);
    writeln!(
        f,
        "{padding}{} DynamicFilterRoute producer_kind={} producer_node={} consumer_kind={} consumer_node={} producer_columns={} consumer_columns={} filter_kinds={} consumer_lowering={} consumer_reduction_stage={} readiness={}",
        route.id,
        route.producer.kind(),
        route.producer.node_id(),
        route.consumer.kind(),
        route.consumer.node_id(),
        format_strings(&route.producer_columns),
        format_strings(&route.consumer_columns),
        route.filter_kinds.iter().map(ToString::to_string).collect::<Vec<_>>().join(","),
        route.consumer_lowering,
        route.consumer_reduction_stage(),
        route.readiness,
    )
}

impl QueryNode {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let padding = "  ".repeat(indent);
        match &self.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => {
                    let row_group_stats = source.diagnostics.selected_row_group_stats.as_ref().map_or_else(
                        || "unknown".to_string(),
                        |stats| {
                            format!(
                                "{}/{} selected_bytes={} total_bytes={}",
                                stats.selected_row_groups,
                                stats.total_row_groups,
                                stats
                                    .selected_source_bytes()
                                    .map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string()),
                                stats
                                    .total_source_bytes()
                                    .map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string())
                            )
                        },
                    );
                    writeln!(
                        f,
                        "{padding}{} LocalParquetSource files={} path={} format={:?} direct_device_read={} known_bytes={} selected_row_groups={} row_group_stats={} read_columns={} projected_output_columns={} pure_filter_columns={} predicate={} byte_range={} partition_columns={} zero_column_count={} scan_limit={} max_rows={} schema_mapping={} rejected_features={} placement={} schema_behavior={}",
                        self.id,
                        source.diagnostics.file_count,
                        source
                            .files
                            .first()
                            .map_or_else(|| "none".to_string(), |file| file.location.diagnostic_location()),
                        source.diagnostics.format,
                        source.diagnostics.direct_device_read,
                        source.diagnostics.known_bytes.map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string()),
                        source
                            .diagnostics
                            .selected_row_groups
                            .map_or_else(|| "all".to_string(), |row_groups| row_groups.to_string()),
                        row_group_stats,
                        source.diagnostics.read_column_count,
                        source.diagnostics.projected_output_column_count,
                        source.diagnostics.pure_filter_column_count,
                        source.diagnostics.predicate_disposition.as_str(),
                        source.diagnostics.byte_range_disposition.as_str(),
                        source.diagnostics.partition_column_disposition.as_str(),
                        source.diagnostics.zero_column_count_disposition.as_str(),
                        source.diagnostics.scan_limit_disposition.as_str(),
                        source.max_rows.map_or_else(|| "unknown".to_string(), |rows| rows.to_string()),
                        source.diagnostics.schema_mapping_disposition.as_str(),
                        source
                            .diagnostics
                            .rejected_features
                            .iter()
                            .map(|feature| feature.as_str())
                            .collect::<Vec<_>>()
                            .join(","),
                        self.facts.placement,
                        self.facts.schema_behavior
                    )
                },
                SourceNode::GpuResident(source) => writeln!(
                    f,
                    "{padding}{} GpuResidentSource input_name={} schema_fields={} placement={} schema_behavior={}",
                    self.id,
                    source.input_name,
                    source.schema.fields().len(),
                    self.facts.placement,
                    self.facts.schema_behavior
                ),
                SourceNode::LiteralBatch(source) => writeln!(
                    f,
                    "{padding}{} LiteralBatchSource batches={} rows={} schema_fields={} placement={} schema_behavior={}",
                    self.id,
                    source.batches.len(),
                    source.row_count(),
                    source.schema.fields().len(),
                    self.facts.placement,
                    self.facts.schema_behavior
                ),
            },
            PlanNode::Filter(node) => writeln!(
                f,
                "{padding}{} Filter input={} predicate={} placement={} schema_behavior={}",
                self.id, node.input, node.predicate, self.facts.placement, self.facts.schema_behavior
            ),
            PlanNode::Projection(node) => writeln!(
                f,
                "{padding}{} Projection input={} output_count={} expressions={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.outputs.len(),
                format_exprs(&node.expressions().cloned().collect::<Vec<_>>()),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::Aggregate(node) => writeln!(
                f,
                "{padding}{} Aggregate input={} group_count={} aggr_count={} grouping_set_count={} grouping_id_alias={} grouping_sets={} groups={} aggrs={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.spec.group_exprs.len(),
                node.spec.aggr_exprs.len(),
                node.spec.grouping_sets.as_ref().map_or(0, |grouping_sets| grouping_sets.groups.len()),
                node.spec
                    .grouping_sets
                    .as_ref()
                    .map_or("none", |grouping_sets| grouping_sets.grouping_id_alias.as_str()),
                format_grouping_sets(node.spec.grouping_sets.as_ref()),
                format_exprs(&node.spec.group_exprs),
                format_aggregate_exprs(&node.spec.aggr_exprs),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::Window(node) => writeln!(
                f,
                "{padding}{} Window input={} partition_count={} order_count={} function_count={} partitions={} order={} functions={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.spec.partition_by.len(),
                node.spec.order_by.len(),
                node.spec.functions.len(),
                format_exprs(&node.spec.partition_by),
                format_sort_exprs(&node.spec.order_by),
                format_window_functions(&node.spec.functions),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::Sort(node) => writeln!(
                f,
                "{padding}{} Sort input={} expr_count={} expressions={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.spec.expressions.len(),
                format_sort_exprs(&node.spec.expressions),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::Limit(node) => writeln!(
                f,
                "{padding}{} Limit input={} skip={} fetch={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.spec.skip,
                node.spec.fetch.map_or_else(|| "none".to_string(), |fetch| fetch.to_string()),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::Join(node) => writeln!(
                f,
                "{padding}{} Join left={} right={} type={} key_count={} residual={} null_equality={} build_side={} build_artifact={} output_projection={} placement={} schema_behavior={}",
                self.id,
                node.inputs[0],
                node.inputs[1],
                node.spec.join_type,
                node.spec.keys.len(),
                node.spec.residual_predicate.as_ref().map_or("none".to_string(), ToString::to_string),
                node.spec.null_equality,
                node.spec.build_side,
                node.spec.build_artifact_id.as_ref().map_or("none", HashJoinBuildArtifactId::as_str),
                format_join_output_projection(&node.spec.output_projection),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::Union(node) => writeln!(
                f,
                "{padding}{} Union inputs={} input_count={} placement={} schema_behavior={}",
                self.id,
                format_node_ids(&node.inputs),
                node.inputs.len(),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::EdgeNormalize(node) => writeln!(
                f,
                "{padding}{} EdgeNormalize input={} src_col={} dst_col={} weight_col={} edge_id_col={} add_reverse={} dedup={} weight_combiner={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.spec.edge_columns.source,
                node.spec.edge_columns.destination,
                node.spec.edge_columns.weight.as_deref().unwrap_or("none"),
                node.spec.edge_columns.edge_id.as_deref().unwrap_or("none"),
                node.spec.add_reverse,
                node.spec.dedup,
                node.spec.weight_combiner,
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::GraphAlgorithm(node) => writeln!(
                f,
                "{padding}{} GraphAlgorithm input={} side_inputs={} algorithm={} src_col={} dst_col={} weight_col={} edge_id_col={} directed={} renumber={} construction_policy={} params={} predicates={} output_schema_fields={} placement={} schema_behavior={}",
                self.id,
                node.input,
                format_node_ids(&node.inputs[1..]),
                node.spec.algorithm,
                node.spec.edge_columns.source,
                node.spec.edge_columns.destination,
                node.spec.edge_columns.weight.as_deref().unwrap_or("none"),
                node.spec.edge_columns.edge_id.as_deref().unwrap_or("none"),
                node.spec.graph_options.directed,
                node.spec.graph_options.renumber,
                node.spec.graph_options.policy,
                format_graph_algorithm_params(&node.spec.params),
                format_graph_algorithm_predicates(&node.spec.params),
                node.output_schema().fields().len(),
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::VectorAlgorithm(node) => writeln!(
                f,
                "{padding}{} VectorAlgorithm inputs={} algorithm={} params={} output_schema_fields={} placement={} schema_behavior={}",
                self.id,
                format_node_ids(&node.inputs),
                node.spec.algorithm,
                format_vector_algorithm_params(&node.spec.params),
                node.output_schema().fields().len(),
                self.facts.placement,
                self.facts.schema_behavior
            ),
        }
    }
}

fn format_vector_relation(binding: &super::VectorRelationBinding) -> String {
    format!(
        "input:{};row_id:{};features:{}",
        binding.input,
        binding.row_id.diagnostic_label(),
        binding.vector.display_label()
    )
}

fn format_vector_algorithm_params(params: &super::VectorAlgorithmParams) -> String {
    match params {
        super::VectorAlgorithmParams::BruteForceKnn(params) => format!(
            "dataset=({}) queries=({}) k={} metric={}",
            format_vector_relation(&params.dataset),
            format_vector_relation(&params.queries),
            params.k,
            params.metric
        ),
        super::VectorAlgorithmParams::KMeans(params) => format!(
            "input=({}) n_clusters={} max_iterations={} tolerance={} n_init={} init={} initial_centroids={} metric={}",
            format_vector_relation(&params.input),
            params.n_clusters,
            params.max_iterations,
            params.tolerance,
            params.n_init,
            params.init,
            params
                .initial_centroids
                .as_ref()
                .map_or_else(|| "none".to_string(), |binding| format!("({})", format_vector_relation(binding))),
            params.metric
        ),
        super::VectorAlgorithmParams::Pca(params) => format!(
            "input=({}) n_components={} whiten={} solver={} tolerance={} n_iterations={} flip_signs_based_on_u={}",
            format_vector_relation(&params.input),
            params.n_components,
            params.whiten,
            params.solver,
            params.tolerance,
            params.n_iterations,
            params.flip_signs_based_on_u
        ),
    }
}

fn format_node_ids(ids: &[super::NodeId]) -> String {
    if ids.is_empty() { "none".to_string() } else { ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",") }
}

fn format_strings(values: &[String]) -> String {
    if values.is_empty() { "none".to_string() } else { values.join(",") }
}

fn format_exprs(expressions: &[Expr]) -> String {
    if expressions.is_empty() {
        "none".to_string()
    } else {
        expressions.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
    }
}

fn format_aggregate_exprs(expressions: &[crate::expr::AggregateExpr]) -> String {
    if expressions.is_empty() {
        "none".to_string()
    } else {
        expressions
            .iter()
            .map(|expr| {
                let distinct = if expr.distinct { "DISTINCT " } else { "" };
                let filter = expr.filter.as_ref().map(|filter| format!(" FILTER ({filter})")).unwrap_or_default();
                format!("{:?}({distinct}{}){filter} AS {}", expr.function, expr.expr, expr.alias)
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn format_grouping_sets(grouping_sets: Option<&crate::expr::GroupingSetsSpec>) -> String {
    let Some(grouping_sets) = grouping_sets else {
        return "none".to_string();
    };
    if grouping_sets.groups.is_empty() {
        return "empty".to_string();
    }
    grouping_sets
        .groups
        .iter()
        .map(|group| group.iter().map(|masked| if *masked { "1" } else { "0" }).collect::<Vec<_>>().join(""))
        .collect::<Vec<_>>()
        .join("|")
}

fn format_window_functions(functions: &[super::WindowFunction]) -> String {
    if functions.is_empty() {
        "none".to_string()
    } else {
        functions
            .iter()
            .map(|function| match function {
                super::WindowFunction::Aggregate { function, expr, alias, frame } => {
                    format!("{function:?}({expr}) {frame:?} AS {alias}")
                },
                super::WindowFunction::Rank { alias } => format!("RANK() AS {alias}"),
                super::WindowFunction::RowNumber { alias, output_type } => {
                    format!("ROW_NUMBER()::{output_type:?} AS {alias}")
                },
                super::WindowFunction::LagLead { expr, offset, default, alias, direction } => {
                    let name = match direction {
                        super::ShiftDirection::Lag => "LAG",
                        super::ShiftDirection::Lead => "LEAD",
                    };
                    match default {
                        Some(value) => format!("{name}({expr}, {offset}, {value}) AS {alias}"),
                        None => format!("{name}({expr}, {offset}) AS {alias}"),
                    }
                },
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn format_sort_exprs(expressions: &[SortExpr]) -> String {
    if expressions.is_empty() {
        "none".to_string()
    } else {
        expressions
            .iter()
            .map(|expr| {
                format!(
                    "{} {} NULLS {}",
                    expr.expr,
                    if expr.descending { "DESC" } else { "ASC" },
                    if expr.nulls_first { "FIRST" } else { "LAST" }
                )
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn format_graph_algorithm_params(params: &super::GraphAlgorithmParams) -> String {
    match params {
        super::GraphAlgorithmParams::PageRank(params) => {
            format!("alpha={}:epsilon={}:max_iterations={}", params.alpha, params.epsilon, params.max_iterations)
        },
        super::GraphAlgorithmParams::PersonalizedPageRank(params) => format!(
            "alpha={}:epsilon={}:max_iterations={}:personalization_node={}:personalization_vertex_col={}:personalization_value_col={}",
            params.pagerank.alpha,
            params.pagerank.epsilon,
            params.pagerank.max_iterations,
            params.personalization.input,
            params.personalization.vertex,
            params.personalization.value,
        ),
        super::GraphAlgorithmParams::Bfs(params) => format!(
            "source_vertex={}:source_vertices={}:depth_limit={}:path_output_rows={}",
            params.source_vertex,
            params
                .source_vertices
                .as_ref()
                .map_or_else(|| "none".to_string(), |input| format!("{}:{}", input.input, input.column)),
            params.depth_limit.map_or_else(|| "none".to_string(), |depth| depth.to_string()),
            params.path_output.map_or_else(|| "none".to_string(), |path| path.max_output_rows.to_string())
        ),
        super::GraphAlgorithmParams::Louvain(params) => {
            format!("max_level={}:threshold={}:resolution={}", params.max_level, params.threshold, params.resolution)
        },
        super::GraphAlgorithmParams::EigenvectorCentrality(params) => {
            format!("epsilon={}:max_iterations={}", params.epsilon, params.max_iterations)
        },
        super::GraphAlgorithmParams::KatzCentrality(params) => format!(
            "alpha={}:beta={}:epsilon={}:max_iterations={}",
            params.alpha, params.beta, params.epsilon, params.max_iterations,
        ),
        super::GraphAlgorithmParams::Hits(params) => format!(
            "epsilon={}:max_iterations={}:normalize={}",
            params.epsilon, params.max_iterations, params.normalize,
        ),
        super::GraphAlgorithmParams::BetweennessCentrality(params) => format!(
            "seeds={}:k={}:normalized={}:include_endpoints={}:exact_vertex_threshold={}",
            betweenness_seeds_label(params.seeds.as_ref()),
            params.k.map_or_else(|| "none".to_string(), |k| k.to_string()),
            params.normalized,
            params.include_endpoints,
            params.exact_vertex_threshold,
        ),
        super::GraphAlgorithmParams::EdgeBetweennessCentrality(params) => format!(
            "seeds={}:k={}:normalized={}:exact_vertex_threshold={}",
            betweenness_seeds_label(params.seeds.as_ref()),
            params.k.map_or_else(|| "none".to_string(), |k| k.to_string()),
            params.normalized,
            params.exact_vertex_threshold,
        ),
        super::GraphAlgorithmParams::Sssp(params) => format!(
            "source_vertex={}:cutoff={}",
            params.source_vertex,
            params.cutoff.map_or_else(|| "none".to_string(), |v| v.to_string()),
        ),
        super::GraphAlgorithmParams::DegreesAll(_) => String::new(),
        super::GraphAlgorithmParams::InDegreesAll(_) => String::new(),
        super::GraphAlgorithmParams::OutDegreesAll(_) => String::new(),
        super::GraphAlgorithmParams::WeaklyConnectedComponents(_) => String::new(),
        super::GraphAlgorithmParams::StronglyConnectedComponents(_) => String::new(),
        super::GraphAlgorithmParams::TriangleCountAll(_) => String::new(),
        super::GraphAlgorithmParams::CoreNumber(params) => {
            format!("degree_type={}", params.degree_type.as_str())
        },
        super::GraphAlgorithmParams::KCore(params) => {
            format!("k={}:degree_type={}", params.k, params.degree_type.as_str())
        },
        super::GraphAlgorithmParams::MinimumSpanningTree(_) => String::new(),
        super::GraphAlgorithmParams::Leiden(params) => format!(
            "max_level={}:resolution={}:theta={}:seed={}",
            params.max_level, params.resolution, params.theta, params.seed,
        ),
        super::GraphAlgorithmParams::Ecg(params) => format!(
            "min_weight={}:ensemble_size={}:max_level={}:threshold={}:resolution={}:seed={}",
            params.min_weight, params.ensemble_size, params.max_level, params.threshold, params.resolution, params.seed,
        ),
        super::GraphAlgorithmParams::SpectralModularityMaximization(params) => format!(
            "n_clusters={}:n_eigenvectors={}:evs_tolerance={}:evs_max_iterations={}:\
             k_means_tolerance={}:k_means_max_iterations={}:seed={}",
            params.n_clusters,
            params.n_eigenvectors,
            params.evs_tolerance,
            params.evs_max_iterations,
            params.k_means_tolerance,
            params.k_means_max_iterations,
            params.seed,
        ),
        super::GraphAlgorithmParams::ForceAtlas2(params) => format!(
            "max_iter={}:outbound_attraction_distribution={}:lin_log_mode={}:prevent_overlapping={}:\
             overlap_scaling_ratio={}:edge_weight_influence={}:jitter_tolerance={}:barnes_hut_optimize={}:\
             barnes_hut_theta={}:scaling_ratio={}:strong_gravity_mode={}:gravity={}:verbose={}:seed={}",
            params.max_iter,
            params.outbound_attraction_distribution,
            params.lin_log_mode,
            params.prevent_overlapping,
            params.overlap_scaling_ratio,
            params.edge_weight_influence,
            params.jitter_tolerance,
            params.barnes_hut_optimize,
            params.barnes_hut_theta,
            params.scaling_ratio,
            params.strong_gravity_mode,
            params.gravity,
            params.verbose,
            params.seed,
        ),
        super::GraphAlgorithmParams::Similarity(params) => format!(
            "vertex_pairs_node={}:first_column={}:second_column={}",
            params.vertex_pairs.input, params.vertex_pairs.first_column, params.vertex_pairs.second_column,
        ),
    }
}

fn betweenness_seeds_label(seeds: Option<&super::GraphBetweennessSeeds>) -> String {
    seeds.map_or_else(|| "all".to_string(), |seeds| seeds.len().to_string())
}

fn format_graph_algorithm_predicates(params: &super::GraphAlgorithmParams) -> String {
    let super::GraphAlgorithmParams::Bfs(params) = params else {
        return "none".to_string();
    };
    let predicates = &params.predicates;
    let mut labels = Vec::new();
    if let Some(input) = &predicates.include_vertices {
        labels.push(format!("include_vertices:{}:{}", input.input, input.column));
    }
    if let Some(input) = &predicates.exclude_vertices {
        labels.push(format!("exclude_vertices:{}:{}", input.input, input.column));
    }
    if let Some(input) = &predicates.target_vertices {
        labels.push(format!("target_vertices:{}:{}", input.input, input.column));
    }
    if let Some(input) = &predicates.include_edge_ids {
        labels.push(format!("include_edge_ids:{}:{}", input.input, input.column));
    }
    if let Some(input) = &predicates.include_edges {
        labels.push(format!(
            "include_edges:{}:{}:{}:{}",
            input.input,
            input.source,
            input.destination,
            input.edge_id.as_deref().unwrap_or("none")
        ));
    }
    if labels.is_empty() { "none".to_string() } else { labels.join("|") }
}

fn format_join_output_projection(projection: &[JoinOutputColumn]) -> String {
    if projection.is_empty() {
        "none".to_string()
    } else {
        projection.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
    }
}
