//! `Display` formatting for [`QueryPlan`] and its nodes.
//!
//! The format is human-oriented diagnostic text used by triage tools, not a
//! stable parser surface. Bump the relevant report TSV schema instead of
//! parsing display output.

#![deny(missing_docs)]

use std::fmt;

use super::{HashJoinBuildArtifactId, JoinOutputColumn, PlanNode, QueryNode, QueryPlan, ResultSink, SortExpr};
use crate::{expr::Expr, source::SourceNode};

impl fmt::Display for QueryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "QueryPlan root={} sink={}", self.root, self.sink.kind_name())?;
        for node in &self.nodes {
            node.fmt_with_indent(f, 1)?;
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

impl QueryNode {
    fn fmt_with_indent(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
        let padding = "  ".repeat(indent);
        match &self.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => {
                    writeln!(
                        f,
                        "{padding}{} LocalParquetSource files={} path={} format={:?} direct_device_read={} known_bytes={} selected_row_groups={} read_columns={} projected_output_columns={} pure_filter_columns={} predicate={} byte_range={} partition_columns={} zero_column_count={} scan_limit={} schema_mapping={} rejected_features={} placement={} schema_behavior={}",
                        self.id,
                        source.diagnostics.file_count,
                        source.files.first().map_or_else(|| "none".to_string(), |file| file.location.reader_path()),
                        source.diagnostics.format,
                        source.diagnostics.direct_device_read,
                        source.diagnostics.known_bytes.map_or_else(|| "unknown".to_string(), |bytes| bytes.to_string()),
                        source
                            .diagnostics
                            .selected_row_groups
                            .map_or_else(|| "all".to_string(), |row_groups| row_groups.to_string()),
                        source.diagnostics.read_column_count,
                        source.diagnostics.projected_output_column_count,
                        source.diagnostics.pure_filter_column_count,
                        source.diagnostics.predicate_disposition.as_str(),
                        source.diagnostics.byte_range_disposition.as_str(),
                        source.diagnostics.partition_column_disposition.as_str(),
                        source.diagnostics.zero_column_count_disposition.as_str(),
                        source.diagnostics.scan_limit_disposition.as_str(),
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
                "{padding}{} Aggregate input={} group_count={} aggr_count={} groups={} aggrs={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.spec.group_exprs.len(),
                node.spec.aggr_exprs.len(),
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
                "{padding}{} EdgeNormalize input={} src_col={} dst_col={} weight_col={} add_reverse={} dedup={} weight_combiner={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.spec.edge_columns.source,
                node.spec.edge_columns.destination,
                node.spec.edge_columns.weight.as_deref().unwrap_or("none"),
                node.spec.add_reverse,
                node.spec.dedup,
                node.spec.weight_combiner,
                self.facts.placement,
                self.facts.schema_behavior
            ),
            PlanNode::GraphAlgorithm(node) => writeln!(
                f,
                "{padding}{} GraphAlgorithm input={} algorithm={} src_col={} dst_col={} weight_col={} directed={} renumber={} construction_policy={} params={} output_schema_fields={} placement={} schema_behavior={}",
                self.id,
                node.input,
                node.spec.algorithm,
                node.spec.edge_columns.source,
                node.spec.edge_columns.destination,
                node.spec.edge_columns.weight.as_deref().unwrap_or("none"),
                node.spec.graph_options.directed,
                node.spec.graph_options.renumber,
                node.spec.graph_options.policy,
                format_graph_algorithm_params(&node.spec.params),
                node.output_schema().fields().len(),
                self.facts.placement,
                self.facts.schema_behavior
            ),
        }
    }
}

fn format_node_ids(ids: &[super::NodeId]) -> String {
    if ids.is_empty() { "none".to_string() } else { ids.iter().map(ToString::to_string).collect::<Vec<_>>().join(",") }
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
                super::WindowFunction::RowNumber { alias } => format!("ROW_NUMBER() AS {alias}"),
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
        super::GraphAlgorithmParams::Bfs(params) => format!(
            "source_vertex={}:depth_limit={}",
            params.source_vertex,
            params.depth_limit.map_or_else(|| "none".to_string(), |depth| depth.to_string())
        ),
        super::GraphAlgorithmParams::Louvain(params) => format!("resolution={}", params.resolution),
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
        super::GraphAlgorithmParams::PageRankStrict(params) => {
            format!("alpha={}:epsilon={}:max_iterations={}", params.alpha, params.epsilon, params.max_iterations)
        },
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
        super::GraphAlgorithmParams::BalancedCutClustering(params) => format!(
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
    }
}

fn format_join_output_projection(projection: &[JoinOutputColumn]) -> String {
    if projection.is_empty() {
        "none".to_string()
    } else {
        projection.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
    }
}
