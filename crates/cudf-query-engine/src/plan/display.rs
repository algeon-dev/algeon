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
                        source.files.first().map_or_else(|| "none".to_string(), |file| file.path.display().to_string()),
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
                self.id, node.input, node.spec.skip, node.spec.fetch, self.facts.placement, self.facts.schema_behavior
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
        }
    }
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
            .map(|expr| format!("{:?}({}) AS {}", expr.function, expr.expr, expr.alias))
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

fn format_join_output_projection(projection: &[JoinOutputColumn]) -> String {
    if projection.is_empty() {
        "none".to_string()
    } else {
        projection.iter().map(ToString::to_string).collect::<Vec<_>>().join(",")
    }
}
