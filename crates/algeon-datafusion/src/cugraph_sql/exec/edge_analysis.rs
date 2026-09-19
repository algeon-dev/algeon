use std::sync::Arc;

use super::{
    AggregateExec, AggregateMode, Column, CugraphAlgorithmSpec, CugraphGraphConstructionPolicy, DataSourceExec,
    ExecutionPlan, FileScanConfig, MemorySourceConfig, ProjectionExec, RepartitionExec, Schema, SortExec, UnionExec,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct CugraphEdgeCanonicalFacts {
    add_reverse_present: bool,
    dedup_present: bool,
}

impl CugraphEdgeCanonicalFacts {
    fn satisfies_python_options(self, spec: &CugraphAlgorithmSpec) -> bool {
        self.dedup_present && spec.weight_col.is_none() && (spec.graph_options.directed || self.add_reverse_present)
    }
}

pub(super) fn cugraph_edge_normalize_inserted(spec: &CugraphAlgorithmSpec, facts: &CugraphEdgeCanonicalFacts) -> bool {
    spec.graph_options.construction_policy == CugraphGraphConstructionPolicy::PythonCugraph
        && !facts.satisfies_python_options(spec)
}

pub(super) fn cugraph_edge_canonical_facts(
    plan: &dyn ExecutionPlan,
    spec: &CugraphAlgorithmSpec,
) -> CugraphEdgeCanonicalFacts {
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        let input_facts = cugraph_edge_canonical_facts(aggregate.input().as_ref(), spec);
        return aggregate_edge_dedup_facts(aggregate, spec, input_facts);
    }

    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        if projection_edge_orientation(projection, spec).is_some() {
            return cugraph_edge_canonical_facts(projection.input().as_ref(), spec);
        }
        return CugraphEdgeCanonicalFacts::default();
    }

    if let Some(union) = plan.downcast_ref::<UnionExec>()
        && union_has_reverse_edge_expansion(union, spec)
    {
        return CugraphEdgeCanonicalFacts { add_reverse_present: true, ..Default::default() };
    }

    if preserves_edge_list_canonical_facts(plan) {
        let children = plan.children();
        if let [child] = children.as_slice() {
            return cugraph_edge_canonical_facts(child.as_ref(), spec);
        }
    }

    if let Some(source) = plan.downcast_ref::<DataSourceExec>()
        && crate::native::datafusion_local_parquet_scan_has_simple_directed_edge_topology(
            source,
            &spec.src_col,
            &spec.dst_col,
        )
    {
        return CugraphEdgeCanonicalFacts { dedup_present: true, ..Default::default() };
    }

    if let Some(native_plan) = super::admission::lowered_native_edge_plan(plan)
        && query_engine::plan::edge_plan_has_simple_directed_edge_topology(native_plan, &spec.src_col, &spec.dst_col)
    {
        return CugraphEdgeCanonicalFacts { dedup_present: true, ..Default::default() };
    }

    CugraphEdgeCanonicalFacts::default()
}

pub(super) fn preserves_edge_list_canonical_facts(plan: &dyn ExecutionPlan) -> bool {
    plan.downcast_ref::<RepartitionExec>().is_some()
        || plan.downcast_ref::<SortExec>().is_some_and(|sort| sort.fetch().is_none())
        || matches!(plan.name(), "CoalesceBatchesExec" | "CoalescePartitionsExec") && plan.fetch().is_none()
}

pub(super) fn aggregate_edge_dedup_facts(
    aggregate: &AggregateExec,
    spec: &CugraphAlgorithmSpec,
    input_facts: CugraphEdgeCanonicalFacts,
) -> CugraphEdgeCanonicalFacts {
    if !matches!(aggregate.mode(), AggregateMode::Single | AggregateMode::Final | AggregateMode::FinalPartitioned) {
        return CugraphEdgeCanonicalFacts::default();
    }
    let group_by = aggregate.group_expr();
    if group_by.has_grouping_set()
        || !group_by.null_expr().is_empty()
        || group_by.groups().len() != 1
        || group_by.groups().first().is_some_and(|group| group.iter().any(|is_null| *is_null))
    {
        return CugraphEdgeCanonicalFacts::default();
    }
    let group_exprs = aggregate.group_expr().expr();
    if group_exprs.len() != 2 {
        return CugraphEdgeCanonicalFacts::default();
    }
    let Some(output_indices) = unique_edge_schema_column_indices(aggregate.schema().as_ref(), spec) else {
        return CugraphEdgeCanonicalFacts::default();
    };
    if output_indices.source >= group_exprs.len() || output_indices.destination >= group_exprs.len() {
        return CugraphEdgeCanonicalFacts::default();
    }
    let Some(input_indices) = unique_edge_schema_column_indices(aggregate.input().schema().as_ref(), spec) else {
        return CugraphEdgeCanonicalFacts::default();
    };
    let Some(source_role) = group_exprs
        .get(output_indices.source)
        .and_then(|(expr, _alias)| edge_column_role(expr.as_ref(), input_indices, spec))
    else {
        return CugraphEdgeCanonicalFacts::default();
    };
    let Some(destination_role) = group_exprs
        .get(output_indices.destination)
        .and_then(|(expr, _alias)| edge_column_role(expr.as_ref(), input_indices, spec))
    else {
        return CugraphEdgeCanonicalFacts::default();
    };
    if source_role == destination_role {
        return CugraphEdgeCanonicalFacts::default();
    }

    CugraphEdgeCanonicalFacts { dedup_present: true, add_reverse_present: input_facts.add_reverse_present }
}

pub(super) fn union_has_reverse_edge_expansion(union: &UnionExec, spec: &CugraphAlgorithmSpec) -> bool {
    let inputs = union.inputs();
    let [left, right] = inputs.as_slice() else {
        return false;
    };
    let Some(left) = edge_projection_provenance(left.as_ref(), spec) else {
        return false;
    };
    let Some(right) = edge_projection_provenance(right.as_ref(), spec) else {
        return false;
    };
    left.input_identity == right.input_identity
        && matches!(
            (left.orientation, right.orientation),
            (EdgeProjectionOrientation::Forward, EdgeProjectionOrientation::Reverse)
                | (EdgeProjectionOrientation::Reverse, EdgeProjectionOrientation::Forward)
        )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EdgeProjectionOrientation {
    Forward,
    Reverse,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct EdgeProjectionProvenance {
    orientation: EdgeProjectionOrientation,
    input_identity: EdgeProjectionInputIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum EdgeProjectionInputIdentity {
    FileScan {
        object_store_url: String,
        file_groups: Vec<Vec<String>>,
        projection: String,
        filter: String,
        fetch: Option<usize>,
    },
    MemorySourcePointer(String),
}

pub(super) fn edge_projection_provenance(
    plan: &dyn ExecutionPlan,
    spec: &CugraphAlgorithmSpec,
) -> Option<EdgeProjectionProvenance> {
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        let orientation = projection_edge_orientation(projection, spec)?;
        let input_identity = edge_projection_input_identity(projection.input().as_ref(), spec)?;
        return Some(EdgeProjectionProvenance { orientation, input_identity });
    }

    if preserves_edge_list_canonical_facts(plan) {
        let children = plan.children();
        if let [child] = children.as_slice() {
            return edge_projection_provenance(child.as_ref(), spec);
        }
    }

    let schema = plan.schema();
    if unique_edge_schema_column_indices(schema.as_ref(), spec).is_some() {
        let input_identity = edge_projection_input_identity(plan, spec)?;
        Some(EdgeProjectionProvenance { orientation: EdgeProjectionOrientation::Forward, input_identity })
    } else {
        None
    }
}

pub(super) fn edge_projection_input_identity(
    plan: &dyn ExecutionPlan,
    spec: &CugraphAlgorithmSpec,
) -> Option<EdgeProjectionInputIdentity> {
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        if projection_edge_orientation(projection, spec).is_some() {
            return edge_projection_input_identity(projection.input().as_ref(), spec);
        }
        return None;
    }
    if preserves_edge_list_canonical_facts(plan) {
        let children = plan.children();
        if let [child] = children.as_slice() {
            return edge_projection_input_identity(child.as_ref(), spec);
        }
        return None;
    }
    let source = plan.downcast_ref::<DataSourceExec>()?;
    let data_source = source.data_source();
    let data_source_any = data_source.as_ref() as &dyn std::any::Any;
    if let Some(file_scan) = data_source_any.downcast_ref::<FileScanConfig>() {
        let file_groups = file_scan
            .file_groups
            .iter()
            .map(|group| {
                group
                    .files()
                    .iter()
                    .map(|file| {
                        format!(
                            "{}:{}:{:?}:{:?}",
                            file.object_meta.location, file.object_meta.size, file.range, file.partition_values
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        return Some(EdgeProjectionInputIdentity::FileScan {
            object_store_url: format!("{:?}", file_scan.object_store_url),
            file_groups,
            projection: format!("{:?}", file_scan.file_source.projection()),
            filter: format!("{:?}", file_scan.file_source.filter()),
            fetch: file_scan.limit,
        });
    }
    if data_source_any.downcast_ref::<MemorySourceConfig>().is_some() {
        return Some(EdgeProjectionInputIdentity::MemorySourcePointer(format!("{:p}", Arc::as_ptr(data_source))));
    }
    None
}

pub(super) fn projection_edge_orientation(
    projection: &ProjectionExec,
    spec: &CugraphAlgorithmSpec,
) -> Option<EdgeProjectionOrientation> {
    let output_indices = unique_edge_schema_column_indices(projection.schema().as_ref(), spec)?;
    let input_indices = unique_edge_schema_column_indices(projection.input().schema().as_ref(), spec)?;
    let source_input =
        projection.expr().get(output_indices.source).and_then(|expr| physical_column(expr.expr.as_ref()))?;
    let destination_input =
        projection.expr().get(output_indices.destination).and_then(|expr| physical_column(expr.expr.as_ref()))?;
    if !projection_preserves_weight(projection, spec) {
        return None;
    }

    if column_matches_edge_role(source_input, input_indices.source, &spec.src_col)
        && column_matches_edge_role(destination_input, input_indices.destination, &spec.dst_col)
    {
        Some(EdgeProjectionOrientation::Forward)
    } else if column_matches_edge_role(source_input, input_indices.destination, &spec.dst_col)
        && column_matches_edge_role(destination_input, input_indices.source, &spec.src_col)
    {
        Some(EdgeProjectionOrientation::Reverse)
    } else {
        None
    }
}

pub(super) fn projection_preserves_weight(projection: &ProjectionExec, spec: &CugraphAlgorithmSpec) -> bool {
    let Some(weight_col) = spec.weight_col.as_deref() else {
        return true;
    };
    let Some(weight_index) = unique_schema_column_index(projection.schema().as_ref(), weight_col) else {
        return false;
    };
    let Some(input_weight_index) = unique_schema_column_index(projection.input().schema().as_ref(), weight_col) else {
        return false;
    };
    projection
        .expr()
        .get(weight_index)
        .and_then(|expr| physical_column(expr.expr.as_ref()))
        .is_some_and(|column| column_matches_edge_role(column, input_weight_index, weight_col))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct EdgeSchemaColumnIndices {
    source: usize,
    destination: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum EdgeColumnRole {
    Source,
    Destination,
}

pub(super) fn unique_edge_schema_column_indices(
    schema: &Schema,
    spec: &CugraphAlgorithmSpec,
) -> Option<EdgeSchemaColumnIndices> {
    let source = unique_schema_column_index(schema, &spec.src_col)?;
    let destination = unique_schema_column_index(schema, &spec.dst_col)?;
    (source != destination).then_some(EdgeSchemaColumnIndices { source, destination })
}

pub(super) fn unique_schema_column_index(schema: &Schema, column_name: &str) -> Option<usize> {
    let mut matching_indices = schema
        .fields()
        .iter()
        .enumerate()
        .filter_map(|(index, field)| same_edge_column(field.name(), column_name).then_some(index));
    let index = matching_indices.next()?;
    if matching_indices.next().is_some() {
        return None;
    }
    Some(index)
}

pub(super) fn edge_column_role(
    expr: &dyn datafusion::physical_expr::PhysicalExpr,
    input_indices: EdgeSchemaColumnIndices,
    spec: &CugraphAlgorithmSpec,
) -> Option<EdgeColumnRole> {
    let column = physical_column(expr)?;
    if column_matches_edge_role(column, input_indices.source, &spec.src_col) {
        Some(EdgeColumnRole::Source)
    } else if column_matches_edge_role(column, input_indices.destination, &spec.dst_col) {
        Some(EdgeColumnRole::Destination)
    } else {
        None
    }
}

pub(super) fn column_matches_edge_role(column: &Column, index: usize, name: &str) -> bool {
    column.index() == index && same_edge_column(column.name(), name)
}

pub(super) fn physical_column(expr: &dyn datafusion::physical_expr::PhysicalExpr) -> Option<&Column> {
    expr.downcast_ref::<Column>()
}

pub(super) fn same_edge_column(actual: &str, expected: &str) -> bool {
    actual == expected
}
