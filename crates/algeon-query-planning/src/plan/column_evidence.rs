use std::collections::{BTreeMap, BTreeSet, HashMap};

use arrow::{
    array::{Array, LargeStringArray, StringArray, StringViewArray},
    datatypes::{DataType, SchemaRef},
};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    expr::{
        AggregateFunction, BinaryOp, Expr, LiteralValue, alias_wrapped_column, representation_cast_preserves_identity,
    },
    plan::{
        AggregateNode, JoinNode, JoinOutputColumn, JoinSide, JoinType, NodeId, PlanNode, QueryNode, QueryPlan,
        WindowFunction, canonical_execution_scope_fingerprint,
        row_bounds::query_plan_node_hard_row_upper_bound_with_resident_rows,
    },
    source::{LiteralBatchSource, LocalParquetSource, SourceColumnRef, SourceNode},
    util::log_safe_fingerprint,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[doc(hidden)]
pub enum StringPayloadLineage {
    Parquet { stable_id: String, physical_column_index: usize, selected_row_groups: Vec<usize> },
    Literal { node_id: NodeId, column_index: usize },
    Computed { node_id: NodeId, column_index: usize },
    PendingResident { execution_scope_fingerprint: String, occurrence_id: String, column_index: usize },
    Resident { execution_scope_fingerprint: String, occurrence_id: String, column_index: usize },
}

impl StringPayloadLineage {
    fn audit_label(&self) -> String {
        let fingerprint = log_safe_fingerprint(&format!("{self:?}"));
        match self {
            Self::Parquet { physical_column_index, selected_row_groups, .. } => format!(
                "kind=parquet,column_index={physical_column_index},selected_row_group_count={},fingerprint={fingerprint}",
                selected_row_groups.len()
            ),
            Self::Literal { node_id, column_index } => {
                format!("kind=literal,node_id={node_id},column_index={column_index},fingerprint={fingerprint}")
            },
            Self::Computed { node_id, column_index } => {
                format!("kind=computed,node_id={node_id},column_index={column_index},fingerprint={fingerprint}")
            },
            Self::PendingResident { column_index, .. } => {
                format!("kind=pending_resident,column_index={column_index},fingerprint={fingerprint}")
            },
            Self::Resident { column_index, .. } => {
                format!("kind=resident,column_index={column_index},fingerprint={fingerprint}")
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct StringPayloadBound {
    #[doc(hidden)]
    pub total_payload_upper_bound_bytes: Option<u64>,
    #[doc(hidden)]
    pub distinct_payload_upper_bound_bytes: Option<u64>,
    #[doc(hidden)]
    pub distinct_value_count_upper_bound: Option<u64>,
    #[doc(hidden)]
    pub row_multiplicity_upper_bound: Option<u64>,
    #[doc(hidden)]
    pub max_single_value_length_bytes: Option<u64>,
    #[doc(hidden)]
    pub lineages: BTreeSet<StringPayloadLineage>,
}

impl StringPayloadBound {
    fn for_union<'a>(bounds: impl IntoIterator<Item = &'a Self>) -> Option<Self> {
        let bounds = bounds.into_iter().collect::<Vec<_>>();
        let total_payload_upper_bound_bytes = bounds
            .iter()
            .all(|bound| bound.total_payload_upper_bound_bytes.is_some())
            .then(|| {
                bounds.iter().try_fold(0_u64, |total, bound| {
                    total.checked_add(bound.total_payload_upper_bound_bytes.expect("all total bounds are present"))
                })
            })
            .flatten();
        let max_single_value_length_bytes = bounds
            .iter()
            .map(|bound| bound.max_single_value_length_bytes)
            .collect::<Option<Vec<_>>>()
            .and_then(|bounds| bounds.into_iter().max());
        let distinct_payload_upper_bound_bytes = bounds
            .iter()
            .all(|bound| bound.distinct_payload_upper_bound_bytes.is_some())
            .then(|| {
                bounds.iter().try_fold(0_u64, |total, bound| {
                    total.checked_add(
                        bound.distinct_payload_upper_bound_bytes.expect("all distinct payload bounds are present"),
                    )
                })
            })
            .flatten();
        let distinct_value_count_upper_bound = bounds
            .iter()
            .all(|bound| bound.distinct_value_count_upper_bound.is_some())
            .then(|| {
                bounds.iter().try_fold(0_u64, |total, bound| {
                    total.checked_add(
                        bound.distinct_value_count_upper_bound.expect("all distinct count bounds are present"),
                    )
                })
            })
            .flatten();
        let row_multiplicity_upper_bound = bounds
            .iter()
            .map(|bound| bound.row_multiplicity_upper_bound)
            .collect::<Option<Vec<_>>>()
            .and_then(|bounds| bounds.into_iter().max());
        let mut lineages = BTreeSet::new();
        for bound in &bounds {
            lineages.extend(bound.lineages.iter().cloned());
        }
        (total_payload_upper_bound_bytes.is_some()
            || distinct_payload_upper_bound_bytes.is_some()
            || max_single_value_length_bytes.is_some())
        .then_some(Self {
            total_payload_upper_bound_bytes,
            distinct_payload_upper_bound_bytes,
            distinct_value_count_upper_bound,
            row_multiplicity_upper_bound,
            max_single_value_length_bytes,
            lineages,
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[doc(hidden)]
pub struct PlanColumnEvidence {
    #[doc(hidden)]
    pub string_payload_bounds: BTreeMap<String, StringPayloadBound>,
    #[doc(hidden)]
    pub unique_keys: Vec<BTreeSet<String>>,
    #[doc(hidden)]
    pub equivalent_columns: Vec<BTreeSet<String>>,
}

#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct ResidentInputEvidence {
    #[doc(hidden)]
    pub schema: SchemaRef,
    #[doc(hidden)]
    pub producer_execution_scope_fingerprint: Option<String>,
    #[doc(hidden)]
    pub row_count: u64,
    #[doc(hidden)]
    pub column_evidence: PlanColumnEvidence,
}

#[doc(hidden)]
pub type ResidentInputEvidenceMap = BTreeMap<String, ResidentInputEvidence>;

impl PlanColumnEvidence {
    #[doc(hidden)]
    pub fn string_payload_bound(&self, column: &str) -> Option<&StringPayloadBound> {
        self.string_payload_bounds.get(column)
    }

    #[doc(hidden)]
    pub fn is_unique_on(&self, columns: &BTreeSet<String>) -> bool {
        self.unique_keys.iter().any(|key| key.is_subset(columns))
    }

    #[doc(hidden)]
    pub fn audit_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for (column, bound) in &self.string_payload_bounds {
            let column = log_safe_fingerprint(column);
            lines.push(format!(
                "string_payload,column={column},total_payload_upper_bound_bytes={},distinct_payload_upper_bound_bytes={},distinct_value_count_upper_bound={},row_multiplicity_upper_bound={},max_single_value_length_bytes={},lineage_count={}",
                optional_bytes(bound.total_payload_upper_bound_bytes),
                optional_bytes(bound.distinct_payload_upper_bound_bytes),
                optional_bytes(bound.distinct_value_count_upper_bound),
                optional_bytes(bound.row_multiplicity_upper_bound),
                optional_bytes(bound.max_single_value_length_bytes),
                bound.lineages.len(),
            ));
            lines.extend(bound.lineages.iter().enumerate().map(|(index, lineage)| {
                format!("string_payload_lineage,column={column},index={index},{}", lineage.audit_label())
            }));
        }
        lines
    }

    fn columns_are_equivalent(&self, left: &str, right: &str) -> bool {
        left == right || self.equivalent_columns.iter().any(|columns| columns.contains(left) && columns.contains(right))
    }
}

fn optional_bytes(value: Option<u64>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| value.to_string())
}

#[derive(Debug, Clone, Default)]
#[doc(hidden)]
pub struct PlanColumnEvidenceAnalysis {
    by_node: HashMap<NodeId, PlanColumnEvidence>,
}

impl PlanColumnEvidenceAnalysis {
    #[doc(hidden)]
    pub fn for_plan(plan: &QueryPlan, resident_outputs: &HashMap<String, PlanColumnEvidence>) -> Result<Self> {
        Self::for_plan_with_resident_rows(plan, resident_outputs, &HashMap::new())
    }

    #[doc(hidden)]
    pub fn for_plan_with_resident_rows(
        plan: &QueryPlan,
        resident_outputs: &HashMap<String, PlanColumnEvidence>,
        resident_row_upper_bounds: &HashMap<String, u64>,
    ) -> Result<Self> {
        let mut analysis = Self::default();
        let unique_keys_by_node = unique_keys_by_node(plan.nodes());
        for node in plan.nodes() {
            let mut evidence =
                analysis.evidence_for_node(plan, node.id, resident_outputs, resident_row_upper_bounds)?;
            evidence.unique_keys = unique_keys_by_node.get(node.id.index()).cloned().unwrap_or_default();
            analysis.by_node.insert(node.id, evidence);
        }
        Ok(analysis)
    }

    #[doc(hidden)]
    pub fn node(&self, node_id: NodeId) -> PlanColumnEvidence {
        self.by_node.get(&node_id).cloned().unwrap_or_default()
    }

    #[doc(hidden)]
    pub fn root(&self, plan: &QueryPlan) -> PlanColumnEvidence {
        self.node(plan.root_id())
    }

    fn evidence_for_node(
        &self,
        plan: &QueryPlan,
        node_id: NodeId,
        resident_outputs: &HashMap<String, PlanColumnEvidence>,
        resident_rows: &HashMap<String, u64>,
    ) -> Result<PlanColumnEvidence> {
        let Some(node) = plan.node(node_id) else {
            return Ok(PlanColumnEvidence::default());
        };
        Ok(match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => {
                    let evidence = local_parquet_evidence(source);
                    match &source.reader_predicate {
                        Some(predicate) => {
                            constrained_filter_evidence(predicate, evidence, source.exact_selected_row_count(), node_id)
                        },
                        None => evidence,
                    }
                },
                SourceNode::GpuResident(source) => {
                    let mut evidence = resident_outputs.get(&source.input_name).cloned().unwrap_or_default();
                    for (column_index, field) in source.schema.fields().iter().enumerate() {
                        let Some(bound) = evidence.string_payload_bounds.get_mut(field.name()) else {
                            continue;
                        };
                        if bound.max_single_value_length_bytes.is_none() {
                            bound.lineages = BTreeSet::from([StringPayloadLineage::PendingResident {
                                execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan),
                                occurrence_id: source.input_name.clone(),
                                column_index,
                            }]);
                        }
                    }
                    evidence
                },
                SourceNode::LiteralBatch(source) => literal_evidence(node_id, source),
            },
            PlanNode::Filter(filter) => constrained_filter_evidence(
                &filter.predicate,
                self.node(filter.input),
                query_plan_node_hard_row_upper_bound_with_resident_rows(plan, filter.input, resident_rows),
                node_id,
            ),
            PlanNode::Projection(projection) => {
                let input = self.node(projection.input);
                let input_schema = plan.node(projection.input).map(|node| node.output_schema());
                let mut output = PlanColumnEvidence::default();
                let output_rows =
                    query_plan_node_hard_row_upper_bound_with_resident_rows(plan, projection.input, resident_rows);
                for (column_index, projection_output) in projection.outputs.iter().enumerate() {
                    if let Some(source_column) = identity_source_column(&projection_output.expr, input_schema) {
                        if let Some(bound) = input.string_payload_bound(source_column) {
                            output.string_payload_bounds.insert(projection_output.field.name().clone(), bound.clone());
                        }
                        continue;
                    }
                    if !is_string(projection_output.field.data_type()) {
                        continue;
                    }
                    let lineage = StringPayloadLineage::Computed { node_id, column_index };
                    let bound = output_rows
                        .and_then(|rows| {
                            computed_string_payload_bound(&projection_output.expr, &input, rows, lineage.clone())
                        })
                        .or_else(|| row_independent_literal_string_payload_bound(&projection_output.expr, lineage));
                    if let Some(bound) = bound {
                        output.string_payload_bounds.insert(projection_output.field.name().clone(), bound);
                    }
                }
                let identity_mappings = projection
                    .outputs
                    .iter()
                    .filter_map(|output| {
                        Some((
                            identity_source_column(&output.expr, input_schema)?.to_owned(),
                            output.field.name().clone(),
                        ))
                    })
                    .collect::<Vec<_>>();
                output.equivalent_columns = remap_equivalent_columns(&input, identity_mappings);
                output
            },
            PlanNode::Aggregate(aggregate) => {
                let input = self.node(aggregate.input);
                let input_schema = plan.node(aggregate.input).map(|node| node.output_schema());
                let input_rows =
                    query_plan_node_hard_row_upper_bound_with_resident_rows(plan, aggregate.input, resident_rows);
                let mut output =
                    grouped_aggregate_evidence(aggregate, node.output_schema(), input_schema, &input, input_rows);
                if aggregate.spec.grouping_sets.is_none()
                    && let Some(input_node) = plan.node(aggregate.input)
                    && let PlanNode::Union(union) = &input_node.kind
                {
                    let branch_outputs = union
                        .inputs
                        .iter()
                        .filter_map(|input_id| {
                            let branch_node = plan.node(*input_id)?;
                            let branch = remap_union_input_evidence(
                                input_node.output_schema(),
                                branch_node.output_schema(),
                                &self.node(*input_id),
                            );
                            Some(grouped_aggregate_evidence(
                                aggregate,
                                node.output_schema(),
                                Some(input_node.output_schema()),
                                &branch,
                                query_plan_node_hard_row_upper_bound_with_resident_rows(plan, *input_id, resident_rows),
                            ))
                        })
                        .collect::<Vec<_>>();
                    if branch_outputs.len() == union.inputs.len() {
                        output = union_evidence_from_inputs(node.output_schema(), &branch_outputs);
                    }
                }
                output
            },
            PlanNode::Window(window) => self.node(window.input),
            PlanNode::Sort(sort) => self.node(sort.input),
            PlanNode::Limit(limit) => self.node(limit.input),
            PlanNode::Union(union) => union_evidence(plan, union.inputs.as_slice(), self),
            PlanNode::Join(join) => join_evidence(
                join,
                &self.node(join.inputs[0]),
                &self.node(join.inputs[1]),
                query_plan_node_hard_row_upper_bound_with_resident_rows(plan, join.inputs[0], resident_rows),
                query_plan_node_hard_row_upper_bound_with_resident_rows(plan, join.inputs[1], resident_rows),
            )
            .map_err(|error| {
                error
                    .with_fact("node_id", node_id)
                    .with_fact("owner_kind", "consumer")
                    .with_fact("owner_node_id", node_id)
                    .with_fact(
                        "owner_execution_scope_fingerprint",
                        log_safe_fingerprint(&canonical_execution_scope_fingerprint(plan)),
                    )
            })?,
            PlanNode::GraphAlgorithm(_) => PlanColumnEvidence::default(),
            _ => PlanColumnEvidence::default(),
        })
    }
}

fn grouped_aggregate_evidence(
    aggregate: &AggregateNode,
    output_schema: &SchemaRef,
    input_schema: Option<&SchemaRef>,
    input: &PlanColumnEvidence,
    input_rows: Option<u64>,
) -> PlanColumnEvidence {
    let mut output = PlanColumnEvidence::default();
    if aggregate.spec.grouping_sets.is_some() {
        return output;
    }
    for (group_index, (expression, field)) in aggregate.spec.group_exprs.iter().zip(output_schema.fields()).enumerate()
    {
        let Some(source_column) = identity_source_column(expression, input_schema) else {
            continue;
        };
        let Some(bound) = input.string_payload_bound(source_column) else {
            continue;
        };
        let mut output_bound = bound.clone();
        let repeated_domain = aggregate
            .spec
            .group_exprs
            .iter()
            .enumerate()
            .filter(|(index, _)| *index != group_index)
            .try_fold(1_u64, |count, (_, other)| {
                let column = identity_source_column(other, input_schema)?;
                count.checked_mul(distinct_value_count_bound(input.string_payload_bound(column)?)?)
            })
            .and_then(|count| bound.distinct_payload_upper_bound_bytes?.checked_mul(count));
        let row_payload =
            input_rows.and_then(|rows| bound.max_single_value_length_bytes.and_then(|max| rows.checked_mul(max)));
        output_bound.total_payload_upper_bound_bytes =
            [bound.total_payload_upper_bound_bytes, repeated_domain, row_payload].into_iter().flatten().min();
        if aggregate.spec.group_exprs.len() == 1 {
            output_bound.total_payload_upper_bound_bytes = bound.distinct_payload_upper_bound_bytes;
        }
        output.string_payload_bounds.insert(field.name().clone(), output_bound);
    }
    for (expression, field) in
        aggregate.spec.aggr_exprs.iter().zip(output_schema.fields().iter().skip(aggregate.spec.group_exprs.len()))
    {
        if !matches!(expression.function, AggregateFunction::Min | AggregateFunction::Max) {
            continue;
        }
        let Some(source_column) = identity_source_column(&expression.expr, input_schema) else {
            continue;
        };
        if let Some(bound) = input.string_payload_bound(source_column) {
            output.string_payload_bounds.insert(field.name().clone(), bound.clone());
        }
    }
    output
}

fn distinct_value_count_bound(bound: &StringPayloadBound) -> Option<u64> {
    bound.distinct_value_count_upper_bound.or_else(|| bound.distinct_payload_upper_bound_bytes?.checked_add(2))
}

fn remap_union_input_evidence(
    union_schema: &SchemaRef,
    input_schema: &SchemaRef,
    input: &PlanColumnEvidence,
) -> PlanColumnEvidence {
    let mut output = PlanColumnEvidence::default();
    for (union_field, input_field) in union_schema.fields().iter().zip(input_schema.fields()) {
        if let Some(bound) = input.string_payload_bound(input_field.name()) {
            output.string_payload_bounds.insert(union_field.name().clone(), bound.clone());
        }
    }
    output
}

fn union_evidence_from_inputs(output_schema: &SchemaRef, inputs: &[PlanColumnEvidence]) -> PlanColumnEvidence {
    let mut output = PlanColumnEvidence::default();
    for field in output_schema.fields() {
        let bounds = inputs.iter().map(|input| input.string_payload_bound(field.name())).collect::<Option<Vec<_>>>();
        if let Some(bounds) = bounds
            && let Some(bound) = StringPayloadBound::for_union(bounds)
        {
            output.string_payload_bounds.insert(field.name().clone(), bound);
        }
    }
    output
}

#[doc(hidden)]
pub fn plan_node_is_unique_on(plan: &QueryPlan, node_id: NodeId, columns: &BTreeSet<String>) -> Result<bool> {
    Ok(partial_plan_node_is_unique_on(plan.nodes(), node_id, columns))
}

fn local_parquet_evidence(source: &LocalParquetSource) -> PlanColumnEvidence {
    let mut output = PlanColumnEvidence::default();
    for field in source.output_schema().fields() {
        if !is_string(field.data_type()) {
            continue;
        }
        let Some(SourceColumnRef::File(physical_column_index)) = source.source_column_ref(field.name()) else {
            continue;
        };
        let mut file_bounds = Vec::with_capacity(source.files.len());
        let mut complete = true;
        for file in &source.files {
            let Some(evidence) = file.variable_width_payload_evidence.get(&physical_column_index) else {
                complete = false;
                break;
            };
            if evidence.physical_column_index != physical_column_index
                || file.row_groups.as_ref().is_some_and(|selected| selected != &evidence.selected_row_groups)
            {
                complete = false;
                break;
            }
            file_bounds.push(StringPayloadBound {
                total_payload_upper_bound_bytes: Some(evidence.payload_upper_bound_bytes),
                distinct_payload_upper_bound_bytes: Some(evidence.payload_upper_bound_bytes),
                distinct_value_count_upper_bound: None,
                row_multiplicity_upper_bound: Some(1),
                max_single_value_length_bytes: None,
                lineages: BTreeSet::from([StringPayloadLineage::Parquet {
                    stable_id: file.stable_id.clone(),
                    physical_column_index,
                    selected_row_groups: evidence.selected_row_groups.clone(),
                }]),
            });
        }
        if complete
            && !file_bounds.is_empty()
            && let Some(bound) = StringPayloadBound::for_union(&file_bounds)
        {
            output.string_payload_bounds.insert(field.name().clone(), bound);
        }
    }
    output
}

fn literal_evidence(node_id: NodeId, source: &LiteralBatchSource) -> PlanColumnEvidence {
    let mut output = PlanColumnEvidence::default();
    for (column_index, field) in source.schema.fields().iter().enumerate() {
        if !is_string(field.data_type()) {
            continue;
        }
        let stats = source.batches.iter().try_fold((0_u64, 0_u64), |(total, max), batch| {
            let array = batch.column(column_index);
            (0..array.len()).try_fold((total, max), |(total, max), row| {
                if array.is_null(row) {
                    return Some((total, max));
                }
                let length = match field.data_type() {
                    DataType::Utf8 => array.as_any().downcast_ref::<StringArray>()?.value(row).len(),
                    DataType::LargeUtf8 => array.as_any().downcast_ref::<LargeStringArray>()?.value(row).len(),
                    DataType::Utf8View => array.as_any().downcast_ref::<StringViewArray>()?.value(row).len(),
                    _ => return None,
                };
                let length = u64::try_from(length).ok()?;
                Some((total.checked_add(length)?, max.max(length)))
            })
        });
        if let Some((total_payload_upper_bound_bytes, max_single_value_length_bytes)) = stats {
            output.string_payload_bounds.insert(
                field.name().clone(),
                StringPayloadBound {
                    total_payload_upper_bound_bytes: Some(total_payload_upper_bound_bytes),
                    distinct_payload_upper_bound_bytes: Some(total_payload_upper_bound_bytes),
                    distinct_value_count_upper_bound: u64::try_from(
                        source.batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
                    )
                    .ok(),
                    row_multiplicity_upper_bound: Some(1),
                    max_single_value_length_bytes: Some(max_single_value_length_bytes),
                    lineages: BTreeSet::from([StringPayloadLineage::Literal { node_id, column_index }]),
                },
            );
        }
    }
    output
}

fn constrained_filter_evidence(
    predicate: &Expr,
    mut output: PlanColumnEvidence,
    input_rows: Option<u64>,
    node_id: NodeId,
) -> PlanColumnEvidence {
    if let Expr::Binary { left, op: BinaryOp::And, right, .. } = predicate {
        return constrained_filter_evidence(
            right,
            constrained_filter_evidence(left, output, input_rows, node_id),
            input_rows,
            node_id,
        );
    }
    let Expr::InList { expr, list, negated: false } = predicate else {
        return output;
    };
    let Some(column) = alias_wrapped_column(expr) else {
        return output;
    };
    let Some(bound) = output.string_payload_bounds.get_mut(column.name()) else {
        return output;
    };
    let Some(values) = list.iter().map(literal_string_value).collect::<Option<BTreeSet<_>>>() else {
        return output;
    };
    let Some(total) = values.iter().try_fold(0_u64, |total, value| total.checked_add(u64::try_from(value.len()).ok()?))
    else {
        return output;
    };
    let max = values.iter().filter_map(|value| u64::try_from(value.len()).ok()).max().unwrap_or(0);
    let occurrence = input_rows.and_then(|rows| rows.checked_mul(max));
    bound.total_payload_upper_bound_bytes = match (bound.total_payload_upper_bound_bytes, occurrence) {
        (Some(existing), Some(constrained)) => Some(existing.min(constrained)),
        (existing, constrained) => existing.or(constrained),
    };
    bound.distinct_payload_upper_bound_bytes =
        Some(bound.distinct_payload_upper_bound_bytes.map_or(total, |existing| existing.min(total)));
    let count = u64::try_from(values.len()).ok();
    bound.distinct_value_count_upper_bound = match (bound.distinct_value_count_upper_bound, count) {
        (Some(existing), Some(constrained)) => Some(existing.min(constrained)),
        (existing, constrained) => existing.or(constrained),
    };
    bound.max_single_value_length_bytes = Some(bound.max_single_value_length_bytes.map_or(max, |value| value.min(max)));
    bound.lineages.insert(StringPayloadLineage::Computed { node_id, column_index: 0 });
    output
}

fn literal_string_value(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Literal(LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value)) => Some(value),
        Expr::Alias { expr, .. } => literal_string_value(expr),
        Expr::Cast { expr, target_type } if is_string(target_type) => literal_string_value(expr),
        _ => None,
    }
}

fn computed_string_payload_bound(
    expr: &Expr,
    input: &PlanColumnEvidence,
    rows: u64,
    lineage: StringPayloadLineage,
) -> Option<StringPayloadBound> {
    if let Some(length) = literal_string_value_length(expr) {
        return Some(StringPayloadBound {
            total_payload_upper_bound_bytes: Some(rows.checked_mul(length)?),
            distinct_payload_upper_bound_bytes: Some(length),
            distinct_value_count_upper_bound: Some(1),
            row_multiplicity_upper_bound: Some(1),
            max_single_value_length_bytes: Some(length),
            lineages: BTreeSet::from([lineage]),
        });
    }

    match expr {
        Expr::Column(column) => input
            .string_payload_bound(column.name())
            .filter(|bound| bound.max_single_value_length_bytes.is_some())
            .cloned(),
        Expr::Alias { expr, .. } => computed_string_payload_bound(expr, input, rows, lineage),
        Expr::Case { branches, else_expr } => {
            let mut bounds = branches
                .iter()
                .map(|branch| computed_string_payload_bound(&branch.then, input, rows, lineage.clone()))
                .chain(std::iter::once(computed_string_payload_bound(else_expr, input, rows, lineage.clone())))
                .collect::<Option<Vec<_>>>()?;
            let max_single_value_length_bytes = bounds
                .iter()
                .map(|bound| bound.max_single_value_length_bytes)
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .max()
                .unwrap_or(0);
            let distinct_payload_upper_bound_bytes = bounds
                .iter()
                .map(|bound| bound.distinct_payload_upper_bound_bytes)
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .try_fold(0_u64, u64::checked_add);
            let distinct_value_count_upper_bound = bounds
                .iter()
                .map(|bound| bound.distinct_value_count_upper_bound)
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .try_fold(0_u64, u64::checked_add);
            let mut lineages = BTreeSet::from([lineage]);
            for bound in bounds.drain(..) {
                lineages.extend(bound.lineages);
            }
            Some(StringPayloadBound {
                total_payload_upper_bound_bytes: Some(rows.checked_mul(max_single_value_length_bytes)?),
                distinct_payload_upper_bound_bytes,
                distinct_value_count_upper_bound,
                row_multiplicity_upper_bound: Some(1),
                max_single_value_length_bytes: Some(max_single_value_length_bytes),
                lineages,
            })
        },
        _ => None,
    }
}

fn literal_string_value_length(expr: &Expr) -> Option<u64> {
    match expr {
        Expr::Literal(LiteralValue::Utf8(value) | LiteralValue::LargeUtf8(value)) => {
            Some(u64::try_from(value.len()).ok()?)
        },
        Expr::Literal(LiteralValue::Null(data_type)) if is_string(data_type) => Some(0),
        Expr::Alias { expr, .. } => literal_string_value_length(expr),
        Expr::Cast { expr, target_type } if is_string(target_type) => literal_string_value_length(expr),
        _ => None,
    }
}

fn row_independent_literal_string_payload_bound(
    expr: &Expr,
    lineage: StringPayloadLineage,
) -> Option<StringPayloadBound> {
    let length = literal_string_value_length(expr)?;
    Some(StringPayloadBound {
        total_payload_upper_bound_bytes: None,
        distinct_payload_upper_bound_bytes: Some(length),
        distinct_value_count_upper_bound: Some(1),
        row_multiplicity_upper_bound: Some(1),
        max_single_value_length_bytes: Some(length),
        lineages: BTreeSet::from([lineage]),
    })
}

fn union_evidence(plan: &QueryPlan, inputs: &[NodeId], analysis: &PlanColumnEvidenceAnalysis) -> PlanColumnEvidence {
    let mut output = PlanColumnEvidence::default();
    let Some(first) = inputs.first().and_then(|input| plan.node(*input)) else {
        return output;
    };
    for (position, output_field) in first.output_schema().fields().iter().enumerate() {
        if !is_string(output_field.data_type()) {
            continue;
        }
        let bounds = inputs
            .iter()
            .map(|input| {
                let node = plan.node(*input)?;
                let field = node.output_schema().fields().get(position)?;
                analysis.by_node.get(input)?.string_payload_bound(field.name())
            })
            .collect::<Option<Vec<_>>>();
        if let Some(bounds) = bounds
            && let Some(bound) = StringPayloadBound::for_union(bounds)
        {
            output.string_payload_bounds.insert(output_field.name().clone(), bound);
        }
    }
    output
}

fn join_evidence(
    join: &JoinNode,
    left: &PlanColumnEvidence,
    right: &PlanColumnEvidence,
    left_rows: Option<u64>,
    right_rows: Option<u64>,
) -> Result<PlanColumnEvidence> {
    let mut output = PlanColumnEvidence::default();
    let bounded_side = [JoinSide::Left, JoinSide::Right]
        .into_iter()
        .find(|side| join.spec.has_at_most_one_match_per_bounded_row(*side));
    let left_is_unique = side_is_unique_on_join_keys(join, JoinSide::Left, left, right);
    let right_is_unique = side_is_unique_on_join_keys(join, JoinSide::Right, left, right);
    let bounded_is_unique = bounded_side.is_some_and(|side| side_is_unique_on_join_keys(join, side, left, right));
    for projected in &join.spec.output_projection {
        let input = match projected.side {
            JoinSide::Left => left,
            JoinSide::Right => right,
        };
        let mut bound = input.string_payload_bound(projected.column.name()).cloned();
        let input_rows_are_not_duplicated = Some(projected.side) == bounded_side
            || matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
                && projected.side == JoinSide::Left
            || match projected.side {
                JoinSide::Left => right_is_unique,
                JoinSide::Right => left_is_unique,
            };
        if !input_rows_are_not_duplicated
            && !bounded_is_unique
            && !join.spec.preserves_total_payload_bound_for_output(&projected.output_name)
        {
            if join.spec.join_type == JoinType::Inner
                && let Some(bounded_side) = bounded_side
                && let Some(equivalent) = equivalent_bounded_join_key_bound(join, projected, left, right, bounded_side)
                && equivalent.total_payload_upper_bound_bytes.is_some()
            {
                if let Some(bound) = &mut bound {
                    bound.total_payload_upper_bound_bytes = equivalent.total_payload_upper_bound_bytes;
                    bound.lineages.extend(equivalent.lineages.iter().cloned());
                } else {
                    bound = Some(equivalent.clone());
                }
            } else if join.spec.join_type == JoinType::Left
                && projected.side == JoinSide::Left
                && left_is_unique
                && let Some(equivalent) =
                    equivalent_bounded_join_key_bound(join, projected, left, right, JoinSide::Right)
                && let Some(bound) = &mut bound
            {
                bound.total_payload_upper_bound_bytes =
                    match (bound.total_payload_upper_bound_bytes, equivalent.total_payload_upper_bound_bytes) {
                        (Some(left_total), Some(right_total)) => {
                            Some(left_total.checked_add(right_total).ok_or_else(|| {
                                payload_row_bound_overflow("left_join_unique_key_payload")
                                    .with_fact("column", &projected.output_name)
                            })?)
                        },
                        _ => None,
                    };
                bound.max_single_value_length_bytes =
                    match (bound.max_single_value_length_bytes, equivalent.max_single_value_length_bytes) {
                        (Some(left_max), Some(right_max)) => Some(left_max.max(right_max)),
                        (left_max, right_max) => left_max.or(right_max),
                    };
                bound.lineages.extend(equivalent.lineages.iter().cloned());
            } else if let Some(bound) = &mut bound {
                let multiplicity = match projected.side {
                    JoinSide::Left => right_rows,
                    JoinSide::Right => left_rows,
                };
                let scaled_multiplicity = match (bound.row_multiplicity_upper_bound, multiplicity) {
                    (Some(lhs), Some(rhs)) => Some(lhs.checked_mul(rhs.max(1)).ok_or_else(|| {
                        payload_row_bound_overflow("join_payload")
                            .with_fact("column", &projected.output_name)
                            .with_fact("lhs_multiplicity", lhs)
                            .with_fact("rhs_multiplicity", rhs)
                    })?),
                    _ => None,
                };
                bound.total_payload_upper_bound_bytes = match (bound.total_payload_upper_bound_bytes, multiplicity) {
                    (Some(total), Some(multiplicity)) => {
                        Some(total.checked_mul(multiplicity.max(1)).ok_or_else(|| {
                            let mut error = payload_row_bound_overflow("join_payload")
                                .with_fact("column", &projected.output_name)
                                .with_fact("input_row_multiplicity", multiplicity);
                            if let Some(lhs) = bound.row_multiplicity_upper_bound {
                                error = error.with_fact("lhs_multiplicity", lhs);
                            }
                            error.with_fact("rhs_multiplicity", multiplicity)
                        })?)
                    },
                    _ => None,
                };
                bound.row_multiplicity_upper_bound = scaled_multiplicity;
            }
        }
        if let Some(bound) = bound
            && (bound.total_payload_upper_bound_bytes.is_some() || bound.max_single_value_length_bytes.is_some())
        {
            output.string_payload_bounds.insert(projected.output_name.clone(), bound);
        }
    }

    let left_mapping = join
        .spec
        .output_projection
        .iter()
        .filter(|column| column.side == JoinSide::Left)
        .map(|column| (column.column.name().to_owned(), column.output_name.clone()))
        .collect::<Vec<_>>();
    let right_mapping = join
        .spec
        .output_projection
        .iter()
        .filter(|column| column.side == JoinSide::Right)
        .map(|column| (column.column.name().to_owned(), column.output_name.clone()))
        .collect::<Vec<_>>();
    output.equivalent_columns.extend(remap_equivalent_columns(left, left_mapping));
    output.equivalent_columns.extend(remap_equivalent_columns(right, right_mapping));
    Ok(output)
}

fn payload_row_bound_overflow(stage: &'static str) -> QueryError {
    QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.admission.string_payload",
        "string payload row multiplicity overflows u64",
    )
    .with_fact("reason", "string_payload_row_multiplicity_overflow")
    .with_fact("stage", stage)
}

fn equivalent_bounded_join_key_bound<'a>(
    join: &JoinNode,
    projected: &JoinOutputColumn,
    left: &'a PlanColumnEvidence,
    right: &'a PlanColumnEvidence,
    bounded_side: JoinSide,
) -> Option<&'a StringPayloadBound> {
    join.spec.keys.iter().find_map(|key| match (projected.side, bounded_side) {
        (JoinSide::Left, JoinSide::Right) if left.columns_are_equivalent(key.left.name(), projected.column.name()) => {
            right.string_payload_bound(key.right.name())
        },
        (JoinSide::Right, JoinSide::Left)
            if right.columns_are_equivalent(key.right.name(), projected.column.name()) =>
        {
            left.string_payload_bound(key.left.name())
        },
        _ => None,
    })
}

fn side_is_unique_on_join_keys(
    join: &JoinNode,
    side: JoinSide,
    left: &PlanColumnEvidence,
    right: &PlanColumnEvidence,
) -> bool {
    let columns = join
        .spec
        .keys
        .iter()
        .map(|key| match side {
            JoinSide::Left => key.left.name().to_owned(),
            JoinSide::Right => key.right.name().to_owned(),
        })
        .collect();
    match side {
        JoinSide::Left => left.is_unique_on(&columns),
        JoinSide::Right => right.is_unique_on(&columns),
    }
}

fn remap_unique_key_sets(
    input: &[BTreeSet<String>],
    mappings: impl IntoIterator<Item = (String, String)>,
) -> Vec<BTreeSet<String>> {
    let mappings = mappings.into_iter().fold(BTreeMap::<_, Vec<_>>::new(), |mut mappings, (input, output)| {
        mappings.entry(input).or_default().push(output);
        mappings
    });
    input
        .iter()
        .flat_map(|key| {
            key.iter()
                .try_fold(vec![BTreeSet::new()], |keys, column| {
                    let outputs = mappings.get(column)?;
                    Some(
                        keys.into_iter()
                            .flat_map(|key| {
                                outputs.iter().map(move |output| {
                                    let mut key = key.clone();
                                    key.insert(output.clone());
                                    key
                                })
                            })
                            .collect(),
                    )
                })
                .unwrap_or_default()
        })
        .collect()
}

#[doc(hidden)]
pub fn partial_plan_node_is_unique_on(nodes: &[QueryNode], node_id: NodeId, columns: &BTreeSet<String>) -> bool {
    unique_keys_by_node(nodes).get(node_id.index()).is_some_and(|keys| keys.iter().any(|key| key.is_subset(columns)))
}

fn unique_keys_by_node(nodes: &[QueryNode]) -> Vec<Vec<BTreeSet<String>>> {
    let mut by_node = Vec::with_capacity(nodes.len());
    for node in nodes {
        let input_keys = |input: NodeId| by_node.get(input.index()).cloned().unwrap_or_default();
        let keys = match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => source.unique_key_columns.clone(),
                SourceNode::GpuResident(source) => source.unique_key_columns.iter().cloned().collect(),
                SourceNode::LiteralBatch(_) => Vec::new(),
            },
            PlanNode::Filter(filter) => input_keys(filter.input),
            PlanNode::Sort(sort) => input_keys(sort.input),
            PlanNode::Limit(limit) => input_keys(limit.input),
            PlanNode::Projection(projection) => {
                let input_schema = nodes.get(projection.input.index()).map(|node| node.output_schema());
                let mappings = projection.outputs.iter().filter_map(|output| {
                    Some((unique_source_column(&output.expr, input_schema)?.to_owned(), output.field.name().clone()))
                });
                remap_unique_key_sets(&input_keys(projection.input), mappings)
            },
            PlanNode::Aggregate(aggregate)
                if aggregate.spec.grouping_sets.is_none() && !aggregate.spec.group_exprs.is_empty() =>
            {
                vec![
                    node.output_schema()
                        .fields()
                        .iter()
                        .take(aggregate.spec.group_exprs.len())
                        .map(|field| field.name().clone())
                        .collect(),
                ]
            },
            PlanNode::Aggregate(_) => Vec::new(),
            PlanNode::Window(window) => {
                let mut keys = input_keys(window.input);
                if window.spec.partition_by.is_empty() {
                    keys.extend(window.spec.functions.iter().filter_map(|function| match function {
                        WindowFunction::RowNumber { alias, .. } => Some(BTreeSet::from([alias.clone()])),
                        _ => None,
                    }));
                }
                keys
            },
            PlanNode::Join(join) => join_unique_keys(join, &input_keys(join.inputs[0]), &input_keys(join.inputs[1])),
            PlanNode::GraphAlgorithm(graph) => {
                let key = graph.spec.unique_output_key_columns();
                if !key.is_empty() && key.iter().all(|column| node.output_schema().field_with_name(column).is_ok()) {
                    vec![key.iter().map(|column| (*column).to_owned()).collect()]
                } else {
                    Vec::new()
                }
            },
            PlanNode::Union(_) | PlanNode::EdgeNormalize(_) | PlanNode::VectorAlgorithm(_) => Vec::new(),
        };
        by_node.push(keys);
    }
    by_node
}

fn join_unique_keys(
    join: &JoinNode,
    left_keys: &[BTreeSet<String>],
    right_keys: &[BTreeSet<String>],
) -> Vec<BTreeSet<String>> {
    let side_is_unique = |side: JoinSide, keys: &[BTreeSet<String>]| {
        let columns = join
            .spec
            .keys
            .iter()
            .map(|key| match side {
                JoinSide::Left => key.left.name().to_owned(),
                JoinSide::Right => key.right.name().to_owned(),
            })
            .collect::<BTreeSet<_>>();
        keys.iter().any(|key| key.is_subset(&columns))
    };
    let bounded_side = [JoinSide::Left, JoinSide::Right]
        .into_iter()
        .find(|side| join.spec.has_at_most_one_match_per_bounded_row(*side));
    let left_is_unique = side_is_unique(JoinSide::Left, left_keys);
    let right_is_unique = side_is_unique(JoinSide::Right, right_keys);
    let bounded_is_unique = bounded_side.is_some_and(|side| match side {
        JoinSide::Left => left_is_unique,
        JoinSide::Right => right_is_unique,
    });
    let mappings = |side| {
        join.spec
            .output_projection
            .iter()
            .filter(move |column| column.side == side)
            .map(|column| (column.column.name().to_owned(), column.output_name.clone()))
    };
    let mut output = Vec::new();
    if bounded_side == Some(JoinSide::Left)
        || bounded_is_unique
        || right_is_unique
        || matches!(join.spec.join_type, JoinType::LeftSemi | JoinType::LeftAnti)
    {
        output.extend(remap_unique_key_sets(left_keys, mappings(JoinSide::Left)));
    }
    if bounded_side == Some(JoinSide::Right) || bounded_is_unique || left_is_unique {
        output.extend(remap_unique_key_sets(right_keys, mappings(JoinSide::Right)));
    }
    output
}

fn remap_equivalent_columns(
    input: &PlanColumnEvidence,
    mappings: impl IntoIterator<Item = (String, String)>,
) -> Vec<BTreeSet<String>> {
    let mappings = mappings.into_iter().fold(BTreeMap::<_, Vec<_>>::new(), |mut mappings, (input, output)| {
        mappings.entry(input).or_default().push(output);
        mappings
    });
    mappings
        .keys()
        .filter_map(|input_column| {
            let outputs = mappings
                .iter()
                .filter(|(candidate, _)| input.columns_are_equivalent(input_column, candidate))
                .flat_map(|(_, outputs)| outputs.iter().cloned())
                .collect::<BTreeSet<_>>();
            (outputs.len() > 1).then_some(outputs)
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[doc(hidden)]
pub fn identity_source_column<'a>(expr: &'a Expr, input_schema: Option<&SchemaRef>) -> Option<&'a str> {
    if let Some(column) = alias_wrapped_column(expr) {
        return Some(column.name());
    }
    match expr {
        Expr::Alias { expr, .. } => identity_source_column(expr, input_schema),
        Expr::Cast { expr, target_type } => {
            let column = identity_source_column(expr, input_schema)?;
            let source_type = input_schema?.field_with_name(column).ok()?.data_type();
            representation_cast_preserves_identity(source_type, target_type).then_some(column)
        },
        _ => None,
    }
}

fn unique_source_column<'a>(expr: &'a Expr, input_schema: Option<&SchemaRef>) -> Option<&'a str> {
    if let Some(column) = identity_source_column(expr, input_schema) {
        return Some(column);
    }
    let Expr::Binary { left, op: BinaryOp::Add, right, .. } = expr else {
        return None;
    };
    let (column, literal) = match (left.as_ref(), right.as_ref()) {
        (Expr::Column(column), Expr::Literal(literal)) | (Expr::Literal(literal), Expr::Column(column)) => {
            (column, literal)
        },
        _ => return None,
    };
    let field = input_schema?.field_with_name(column.name()).ok()?;
    let source_type = field.data_type();
    (!field.is_nullable()
        && source_type.is_integer()
        && literal.data_type() == *source_type
        && matches!(
            literal,
            LiteralValue::Int8(0)
                | LiteralValue::Int16(0)
                | LiteralValue::Int32(0)
                | LiteralValue::Int64(0)
                | LiteralValue::UInt8(0)
                | LiteralValue::UInt16(0)
                | LiteralValue::UInt32(0)
                | LiteralValue::UInt64(0)
        ))
    .then_some(column.name())
}

fn is_string(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use arrow::{
        array::Int32Array,
        datatypes::{Field, Schema},
        record_batch::RecordBatch,
    };

    use super::*;
    use crate::{expr::ColumnRef, source::GpuResidentSource};

    #[test]
    fn builder_answers_partial_uniqueness_without_finishing_a_temporary_plan() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(vec![Expr::column("sourceId")], Vec::new()),
            Arc::new(Schema::new(vec![Field::new("sourceId", DataType::Utf8, false)])),
        );

        assert!(builder.output_is_unique_on(aggregate, &BTreeSet::from(["sourceId".to_owned()])).unwrap());
        assert!(!builder.output_is_unique_on(source, &BTreeSet::from(["targetId".to_owned()])).unwrap());
    }
    use crate::{
        expr::{AggregateExpr, AggregateSpec, CaseWhen, LiteralValue, SortExpr},
        parquet_stats::{ParquetVariableWidthEncodingProof, ParquetVariableWidthPayloadEvidence},
        plan::{JoinKey, JoinSpec, ProjectionOutput, QueryPlanBuilder, WindowSpec},
        source::LocalParquetFile,
    };

    fn production_endpoint_source() -> LocalParquetSource {
        let schema = Arc::new(Schema::new(vec![
            Field::new("sourceId", DataType::Utf8, false),
            Field::new("targetId", DataType::Utf8, false),
        ]));
        let evidence = BTreeMap::from([
            (
                0,
                ParquetVariableWidthPayloadEvidence::new(
                    0,
                    (0..36).collect(),
                    1_982_672_200,
                    ParquetVariableWidthEncodingProof::PlainAndDictionary,
                ),
            ),
            (
                1,
                ParquetVariableWidthPayloadEvidence::new(
                    1,
                    (0..36).collect(),
                    2_081_894_503,
                    ParquetVariableWidthEncodingProof::PlainAndDictionary,
                ),
            ),
        ]);
        let file = LocalParquetFile::new("/tmp/production-string-edges.parquet")
            .with_row_group_rows(vec![1; 36])
            .with_variable_width_payload_evidence(evidence);
        LocalParquetSource::from_files(vec![file], schema, vec!["sourceId".to_owned(), "targetId".to_owned()])
    }

    #[test]
    fn endpoint_payload_bound_survives_codebook_union_grouping_and_dense_id() {
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
        let subject = builder.add_projection(
            source,
            vec![ProjectionOutput::new(Expr::column("sourceId"), Field::new("vertex", DataType::Utf8, false))],
        );
        let object = builder.add_projection(
            source,
            vec![ProjectionOutput::new(Expr::column("targetId"), Field::new("vertex", DataType::Utf8, false))],
        );
        let vertex_schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Utf8, false)]));
        let endpoints = builder.add_union(vec![subject, object], Arc::clone(&vertex_schema));
        let codebook = builder.add_aggregate(
            endpoints,
            AggregateSpec::new(vec![Expr::column("vertex")], Vec::new()),
            Arc::clone(&vertex_schema),
        );
        let dense_schema = Arc::new(Schema::new(vec![
            Field::new("vertex", DataType::Utf8, false),
            Field::new("__vertex_id", DataType::Int64, false),
        ]));
        let dense = builder.add_window(
            codebook,
            WindowSpec::new(
                Vec::new(),
                vec![SortExpr::new(Expr::column("vertex"), true, false)],
                vec![WindowFunction::row_number_i64("__vertex_id")],
            ),
            dense_schema,
        );
        let plan = builder.finish_gpu_resident(dense);
        let output = PlanColumnEvidenceAnalysis::for_plan(&plan, &HashMap::new())
            .expect("codebook payload evidence analysis must succeed")
            .root(&plan);

        let bound = output.string_payload_bound("vertex").expect("codebook payload must retain footer lineage");
        assert_eq!(bound.total_payload_upper_bound_bytes, Some(4_064_566_703));
        assert_eq!(bound.max_single_value_length_bytes, None);
        assert_eq!(bound.lineages.len(), 2);
        assert!(output.is_unique_on(&BTreeSet::from(["__vertex_id".to_owned()])));
    }

    #[test]
    fn grouped_string_min_max_outputs_retain_the_input_payload_bound() {
        let source = production_endpoint_source();
        let output_schema = Arc::new(Schema::new(vec![
            Field::new("sourceId", DataType::Utf8, false),
            Field::new("object_min", DataType::Utf8, true),
            Field::new("object_max", DataType::Utf8, true),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(source));
        let aggregate = builder.add_aggregate(
            source,
            AggregateSpec::new(
                vec![Expr::column("sourceId")],
                vec![
                    AggregateExpr::new(AggregateFunction::Min, Expr::column("targetId"), "object_min"),
                    AggregateExpr::new(AggregateFunction::Max, Expr::column("targetId"), "object_max"),
                ],
            ),
            output_schema,
        );
        let plan = builder.finish_gpu_resident(aggregate);
        let output = PlanColumnEvidenceAnalysis::for_plan(&plan, &HashMap::new())
            .expect("grouped MIN/MAX payload evidence must resolve")
            .root(&plan);

        for column in ["object_min", "object_max"] {
            let bound = output.string_payload_bound(column).expect("MIN/MAX must retain source payload evidence");
            assert_eq!(bound.total_payload_upper_bound_bytes, Some(2_081_894_503));
            assert_eq!(bound.lineages.len(), 1);
        }
    }

    #[test]
    fn unbounded_path_index_addition_does_not_claim_injective_join_keys() {
        let path_schema = Arc::new(Schema::new(vec![
            Field::new("path_index", DataType::Int64, false),
            Field::new("vertex", DataType::Utf8, false),
        ]));
        let output_schema = Arc::new(Schema::new(vec![
            Field::new("object_id", DataType::Utf8, false),
            Field::new("subject_id", DataType::Utf8, false),
        ]));
        let mut builder = QueryPlanBuilder::new();
        let current = builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new("current", Arc::clone(&path_schema)).with_row_count_upper_bound(49),
        ));
        let previous = builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new("previous", Arc::clone(&path_schema)).with_row_count_upper_bound(49),
        ));
        let previous = builder.add_projection(
            previous,
            vec![
                ProjectionOutput::new(
                    Expr::binary(Expr::column("path_index"), BinaryOp::Add, Expr::literal(LiteralValue::Int64(1))),
                    Field::new("next_path_index", DataType::Int64, false),
                ),
                ProjectionOutput::new(Expr::column("vertex"), Field::new("vertex", DataType::Utf8, false)),
            ],
        );
        let join = builder.add_join(
            current,
            previous,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "path_index", "path_index"),
                    ColumnRef::input(1, "next_path_index", "next_path_index"),
                )],
            )
            .with_output_projection(vec![
                JoinOutputColumn::left(ColumnRef::input(0, "vertex", "vertex"), "object_id"),
                JoinOutputColumn::right(ColumnRef::input(1, "vertex", "vertex"), "subject_id"),
            ]),
            output_schema,
        );
        let plan = builder.finish_with_host_sink(join);
        let path_evidence =
            PlanColumnEvidence { unique_keys: vec![BTreeSet::from(["path_index".to_owned()])], ..Default::default() };
        let resident_outputs =
            HashMap::from([("current".to_owned(), path_evidence.clone()), ("previous".to_owned(), path_evidence)]);
        let output = PlanColumnEvidenceAnalysis::for_plan(&plan, &resident_outputs)
            .expect("bounded path join evidence must resolve")
            .root(&plan);

        assert!(!output.is_unique_on(&BTreeSet::from(["object_id".to_owned()])));
        assert!(!output.is_unique_on(&BTreeSet::from(["subject_id".to_owned()])));
    }

    #[test]
    fn integer_addition_requires_non_nullable_identity_without_overflow() {
        let non_nullable = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let nullable = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, true)]));
        let add =
            |literal| Expr::binary(Expr::column("value"), BinaryOp::Add, Expr::literal(LiteralValue::Int64(literal)));
        assert_eq!(unique_source_column(&add(0), Some(&non_nullable)), Some("value"));
        assert_eq!(unique_source_column(&add(1), Some(&non_nullable)), None);
        assert_eq!(unique_source_column(&add(0), Some(&nullable)), None);
    }

    #[test]
    fn payload_lineage_stops_at_string_transform_and_scales_for_finite_join_multiplicity() {
        let mut transform_builder = QueryPlanBuilder::new();
        let source = transform_builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
        let transformed = transform_builder.add_projection(
            source,
            vec![ProjectionOutput::new(
                Expr::substring(Expr::column("sourceId"), 1, Some(8)),
                Field::new("sourceId", DataType::Utf8, false),
            )],
        );
        let transform_plan = transform_builder.finish_gpu_resident(transformed);
        assert!(
            PlanColumnEvidenceAnalysis::for_plan(&transform_plan, &HashMap::new())
                .expect("transform payload evidence analysis must succeed")
                .root(&transform_plan)
                .string_payload_bound("sourceId")
                .is_none()
        );

        let mut join_builder = QueryPlanBuilder::new();
        let left = join_builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
        let right = join_builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
        let output_schema = Arc::new(Schema::new(vec![Field::new("sourceId", DataType::Utf8, false)]));
        let spec = JoinSpec::new(
            JoinType::Inner,
            vec![JoinKey::new(
                ColumnRef::input(0, "sourceId", "sourceId"),
                ColumnRef::input(1, "sourceId", "sourceId"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            ColumnRef::input(0, "sourceId", "sourceId"),
            "sourceId",
        )]);
        let join = join_builder.add_join(left, right, spec, output_schema);
        let join_plan = join_builder.finish_gpu_resident(join);
        let PlanNode::Join(join) = &join_plan.node(join_plan.root_id()).unwrap().kind else { unreachable!() };
        let join_bound = PlanColumnEvidenceAnalysis::for_plan(&join_plan, &HashMap::new())
            .expect("join payload evidence analysis must succeed")
            .root(&join_plan)
            .string_payload_bound("sourceId")
            .cloned()
            .expect("finite input rows must scale the source total through a duplication-capable join");
        assert_eq!(join_bound.total_payload_upper_bound_bytes, Some(1_982_672_200 * 36));
        assert_eq!(join_bound.max_single_value_length_bytes, None);
        assert_eq!(join_bound.row_multiplicity_upper_bound, Some(36));

        let mut unknown_multiplicity = join_bound.clone();
        unknown_multiplicity.row_multiplicity_upper_bound = None;
        unknown_multiplicity.total_payload_upper_bound_bytes = Some(128);
        let unknown_evidence = PlanColumnEvidence {
            string_payload_bounds: BTreeMap::from([("sourceId".to_owned(), unknown_multiplicity)]),
            ..Default::default()
        };
        let unknown_output = join_evidence(join, &unknown_evidence, &unknown_evidence, Some(2), Some(2))
            .expect("unknown input multiplicity must remain unknown after join scaling");
        assert_eq!(unknown_output.string_payload_bound("sourceId").unwrap().row_multiplicity_upper_bound, None);

        let mut overflowing = join_bound;
        overflowing.total_payload_upper_bound_bytes = Some(u64::MAX);
        let evidence = PlanColumnEvidence {
            string_payload_bounds: BTreeMap::from([("sourceId".to_owned(), overflowing)]),
            ..Default::default()
        };
        let error = join_evidence(join, &evidence, &evidence, Some(2), Some(2))
            .expect_err("finite row counts must not turn overflowing payload multiplication into a bound");
        assert_eq!(error.fact_value("reason"), Some("string_payload_row_multiplicity_overflow"));
        assert_eq!(error.fact_value("column"), Some("sourceId"));
        assert_eq!(error.fact_value("input_row_multiplicity"), Some("2"));
        assert_eq!(error.fact_value("lhs_multiplicity"), Some("36"));
        assert_eq!(error.fact_value("rhs_multiplicity"), Some("2"));

        let mut qualified = join.clone();
        qualified.spec = qualified.spec.with_at_most_one_match_per_bounded_row(JoinSide::Right);
        let mut bounded_key = evidence.clone();
        bounded_key.string_payload_bounds.get_mut("sourceId").unwrap().total_payload_upper_bound_bytes = Some(128);
        let qualified = join_evidence(&qualified, &evidence, &bounded_key, Some(2), Some(2))
            .expect("an equivalent bounded key must be used before an unnecessary overflowing multiplication");
        assert_eq!(qualified.string_payload_bound("sourceId").unwrap().total_payload_upper_bound_bytes, Some(128));
    }

    #[test]
    fn bounded_literal_and_case_projection_produce_exact_string_payload_evidence() {
        let input_schema = Arc::new(Schema::new(vec![Field::new("selector", DataType::Int32, false)]));
        let input_batch =
            RecordBatch::try_new(Arc::clone(&input_schema), vec![Arc::new(Int32Array::from(vec![0, 1, 2]))]).unwrap();
        let mut builder = QueryPlanBuilder::new();
        let source =
            builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(input_schema, vec![input_batch])));
        let projection = builder.add_projection(
            source,
            vec![
                ProjectionOutput::new(
                    Expr::literal(LiteralValue::Utf8("seed".to_owned())),
                    Field::new("constant_seed", DataType::Utf8, false),
                ),
                ProjectionOutput::new(
                    Expr::case(
                        vec![CaseWhen::new(
                            Expr::binary(Expr::column("selector"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(0))),
                            Expr::literal(LiteralValue::Utf8("long".to_owned())),
                        )],
                        Expr::literal(LiteralValue::Utf8("ok".to_owned())),
                    ),
                    Field::new("computed_seed", DataType::Utf8, false),
                ),
            ],
        );
        let plan = builder.finish_gpu_resident(projection);
        let output = PlanColumnEvidenceAnalysis::for_plan(&plan, &HashMap::new()).unwrap().root(&plan);

        let constant = output.string_payload_bound("constant_seed").unwrap();
        assert_eq!(constant.total_payload_upper_bound_bytes, Some(12));
        assert_eq!(constant.max_single_value_length_bytes, Some(4));
        let computed = output.string_payload_bound("computed_seed").unwrap();
        assert_eq!(computed.total_payload_upper_bound_bytes, Some(12));
        assert_eq!(computed.max_single_value_length_bytes, Some(4));
        assert!(computed.lineages.contains(&StringPayloadLineage::Computed { node_id: projection, column_index: 1 }));
    }

    #[test]
    fn computed_string_payload_evidence_requires_a_hard_row_bound_and_complete_case_arms() {
        let unknown_source = LocalParquetSource::new(
            "/tmp/unknown-rows.parquet",
            Arc::new(Schema::new(vec![Field::new("selector", DataType::Int32, false)])),
            vec!["selector".to_owned()],
        );
        let mut unknown_builder = QueryPlanBuilder::new();
        let source = unknown_builder.add_source(SourceNode::LocalParquet(unknown_source));
        let projection = unknown_builder.add_projection(
            source,
            vec![ProjectionOutput::new(
                Expr::literal(LiteralValue::Utf8("seed".to_owned())),
                Field::new("seed", DataType::Utf8, false),
            )],
        );
        let unknown_plan = unknown_builder.finish_gpu_resident(projection);
        let unknown_analysis = PlanColumnEvidenceAnalysis::for_plan(&unknown_plan, &HashMap::new()).unwrap();
        let unknown_root = unknown_analysis.root(&unknown_plan);
        let unknown = unknown_root.string_payload_bound("seed").unwrap();
        assert_eq!(unknown.total_payload_upper_bound_bytes, None);
        assert_eq!(unknown.max_single_value_length_bytes, Some(4));

        let input_schema = Arc::new(Schema::new(vec![Field::new("selector", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(Arc::clone(&input_schema), vec![Arc::new(Int32Array::from(vec![0]))]).unwrap();
        let mut incomplete_builder = QueryPlanBuilder::new();
        let source =
            incomplete_builder.add_source(SourceNode::LiteralBatch(LiteralBatchSource::new(input_schema, vec![batch])));
        let projection = incomplete_builder.add_projection(
            source,
            vec![ProjectionOutput::new(
                Expr::case(
                    vec![CaseWhen::new(
                        Expr::literal(LiteralValue::Boolean(true)),
                        Expr::literal(LiteralValue::Utf8("bounded".to_owned())),
                    )],
                    Expr::substring(Expr::literal(LiteralValue::Utf8("unsupported".to_owned())), 1, Some(2)),
                ),
                Field::new("seed", DataType::Utf8, false),
            )],
        );
        let incomplete_plan = incomplete_builder.finish_gpu_resident(projection);
        assert!(
            PlanColumnEvidenceAnalysis::for_plan(&incomplete_plan, &HashMap::new())
                .unwrap()
                .root(&incomplete_plan)
                .string_payload_bound("seed")
                .is_none()
        );
    }

    #[test]
    fn bounded_join_retains_total_payload_only_with_a_multiplicity_proof() {
        let mut encode_builder = QueryPlanBuilder::new();
        let relation = encode_builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
        let codebook = encode_builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
        let output_schema = Arc::new(Schema::new(vec![Field::new("targetId", DataType::Utf8, false)]));
        let encode = encode_builder.add_join(
            relation,
            codebook,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "sourceId", "sourceId"),
                    ColumnRef::input(1, "sourceId", "sourceId"),
                )],
            )
            .with_at_most_one_match_per_bounded_row(JoinSide::Left)
            .with_output_projection(vec![JoinOutputColumn::left(
                ColumnRef::input(0, "targetId", "targetId"),
                "targetId",
            )]),
            Arc::clone(&output_schema),
        );
        let encode_plan = encode_builder.finish_gpu_resident(encode);
        let encode_bound = PlanColumnEvidenceAnalysis::for_plan(&encode_plan, &HashMap::new())
            .expect("encoding payload evidence analysis must succeed")
            .root(&encode_plan)
            .string_payload_bound("targetId")
            .cloned()
            .expect("encoding cannot duplicate rows from its bounded relation");
        assert_eq!(encode_bound.total_payload_upper_bound_bytes, Some(2_081_894_503));

        let mut decode_builder = QueryPlanBuilder::new();
        let relation = decode_builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
        let unique_relation = decode_builder.add_aggregate(
            relation,
            AggregateSpec::new(vec![Expr::column("sourceId")], Vec::new()),
            Arc::new(Schema::new(vec![Field::new("sourceId", DataType::Utf8, false)])),
        );
        let codebook = decode_builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
        let decode = decode_builder.add_join(
            unique_relation,
            codebook,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "sourceId", "sourceId"),
                    ColumnRef::input(1, "sourceId", "sourceId"),
                )],
            )
            .with_at_most_one_match_per_bounded_row(JoinSide::Left)
            .with_output_projection(vec![JoinOutputColumn::right(
                ColumnRef::input(1, "targetId", "targetId"),
                "targetId",
            )]),
            Arc::clone(&output_schema),
        );
        let decode_plan = decode_builder.finish_gpu_resident(decode);
        let decode_bound = PlanColumnEvidenceAnalysis::for_plan(&decode_plan, &HashMap::new())
            .expect("decoding payload evidence analysis must succeed")
            .root(&decode_plan)
            .string_payload_bound("targetId")
            .cloned()
            .expect("a unique bounded key prevents decode-side value duplication");
        assert_eq!(decode_bound.total_payload_upper_bound_bytes, Some(2_081_894_503));

        for join_type in [JoinType::LeftSemi, JoinType::LeftAnti] {
            let mut builder = QueryPlanBuilder::new();
            let relation = builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
            let unique_relation = builder.add_aggregate(
                relation,
                AggregateSpec::new(vec![Expr::column("sourceId"), Expr::column("targetId")], Vec::new()),
                Arc::new(Schema::new(vec![
                    Field::new("sourceId", DataType::Utf8, false),
                    Field::new("targetId", DataType::Utf8, false),
                ])),
            );
            let lookup = builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
            let selected = builder.add_join(
                unique_relation,
                lookup,
                JoinSpec::new(
                    join_type,
                    vec![JoinKey::new(
                        ColumnRef::input(0, "sourceId", "sourceId"),
                        ColumnRef::input(1, "sourceId", "sourceId"),
                    )],
                )
                .with_output_projection(vec![
                    JoinOutputColumn::left(ColumnRef::input(0, "sourceId", "sourceId"), "sourceId"),
                    JoinOutputColumn::left(ColumnRef::input(0, "targetId", "targetId"), "targetId"),
                ]),
                Arc::new(Schema::new(vec![
                    Field::new("sourceId", DataType::Utf8, false),
                    Field::new("targetId", DataType::Utf8, false),
                ])),
            );
            let plan = builder.finish_gpu_resident(selected);
            let analysis = PlanColumnEvidenceAnalysis::for_plan(&plan, &HashMap::new()).unwrap();
            let input = analysis.node(unique_relation);
            let output = analysis.root(&plan);
            assert_eq!(output.string_payload_bound("targetId"), input.string_payload_bound("targetId"));
            assert!(output.is_unique_on(&BTreeSet::from(["sourceId".to_owned(), "targetId".to_owned()])));
        }

        let mut duplicating_builder = QueryPlanBuilder::new();
        let relation = duplicating_builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
        let lookup = duplicating_builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
        let gathered = duplicating_builder.add_join(
            relation,
            lookup,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    ColumnRef::input(0, "sourceId", "sourceId"),
                    ColumnRef::input(1, "sourceId", "sourceId"),
                )],
            )
            .with_at_most_one_match_per_bounded_row(JoinSide::Left)
            .with_output_projection(vec![JoinOutputColumn::right(
                ColumnRef::input(1, "targetId", "targetId"),
                "targetId",
            )]),
            output_schema,
        );
        let duplicating_plan = duplicating_builder.finish_gpu_resident(gathered);
        assert_eq!(
            PlanColumnEvidenceAnalysis::for_plan(&duplicating_plan, &HashMap::new())
                .expect("finite duplicated payload evidence analysis must succeed")
                .root(&duplicating_plan)
                .string_payload_bound("targetId")
                .and_then(|bound| bound.total_payload_upper_bound_bytes),
            Some(2_081_894_503 * 36),
            "a duplicated lookup-side payload must scale its input total cap by finite row multiplicity"
        );
    }
}
