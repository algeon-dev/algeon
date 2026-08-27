use std::collections::{BTreeMap, BTreeSet, HashMap};

use arrow::{
    array::{Array, LargeStringArray, StringArray, StringViewArray},
    datatypes::DataType,
};

use crate::{
    expr::{Expr, alias_wrapped_column, representation_cast_preserves_identity},
    plan::{GraphAlgorithm, JoinNode, JoinSide, NodeId, PlanNode, QueryPlan, WindowFunction},
    source::{SourceColumnRef, SourceNode},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum StringPayloadLineage {
    Parquet { stable_id: String, physical_column_index: usize, selected_row_groups: Vec<usize> },
    Literal { node_id: NodeId, column_index: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct StringPayloadBound {
    pub(super) total_payload_upper_bound_bytes: Option<u64>,
    pub(super) max_single_value_length_bytes: Option<u64>,
    pub(super) lineages: BTreeSet<StringPayloadLineage>,
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
        let mut lineages = BTreeSet::new();
        for bound in &bounds {
            lineages.extend(bound.lineages.iter().cloned());
        }
        (total_payload_upper_bound_bytes.is_some() || max_single_value_length_bytes.is_some()).then_some(Self {
            total_payload_upper_bound_bytes,
            max_single_value_length_bytes,
            lineages,
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct PlanColumnEvidence {
    pub(super) string_payload_bounds: BTreeMap<String, StringPayloadBound>,
    pub(super) unique_keys: Vec<BTreeSet<String>>,
}

impl PlanColumnEvidence {
    pub(super) fn string_payload_bound(&self, column: &str) -> Option<&StringPayloadBound> {
        self.string_payload_bounds.get(column)
    }

    pub(super) fn is_unique_on(&self, columns: &BTreeSet<String>) -> bool {
        self.unique_keys.iter().any(|key| key.is_subset(columns))
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct PlanColumnEvidenceAnalysis {
    by_node: HashMap<NodeId, PlanColumnEvidence>,
}

impl PlanColumnEvidenceAnalysis {
    pub(super) fn for_plan(plan: &QueryPlan, resident_outputs: &HashMap<String, PlanColumnEvidence>) -> Self {
        let mut analysis = Self::default();
        for node in plan.nodes() {
            let evidence = analysis.evidence_for_node(plan, node.id, resident_outputs);
            analysis.by_node.insert(node.id, evidence);
        }
        analysis
    }

    pub(super) fn node(&self, node_id: NodeId) -> PlanColumnEvidence {
        self.by_node.get(&node_id).cloned().unwrap_or_default()
    }

    pub(super) fn root(&self, plan: &QueryPlan) -> PlanColumnEvidence {
        self.node(plan.root_id())
    }

    fn evidence_for_node(
        &self,
        plan: &QueryPlan,
        node_id: NodeId,
        resident_outputs: &HashMap<String, PlanColumnEvidence>,
    ) -> PlanColumnEvidence {
        let Some(node) = plan.node(node_id) else {
            return PlanColumnEvidence::default();
        };
        match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => local_parquet_evidence(source),
                SourceNode::GpuResident(source) => {
                    let mut evidence = resident_outputs.get(&source.input_name).cloned().unwrap_or_default();
                    if let Some(unique_key) = &source.unique_key_columns {
                        evidence.unique_keys.push(unique_key.clone());
                    }
                    evidence
                },
                SourceNode::LiteralBatch(source) => literal_evidence(node_id, source),
            },
            PlanNode::Filter(filter) => self.node(filter.input),
            PlanNode::Projection(projection) => {
                let input = self.node(projection.input);
                let input_schema = plan.node(projection.input).map(|node| node.output_schema());
                let mut output = PlanColumnEvidence::default();
                for projection_output in &projection.outputs {
                    let Some(source_column) = identity_source_column(&projection_output.expr, input_schema) else {
                        continue;
                    };
                    if let Some(bound) = input.string_payload_bound(source_column) {
                        output.string_payload_bounds.insert(projection_output.field.name().clone(), bound.clone());
                    }
                }
                output.unique_keys = remap_unique_keys(
                    &input,
                    projection.outputs.iter().filter_map(|output| {
                        Some((
                            identity_source_column(&output.expr, input_schema)?.to_owned(),
                            output.field.name().clone(),
                        ))
                    }),
                );
                output
            },
            PlanNode::Aggregate(aggregate) => {
                let input = self.node(aggregate.input);
                let input_schema = plan.node(aggregate.input).map(|node| node.output_schema());
                let mut output = PlanColumnEvidence::default();
                if aggregate.spec.grouping_sets.is_none() {
                    for (expression, field) in aggregate.spec.group_exprs.iter().zip(node.output_schema().fields()) {
                        let Some(source_column) = identity_source_column(expression, input_schema) else {
                            continue;
                        };
                        if let Some(bound) = input.string_payload_bound(source_column) {
                            output.string_payload_bounds.insert(field.name().clone(), bound.clone());
                        }
                    }
                    if !aggregate.spec.group_exprs.is_empty() {
                        output.unique_keys.push(
                            node.output_schema()
                                .fields()
                                .iter()
                                .take(aggregate.spec.group_exprs.len())
                                .map(|field| field.name().clone())
                                .collect(),
                        );
                    }
                }
                output
            },
            PlanNode::Window(window) => {
                let mut output = self.node(window.input);
                if window.spec.partition_by.is_empty() {
                    output.unique_keys.extend(window.spec.functions.iter().filter_map(|function| match function {
                        WindowFunction::RowNumber { alias, .. } => Some(BTreeSet::from([alias.clone()])),
                        _ => None,
                    }));
                }
                output
            },
            PlanNode::Sort(sort) => self.node(sort.input),
            PlanNode::Limit(limit) => self.node(limit.input),
            PlanNode::Union(union) => union_evidence(plan, union.inputs.as_slice(), self),
            PlanNode::Join(join) => join_evidence(join, &self.node(join.inputs[0]), &self.node(join.inputs[1])),
            PlanNode::GraphAlgorithm(graph) => {
                let mut output = PlanColumnEvidence::default();
                if matches!(
                    graph.spec.algorithm,
                    GraphAlgorithm::PageRank | GraphAlgorithm::PersonalizedPageRank | GraphAlgorithm::Bfs
                ) && node.output_schema().field_with_name("vertex").is_ok()
                {
                    output.unique_keys.push(BTreeSet::from(["vertex".to_owned()]));
                }
                output
            },
            _ => PlanColumnEvidence::default(),
        }
    }
}

fn local_parquet_evidence(source: &crate::source::LocalParquetSource) -> PlanColumnEvidence {
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

fn literal_evidence(node_id: NodeId, source: &crate::source::LiteralBatchSource) -> PlanColumnEvidence {
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
                    max_single_value_length_bytes: Some(max_single_value_length_bytes),
                    lineages: BTreeSet::from([StringPayloadLineage::Literal { node_id, column_index }]),
                },
            );
        }
    }
    output
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

fn join_evidence(join: &JoinNode, left: &PlanColumnEvidence, right: &PlanColumnEvidence) -> PlanColumnEvidence {
    let mut output = PlanColumnEvidence::default();
    let bounded_side = [JoinSide::Left, JoinSide::Right]
        .into_iter()
        .find(|side| join.spec.has_at_most_one_match_per_bounded_row(*side));
    let Some(bounded_side) = bounded_side else {
        return output;
    };
    let bounded_is_unique = side_is_unique_on_join_keys(join, bounded_side, left, right);
    for projected in &join.spec.output_projection {
        let input = match projected.side {
            JoinSide::Left => left,
            JoinSide::Right => right,
        };
        let mut bound = input.string_payload_bound(projected.column.name()).cloned();
        if projected.side != bounded_side
            && !bounded_is_unique
            && !join.spec.preserves_total_payload_bound_for_output(&projected.output_name)
        {
            if let Some(bound) = &mut bound {
                bound.total_payload_upper_bound_bytes = None;
            }
            if join.spec.join_type == crate::plan::JoinType::Inner
                && let Some(equivalent) = equivalent_bounded_join_key_bound(join, projected, left, right, bounded_side)
                && equivalent.total_payload_upper_bound_bytes.is_some()
            {
                if let Some(bound) = &mut bound {
                    bound.total_payload_upper_bound_bytes = equivalent.total_payload_upper_bound_bytes;
                    bound.lineages.extend(equivalent.lineages.iter().cloned());
                } else {
                    bound = Some(equivalent.clone());
                }
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
        .map(|column| (column.column.name().to_owned(), column.output_name.clone()));
    let right_mapping = join
        .spec
        .output_projection
        .iter()
        .filter(|column| column.side == JoinSide::Right)
        .map(|column| (column.column.name().to_owned(), column.output_name.clone()));
    if bounded_side == JoinSide::Left || bounded_is_unique {
        output.unique_keys.extend(remap_unique_keys(left, left_mapping));
    }
    if bounded_side == JoinSide::Right || bounded_is_unique {
        output.unique_keys.extend(remap_unique_keys(right, right_mapping));
    }
    output
}

fn equivalent_bounded_join_key_bound<'a>(
    join: &JoinNode,
    projected: &crate::plan::JoinOutputColumn,
    left: &'a PlanColumnEvidence,
    right: &'a PlanColumnEvidence,
    bounded_side: JoinSide,
) -> Option<&'a StringPayloadBound> {
    join.spec.keys.iter().find_map(|key| match (projected.side, bounded_side) {
        (JoinSide::Left, JoinSide::Right) if key.left.name() == projected.column.name() => {
            right.string_payload_bound(key.right.name())
        },
        (JoinSide::Right, JoinSide::Left) if key.right.name() == projected.column.name() => {
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

fn remap_unique_keys(
    input: &PlanColumnEvidence,
    mappings: impl IntoIterator<Item = (String, String)>,
) -> Vec<BTreeSet<String>> {
    let mappings = mappings.into_iter().collect::<BTreeMap<_, _>>();
    input
        .unique_keys
        .iter()
        .filter_map(|key| key.iter().map(|column| mappings.get(column).cloned()).collect::<Option<BTreeSet<_>>>())
        .collect()
}

fn identity_source_column<'a>(expr: &'a Expr, input_schema: Option<&arrow::datatypes::SchemaRef>) -> Option<&'a str> {
    if let Some(column) = alias_wrapped_column(expr) {
        return Some(column.name());
    }
    match expr {
        Expr::Alias { expr, .. } => identity_source_column(expr, input_schema),
        Expr::Cast { expr, target_type } => {
            let column = alias_wrapped_column(expr)?;
            let source_type = input_schema?.field_with_name(column.name()).ok()?.data_type();
            representation_cast_preserves_identity(source_type, target_type).then_some(column.name())
        },
        _ => None,
    }
}

fn is_string(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use arrow::datatypes::{Field, Schema};

    use super::*;
    use crate::{
        expr::{AggregateSpec, SortExpr},
        parquet_stats::{ParquetVariableWidthEncodingProof, ParquetVariableWidthPayloadEvidence},
        plan::{JoinKey, JoinOutputColumn, JoinSpec, JoinType, ProjectionOutput, QueryPlanBuilder, WindowSpec},
        source::{LocalParquetFile, LocalParquetSource},
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
        let output = PlanColumnEvidenceAnalysis::for_plan(&plan, &HashMap::new()).root(&plan);

        let bound = output.string_payload_bound("vertex").expect("codebook payload must retain footer lineage");
        assert_eq!(bound.total_payload_upper_bound_bytes, Some(4_064_566_703));
        assert_eq!(bound.max_single_value_length_bytes, None);
        assert_eq!(bound.lineages.len(), 2);
        assert!(output.is_unique_on(&BTreeSet::from(["__vertex_id".to_owned()])));
    }

    #[test]
    fn payload_lineage_stops_at_string_transform_and_duplication_capable_join() {
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
                crate::expr::ColumnRef::input(0, "sourceId", "sourceId"),
                crate::expr::ColumnRef::input(1, "sourceId", "sourceId"),
            )],
        )
        .with_output_projection(vec![JoinOutputColumn::left(
            crate::expr::ColumnRef::input(0, "sourceId", "sourceId"),
            "sourceId",
        )]);
        let join = join_builder.add_join(left, right, spec, output_schema);
        let join_plan = join_builder.finish_gpu_resident(join);
        assert!(
            PlanColumnEvidenceAnalysis::for_plan(&join_plan, &HashMap::new())
                .root(&join_plan)
                .string_payload_bound("sourceId")
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
                    crate::expr::ColumnRef::input(0, "sourceId", "sourceId"),
                    crate::expr::ColumnRef::input(1, "sourceId", "sourceId"),
                )],
            )
            .with_at_most_one_match_per_bounded_row(JoinSide::Left)
            .with_output_projection(vec![JoinOutputColumn::left(
                crate::expr::ColumnRef::input(0, "targetId", "targetId"),
                "targetId",
            )]),
            Arc::clone(&output_schema),
        );
        let encode_plan = encode_builder.finish_gpu_resident(encode);
        let encode_bound = PlanColumnEvidenceAnalysis::for_plan(&encode_plan, &HashMap::new())
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
                    crate::expr::ColumnRef::input(0, "sourceId", "sourceId"),
                    crate::expr::ColumnRef::input(1, "sourceId", "sourceId"),
                )],
            )
            .with_at_most_one_match_per_bounded_row(JoinSide::Left)
            .with_output_projection(vec![JoinOutputColumn::right(
                crate::expr::ColumnRef::input(1, "targetId", "targetId"),
                "targetId",
            )]),
            Arc::clone(&output_schema),
        );
        let decode_plan = decode_builder.finish_gpu_resident(decode);
        let decode_bound = PlanColumnEvidenceAnalysis::for_plan(&decode_plan, &HashMap::new())
            .root(&decode_plan)
            .string_payload_bound("targetId")
            .cloned()
            .expect("a unique bounded key prevents decode-side value duplication");
        assert_eq!(decode_bound.total_payload_upper_bound_bytes, Some(2_081_894_503));

        let mut unsafe_builder = QueryPlanBuilder::new();
        let relation = unsafe_builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
        let lookup = unsafe_builder.add_source(SourceNode::LocalParquet(production_endpoint_source()));
        let gathered = unsafe_builder.add_join(
            relation,
            lookup,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(
                    crate::expr::ColumnRef::input(0, "sourceId", "sourceId"),
                    crate::expr::ColumnRef::input(1, "sourceId", "sourceId"),
                )],
            )
            .with_at_most_one_match_per_bounded_row(JoinSide::Left)
            .with_output_projection(vec![JoinOutputColumn::right(
                crate::expr::ColumnRef::input(1, "targetId", "targetId"),
                "targetId",
            )]),
            output_schema,
        );
        let unsafe_plan = unsafe_builder.finish_gpu_resident(gathered);
        assert!(
            PlanColumnEvidenceAnalysis::for_plan(&unsafe_plan, &HashMap::new())
                .root(&unsafe_plan)
                .string_payload_bound("targetId")
                .is_none(),
            "a duplicated lookup-side payload cannot retain its input total cap"
        );
    }
}
