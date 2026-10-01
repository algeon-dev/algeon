use std::{collections::BTreeSet, sync::Arc};

use crate::{
    plan::{
        NodeId, PlanFacts, PlanFingerprint, PlanNode, QueryPlan, RewrittenNodes, SchemaBehavior,
        row_bounds::query_plan_hard_row_upper_bound,
    },
    source::{GpuResidentSource, SourceNode},
};

/// Stable optimizer rule name for shared-resident extraction diagnostics.
pub const SHARED_RESIDENT_EXTRACTION_RULE_NAME: &str = "shared_resident_extraction";

/// Decision emitted for one extraction candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedResidentExtractionDecision {
    /// The repeated closed subplan became a resident producer.
    Extracted,
    /// The candidate was left in the consumer plan.
    Skipped,
    /// The repeated closed candidate could not preserve the resident contract.
    Rejected,
}

/// Stable reason for an extraction decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedResidentExtractionReason {
    /// The node has fewer than two consumer occurrences.
    NotRepeated,
    /// Dynamic-filter route node identities cannot yet be remapped.
    DynamicFilterRoutesUnremappable,
    /// The subtree already consumes a resident source.
    ContainsGpuResidentSource,
    /// A descendant is consumed outside the candidate subtree.
    SubplanNotClosed,
    /// An enclosing selected subtree already contains the candidate.
    NestedUnderSelectedSubplan,
    /// The resident source would not carry a hard output row bound.
    ResidentRowBoundUnavailable,
    /// The candidate became a resident producer.
    Extracted,
}

/// Structured decision for one inspected native-plan node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedResidentExtractionCandidateDiagnostics {
    /// Candidate root node.
    pub node_id: NodeId,
    /// Execution fingerprint of the exact candidate subplan.
    pub subplan_fingerprint: PlanFingerprint,
    /// Plan nodes that consume the candidate root.
    pub consumer_occurrences: Vec<NodeId>,
    /// Extraction disposition.
    pub decision: SharedResidentExtractionDecision,
    /// Stable decision reason.
    pub reason: SharedResidentExtractionReason,
    /// Resident source identity created for an extracted candidate.
    pub resident_input_name: Option<String>,
}

/// Diagnostics emitted by the named shared-resident extraction stage.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SharedResidentExtractionDiagnostics {
    /// Number of native-plan nodes inspected.
    pub considered_count: usize,
    /// Number of resident producers created.
    pub extracted_count: usize,
    /// Number of repeated closed candidates rejected by the resident contract.
    pub rejected_count: usize,
    /// Decisions in candidate node order.
    pub candidates: Vec<SharedResidentExtractionCandidateDiagnostics>,
}

impl SharedResidentExtractionDiagnostics {
    /// Returns whether the stage inspected no nodes.
    pub(crate) fn is_empty(&self) -> bool {
        self.considered_count == 0
            && self.extracted_count == 0
            && self.rejected_count == 0
            && self.candidates.is_empty()
    }

    /// Merges decisions from another invocation of the same stage.
    pub(crate) fn merge(&mut self, other: Self) {
        self.considered_count += other.considered_count;
        self.extracted_count += other.extracted_count;
        self.rejected_count += other.rejected_count;
        self.candidates.extend(other.candidates);
    }
}

/// One closed shared DAG fragment extracted behind a GPU-resident boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedResidentProducerPlan {
    /// Runtime occurrence identity used by the rewritten consumer source.
    pub input_name: String,
    /// Exact producer plan whose output remains device-resident.
    pub plan: QueryPlan,
}

/// A consumer plan and the shared producer fragments it consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedResidentPlan {
    /// Rewritten plan whose repeated closed subplans are resident sources.
    pub consumer: QueryPlan,
    /// Producer plans in deterministic consumer dependency order.
    pub producers: Vec<SharedResidentProducerPlan>,
    /// Structured decisions from the named extraction stage.
    pub diagnostics: SharedResidentExtractionDiagnostics,
}

/// Runs the shared-resident extraction stage for a composed native plan.
pub fn optimize_shared_resident_extraction(plan: QueryPlan) -> SharedResidentPlan {
    if !plan.nodes().iter().any(|node| {
        matches!(
            &node.kind,
            PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(_))
        )
    }) {
        return SharedResidentPlan {
            consumer: plan,
            producers: Vec::new(),
            diagnostics: SharedResidentExtractionDiagnostics::default(),
        };
    }
    extract_shared_resident_subplans(plan)
}

fn extract_shared_resident_subplans(plan: QueryPlan) -> SharedResidentPlan {
    let parents = parents_by_node(&plan);
    let mut selected = Vec::new();
    let mut diagnostics = SharedResidentExtractionDiagnostics {
        considered_count: plan.node_count(),
        extracted_count: 0,
        rejected_count: 0,
        candidates: Vec::with_capacity(plan.node_count()),
    };
    for node in plan.nodes().iter().rev() {
        let producer = plan.resident_subplan(node.id);
        let reason = if parents[node.id.index()].len() < 2 {
            SharedResidentExtractionReason::NotRepeated
        } else if !plan.dynamic_filter_routes().is_empty() {
            SharedResidentExtractionReason::DynamicFilterRoutesUnremappable
        } else if subtree_contains_gpu_resident_source(&plan, node.id) {
            SharedResidentExtractionReason::ContainsGpuResidentSource
        } else if !subtree_is_closed(&plan, node.id, &parents) {
            SharedResidentExtractionReason::SubplanNotClosed
        } else if selected.iter().any(|selected_root| subtree_contains(&plan, *selected_root, node.id)) {
            SharedResidentExtractionReason::NestedUnderSelectedSubplan
        } else if query_plan_hard_row_upper_bound(&producer).is_none() {
            SharedResidentExtractionReason::ResidentRowBoundUnavailable
        } else {
            selected.push(node.id);
            SharedResidentExtractionReason::Extracted
        };
        diagnostics.candidates.push(SharedResidentExtractionCandidateDiagnostics {
            node_id: node.id,
            subplan_fingerprint: producer.execution_fingerprint(),
            consumer_occurrences: parents[node.id.index()].clone(),
            decision: match reason {
                SharedResidentExtractionReason::Extracted => SharedResidentExtractionDecision::Extracted,
                SharedResidentExtractionReason::ResidentRowBoundUnavailable => {
                    SharedResidentExtractionDecision::Rejected
                },
                _ => SharedResidentExtractionDecision::Skipped,
            },
            reason,
            resident_input_name: None,
        });
        if reason == SharedResidentExtractionReason::ResidentRowBoundUnavailable {
            diagnostics.rejected_count += 1;
        }
    }
    selected.sort();

    let mut nodes = plan.nodes().to_vec();
    let mut producers = Vec::with_capacity(selected.len());
    let mut input_names = plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::GpuResident(source) => Some(source.input_name.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let mut input_index = 0;
    for root in selected {
        let producer = plan.resident_subplan(root);
        let input_name = loop {
            let candidate = format!("__algeon_shared_resident_{input_index}");
            input_index += 1;
            if input_names.insert(candidate.clone()) {
                break candidate;
            }
        };
        let schema = Arc::clone(producer.output_schema());
        let mut source = GpuResidentSource::new(&input_name, Arc::clone(&schema));
        let row_count_upper_bound = query_plan_hard_row_upper_bound(&producer)
            .expect("selected shared-resident producers have a hard row bound");
        source = source.with_row_count_upper_bound(row_count_upper_bound);
        nodes[root.index()].kind = PlanNode::Source(Box::new(SourceNode::GpuResident(source)));
        nodes[root.index()].facts = PlanFacts::device(schema, SchemaBehavior::SourceOutput);
        producers.push(SharedResidentProducerPlan { input_name, plan: producer });
        let decision = diagnostics
            .candidates
            .iter_mut()
            .find(|candidate| candidate.node_id == root)
            .expect("selected extraction root retains diagnostics");
        decision.resident_input_name = Some(producers.last().expect("producer was appended").input_name.clone());
        diagnostics.extracted_count += 1;
    }

    let consumer = plan.rewrite(RewrittenNodes::in_place(nodes, plan.root_id()));
    diagnostics.candidates.sort_by_key(|candidate| candidate.node_id);
    SharedResidentPlan { consumer, producers, diagnostics }
}

fn parents_by_node(plan: &QueryPlan) -> Vec<Vec<NodeId>> {
    let mut parents = vec![Vec::new(); plan.node_count()];
    for node in plan.nodes() {
        for input in node.input_ids() {
            parents[input.index()].push(node.id);
        }
    }
    parents
}

fn subtree_is_closed(plan: &QueryPlan, root: NodeId, parents: &[Vec<NodeId>]) -> bool {
    let subtree = subtree_nodes(plan, root);
    subtree.iter().all(|node| *node == root || parents[node.index()].iter().all(|parent| subtree.contains(parent)))
}

fn subtree_contains_gpu_resident_source(plan: &QueryPlan, root: NodeId) -> bool {
    subtree_nodes(plan, root).into_iter().any(|node| {
        matches!(
            &plan.node(node).expect("reachable shared node must exist").kind,
            PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::GpuResident(_))
        )
    })
}

fn subtree_contains(plan: &QueryPlan, root: NodeId, candidate: NodeId) -> bool {
    subtree_nodes(plan, root).contains(&candidate)
}

fn subtree_nodes(plan: &QueryPlan, root: NodeId) -> BTreeSet<NodeId> {
    let mut nodes = BTreeSet::new();
    let mut stack = vec![root];
    while let Some(node_id) = stack.pop() {
        if nodes.insert(node_id) {
            stack.extend(plan.node(node_id).expect("reachable shared node must exist").input_ids().iter().copied());
        }
    }
    nodes
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        dynamic_filter::{
            DynamicFilterConsumer, DynamicFilterConsumerLowering, DynamicFilterId, DynamicFilterKind,
            DynamicFilterProducer, DynamicFilterRoute,
        },
        expr::{BinaryOp, ColumnRef, Expr, LiteralValue},
        plan::{JoinKey, JoinOutputColumn, JoinSide, JoinSpec, JoinType, QueryPlanBuilder},
        source::{LocalParquetFile, LocalParquetSource},
    };

    #[test]
    fn repeated_closed_candidate_without_a_hard_row_bound_is_rejected() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/shared-resident-unbounded.parquet")],
            Arc::clone(&schema),
            vec!["id".to_string()],
        )));
        let filtered = builder.add_filter(
            source,
            Expr::binary(Expr::column("id"), BinaryOp::Eq, Expr::Literal(LiteralValue::Int64(1))),
            Arc::clone(&schema),
        );
        let repeated = builder.add_join(
            filtered,
            filtered,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::unqualified("id"), ColumnRef::unqualified("id"))],
            )
            .with_output_projection(vec![JoinOutputColumn::new(
                JoinSide::Left,
                ColumnRef::unqualified("id"),
                "id",
            )]),
            Arc::clone(&schema),
        );
        let resident = builder.add_source(SourceNode::GpuResident(
            GpuResidentSource::new("existing-resident", Arc::clone(&schema)).with_row_count_upper_bound(1),
        ));
        let root = builder.add_join(
            repeated,
            resident,
            JoinSpec::new(
                JoinType::Inner,
                vec![JoinKey::new(ColumnRef::unqualified("id"), ColumnRef::unqualified("id"))],
            )
            .with_output_projection(vec![JoinOutputColumn::new(
                JoinSide::Left,
                ColumnRef::unqualified("id"),
                "id",
            )]),
            Arc::clone(&schema),
        );
        let split = optimize_shared_resident_extraction(builder.finish_with_host_sink_schema(root, schema));

        assert!(split.producers.is_empty());
        assert_eq!(split.diagnostics.rejected_count, 1);
        let rejected = split
            .diagnostics
            .candidates
            .iter()
            .find(|candidate| candidate.node_id == filtered)
            .expect("the repeated closed candidate must retain its rejection");
        assert_eq!(rejected.decision, SharedResidentExtractionDecision::Rejected);
        assert_eq!(rejected.reason, SharedResidentExtractionReason::ResidentRowBoundUnavailable);
    }

    #[test]
    fn shared_resident_extraction_remaps_dynamic_filter_routes() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let _unreachable =
            builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("x", Arc::clone(&schema))));
        let a = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("a", Arc::clone(&schema))));
        let b = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("b", Arc::clone(&schema))));
        let join_keys = || vec![JoinKey::new(ColumnRef::unqualified("id"), ColumnRef::unqualified("id"))];
        let join_output = || vec![JoinOutputColumn::new(JoinSide::Left, ColumnRef::unqualified("id"), "id")];
        let first_join = builder.add_join(
            a,
            b,
            JoinSpec::new(JoinType::Inner, join_keys()).with_output_projection(join_output()),
            Arc::clone(&schema),
        );
        let c = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new("c", Arc::clone(&schema))));
        let second_join = builder.add_join(
            first_join,
            c,
            JoinSpec::new(JoinType::Inner, join_keys()).with_output_projection(join_output()),
            Arc::clone(&schema),
        );
        let route = DynamicFilterRoute::new(
            DynamicFilterId::new(0),
            DynamicFilterProducer::HashJoinBuild {
                join_node_id: first_join,
                build_side: JoinSide::Right,
                build_input: b,
            },
            DynamicFilterConsumer::HashJoinProbe { join_node_id: second_join, probe_side: JoinSide::Left },
            vec!["id".to_string()],
            vec!["id".to_string()],
            vec![DynamicFilterKind::Range],
            DynamicFilterConsumerLowering::PostDecodeOnly,
        );
        let plan = builder
            .finish_with_host_sink_schema(second_join, Arc::clone(&schema))
            .with_dynamic_filter_routes(vec![route]);
        assert!(plan.unreachable_node_ids().len() == 1);

        let split = optimize_shared_resident_extraction(plan);

        // One unreachable node before every endpoint shifts each survivor down
        // by one index; the route follows its joins instead of the old ids.
        let shifted = |id: NodeId| NodeId::from_index_for_internal_use(id.index() - 1);
        let routes = split.consumer.dynamic_filter_routes();
        assert_eq!(routes.len(), 1, "pure renumbering must keep the route");
        let DynamicFilterProducer::HashJoinBuild { join_node_id, build_input, .. } = &routes[0].producer else {
            panic!("producer kind must survive the remap");
        };
        assert_eq!(*join_node_id, shifted(first_join));
        assert_eq!(*build_input, shifted(b));
        let DynamicFilterConsumer::HashJoinProbe { join_node_id, .. } = &routes[0].consumer else {
            panic!("consumer kind must survive the remap");
        };
        assert_eq!(*join_node_id, shifted(second_join));
    }
}
