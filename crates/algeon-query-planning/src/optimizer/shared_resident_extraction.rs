use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use crate::{
    error::Result,
    plan::{NodeId, PlanFingerprint, PlanNode, QueryPlan, RewrittenNodes, row_bounds::query_plan_hard_row_upper_bound},
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
    /// The repeated closed candidate stays in every consumer because its
    /// root does not reduce rows or admission refused its resident plans.
    Rejected,
}

/// Stable reason for an extraction decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedResidentExtractionReason {
    /// The subplan has fewer than two consumers across the inspected plans.
    NotRepeated,
    /// The subtree already consumes a resident source.
    ContainsGpuResidentSource,
    /// A descendant is consumed outside the candidate subtree in its own
    /// plan, or a descendant that would itself be retained also occurs
    /// outside the candidate's occurrences.
    SubplanNotClosed,
    /// An enclosing selected subtree already contains the candidate.
    NestedUnderSelectedSubplan,
    /// The candidate root does not reduce rows, so each consumer recomputes
    /// the subplan instead of retaining its result.
    RootDoesNotReduceRows,
    /// Admission refused the resident producer or a plan rewritten to read
    /// it, so every consumer keeps computing the subplan.
    ResidentPlanNotAdmitted,
    /// The candidate became a resident producer.
    Extracted,
}

/// Structured decision for one inspected native-plan node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedResidentExtractionCandidateDiagnostics {
    /// Index of the node's plan among the fragments of a composed attempt
    /// when the stage compared fragments, or `None` when it ran over one plan
    /// while that plan was optimized.
    pub attempt_fragment: Option<usize>,
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
    /// Number of inspected nodes whose repeated closed subplan was rejected.
    pub rejected_count: usize,
    /// Decisions in fragment order, then node order.
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
    /// Indexes, in the stage's input order, of the plans rewritten to read
    /// this output.
    pub consumer_fragments: Vec<usize>,
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

/// The fragments of one composed attempt after shared subplans moved into
/// resident producers that every reading fragment consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedResidentFragments {
    /// The input fragments in input order, each rewritten to read the
    /// producers it shares as GPU-resident sources.
    pub fragments: Vec<QueryPlan>,
    /// Producer plans in deterministic dependency order.
    pub producers: Vec<SharedResidentProducerPlan>,
    /// Structured decisions, one per node of every input fragment.
    pub diagnostics: SharedResidentExtractionDiagnostics,
}

/// Runs the shared-resident extraction stage for a composed native plan.
pub(crate) fn optimize_shared_resident_extraction(plan: QueryPlan) -> Result<SharedResidentPlan> {
    if gpu_resident_source_names(&plan).is_empty() {
        return Ok(SharedResidentPlan {
            consumer: plan,
            producers: Vec::new(),
            diagnostics: SharedResidentExtractionDiagnostics::default(),
        });
    }
    let SharedResidentFragments { mut fragments, producers, diagnostics } =
        extract_shared_resident_subplans(vec![plan], FragmentLabels::OnePlan)?;
    let consumer = fragments.pop().expect("extraction returns its one input plan");
    Ok(SharedResidentPlan { consumer, producers, diagnostics })
}

/// Rewrites the native fragments of one composed attempt so that one
/// resident producer computes each subplan they repeat.
///
/// `fragments` are the native plans that run in one admitted context: GPU
/// function input producers, the plans that read their results, and the
/// root. A subplan qualifies when its execution fingerprint occurs in two or
/// more fragments, or has two consumers in one, under the same rule as the
/// single-plan stage. Each selected subplan becomes one producer, and every
/// occurrence is replaced by a GPU-resident source with the producer's input
/// name and output facts. A consumer that needs an ordering keeps its own
/// sort above that source.
pub fn extract_shared_resident_fragments(fragments: Vec<QueryPlan>) -> Result<SharedResidentFragments> {
    extract_shared_resident_subplans(fragments, FragmentLabels::AttemptFragments)
}

/// Whether a shared subplan is retained as one resident result rather than
/// recomputed by each consumer.
///
/// Native planning has no cardinality estimates. A retained result holds
/// grant memory until its last consumer finishes, while recomputing a chain
/// of scans, filters, projections and joins streams chunk by chunk and holds
/// little memory. A root that reduces rows bounds the retained result below
/// its input and makes recomputation the expensive side: an aggregate emits
/// one row per group, and a limit with a fetch emits at most that many rows.
/// Every other root is recomputed.
///
/// The stage used to require a hard output row bound. That bound priced
/// resident sources for bounded composition admission, which no longer
/// predicts memory, and Parquet footer row counts give almost every scan
/// chain one, so it did not separate the two cases.
fn retains_resident_result(subplan: &QueryPlan) -> bool {
    match &subplan.root_node().kind {
        PlanNode::Aggregate(_) => true,
        PlanNode::Limit(limit) => limit.spec.fetch.is_some(),
        _ => false,
    }
}

/// How diagnostics label the plan of each inspected node.
#[derive(Debug, Clone, Copy)]
enum FragmentLabels {
    OnePlan,
    AttemptFragments,
}

impl FragmentLabels {
    fn label(self, fragment: usize) -> Option<usize> {
        match self {
            Self::OnePlan => None,
            Self::AttemptFragments => Some(fragment),
        }
    }
}

/// One node of one input plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Occurrence {
    fragment: usize,
    node: NodeId,
}

/// Facts of one input plan that every candidate decision reads.
struct FragmentAnalysis {
    parents: Vec<Vec<NodeId>>,
    fingerprints: Vec<PlanFingerprint>,
    subplan_sizes: Vec<usize>,
}

/// Every occurrence of one subplan fingerprint and its decision.
struct CandidateGroup {
    occurrences: Vec<Occurrence>,
    consumer_count: usize,
    subplan_size: usize,
    reason: Option<SharedResidentExtractionReason>,
    resident_input_name: Option<String>,
}

fn extract_shared_resident_subplans(
    fragments: Vec<QueryPlan>,
    labels: FragmentLabels,
) -> Result<SharedResidentFragments> {
    let analyses = fragments.iter().map(analyze_fragment).collect::<Result<Vec<_>>>()?;
    let mut groups = BTreeMap::<PlanFingerprint, CandidateGroup>::new();
    for (fragment, analysis) in analyses.iter().enumerate() {
        for (index, fingerprint) in analysis.fingerprints.iter().enumerate() {
            let group = groups.entry(*fingerprint).or_insert_with(|| CandidateGroup {
                occurrences: Vec::new(),
                consumer_count: 0,
                subplan_size: analysis.subplan_sizes[index],
                reason: None,
                resident_input_name: None,
            });
            group.occurrences.push(Occurrence { fragment, node: NodeId(index) });
            group.consumer_count += analysis.parents[index].len().max(1);
        }
    }

    // Larger subplans first: an ancestor always has more nodes than its
    // descendants, so every enclosing candidate is decided before the
    // subplans it contains.
    let mut order = groups.keys().copied().collect::<Vec<_>>();
    order.sort_by_key(|fingerprint| {
        let group = &groups[fingerprint];
        (std::cmp::Reverse(group.subplan_size), group.occurrences[0])
    });
    let mut selected = Vec::<Occurrence>::new();
    for fingerprint in &order {
        let reason = decide_group(&fragments, &analyses, &groups, *fingerprint, &selected)?;
        let group = groups.get_mut(fingerprint).expect("ordered fingerprints name existing groups");
        if reason == SharedResidentExtractionReason::Extracted {
            selected.extend(group.occurrences.iter().copied());
        }
        group.reason = Some(reason);
    }

    let mut reserved_names = fragments.iter().flat_map(gpu_resident_source_names).collect::<BTreeSet<_>>();
    let mut next_input_index = 0;
    let mut nodes = fragments.iter().map(|plan| plan.nodes().to_vec()).collect::<Vec<_>>();
    let mut changed = vec![false; fragments.len()];
    let mut producers = Vec::new();
    let mut extracted = groups
        .iter()
        .filter(|(_, group)| group.reason == Some(SharedResidentExtractionReason::Extracted))
        .map(|(fingerprint, group)| (group.occurrences[0], *fingerprint))
        .collect::<Vec<_>>();
    extracted.sort();
    for (first, fingerprint) in extracted {
        let producer = fragments[first.fragment].resident_subplan(first.node)?;
        let input_name = loop {
            let candidate = format!("__algeon_shared_resident_{next_input_index}");
            next_input_index += 1;
            if reserved_names.insert(candidate.clone()) {
                break candidate;
            }
        };
        let root_facts =
            fragments[first.fragment].node(first.node).expect("an occurrence names a node of its plan").facts.clone();
        let mut source = GpuResidentSource::new(&input_name, Arc::clone(&root_facts.output_schema));
        if let Some(rows) = query_plan_hard_row_upper_bound(&producer) {
            source = source.with_row_count_upper_bound(rows);
        }
        let group = groups.get_mut(&fingerprint).expect("extracted fingerprints name existing groups");
        let mut consumer_fragments = BTreeSet::new();
        for occurrence in &group.occurrences {
            let node = &mut nodes[occurrence.fragment][occurrence.node.index()];
            node.kind = PlanNode::Source(Box::new(SourceNode::GpuResident(source.clone())));
            node.facts = root_facts.clone();
            changed[occurrence.fragment] = true;
            consumer_fragments.insert(occurrence.fragment);
        }
        group.resident_input_name = Some(input_name.clone());
        producers.push(SharedResidentProducerPlan {
            input_name,
            plan: producer,
            consumer_fragments: consumer_fragments.into_iter().collect(),
        });
    }

    let mut diagnostics = SharedResidentExtractionDiagnostics {
        considered_count: analyses.iter().map(|analysis| analysis.fingerprints.len()).sum(),
        extracted_count: producers.len(),
        rejected_count: 0,
        candidates: Vec::with_capacity(analyses.iter().map(|analysis| analysis.fingerprints.len()).sum()),
    };
    for (fragment, analysis) in analyses.iter().enumerate() {
        for (index, fingerprint) in analysis.fingerprints.iter().enumerate() {
            let group = &groups[fingerprint];
            let reason = group.reason.expect("every group is decided");
            let decision = match reason {
                SharedResidentExtractionReason::Extracted => SharedResidentExtractionDecision::Extracted,
                SharedResidentExtractionReason::RootDoesNotReduceRows
                | SharedResidentExtractionReason::ResidentPlanNotAdmitted => SharedResidentExtractionDecision::Rejected,
                _ => SharedResidentExtractionDecision::Skipped,
            };
            if decision == SharedResidentExtractionDecision::Rejected {
                diagnostics.rejected_count += 1;
            }
            diagnostics.candidates.push(SharedResidentExtractionCandidateDiagnostics {
                attempt_fragment: labels.label(fragment),
                node_id: NodeId(index),
                subplan_fingerprint: *fingerprint,
                consumer_occurrences: analysis.parents[index].clone(),
                decision,
                reason,
                resident_input_name: group.resident_input_name.clone(),
            });
        }
    }

    let fragments = fragments
        .into_iter()
        .zip(nodes)
        .zip(changed)
        .map(
            |((plan, nodes), changed)| {
                if changed { plan.rewrite(RewrittenNodes::in_place(nodes, plan.root_id())) } else { Ok(plan) }
            },
        )
        .collect::<Result<Vec<_>>>()?;
    Ok(SharedResidentFragments { fragments, producers, diagnostics })
}

fn analyze_fragment(plan: &QueryPlan) -> Result<FragmentAnalysis> {
    let mut fingerprints = Vec::with_capacity(plan.node_count());
    let mut subplan_sizes = Vec::with_capacity(plan.node_count());
    for node in plan.nodes() {
        let subplan = plan.resident_subplan(node.id)?;
        fingerprints.push(subplan.execution_fingerprint());
        subplan_sizes.push(subplan.node_count());
    }
    Ok(FragmentAnalysis { parents: parents_by_node(plan), fingerprints, subplan_sizes })
}

fn decide_group(
    fragments: &[QueryPlan],
    analyses: &[FragmentAnalysis],
    groups: &BTreeMap<PlanFingerprint, CandidateGroup>,
    fingerprint: PlanFingerprint,
    selected: &[Occurrence],
) -> Result<SharedResidentExtractionReason> {
    let group = &groups[&fingerprint];
    let first = group.occurrences[0];
    if group.consumer_count < 2 {
        return Ok(SharedResidentExtractionReason::NotRepeated);
    }
    if subtree_contains_gpu_resident_source(&fragments[first.fragment], first.node) {
        return Ok(SharedResidentExtractionReason::ContainsGpuResidentSource);
    }
    if !occurrences_are_closed(fragments, analyses, group) {
        return Ok(SharedResidentExtractionReason::SubplanNotClosed);
    }
    if group.occurrences.iter().any(|occurrence| {
        selected.iter().any(|root| {
            root.fragment == occurrence.fragment
                && subtree_nodes(&fragments[root.fragment], root.node).contains(&occurrence.node)
        })
    }) {
        return Ok(SharedResidentExtractionReason::NestedUnderSelectedSubplan);
    }
    if !retains_resident_result(&fragments[first.fragment].resident_subplan(first.node)?) {
        return Ok(SharedResidentExtractionReason::RootDoesNotReduceRows);
    }
    if retained_descendant_occurs_elsewhere(fragments, analyses, groups, group)? {
        return Ok(SharedResidentExtractionReason::SubplanNotClosed);
    }
    Ok(SharedResidentExtractionReason::Extracted)
}

/// Whether a descendant subplan that would itself be retained also occurs
/// outside the candidate's occurrences. Retaining the candidate would then
/// compute that descendant once inside the producer and again where it
/// occurs elsewhere, so the descendant is retained instead.
fn retained_descendant_occurs_elsewhere(
    fragments: &[QueryPlan],
    analyses: &[FragmentAnalysis],
    groups: &BTreeMap<PlanFingerprint, CandidateGroup>,
    group: &CandidateGroup,
) -> Result<bool> {
    let mut covered = BTreeMap::<PlanFingerprint, BTreeSet<Occurrence>>::new();
    for occurrence in &group.occurrences {
        for node in subtree_nodes(&fragments[occurrence.fragment], occurrence.node) {
            if node != occurrence.node {
                covered
                    .entry(analyses[occurrence.fragment].fingerprints[node.index()])
                    .or_default()
                    .insert(Occurrence { fragment: occurrence.fragment, node });
            }
        }
    }
    for (fingerprint, covered_occurrences) in covered {
        let descendant = &groups[&fingerprint];
        if descendant.occurrences.len() == covered_occurrences.len()
            || descendant.consumer_count < 2
            || !occurrences_are_closed(fragments, analyses, descendant)
        {
            continue;
        }
        let first = descendant.occurrences[0];
        if retains_resident_result(&fragments[first.fragment].resident_subplan(first.node)?) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn occurrences_are_closed(fragments: &[QueryPlan], analyses: &[FragmentAnalysis], group: &CandidateGroup) -> bool {
    group.occurrences.iter().all(|occurrence| {
        subtree_is_closed(&fragments[occurrence.fragment], occurrence.node, &analyses[occurrence.fragment].parents)
    })
}

fn gpu_resident_source_names(plan: &QueryPlan) -> BTreeSet<String> {
    plan.nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::GpuResident(source) => Some(source.input_name.clone()),
                _ => None,
            },
            _ => None,
        })
        .collect()
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
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        expr::{AggregateExpr, AggregateFunction, AggregateSpec, BinaryOp, Expr, LiteralValue},
        plan::{ProjectionOutput, QueryPlanBuilder},
        source::{LocalParquetFile, LocalParquetSource},
    };

    /// Adds `SUM(v) GROUP BY k` over one Parquet scan.
    fn add_grouped_sum(builder: &mut QueryPlanBuilder) -> NodeId {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let source = builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/shared-resident-fragments-not-read.parquet")],
            Arc::clone(&schema),
            vec!["k".to_string(), "v".to_string()],
        )));
        builder
            .add_aggregate(
                source,
                AggregateSpec::new(
                    vec![Expr::column("k")],
                    vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "v")],
                ),
                Arc::new(Schema::new(vec![
                    Field::new("k", DataType::Int64, false),
                    Field::new("v", DataType::Int64, true),
                ])),
            )
            .unwrap()
    }

    /// Adds `SUM(v)` over the grouped sum.
    fn add_total(builder: &mut QueryPlanBuilder, grouped: NodeId) -> NodeId {
        builder
            .add_aggregate(
                grouped,
                AggregateSpec::new(
                    Vec::new(),
                    vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "total")],
                ),
                Arc::new(Schema::new(vec![Field::new("total", DataType::Int64, true)])),
            )
            .unwrap()
    }

    fn resident_source_names(plan: &QueryPlan) -> BTreeSet<String> {
        gpu_resident_source_names(plan)
    }

    #[test]
    fn a_retained_descendant_repeated_elsewhere_is_shared_instead_of_its_ancestor() {
        // The total repeats in two fragments, but the grouped sum below it
        // also feeds a third. Sharing the total would compute the grouped
        // sum once inside that producer and again for the third fragment.
        let mut filtered_total = QueryPlanBuilder::new();
        let grouped = add_grouped_sum(&mut filtered_total);
        let total = add_total(&mut filtered_total, grouped);
        let positive = filtered_total
            .add_filter(total, Expr::binary(Expr::column("total"), BinaryOp::Gt, Expr::Literal(LiteralValue::Int64(0))))
            .unwrap();
        let filtered_total = filtered_total.finish_with_host_sink(positive).unwrap();
        let mut bare_total = QueryPlanBuilder::new();
        let grouped = add_grouped_sum(&mut bare_total);
        let total = add_total(&mut bare_total, grouped);
        let bare_total = bare_total.finish_with_host_sink(total).unwrap().clone_with_gpu_resident_sink();
        let mut keys = QueryPlanBuilder::new();
        let grouped = add_grouped_sum(&mut keys);
        let key = keys
            .add_projection(
                grouped,
                vec![ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int64, false))],
            )
            .unwrap();
        let keys = keys.finish_with_host_sink(key).unwrap().clone_with_gpu_resident_sink();

        let shared = extract_shared_resident_fragments(vec![filtered_total, bare_total, keys]).unwrap();

        let [producer] = shared.producers.as_slice() else { panic!("one shared producer") };
        assert!(matches!(
            &producer.plan.root_node().kind,
            PlanNode::Aggregate(aggregate) if !aggregate.spec.group_exprs.is_empty()
        ));
        assert_eq!(producer.consumer_fragments, [0, 1, 2]);
        for fragment in &shared.fragments {
            assert_eq!(resident_source_names(fragment), BTreeSet::from([producer.input_name.clone()]));
        }
        let skipped_totals = shared
            .diagnostics
            .candidates
            .iter()
            .filter(|candidate| candidate.reason == SharedResidentExtractionReason::SubplanNotClosed)
            .map(|candidate| (candidate.attempt_fragment, candidate.decision))
            .collect::<Vec<_>>();
        assert_eq!(
            skipped_totals,
            [
                (Some(0), SharedResidentExtractionDecision::Skipped),
                (Some(1), SharedResidentExtractionDecision::Skipped)
            ]
        );
    }
}
