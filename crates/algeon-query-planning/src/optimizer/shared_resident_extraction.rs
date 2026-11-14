//! Shared-resident extraction: the stage after the optimizer pass table that
//! moves a repeated closed subplan into one GPU-resident producer.
//!
//! The rule matches a subplan whose comparison fingerprint has two or more
//! consumers, within one plan or across the native units of one composed island,
//! whose root reduces rows (an aggregate, or a limit with a fetch), and which is
//! closed: no node below the root is read from outside it. In the rewrite, every
//! occurrence becomes a `SourceNode::GpuResident` named
//! `__algeon_shared_resident_{n}` with the root's facts, and the subplan becomes
//! a producer plan under a `ResultSink::GpuResident` sink. The consumer keeps its
//! own sink. The producer is a separate native plan, so the adapter admits it on
//! its own, executes it once in the island root's context and releases the result
//! after its last reading unit; a producer admission refuses is a refused
//! candidate. Semantics are preserved by construction, since the occurrence read
//! the same rows the producer now emits. Side state: occurrences are replaced in
//! place, so routes and advisory bindings elsewhere survive and `rewrite` drops
//! those bound to a replaced node.
//!
//! Retention is a memory decision made without cardinality estimates; see
//! `retains_resident_result`. Every inspected node gets one diagnostics row.
//!
//! Aggregate copies whose fingerprints differ only in what their local
//! Parquet sources select (row groups, reader predicate, the filter directly
//! above the source) can also share one producer; see
//! `decide_residual_families`. Each such copy reads the producer through a
//! filter over the producer's group outputs, which keeps exactly the groups
//! the copy computed.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use arrow::datatypes::DataType;

use super::{
    access::{local_parquet_source, projection_only_selects_columns, subplan_can_fail},
    lineage::substitute_unqualified_columns,
    predicate_factoring::{
        GroupKeyDefinition, GroupKeyOutput, conjunction, factor_selections, is_lossless_integer_widening, map_residual,
        split_conjuncts,
    },
};
use crate::{
    dynamic_filter::{DynamicFilterConsumer, DynamicFilterId, DynamicFilterProducer, DynamicFilterRoute},
    error::Result,
    expr::Expr,
    plan::{
        FilterNode, JoinSide, NodeId, PlanFingerprint, PlanNode, ProjectionNode, ProjectionOutput, QueryNode,
        QueryPlan, RewrittenNodes, clone_plan_node_with_inputs, row_bounds::query_plan_hard_row_upper_bound,
    },
    source::{GpuResidentSource, LocalParquetSource, SourceNode},
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
    /// the subplan and does not retain its result.
    RootDoesNotReduceRows,
    /// Admission refused the resident producer or a plan rewritten to read
    /// it, so every consumer keeps computing the subplan.
    ResidentPlanNotAdmitted,
    /// The candidate became a resident producer.
    Extracted,
    /// The candidate and copies that differ from it only in source selection
    /// predicates and row groups became one resident producer, which this
    /// candidate reads under a filter over the producer's group outputs.
    ExtractedUnderResidualFilter,
    /// Copies that differ only in source selection predicates and row groups
    /// stay separate because, for some source, no copy's scan selects every
    /// row group the others select and admits every row the widened
    /// selection needs.
    WidenedScanNotCovered,
    /// Copies that differ only in source selection predicates stay separate
    /// because a predicate one of them does not share has no restatement over
    /// the aggregate's group outputs.
    ResidualNotMappedToOutput,
    /// Copies that differ in the selection of two or more sources stay
    /// separate because the subplan has an expression that [can
    /// fail](crate::expr::Expr::can_fail). The producer would pair rows of
    /// those sources that no copy pairs and evaluate it there.
    WidenedRowsCanFail,
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
    /// Execution fingerprint after normalizing private projection names for comparison.
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
///
/// Only a plan that already reads a GPU-resident source, the shape a root
/// lowered over GPU function inputs has, is inspected; any other plan is
/// returned unchanged with empty diagnostics.
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
/// root. A subplan qualifies when its comparison fingerprint occurs in two or
/// more fragments, or has two consumers in one, under the same rule as the
/// single-plan stage. Each selected subplan becomes one producer, and every
/// occurrence is replaced by a GPU-resident source with the producer's input
/// name and output facts. A consumer that needs an ordering keeps its own
/// sort above that source. Comparison normalizes intermediate projection names
/// read only by projections; source bindings and root output names remain exact.
pub fn extract_shared_resident_fragments(fragments: Vec<QueryPlan>) -> Result<SharedResidentFragments> {
    extract_shared_resident_subplans(fragments, FragmentLabels::AttemptFragments)
}

/// Whether a shared subplan is retained as one resident result. A subplan that is
/// not retained is recomputed by each consumer.
///
/// Native planning has no cardinality estimates. A retained result holds
/// grant memory until its last consumer finishes, while recomputing a chain
/// of scans, filters, projections and joins streams chunk by chunk and holds
/// little memory. A root that reduces rows bounds the retained result below
/// its input and makes recomputation the expensive side: an aggregate emits
/// one row per group, and a limit with a fetch emits at most that many rows.
/// Every other root is recomputed.
///
/// A hard output row bound does not decide retention. Parquet footer row
/// counts give almost every scan chain such a bound, so it would not separate
/// retained results from recomputed ones.
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
    /// Comparison fingerprint of the subplan rooted at each node, under a
    /// GPU-resident sink so sink kind and host schema do not separate
    /// otherwise equal subplans.
    fingerprints: Vec<PlanFingerprint>,
    subplan_sizes: Vec<usize>,
}

/// Every occurrence of one subplan fingerprint and its decision.
struct CandidateGroup {
    occurrences: Vec<Occurrence>,
    /// Parents over all occurrences; a root with no parent counts once for
    /// its sink, so a subplan that is a whole fragment is a consumer.
    consumer_count: usize,
    subplan_size: usize,
    /// Set once by the decision loop; read when rows are emitted.
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

    let families = decide_residual_families(&fragments, &analyses, &mut groups, &mut selected)?;

    // Producer names share the namespace of the fragments' existing
    // GPU-resident inputs, which the adapter also binds by name.
    let mut reserved_names = fragments.iter().flat_map(gpu_resident_source_names).collect::<BTreeSet<_>>();
    let mut next_input_index = 0;
    let mut nodes = fragments.iter().map(|plan| plan.nodes().to_vec()).collect::<Vec<_>>();
    let mut changed = vec![false; fragments.len()];
    let family_members = families.iter().flat_map(|family| family.members.iter().map(|(fingerprint, _)| *fingerprint));
    let family_members = family_members.collect::<BTreeSet<_>>();
    let mut extracted = groups
        .iter()
        .filter(|(fingerprint, group)| {
            group.reason == Some(SharedResidentExtractionReason::Extracted) && !family_members.contains(*fingerprint)
        })
        .map(|(fingerprint, group)| {
            // Equal comparison fingerprints preserve the root schema and rows;
            // the producer keeps the first occurrence's executable bindings.
            let first = group.occurrences[0];
            Ok((first, fragments[first.fragment].resident_subplan(first.node)?, vec![(*fingerprint, None)]))
        })
        .collect::<Result<Vec<_>>>()?;
    for family in families {
        let first = family
            .members
            .iter()
            .map(|(fingerprint, _)| groups[fingerprint].occurrences[0])
            .min()
            .expect("a residual family has members");
        extracted.push((first, family.producer, family.members));
    }
    extracted.sort_by_key(|(first, _, _)| *first);
    let mut producers = Vec::new();
    for (_, producer, members) in extracted {
        let input_name = loop {
            let candidate = format!("__algeon_shared_resident_{next_input_index}");
            next_input_index += 1;
            if reserved_names.insert(candidate.clone()) {
                break candidate;
            }
        };
        let root_facts = producer.root_node().facts.clone();
        let mut source = GpuResidentSource::new(&input_name, Arc::clone(&root_facts.output_schema));
        if let Some(rows) = query_plan_hard_row_upper_bound(&producer) {
            source = source.with_row_count_upper_bound(rows);
        }
        let source_kind = PlanNode::Source(Box::new(SourceNode::GpuResident(source)));
        let mut consumer_fragments = BTreeSet::new();
        for (fingerprint, residual) in members {
            let group = groups.get_mut(&fingerprint).expect("extracted fingerprints name existing groups");
            for occurrence in &group.occurrences {
                let fragment_nodes = &mut nodes[occurrence.fragment];
                match &residual {
                    None => {
                        fragment_nodes[occurrence.node.index()].kind = source_kind.clone();
                    },
                    Some(predicate) => {
                        // The occurrence's subtree is closed and about to be
                        // dropped, so its lowest node, which precedes the
                        // occurrence, holds the resident source the residual
                        // filter reads.
                        let slot = *subtree_nodes(&fragments[occurrence.fragment], occurrence.node)
                            .first()
                            .expect("a subtree contains its root");
                        debug_assert_ne!(slot, occurrence.node, "an aggregate occurrence has an input");
                        fragment_nodes[slot.index()].kind = source_kind.clone();
                        fragment_nodes[slot.index()].facts = root_facts.clone();
                        fragment_nodes[occurrence.node.index()].kind =
                            PlanNode::Filter(FilterNode { input: slot, predicate: predicate.clone() });
                    },
                }
                fragment_nodes[occurrence.node.index()].facts = root_facts.clone();
                changed[occurrence.fragment] = true;
                consumer_fragments.insert(occurrence.fragment);
            }
            group.resident_input_name = Some(input_name.clone());
        }
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
                SharedResidentExtractionReason::Extracted
                | SharedResidentExtractionReason::ExtractedUnderResidualFilter => {
                    SharedResidentExtractionDecision::Extracted
                },
                SharedResidentExtractionReason::RootDoesNotReduceRows
                | SharedResidentExtractionReason::ResidentPlanNotAdmitted
                | SharedResidentExtractionReason::WidenedScanNotCovered
                | SharedResidentExtractionReason::ResidualNotMappedToOutput
                | SharedResidentExtractionReason::WidenedRowsCanFail => SharedResidentExtractionDecision::Rejected,
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

/// Copies of one aggregate that differ only in source selections, fused into
/// one producer.
struct ResidualFamily {
    producer: QueryPlan,
    /// Member fingerprints in first-occurrence order, each with the filter
    /// over the producer's output that its occurrences read under.
    members: Vec<(PlanFingerprint, Option<Expr>)>,
}

/// Fuses aggregate groups whose subplans are equal apart from what their
/// local Parquet sources select: each source's row groups and its selection
/// predicate, which is the reader predicate AND the filter directly above
/// the source (through projections that only select columns).
///
/// Runs after the identical-fingerprint decisions. A family takes members
/// that were extracted on their own or are not repeated, and is skipped when
/// a member is not closed, contains a resident source, overlaps a selected
/// subplan other than its own members, or retains a descendant read
/// elsewhere. A fused family's producer is cheaper than its members together
/// only when its scans are scans one member already performs; the covering
/// check in [`fuse_residual_family`] holds it to that. A refused family
/// reports its reason on members that would otherwise be `NotRepeated`.
fn decide_residual_families(
    fragments: &[QueryPlan],
    analyses: &[FragmentAnalysis],
    groups: &mut BTreeMap<PlanFingerprint, CandidateGroup>,
    selected: &mut Vec<Occurrence>,
) -> Result<Vec<ResidualFamily>> {
    let mut buckets = BTreeMap::<PlanFingerprint, Vec<PlanFingerprint>>::new();
    for (fingerprint, group) in groups.iter() {
        if !matches!(
            group.reason,
            Some(SharedResidentExtractionReason::Extracted | SharedResidentExtractionReason::NotRepeated)
        ) {
            continue;
        }
        let first = group.occurrences[0];
        let subplan = fragments[first.fragment].resident_subplan(first.node)?;
        if !matches!(subplan.root_node().kind, PlanNode::Aggregate(_)) || source_selections(&subplan).is_empty() {
            continue;
        }
        buckets.entry(selection_wildcard_fingerprint(subplan)?).or_default().push(*fingerprint);
    }
    let mut ordered = buckets.into_values().filter(|members| members.len() >= 2).collect::<Vec<_>>();
    for members in &mut ordered {
        members.sort_by_key(|fingerprint| groups[fingerprint].occurrences[0]);
    }
    ordered.sort_by_key(|members| {
        let group = &groups[&members[0]];
        (std::cmp::Reverse(group.subplan_size), group.occurrences[0])
    });

    let mut families = Vec::new();
    for members in ordered {
        let occurrences =
            members.iter().flat_map(|fingerprint| groups[fingerprint].occurrences.iter().copied()).collect::<Vec<_>>();
        let union = CandidateGroup {
            occurrences: occurrences.clone(),
            consumer_count: members.iter().map(|fingerprint| groups[fingerprint].consumer_count).sum(),
            subplan_size: groups[&members[0]].subplan_size,
            reason: None,
            resident_input_name: None,
        };
        let own = occurrences.iter().copied().collect::<BTreeSet<_>>();
        let overlaps_selected = selected.iter().filter(|root| !own.contains(root)).any(|root| {
            occurrences.iter().any(|occurrence| {
                occurrence.fragment == root.fragment
                    && (subtree_nodes(&fragments[root.fragment], root.node).contains(&occurrence.node)
                        || subtree_nodes(&fragments[root.fragment], occurrence.node).contains(&root.node))
            })
        });
        if overlaps_selected
            || occurrences.iter().any(|occurrence| {
                subtree_contains_gpu_resident_source(&fragments[occurrence.fragment], occurrence.node)
            })
            || !occurrences_are_closed(fragments, analyses, &union)
            || retained_descendant_occurs_elsewhere(fragments, analyses, groups, &union)?
        {
            continue;
        }
        match fuse_residual_family(fragments, groups, &members)? {
            Ok(family) => {
                for (fingerprint, residual) in &family.members {
                    groups.get_mut(fingerprint).expect("family members name existing groups").reason =
                        Some(if residual.is_some() {
                            SharedResidentExtractionReason::ExtractedUnderResidualFilter
                        } else {
                            SharedResidentExtractionReason::Extracted
                        });
                }
                let newly_selected =
                    occurrences.into_iter().filter(|occurrence| !selected.contains(occurrence)).collect::<Vec<_>>();
                selected.extend(newly_selected);
                families.push(family);
            },
            Err(reason) => {
                for fingerprint in &members {
                    let group = groups.get_mut(fingerprint).expect("family members name existing groups");
                    if group.reason == Some(SharedResidentExtractionReason::NotRepeated) {
                        group.reason = Some(reason);
                    }
                }
            },
        }
    }
    Ok(families)
}

/// Builds the producer of one family, or names the reason it cannot be
/// shared.
///
/// Members list their local Parquet sources in the same order, since their
/// subplans match once selections are cleared. For each source,
/// [`factor_selections`] splits the members' selection conjuncts, and each
/// member's differing conjuncts are restated over the aggregate's group
/// outputs ([`map_residual`]), which needs every node from a source whose
/// selections differ up to the root to have one reader. The producer is the
/// first member's subplan with each source replaced, unchanged, by a covering
/// member's source node. A member's source covers when its row groups contain
/// every other member's, file by file (`None` selects all), and its reader
/// predicate holds only common conjuncts. The widened predicate replaces the representative's
/// selection filter, which is removed when no predicate remains; a source
/// without a selection filter in any member needs a covering member whose
/// reader predicate already selects the widened rows. The source is never
/// rewritten, so its reader predicate stays one the reader accepted and its
/// `max_rows` bound stays proven. A source with a scan limit or a count-star
/// probe takes part only when every member selects the same rows from it.
/// Selected row groups are pruning results of predicates and joins every
/// member applies, so reading more of them changes no result. It can change
/// whether the query fails: with two or more widened sources the producer
/// pairs rows no copy pairs, so such a family needs a subplan in which no
/// expression can fail.
fn fuse_residual_family(
    fragments: &[QueryPlan],
    groups: &BTreeMap<PlanFingerprint, CandidateGroup>,
    members: &[PlanFingerprint],
) -> Result<std::result::Result<ResidualFamily, SharedResidentExtractionReason>> {
    let plans = members
        .iter()
        .map(|fingerprint| {
            let first = groups[fingerprint].occurrences[0];
            fragments[first.fragment].resident_subplan(first.node)
        })
        .collect::<Result<Vec<_>>>()?;
    let representative = &plans[0];
    let PlanNode::Aggregate(aggregate) = &representative.root_node().kind else {
        unreachable!("residual families are aggregate-rooted");
    };
    if aggregate.spec.grouping_sets.is_some() {
        return Ok(Err(SharedResidentExtractionReason::ResidualNotMappedToOutput));
    }
    let group_keys = group_key_outputs(representative, aggregate.input, &aggregate.spec.group_exprs);
    let member_selections = plans.iter().map(source_selections).collect::<Vec<_>>();
    let member_parents = plans.iter().map(parents_by_node).collect::<Vec<_>>();

    let mut residuals = vec![Vec::<Expr>::new(); plans.len()];
    let mut factored_sources = Vec::with_capacity(member_selections[0].len());
    for (position, selection) in member_selections[0].iter().enumerate() {
        let selections = member_selections.iter().map(|selections| &selections[position]).collect::<Vec<_>>();
        let sources = plans
            .iter()
            .zip(&selections)
            .map(|(plan, selection)| local_parquet_source_at(plan, selection.source))
            .collect::<Vec<_>>();
        let readers = sources
            .iter()
            .map(|source| source.reader_predicate.clone().map(split_conjuncts).unwrap_or_default())
            .collect::<Vec<_>>();
        let conjuncts = plans
            .iter()
            .zip(&selections)
            .zip(&readers)
            .map(|((plan, selection), reader)| {
                let mut conjuncts = reader.clone();
                if let Some(filter) = selection.filter {
                    conjuncts.extend(split_conjuncts(filter_predicate_at(plan, filter).clone()));
                }
                conjuncts
            })
            .collect::<Vec<_>>();
        let schema = Arc::clone(sources[0].output_schema());
        let factored = factor_selections(&conjuncts, &schema);
        // A residual over group outputs restates the selection for the one
        // path its rows take to the aggregate. A source read along several
        // paths, such as both inputs of a self-join, would have every other
        // path read the widened rows.
        if factored.differing.iter().any(|differing| !differing.is_empty())
            && !plans
                .iter()
                .zip(&member_parents)
                .zip(&selections)
                .all(|((plan, parents), selection)| has_single_reader_path(plan, parents, selection.source))
        {
            return Ok(Err(SharedResidentExtractionReason::ResidualNotMappedToOutput));
        }
        let outputs = group_keys.get(&selection.source).map_or(&[][..], Vec::as_slice);
        for (residual, differing) in residuals.iter_mut().zip(&factored.differing) {
            let Some(mapped) = map_residual(differing, outputs, &schema) else {
                return Ok(Err(SharedResidentExtractionReason::ResidualNotMappedToOutput));
            };
            residual.extend(mapped);
        }
        let any_filter = selections.iter().any(|selection| selection.filter.is_some());
        factored_sources.push((sources, readers, factored, any_filter));
    }

    let mut nodes = representative.nodes().to_vec();
    let mut removed_filters = BTreeSet::new();
    let mut widened_sources = 0_usize;
    for (selection, (sources, readers, factored, any_filter)) in member_selections[0].iter().zip(factored_sources) {
        let selections_differ = factored.differing.iter().any(|differing| !differing.is_empty())
            || sources.iter().any(|other| {
                other.files.iter().map(|file| &file.row_groups).ne(sources[0].files.iter().map(|file| &file.row_groups))
            });
        widened_sources += usize::from(selections_differ);
        if selections_differ && sources.iter().any(|source| !source.selection_can_widen()) {
            return Ok(Err(SharedResidentExtractionReason::WidenedScanNotCovered));
        }
        // The widened predicate goes into the representative's selection
        // filter. Without one, the covering copy's own reader predicate must
        // already select exactly the widened rows.
        let in_filter = selection.filter.is_some() && factored.widened.is_some();
        if selection.filter.is_none() && any_filter && factored.widened.is_some() {
            return Ok(Err(SharedResidentExtractionReason::WidenedScanNotCovered));
        }
        if let (true, Some(filter), Some(widened)) = (in_filter, selection.filter, &factored.widened) {
            let PlanNode::Filter(filter) = &nodes[filter.index()].kind else {
                unreachable!("a selection filter is a filter node");
            };
            let input_schema = Arc::clone(nodes[filter.input.index()].output_schema());
            if widened.columns().any(|column| input_schema.field_with_name(column.name()).is_err()) {
                return Ok(Err(SharedResidentExtractionReason::WidenedScanNotCovered));
            }
        }
        let covering = (0..sources.len()).find(|&candidate| {
            let is_common = |conjunct: &Expr| {
                !factored.differing[candidate].iter().any(|differing| differing.to_string() == conjunct.to_string())
            };
            let reader_is_common = readers[candidate].iter().all(is_common);
            let reader_selects_widened =
                in_filter || factored.widened.is_none() || factored.differing[candidate].is_empty();
            reader_is_common
                && reader_selects_widened
                && sources.iter().all(|other| selects_row_group_superset(sources[candidate], other))
        });
        let Some(covering) = covering else {
            return Ok(Err(SharedResidentExtractionReason::WidenedScanNotCovered));
        };
        match (selection.filter, factored.widened) {
            (Some(filter), Some(widened)) => {
                let PlanNode::Filter(filter) = &mut nodes[filter.index()].kind else {
                    unreachable!("a selection filter is a filter node");
                };
                filter.predicate = widened;
            },
            (filter, _) => removed_filters.extend(filter),
        }
        let source = sources[covering].clone();
        nodes[selection.source.index()].kind = PlanNode::Source(Box::new(SourceNode::LocalParquet(source)));
    }
    // A copy reads the product of its own selections, and the producer the
    // product of their unions; with one widened source every row combination
    // is one some copy reads.
    if widened_sources > 1 && subplan_can_fail(representative, representative.root_id()) {
        return Ok(Err(SharedResidentExtractionReason::WidenedRowsCanFail));
    }
    // The node list keeps its structure, so the representative's routes keep
    // their endpoints; a wider build input only widens the runtime filter.
    let producer = representative
        .rewrite(RewrittenNodes::in_place(nodes, representative.root_id()))?
        .with_dynamic_filter_routes(representative.dynamic_filter_routes().to_vec());
    Ok(Ok(ResidualFamily {
        producer: without_filters(&producer, &removed_filters)?,
        members: members.iter().copied().zip(residuals.into_iter().map(conjunction)).collect(),
    }))
}

/// `plan` with each listed filter removed and its readers reading the
/// filter's input. Routes move with their nodes; a route through a removed
/// filter moves to that filter's input, and a route the filter consumed is
/// dropped.
fn without_filters(plan: &QueryPlan, filters: &BTreeSet<NodeId>) -> Result<QueryPlan> {
    if filters.is_empty() {
        return Ok(plan.clone());
    }
    let bypass = |mut node: NodeId| {
        while filters.contains(&node) {
            let PlanNode::Filter(filter) = &plan.node(node).expect("removed filters exist").kind else {
                unreachable!("only filters are removed");
            };
            node = filter.input;
        }
        node
    };
    let nodes = plan
        .nodes()
        .iter()
        .map(|node| QueryNode {
            id: node.id,
            kind: clone_plan_node_with_inputs(&node.kind, bypass),
            facts: node.facts.clone(),
        })
        .collect();
    // Every other node of the closed subplan stays reachable, so compaction
    // renumbers them in order.
    let compacted = |node: NodeId| {
        let node = bypass(node);
        NodeId(node.index() - filters.iter().filter(|filter| filter.index() < node.index()).count())
    };
    let routes = plan
        .dynamic_filter_routes()
        .iter()
        .filter_map(|route| {
            let producer = match route.producer {
                DynamicFilterProducer::HashJoinBuild { join_node_id, build_side, build_input } => {
                    DynamicFilterProducer::HashJoinBuild {
                        join_node_id: compacted(join_node_id),
                        build_side,
                        build_input: compacted(build_input),
                    }
                },
                DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id } => {
                    DynamicFilterProducer::AggregateGroupKeys { aggregate_node_id: compacted(aggregate_node_id) }
                },
            };
            let consumer = match route.consumer {
                DynamicFilterConsumer::Filter { filter_node_id } if filters.contains(&filter_node_id) => return None,
                DynamicFilterConsumer::Filter { filter_node_id } => {
                    DynamicFilterConsumer::Filter { filter_node_id: compacted(filter_node_id) }
                },
                DynamicFilterConsumer::LocalParquetSource { source_node_id } => {
                    DynamicFilterConsumer::LocalParquetSource { source_node_id: compacted(source_node_id) }
                },
                DynamicFilterConsumer::HashJoinProbe { join_node_id, probe_side } => {
                    DynamicFilterConsumer::HashJoinProbe { join_node_id: compacted(join_node_id), probe_side }
                },
            };
            Some(DynamicFilterRoute { producer, consumer, ..route.clone() })
        })
        .collect();
    Ok(plan.rewrite(RewrittenNodes::in_place(nodes, plan.root_id()))?.with_dynamic_filter_routes(routes))
}

/// One local Parquet source of a subplan and the filter that selects its
/// rows, when one sits directly above it.
struct SourceSelection {
    source: NodeId,
    filter: Option<NodeId>,
}

fn source_selections(plan: &QueryPlan) -> Vec<SourceSelection> {
    let parents = parents_by_node(plan);
    plan.nodes()
        .iter()
        .filter(|node| {
            matches!(&node.kind, PlanNode::Source(source) if matches!(source.as_ref(), SourceNode::LocalParquet(_)))
        })
        .map(|node| SourceSelection { source: node.id, filter: selection_filter(plan, &parents, node.id) })
        .collect()
}

/// The filter that is the only reader of `source`, possibly through
/// projections that only select columns under their own names, so its
/// columns are source columns.
fn selection_filter(plan: &QueryPlan, parents: &[Vec<NodeId>], source: NodeId) -> Option<NodeId> {
    let mut current = source;
    loop {
        let [parent] = parents[current.index()].as_slice() else {
            return None;
        };
        match &plan.node(*parent)?.kind {
            PlanNode::Filter(_) => return Some(*parent),
            PlanNode::Projection(projection) if projection_only_selects_columns(projection) => current = *parent,
            _ => return None,
        }
    }
}

/// Whether every node from `source` up to the subplan root, the source
/// included and the root excluded, has exactly one reader.
fn has_single_reader_path(plan: &QueryPlan, parents: &[Vec<NodeId>], source: NodeId) -> bool {
    let mut current = source;
    while current != plan.root_id() {
        let [parent] = parents[current.index()].as_slice() else {
            return false;
        };
        current = *parent;
    }
    true
}

/// Comparison fingerprint with every source's row groups and reader
/// predicate cleared, every selection filter removed and routes numbered in
/// order, so copies that select differently, or select only in some copies,
/// compare equal.
fn selection_wildcard_fingerprint(subplan: QueryPlan) -> Result<PlanFingerprint> {
    let plan = comparison_subplan(subplan)?;
    let mut nodes = plan.nodes().to_vec();
    let mut filters = BTreeSet::new();
    for selection in source_selections(&plan) {
        if let PlanNode::Source(source) = &mut nodes[selection.source.index()].kind
            && let SourceNode::LocalParquet(source) = source.as_mut()
        {
            source.reader_predicate = None;
            for file in &mut source.files {
                file.row_groups = None;
            }
        }
        filters.extend(selection.filter);
    }
    let cleared = plan
        .rewrite(RewrittenNodes::in_place(nodes, plan.root_id()))?
        .with_dynamic_filter_routes(plan.dynamic_filter_routes().to_vec());
    let wildcard = without_filters(&cleared, &filters)?;
    // Route ids number routes within one plan, so copies planned in
    // different units carry different ids for the same route.
    let routes = wildcard
        .dynamic_filter_routes()
        .iter()
        .enumerate()
        .map(|(index, route)| DynamicFilterRoute { id: DynamicFilterId::new(index), ..route.clone() })
        .collect();
    Ok(wildcard.with_dynamic_filter_routes(routes).execution_fingerprint())
}

fn local_parquet_source_at(plan: &QueryPlan, node: NodeId) -> &LocalParquetSource {
    local_parquet_source(plan, node).expect("family members share local Parquet source positions")
}

fn filter_predicate_at(plan: &QueryPlan, node: NodeId) -> &Expr {
    match &plan.node(node).expect("family members share node positions").kind {
        PlanNode::Filter(filter) => &filter.predicate,
        _ => unreachable!("family members share selection filter positions"),
    }
}

/// Whether `source` selects every row group `other` selects, file by file.
fn selects_row_group_superset(source: &LocalParquetSource, other: &LocalParquetSource) -> bool {
    source.files.len() == other.files.len()
        && source.files.iter().zip(&other.files).all(|(file, other)| match (&file.row_groups, &other.row_groups) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(selected), Some(other)) => other.iter().all(|row_group| selected.contains(row_group)),
        })
}

/// Group outputs of an aggregate over `input`, keyed by the local Parquet
/// source whose columns define them. Only definitions [`map_residual`] can
/// restate a conjunct over are listed.
fn group_key_outputs(plan: &QueryPlan, input: NodeId, group_exprs: &[Expr]) -> BTreeMap<NodeId, Vec<GroupKeyOutput>> {
    let mut outputs = BTreeMap::<NodeId, Vec<GroupKeyOutput>>::new();
    let output_fields = plan.root_node().output_schema().fields().clone();
    for (expr, field) in group_exprs.iter().zip(output_fields.iter()) {
        let Some((source, mut definition)) = source_expr(plan, input, expr) else {
            continue;
        };
        while let Expr::Alias { expr, .. } = definition {
            definition = *expr;
        }
        let schema = local_parquet_source_at(plan, source).output_schema();
        let definition = match definition {
            Expr::Column(column) => GroupKeyDefinition::Column(column.name().to_string()),
            Expr::Cast { expr, target_type } => {
                let Expr::Column(column) = expr.as_ref() else { continue };
                let Ok(source_field) = schema.field_with_name(column.name()) else { continue };
                if *source_field.data_type() == target_type {
                    GroupKeyDefinition::Column(column.name().to_string())
                } else if is_lossless_integer_widening(source_field.data_type(), &target_type) {
                    GroupKeyDefinition::WideningCast { column: column.name().to_string(), target: target_type }
                } else {
                    continue;
                }
            },
            definition if *field.data_type() == DataType::Boolean => GroupKeyDefinition::Boolean(definition),
            _ => continue,
        };
        outputs.entry(source).or_default().push(GroupKeyOutput { output: field.name().clone(), definition });
    }
    outputs
}

/// `expr` at `node` restated over the columns of the one local Parquet
/// source every column it reads comes from, through filters, sorts,
/// projections and joins that pass that source's rows through.
fn source_expr(plan: &QueryPlan, node: NodeId, expr: &Expr) -> Option<(NodeId, Expr)> {
    let mut source = None;
    let mut columns = BTreeMap::new();
    for column in expr.columns() {
        if column.qualifier().is_some() {
            return None;
        }
        let (column_source, resolved) = source_column_expr(plan, node, column.name())?;
        if *source.get_or_insert(column_source) != column_source {
            return None;
        }
        columns.insert(column.name().to_string(), resolved);
    }
    Some((source?, substitute_unqualified_columns(expr.clone(), &|name| columns.get(name).cloned())))
}

fn source_column_expr(plan: &QueryPlan, node_id: NodeId, name: &str) -> Option<(NodeId, Expr)> {
    match &plan.node(node_id)?.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) if source.output_schema().field_with_name(name).is_ok() => {
                Some((node_id, Expr::column(name)))
            },
            _ => None,
        },
        PlanNode::Filter(filter) => source_column_expr(plan, filter.input, name),
        PlanNode::Sort(sort) => source_column_expr(plan, sort.input, name),
        PlanNode::Projection(projection) => {
            let output = projection.outputs.iter().find(|output| output.field.name() == name)?;
            source_expr(plan, projection.input, &output.expr)
        },
        PlanNode::Join(join) => {
            let output = join.spec.output_projection.iter().find(|output| output.output_name == name)?;
            if !join.spec.join_type.passes_through_rows_of(output.side) {
                return None;
            }
            let input = match output.side {
                JoinSide::Left => join.inputs[0],
                JoinSide::Right => join.inputs[1],
            };
            source_column_expr(plan, input, output.column.name())
        },
        _ => None,
    }
}

fn analyze_fragment(plan: &QueryPlan) -> Result<FragmentAnalysis> {
    let mut fingerprints = Vec::with_capacity(plan.node_count());
    let mut subplan_sizes = Vec::with_capacity(plan.node_count());
    for node in plan.nodes() {
        let subplan = plan.resident_subplan(node.id)?;
        subplan_sizes.push(subplan.node_count());
        fingerprints.push(comparison_subplan(subplan)?.execution_fingerprint());
    }
    Ok(FragmentAnalysis { parents: parents_by_node(plan), fingerprints, subplan_sizes })
}

/// Projection names read only by other projections are private bindings. Rename
/// them by position in a comparison copy, retaining each evaluation boundary and
/// expression. Executable producers and their root output names remain unchanged.
fn comparison_subplan(plan: QueryPlan) -> Result<QueryPlan> {
    let parents = parents_by_node(&plan);
    let mut nodes = plan.nodes().to_vec();
    let mut changed = false;
    for index in 0..nodes.len() {
        let readers = &parents[index];
        if readers.is_empty()
            || !readers.iter().all(|reader| match &nodes[reader.index()].kind {
                PlanNode::Projection(projection) => {
                    projection.expressions().all(|expr| expr.columns().all(|column| column.qualifier().is_none()))
                },
                _ => false,
            })
        {
            continue;
        }
        let PlanNode::Projection(projection) = &nodes[index].kind else { continue };
        let names = projection
            .outputs
            .iter()
            .enumerate()
            .map(|(position, output)| (output.field.name().clone(), format!("__algeon_shared_projection_{position}")))
            .collect::<BTreeMap<_, _>>();
        if names.len() != projection.outputs.len() {
            continue;
        }
        let outputs = projection
            .outputs
            .iter()
            .map(|output| {
                let mut expr = output.expr.clone();
                while let Expr::Alias { expr: inner, .. } = expr {
                    expr = *inner;
                }
                ProjectionOutput::new(expr, output.field.clone().with_name(&names[output.field.name()]))
            })
            .collect();
        nodes[index].kind = PlanNode::Projection(ProjectionNode::new(projection.input, outputs));
        for ordering in &mut nodes[index].facts.output_ordering {
            ordering.expr =
                substitute_unqualified_columns(ordering.expr.clone(), &|name| names.get(name).map(Expr::column));
        }
        for reader in readers {
            let PlanNode::Projection(projection) = &nodes[reader.index()].kind else {
                unreachable!("only projection readers are normalized");
            };
            let outputs = projection
                .outputs
                .iter()
                .map(|output| {
                    let expr =
                        substitute_unqualified_columns(output.expr.clone(), &|name| names.get(name).map(Expr::column));
                    ProjectionOutput::new(expr, output.field.clone())
                })
                .collect();
            nodes[reader.index()].kind = PlanNode::Projection(ProjectionNode::new(projection.input, outputs));
        }
        changed = true;
    }
    if changed { plan.rewrite(RewrittenNodes::in_place(nodes, plan.root_id())) } else { Ok(plan) }
}

/// Decides one group. The checks run from structural to cost-based, and the
/// first failing one is the reported reason, so a repeated subplan under an
/// already selected ancestor reports nesting even when it would not retain.
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

/// Whether every node below `root` is read only from inside the subtree. An
/// open subtree cannot be replaced by one source, because the outside reader
/// would lose the node it reads.
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

    fn add_parquet_source(builder: &mut QueryPlanBuilder) -> NodeId {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        builder.add_source(SourceNode::LocalParquet(LocalParquetSource::from_files(
            vec![LocalParquetFile::new("/tmp/shared-resident-fragments-not-read.parquet")],
            Arc::clone(&schema),
            vec!["k".to_string(), "v".to_string()],
        )))
    }

    /// Adds `SUM(v) GROUP BY k` over one Parquet scan.
    fn add_grouped_sum(builder: &mut QueryPlanBuilder) -> NodeId {
        let source = add_parquet_source(builder);
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

    fn grouped_computed_sum(alias: &str, increment: i64) -> QueryPlan {
        let mut builder = QueryPlanBuilder::new();
        let source = add_parquet_source(&mut builder);
        let inner = builder
            .add_projection(
                source,
                vec![
                    ProjectionOutput::new(
                        Expr::binary(Expr::column("v"), BinaryOp::Add, Expr::literal(LiteralValue::Int64(increment)))
                            .alias(alias),
                        Field::new(alias, DataType::Int64, false),
                    ),
                    ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int64, false)),
                ],
            )
            .unwrap();
        let outer = builder
            .add_projection(
                inner,
                vec![
                    ProjectionOutput::new(Expr::column("k"), Field::new("k", DataType::Int64, false)),
                    ProjectionOutput::new(Expr::column(alias), Field::new("v", DataType::Int64, false)),
                ],
            )
            .unwrap();
        let aggregate = builder
            .add_aggregate(
                outer,
                AggregateSpec::new(
                    vec![Expr::column("k")],
                    vec![AggregateExpr::new(AggregateFunction::Sum, Expr::column("v"), "v")],
                ),
                Arc::new(Schema::new(vec![
                    Field::new("k", DataType::Int64, false),
                    Field::new("v", DataType::Int64, true),
                ])),
            )
            .unwrap();
        builder.finish_gpu_resident(aggregate).unwrap()
    }

    #[test]
    fn shares_private_projection_bindings_without_changing_executable_expressions() {
        let first = grouped_computed_sum("__common_expr_3", 1);
        let second = grouped_computed_sum("__common_expr_7", 1);
        let collision = grouped_computed_sum("__algeon_shared_projection_1", 1);
        let different = grouped_computed_sum("__common_expr_6", 2);
        let fingerprint = first.execution_fingerprint();
        let schema = Arc::clone(first.root_node().output_schema());
        assert_ne!(fingerprint, second.execution_fingerprint());

        let shared = extract_shared_resident_fragments(vec![first, second, collision, different]).unwrap();

        let [producer] = shared.producers.as_slice() else { panic!("one shared aggregate producer") };
        assert_eq!(producer.consumer_fragments, [0, 1, 2]);
        assert_eq!(producer.plan.execution_fingerprint(), fingerprint);
        for fragment in &shared.fragments[..3] {
            assert!(matches!(fragment.root_node().kind, PlanNode::Source(_)));
            assert_eq!(fragment.root_node().output_schema(), &schema);
        }
        assert!(matches!(shared.fragments[3].root_node().kind, PlanNode::Aggregate(_)));
        assert!(resident_source_names(&shared.fragments[3]).is_empty());
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
