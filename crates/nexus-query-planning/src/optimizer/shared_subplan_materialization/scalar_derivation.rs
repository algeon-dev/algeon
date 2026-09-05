#[derive(Debug, Clone)]
struct ScalarAggregateDerivationRewrite {
    scalar_aggregate_node_id: NodeId,
    grouped_aggregate_node_id: NodeId,
    derived_spec: AggregateSpec,
}

#[doc(hidden)]
pub fn optimize_scalar_aggregate_from_grouped_aggregate(plan: QueryPlan) -> SharedSubplanMaterializationResult {
    let discovery = discover_scalar_aggregate_from_grouped_aggregate_candidates(&plan);
    if discovery.rewrites.is_empty() {
        return SharedSubplanMaterializationResult {
            plan,
            change: super::NativeOptimizerChange::Unchanged,
            diagnostics: discovery.diagnostics,
        };
    }

    let root = plan.root_id();
    let sink = plan.sink().clone();
    let mut rewriter = ScalarAggregateDerivationRewriter::new(&plan, discovery.rewrites);
    let new_root = rewriter.rewrite_subtree(root);
    let rewritten_plan = rewrite::finish_rewritten_plan(rewriter.nodes, new_root, &sink)
        .with_keyed_hash_join_strategy_options_from(&plan)
            .with_advisory_outputs_from(&plan);
    SharedSubplanMaterializationResult {
        plan: rewritten_plan,
        change: super::NativeOptimizerChange::Changed,
        diagnostics: discovery.diagnostics,
    }
}

struct ScalarAggregateDerivationDiscovery {
    diagnostics: SharedSubplanMaterializationDiagnostics,
    rewrites: Vec<ScalarAggregateDerivationRewrite>,
}

fn discover_scalar_aggregate_from_grouped_aggregate_candidates(plan: &QueryPlan) -> ScalarAggregateDerivationDiscovery {
    let reachable = rewrite::sink_reachable_nodes(plan);
    let aggregates = plan
        .nodes()
        .iter()
        .filter_map(|node| match &node.kind {
            PlanNode::Aggregate(aggregate) if reachable.contains(&node.id) => Some((node.id, aggregate)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let grouped =
        aggregates.iter().filter(|(_, aggregate)| derivable_grouped_aggregate(aggregate)).copied().collect::<Vec<_>>();
    let mut rowset_signatures = BTreeMap::new();
    let mut diagnostics = SharedSubplanMaterializationDiagnostics::default();
    let mut rewrites = Vec::new();

    for (scalar_id, scalar_aggregate) in aggregates {
        if !derivable_scalar_aggregate(scalar_aggregate) {
            continue;
        }
        diagnostics.considered_count += 1;
        let Some(scalar_rowset_signature) =
            rowset_signature(plan, scalar_aggregate.input, &mut rowset_signatures, &mut BTreeSet::new())
        else {
            continue;
        };
        for (grouped_id, grouped_aggregate) in &grouped {
            if *grouped_id == scalar_id {
                continue;
            }
            let Some(grouped_rowset_signature) =
                rowset_signature(plan, grouped_aggregate.input, &mut rowset_signatures, &mut BTreeSet::new())
            else {
                continue;
            };
            if grouped_rowset_signature != scalar_rowset_signature {
                continue;
            }
            let Some(derived_spec) =
                derived_scalar_aggregate_spec(plan, *grouped_id, grouped_aggregate, scalar_id, scalar_aggregate)
            else {
                continue;
            };

            let repeated_selected_source_bytes = sum_optional_u64([
                selected_source_bytes_below(plan, grouped_aggregate.input),
                selected_source_bytes_below(plan, scalar_aggregate.input),
            ]);
            let duplicate_selected_source_bytes = repeated_selected_source_bytes
                .zip(
                    selected_source_bytes_below(plan, grouped_aggregate.input)
                        .max(selected_source_bytes_below(plan, scalar_aggregate.input)),
                )
                .map(|(total, largest)| total.saturating_sub(largest));
            let exact_source_variant_count =
                exact_source_variant_count_for_inputs(plan, grouped_aggregate.input, scalar_aggregate.input);

            diagnostics.candidate_count += 1;
            diagnostics.applied_count += 1;
            diagnostics.candidates.push(SharedSubplanMaterializationCandidateDiagnostics {
                rule_name: SHARED_SUBPLAN_MATERIALIZATION_RULE_NAME,
                family_key: format!(
                    "shared-grouped-scalar-v1-fnv64-{:016x}",
                    fnv1a64(scalar_rowset_signature.as_bytes())
                ),
                aggregate_node_ids: vec![*grouped_id, scalar_id],
                reuse_count: 2,
                group_exprs: grouped_aggregate.spec.group_exprs.iter().map(ToString::to_string).collect(),
                aggregate_exprs: scalar_aggregate
                    .spec
                    .aggr_exprs
                    .iter()
                    .map(|expr| aggregate_expr_signature(plan, scalar_aggregate.input, expr))
                    .collect(),
                filtered_aggregate_count: 0,
                filtered_aggregate_memory_shape: "not_applicable".to_string(),
                filtered_aggregate_estimation_input_rows: None,
                filtered_aggregate_estimated_materialization_bytes: None,
                filtered_aggregate_memory_budget_bytes: None,
                filtered_aggregate_memory_threshold_bytes: None,
                filtered_aggregate_memory_threshold_policy: "not_applicable".to_string(),
                source_fingerprints: vec![scalar_rowset_signature],
                exact_source_variant_count,
                repeated_selected_source_bytes,
                duplicate_selected_source_bytes,
                applied: true,
                rejected_reason: None,
            });
            rewrites.push(ScalarAggregateDerivationRewrite {
                scalar_aggregate_node_id: scalar_id,
                grouped_aggregate_node_id: *grouped_id,
                derived_spec,
            });
            break;
        }
    }

    ScalarAggregateDerivationDiscovery { diagnostics, rewrites }
}

fn derivable_grouped_aggregate(aggregate: &AggregateNode) -> bool {
    !aggregate.spec.group_exprs.is_empty()
        && aggregate.spec.grouping_sets.is_none()
        && aggregate.spec.aggr_exprs.iter().any(derivable_sum_expr)
}

fn derivable_scalar_aggregate(aggregate: &AggregateNode) -> bool {
    aggregate.spec.group_exprs.is_empty()
        && aggregate.spec.grouping_sets.is_none()
        && !aggregate.spec.aggr_exprs.is_empty()
        && aggregate.spec.aggr_exprs.iter().all(derivable_sum_expr)
}

fn derivable_sum_expr(expr: &AggregateExpr) -> bool {
    expr.function == AggregateFunction::Sum && !expr.distinct && expr.filter.is_none()
}

fn derived_scalar_aggregate_spec(
    plan: &QueryPlan,
    grouped_aggregate_node_id: NodeId,
    grouped_aggregate: &AggregateNode,
    scalar_aggregate_node_id: NodeId,
    scalar_aggregate: &AggregateNode,
) -> Option<AggregateSpec> {
    let mut used_group_aliases = BTreeSet::new();
    let mut aggr_exprs = Vec::with_capacity(scalar_aggregate.spec.aggr_exprs.len());
    for scalar_expr in &scalar_aggregate.spec.aggr_exprs {
        if !sum_output_is_exact(plan, scalar_aggregate_node_id, &scalar_expr.alias) {
            return None;
        }
        let scalar_arg_signature =
            normalized_aggregate_argument_signature(plan, scalar_aggregate.input, &scalar_expr.expr);
        let grouped_expr = grouped_aggregate.spec.aggr_exprs.iter().find(|grouped_expr| {
            derivable_sum_expr(grouped_expr)
                && grouped_expr.function == scalar_expr.function
                && sum_output_is_exact(plan, grouped_aggregate_node_id, &grouped_expr.alias)
                && normalized_aggregate_argument_signature(plan, grouped_aggregate.input, &grouped_expr.expr)
                    == scalar_arg_signature
                && used_group_aliases.insert(grouped_expr.alias.clone())
        })?;
        aggr_exprs.push(AggregateExpr::new(
            AggregateFunction::Sum,
            Expr::column(grouped_expr.alias.clone()),
            scalar_expr.alias.clone(),
        ));
    }
    Some(AggregateSpec::new(Vec::new(), aggr_exprs))
}

fn sum_output_is_exact(plan: &QueryPlan, aggregate_node_id: NodeId, alias: &str) -> bool {
    let Some(data_type) = plan.node(aggregate_node_id).and_then(|node| {
        node.output_schema().fields().iter().find(|field| field.name() == alias).map(|field| field.data_type())
    }) else {
        return false;
    };
    !matches!(data_type, DataType::Float16 | DataType::Float32 | DataType::Float64)
}

fn normalized_aggregate_argument_signature(plan: &QueryPlan, input: NodeId, expr: &Expr) -> String {
    if let Expr::Column(column) = expr
        && let Some(node) = plan.node(input)
        && let PlanNode::Projection(projection) = &node.kind
        && let Some(output) = projection.outputs.iter().find(|output| output.field.name() == column.name())
    {
        return output.expr.to_string();
    }
    expr.to_string()
}

fn rowset_signature(
    plan: &QueryPlan,
    node_id: NodeId,
    memo: &mut BTreeMap<NodeId, Option<String>>,
    visiting: &mut BTreeSet<NodeId>,
) -> Option<String> {
    if let Some(signature) = memo.get(&node_id) {
        return signature.clone();
    }
    if !visiting.insert(node_id) {
        memo.insert(node_id, None);
        return None;
    }
    let signature = rowset_signature_uncached(plan, node_id, memo, visiting);
    visiting.remove(&node_id);
    memo.insert(node_id, signature.clone());
    signature
}

fn rowset_signature_uncached(
    plan: &QueryPlan,
    node_id: NodeId,
    memo: &mut BTreeMap<NodeId, Option<String>>,
    visiting: &mut BTreeSet<NodeId>,
) -> Option<String> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => Some(local_parquet_rowset_signature(source)),
            SourceNode::GpuResident(source) => Some(format!("gpu_resident:{}", source.input_name)),
            SourceNode::LiteralBatch(source) => Some(format!("literal_batch:{:?}", source.batches)),
        },
        PlanNode::Filter(filter) => rowset_signature(plan, filter.input, memo, visiting)
            .map(|input| format!("filter({input};predicate={})", filter.predicate)),
        PlanNode::Projection(projection) => rowset_signature(plan, projection.input, memo, visiting),
        PlanNode::Join(join) => {
            if !matches!(join.spec.join_type, crate::plan::JoinType::Inner | crate::plan::JoinType::Cross) {
                return None;
            }
            let left = rowset_signature(plan, join.inputs[0], memo, visiting)?;
            let right = rowset_signature(plan, join.inputs[1], memo, visiting)?;
            let keys =
                join.spec.keys.iter().map(|key| format!("{}={}", key.left, key.right)).collect::<Vec<_>>().join(",");
            let residual =
                join.spec.residual_predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string);
            Some(format!(
                "join(type={},keys={keys},residual={residual},nulls={:?},left={left},right={right})",
                join.spec.join_type, join.spec.null_equality
            ))
        },
        PlanNode::Sort(sort) => rowset_signature(plan, sort.input, memo, visiting),
        _ => None,
    }
}

fn local_parquet_rowset_signature(source: &LocalParquetSource) -> String {
    let mut file_signatures = source.files.iter().map(local_parquet_file_rowset_signature).collect::<Vec<_>>();
    file_signatures.sort();
    let files = file_signatures.join("+");
    let predicate = source.reader_predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string);
    format!(
        "local_parquet(files={files};predicate={predicate};scan_limit={};max_rows={};zero_column_count={})",
        source.scan_limit.map_or_else(|| "none".to_string(), |limit| limit.to_string()),
        source.max_rows.map_or_else(|| "none".to_string(), |rows| rows.to_string()),
        source.is_zero_column_count_source(),
    )
}

fn local_parquet_file_rowset_signature(file: &crate::source::LocalParquetFile) -> String {
    let row_groups = file
        .row_groups
        .as_ref()
        .map(|row_groups| row_groups.iter().map(ToString::to_string).collect::<Vec<_>>().join(","))
        .unwrap_or_else(|| "all".to_string());
    format!(
        "stable_id={};location={};resolved_contract={};row_groups={};partition_values={:?}",
        file.location.diagnostic_stable_id(file.stable_id.as_str()),
        file.location.diagnostic_location(),
        resolved_source_contract_signature(&file.resolved_source),
        row_groups,
        file.partition_values
    )
}

fn selected_source_bytes_below(plan: &QueryPlan, input: NodeId) -> Option<u64> {
    let mut sources = Vec::new();
    collect_local_parquet_sources(plan, input, &mut BTreeSet::new(), &mut sources);
    sum_optional_u64(sources.iter().map(|source| source.selected_source_bytes))
}

fn exact_source_variant_count_for_inputs(plan: &QueryPlan, left: NodeId, right: NodeId) -> usize {
    let mut variants = BTreeSet::new();
    for input in [left, right] {
        let mut sources = Vec::new();
        collect_local_parquet_sources(plan, input, &mut BTreeSet::new(), &mut sources);
        variants.insert(sources.iter().map(|source| source.exact_fingerprint.as_str()).collect::<Vec<_>>().join("|"));
    }
    variants.len()
}

struct ScalarAggregateDerivationRewriter<'a> {
    original: &'a QueryPlan,
    replacements: BTreeMap<NodeId, ScalarAggregateDerivationRewrite>,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<NodeId, NodeId>,
}

impl<'a> ScalarAggregateDerivationRewriter<'a> {
    fn new(original: &'a QueryPlan, rewrites: Vec<ScalarAggregateDerivationRewrite>) -> Self {
        Self {
            original,
            replacements: rewrites.into_iter().map(|rewrite| (rewrite.scalar_aggregate_node_id, rewrite)).collect(),
            nodes: Vec::new(),
            memo: BTreeMap::new(),
        }
    }

    fn rewrite_subtree(&mut self, old_id: NodeId) -> NodeId {
        if let Some(new_id) = self.memo.get(&old_id) {
            return *new_id;
        }

        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        let new_kind = if let Some(replacement) = self.replacements.get(&old_id).cloned() {
            let input = self.rewrite_subtree(replacement.grouped_aggregate_node_id);
            PlanNode::Aggregate(AggregateNode { input, spec: replacement.derived_spec })
        } else {
            self.clone_kind_with_rewritten_inputs(&old_node.kind)
        };
        let new_id = self.push_node(new_kind, old_node.facts.clone());
        self.memo.insert(old_id, new_id);
        new_id
    }

    fn clone_kind_with_rewritten_inputs(&mut self, kind: &PlanNode) -> PlanNode {
        rewrite::clone_plan_node_with_inputs(kind, |input| self.rewrite_subtree(input))
    }

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        if let Some(existing) = self.nodes.iter().find(|node| node.kind == kind && node.facts == facts) {
            return existing.id;
        }
        let id = NodeId::from_index_for_internal_use(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}
