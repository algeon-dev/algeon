fn scalar_materialization_input_shape(
    plan: &QueryPlan,
    node_id: NodeId,
    allow_column_projection_elision: bool,
) -> Option<String> {
    scalar_materialization_input_shape_inner(plan, node_id, true, allow_column_projection_elision)
}

fn scalar_materialization_input_shape_inner(
    plan: &QueryPlan,
    node_id: NodeId,
    allow_count_star_projection_elision: bool,
    allow_column_projection_elision: bool,
) -> Option<String> {
    let node = plan.node(node_id)?;
    match &node.kind {
        PlanNode::Source(source) => match source.as_ref() {
            SourceNode::LocalParquet(source) => {
                let signature = local_source_materialization_signature(source);
                Some(format!("local_parquet({})", signature.rowset_fingerprint))
            },
            _ => None,
        },
        PlanNode::Filter(filter) if source_reader_predicate_filter_is_elidable(plan, filter) => {
            scalar_materialization_input_shape_inner(
                plan,
                filter.input,
                allow_count_star_projection_elision,
                allow_column_projection_elision,
            )
        },
        PlanNode::Filter(filter) => Some(format!(
            "filter(predicate={};input={})",
            filter.predicate,
            scalar_materialization_input_shape_inner(plan, filter.input, false, allow_column_projection_elision)?
        )),
        PlanNode::Projection(projection) if identity_projection_is_elidable(projection) => {
            scalar_materialization_input_shape_inner(
                plan,
                projection.input,
                allow_count_star_projection_elision,
                allow_column_projection_elision,
            )
        },
        PlanNode::Projection(projection) if allow_column_projection_elision => {
            scalar_materialization_input_shape_inner(
                plan,
                projection.input,
                allow_count_star_projection_elision,
                allow_column_projection_elision,
            )
        },
        PlanNode::Projection(projection)
            if allow_count_star_projection_elision && count_star_projection_is_elidable(projection) =>
        {
            scalar_materialization_input_shape_inner(plan, projection.input, true, allow_column_projection_elision)
        },
        PlanNode::Projection(projection) => {
            let outputs = projection
                .outputs
                .iter()
                .map(|output| {
                    format!(
                        "{}:{:?}:nullable={}={}",
                        output.field.name(),
                        output.field.data_type(),
                        output.field.is_nullable(),
                        output.expr
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            Some(format!(
                "projection(outputs=[{outputs}];input={})",
                scalar_materialization_input_shape_inner(
                    plan,
                    projection.input,
                    false,
                    allow_column_projection_elision
                )?
            ))
        },
        PlanNode::Join(join) => Some(format!(
            "join(spec={:?};input0={};input1={})",
            join.spec,
            scalar_materialization_input_shape_inner(plan, join.inputs[0], false, allow_column_projection_elision)?,
            scalar_materialization_input_shape_inner(plan, join.inputs[1], false, allow_column_projection_elision)?
        )),
        PlanNode::Sort(sort) => Some(format!(
            "sort(spec={:?};input={})",
            sort.spec,
            scalar_materialization_input_shape_inner(plan, sort.input, false, allow_column_projection_elision)?
        )),
        PlanNode::Limit(_) => None,
        PlanNode::Aggregate(_)
        | PlanNode::Window(_)
        | PlanNode::Union(_)
        | PlanNode::EdgeNormalize(_)
        | PlanNode::GraphAlgorithm(_) => None,
    }
}

fn identity_projection_is_elidable(projection: &ProjectionNode) -> bool {
    !projection.outputs.is_empty()
        && projection.outputs.iter().all(|output| projection_output_preserves_column(output, output.field.name()))
}

fn count_star_projection_is_elidable(projection: &ProjectionNode) -> bool {
    projection.outputs.is_empty()
        || projection.outputs.iter().all(|output| {
            scalar_count_star_expr(&output.expr) || projection_output_preserves_column(output, output.field.name())
        })
}

#[derive(Debug, Clone)]
struct SharedAggregateFamilyRewrite {
    family_key: String,
    representative_aggregate_node_id: NodeId,
    widened_source_predicates: BTreeMap<String, Expr>,
    widened_source_columns: BTreeMap<String, BTreeMap<String, SourceColumnMaterialization>>,
    kind: SharedAggregateFamilyRewriteKind,
}

#[derive(Debug, Clone)]
enum SharedAggregateFamilyRewriteKind {
    Grouped { branch_filters: BTreeMap<NodeId, Option<Expr>> },
    ScalarFiltered { required_aggregate_input_columns: BTreeSet<String>, branches: Vec<ScalarFilteredAggregateBranch> },
}

#[derive(Debug, Clone)]
struct ScalarFilteredAggregateBranch {
    aggregate_node_id: NodeId,
    outputs: Vec<ScalarFilteredAggregateOutput>,
}

#[derive(Debug, Clone)]
struct ScalarFilteredAggregateOutput {
    original_output_field: Field,
    materialized_alias: String,
    aggregate_expr: AggregateExpr,
}

fn shared_aggregate_family_rewrite_plan(
    plan: &QueryPlan,
    family: &[AggregateMaterializationSignature],
) -> Result<SharedAggregateFamilyRewrite, SharedSubplanMaterializationRejectedReason> {
    let representative =
        family.first().ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
    if representative.group_exprs.is_empty() {
        return shared_scalar_filtered_aggregate_rewrite_plan(plan, family);
    }
    if family.iter().any(|signature| !source_materialization_sources_are_unique(signature)) {
        return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
    }
    // Reject disconnected candidates early so diagnostics do not claim an applied
    // rewrite that the sink-reachable clone cannot observe.
    let reachable = rewrite::sink_reachable_nodes(plan);
    if !family.iter().all(|signature| reachable.contains(&signature.aggregate_node_id)) {
        return Err(SharedSubplanMaterializationRejectedReason::NotReachableFromSink);
    }
    let mut predicates_by_source = BTreeMap::<String, BTreeSet<String>>::new();
    for signature in family {
        for source in &signature.source_predicates {
            predicates_by_source
                .entry(source.abstract_fingerprint.clone())
                .or_default()
                .insert(source.predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string));
        }
    }
    let widened_source_fingerprints = predicates_by_source
        .into_iter()
        .filter_map(|(fingerprint, predicates)| (predicates.len() > 1).then_some(fingerprint))
        .collect::<BTreeSet<_>>();
    if widened_source_fingerprints.is_empty() {
        return Ok(SharedAggregateFamilyRewrite {
            family_key: representative.family_key.clone(),
            representative_aggregate_node_id: representative.aggregate_node_id,
            widened_source_predicates: BTreeMap::new(),
            widened_source_columns: BTreeMap::new(),
            kind: SharedAggregateFamilyRewriteKind::Grouped {
                branch_filters: family.iter().map(|signature| (signature.aggregate_node_id, None)).collect(),
            },
        });
    }

    let mut widened_source_predicates = BTreeMap::new();
    let mut branch_filters = BTreeMap::new();
    for fingerprint in &widened_source_fingerprints {
        let mut predicates = Vec::new();
        let mut seen = BTreeSet::new();
        for signature in family {
            let source = signature
                .source_predicates
                .iter()
                .find(|source| &source.abstract_fingerprint == fingerprint)
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
            let predicate = source
                .predicate
                .clone()
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
            if !predicate_references_only_source_filterable_columns(&predicate, signature, fingerprint) {
                return Err(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable);
            }
            if seen.insert(predicate.to_string()) {
                predicates.push(predicate);
            }
        }
        widened_source_predicates.insert(
            fingerprint.clone(),
            disjunction(predicates)
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?,
        );
    }
    for signature in family {
        let mut predicates = Vec::new();
        for source in &signature.source_predicates {
            if !widened_source_fingerprints.contains(&source.abstract_fingerprint) {
                continue;
            }
            let predicate = source
                .predicate
                .clone()
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
            if !predicate_references_only_source_filterable_columns(&predicate, signature, &source.abstract_fingerprint)
            {
                return Err(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable);
            }
            predicates.push(predicate);
        }
        if predicates.is_empty() {
            return Err(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable);
        }
        branch_filters.insert(
            signature.aggregate_node_id,
            Some(
                conjunction(predicates)
                    .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?,
            ),
        );
    }

    Ok(SharedAggregateFamilyRewrite {
        family_key: representative.family_key.clone(),
        representative_aggregate_node_id: representative.aggregate_node_id,
        widened_source_predicates,
        widened_source_columns: BTreeMap::new(),
        kind: SharedAggregateFamilyRewriteKind::Grouped { branch_filters },
    })
}

fn shared_scalar_filtered_aggregate_rewrite_plan(
    plan: &QueryPlan,
    family: &[AggregateMaterializationSignature],
) -> Result<SharedAggregateFamilyRewrite, SharedSubplanMaterializationRejectedReason> {
    let representative =
        family.first().ok_or(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape)?;
    let representative_node = plan
        .node(representative.aggregate_node_id)
        .ok_or(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape)?;
    let PlanNode::Aggregate(representative_aggregate) = &representative_node.kind else {
        return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
    };
    if !scalar_filtered_aggregate_supported(&representative_aggregate.spec) {
        return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
    }
    if !scalar_filtered_aggregate_streaming_supported(&representative_aggregate.spec) {
        return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
    }
    if family.iter().any(|signature| !source_materialization_sources_are_unique(signature)) {
        return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
    }

    let mut predicates_by_source = BTreeMap::<String, BTreeSet<String>>::new();
    for signature in family {
        let node = plan
            .node(signature.aggregate_node_id)
            .ok_or(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape)?;
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
        };
        if !scalar_filtered_aggregate_supported(&aggregate.spec) {
            return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
        }
        if !scalar_filtered_aggregate_streaming_supported(&aggregate.spec) {
            return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
        }
        for aggregate_expr in &aggregate.spec.aggr_exprs {
            if scalar_count_star_aggregate_expr(plan, aggregate.input, &aggregate_expr.expr) {
                continue;
            }
            let Some(branch_signature) =
                scalar_input_expr_lineage_signature(plan, aggregate.input, &aggregate_expr.expr)
            else {
                return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
            };
            match scalar_input_expr_lineage_signature(plan, representative_aggregate.input, &aggregate_expr.expr) {
                Some(representative_signature) if branch_signature != representative_signature => {
                    return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
                },
                Some(_) => {},
                None => {
                    if scalar_input_expr_name_collides_with_unresolved_output(
                        plan,
                        representative_aggregate.input,
                        &aggregate_expr.expr,
                    ) {
                        return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
                    }
                },
            }
        }
        for source in &signature.source_predicates {
            predicates_by_source
                .entry(source.abstract_fingerprint.clone())
                .or_default()
                .insert(source.predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string));
        }
    }

    let widened_source_fingerprints = predicates_by_source
        .into_iter()
        .filter_map(|(fingerprint, predicates)| (predicates.len() > 1).then_some(fingerprint))
        .collect::<BTreeSet<_>>();
    if widened_source_fingerprints.is_empty() {
        return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
    }

    let mut widened_source_predicates = BTreeMap::new();
    let mut required_source_columns = BTreeMap::<String, BTreeSet<String>>::new();
    let mut widened_source_columns = BTreeMap::<String, BTreeMap<String, SourceColumnMaterialization>>::new();
    for fingerprint in &widened_source_fingerprints {
        let mut predicates = Vec::new();
        let mut seen = BTreeSet::new();
        let mut required_columns = BTreeSet::new();
        for signature in family {
            if let Some(columns) = signature.source_columns.get(fingerprint) {
                merge_source_columns(widened_source_columns.entry(fingerprint.clone()).or_default(), columns)?;
            }
            let source = signature
                .source_predicates
                .iter()
                .find(|source| &source.abstract_fingerprint == fingerprint)
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
            let predicate = source
                .predicate
                .clone()
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
            collect_expr_column_names(&predicate, &mut required_columns);
            if seen.insert(predicate.to_string()) {
                predicates.push(predicate);
            }
        }
        required_source_columns.insert(fingerprint.clone(), required_columns);
        widened_source_predicates.insert(
            fingerprint.clone(),
            disjunction(predicates)
                .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?,
        );
    }

    let mut required_aggregate_input_columns = BTreeSet::new();
    for columns in required_source_columns.values() {
        for column in columns {
            if !subtree_can_propagate_materialized_column(
                plan,
                representative_aggregate.input,
                column,
                &widened_source_columns,
            ) {
                return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
            }
            required_aggregate_input_columns.insert(column.clone());
        }
    }
    for signature in family {
        let node = plan
            .node(signature.aggregate_node_id)
            .ok_or(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape)?;
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
        };
        for aggregate_expr in &aggregate.spec.aggr_exprs {
            if scalar_count_star_aggregate_expr(plan, aggregate.input, &aggregate_expr.expr) {
                continue;
            }
            collect_expr_column_names(&aggregate_expr.expr, &mut required_aggregate_input_columns);
        }
        for source in &signature.source_predicates {
            if let Some(columns) = signature.source_columns.get(&source.abstract_fingerprint) {
                merge_source_columns(
                    widened_source_columns.entry(source.abstract_fingerprint.clone()).or_default(),
                    columns,
                )?;
            }
        }
    }
    if !source_column_names_are_unique_across_sources(&widened_source_columns) {
        return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
    }
    for column in &required_aggregate_input_columns {
        if !subtree_can_propagate_materialized_column(
            plan,
            representative_aggregate.input,
            column,
            &widened_source_columns,
        ) {
            return Err(SharedSubplanMaterializationRejectedReason::PredicateColumnPropagationUnsupported);
        }
    }

    let mut branches = Vec::with_capacity(family.len());
    for signature in family {
        let node = plan
            .node(signature.aggregate_node_id)
            .ok_or(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape)?;
        let PlanNode::Aggregate(aggregate) = &node.kind else {
            return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
        };
        if aggregate.spec.aggr_exprs.len() != node.output_schema().fields().len() {
            return Err(SharedSubplanMaterializationRejectedReason::UnsupportedScalarAggregateShape);
        }
        let mut predicates = Vec::new();
        for source in &signature.source_predicates {
            if !widened_source_fingerprints.contains(&source.abstract_fingerprint) {
                continue;
            }
            predicates.push(
                source
                    .predicate
                    .clone()
                    .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?,
            );
        }
        let filter = conjunction(predicates)
            .ok_or(SharedSubplanMaterializationRejectedReason::PredicateNotPostAggregateFilterable)?;
        let outputs = aggregate
            .spec
            .aggr_exprs
            .iter()
            .zip(node.output_schema().fields())
            .enumerate()
            .map(|(index, (aggregate_expr, field))| {
                let count_star = scalar_count_star_aggregate_expr(plan, aggregate.input, &aggregate_expr.expr);
                let mut aggregate_expr = aggregate_expr.clone();
                aggregate_expr.alias =
                    format!("__cudf_shared_scalar_{}_{}", signature.aggregate_node_id.index(), index);
                aggregate_expr.filter = Some(filter.clone());
                if count_star {
                    aggregate_expr.expr = Expr::literal(crate::expr::LiteralValue::Int64(1));
                }
                ScalarFilteredAggregateOutput {
                    original_output_field: field.as_ref().clone(),
                    materialized_alias: aggregate_expr.alias.clone(),
                    aggregate_expr,
                }
            })
            .collect();
        branches.push(ScalarFilteredAggregateBranch { aggregate_node_id: signature.aggregate_node_id, outputs });
    }

    let reachable = rewrite::sink_reachable_nodes(plan);
    if !family.iter().all(|signature| reachable.contains(&signature.aggregate_node_id)) {
        return Err(SharedSubplanMaterializationRejectedReason::NotReachableFromSink);
    }

    Ok(SharedAggregateFamilyRewrite {
        family_key: representative.family_key.clone(),
        representative_aggregate_node_id: representative.aggregate_node_id,
        widened_source_predicates,
        widened_source_columns,
        kind: SharedAggregateFamilyRewriteKind::ScalarFiltered { required_aggregate_input_columns, branches },
    })
}

struct SharedSubplanMaterializationRewriter<'a> {
    original: &'a QueryPlan,
    families_by_key: BTreeMap<String, SharedAggregateFamilyRewrite>,
    replacements: BTreeMap<NodeId, String>,
    materialized_nodes: BTreeMap<String, NodeId>,
    branch_nodes: BTreeMap<NodeId, NodeId>,
    nodes: Vec<QueryNode>,
    memo: BTreeMap<NodeId, NodeId>,
    diagnostics: SharedSubplanMaterializationDiagnostics,
    changed: bool,
}

impl<'a> SharedSubplanMaterializationRewriter<'a> {
    fn new(
        original: &'a QueryPlan,
        families: Vec<SharedAggregateFamilyRewrite>,
        diagnostics: SharedSubplanMaterializationDiagnostics,
    ) -> Self {
        let mut replacements = BTreeMap::new();
        let mut families_by_key = BTreeMap::new();
        for family in families {
            match &family.kind {
                SharedAggregateFamilyRewriteKind::Grouped { branch_filters } => {
                    for aggregate_node_id in branch_filters.keys() {
                        replacements.insert(*aggregate_node_id, family.family_key.clone());
                    }
                },
                SharedAggregateFamilyRewriteKind::ScalarFiltered { branches, .. } => {
                    for branch in branches {
                        replacements.insert(branch.aggregate_node_id, family.family_key.clone());
                    }
                },
            }
            families_by_key.insert(family.family_key.clone(), family);
        }
        Self {
            original,
            families_by_key,
            replacements,
            materialized_nodes: BTreeMap::new(),
            branch_nodes: BTreeMap::new(),
            nodes: Vec::new(),
            memo: BTreeMap::new(),
            diagnostics,
            changed: false,
        }
    }

    fn rewrite_subtree(&mut self, old_id: NodeId) -> NodeId {
        if self.replacements.contains_key(&old_id) {
            self.changed = true;
            return self.branch_node(old_id);
        }
        if let Some(new_id) = self.memo.get(&old_id) {
            return *new_id;
        }

        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        let new_kind = self.clone_kind_with_rewritten_inputs(&old_node.kind);
        let new_id = self.push_node(new_kind, old_node.facts.clone());
        self.memo.insert(old_id, new_id);
        new_id
    }

    fn branch_node(&mut self, old_aggregate_id: NodeId) -> NodeId {
        if let Some(new_id) = self.branch_nodes.get(&old_aggregate_id) {
            return *new_id;
        }
        let family_key = self.replacements.get(&old_aggregate_id).expect("replacement family must exist").clone();
        let materialized_id = self.materialized_node(&family_key);
        let family = self.families_by_key.get(&family_key).expect("replacement family must exist").clone();
        let new_id = match &family.kind {
            SharedAggregateFamilyRewriteKind::Grouped { branch_filters } => {
                let predicate = branch_filters.get(&old_aggregate_id).cloned().flatten();
                if let Some(predicate) = predicate {
                    let mut facts =
                        self.nodes.get(materialized_id.index()).expect("materialized node must exist").facts.clone();
                    facts.schema_behavior = SchemaBehavior::PreservesInput;
                    self.push_node(PlanNode::Filter(FilterNode { input: materialized_id, predicate }), facts)
                } else {
                    materialized_id
                }
            },
            SharedAggregateFamilyRewriteKind::ScalarFiltered { branches, .. } => {
                let branch = branches
                    .iter()
                    .find(|branch| branch.aggregate_node_id == old_aggregate_id)
                    .expect("scalar filtered branch must exist");
                let outputs = branch
                    .outputs
                    .iter()
                    .map(|output| {
                        ProjectionOutput::new(
                            Expr::column(output.materialized_alias.clone())
                                .alias(output.original_output_field.name().clone()),
                            output.original_output_field.clone(),
                        )
                    })
                    .collect();
                let projection = ProjectionNode::new(materialized_id, outputs);
                let facts = PlanFacts::device(Arc::clone(projection.output_schema()), SchemaBehavior::Projection);
                self.push_node(PlanNode::Projection(projection), facts)
            },
        };
        self.branch_nodes.insert(old_aggregate_id, new_id);
        new_id
    }

    fn materialized_node(&mut self, family_key: &str) -> NodeId {
        if let Some(new_id) = self.materialized_nodes.get(family_key) {
            return *new_id;
        }
        let family = self.families_by_key.get(family_key).expect("materialization family must exist").clone();
        let new_id = self.clone_materialized_subtree(family.representative_aggregate_node_id, &family);
        self.materialized_nodes.insert(family_key.to_string(), new_id);
        new_id
    }

    fn clone_materialized_subtree(&mut self, old_id: NodeId, family: &SharedAggregateFamilyRewrite) -> NodeId {
        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        if old_id == family.representative_aggregate_node_id
            && let SharedAggregateFamilyRewriteKind::ScalarFiltered {
                required_aggregate_input_columns, branches, ..
            } = &family.kind
        {
            let PlanNode::Aggregate(aggregate) = &old_node.kind else {
                panic!("scalar filtered materialization root must be an aggregate");
            };
            let input = self.clone_materialized_subtree_with_required_columns(
                aggregate.input,
                family,
                required_aggregate_input_columns,
            );
            let spec = AggregateSpec::new(
                Vec::new(),
                branches
                    .iter()
                    .flat_map(|branch| branch.outputs.iter().map(|output| output.aggregate_expr.clone()))
                    .collect(),
            );
            let output_schema = Arc::new(Schema::new(
                branches
                    .iter()
                    .flat_map(|branch| {
                        branch.outputs.iter().map(|output| {
                            Field::new(
                                output.materialized_alias.clone(),
                                output.original_output_field.data_type().clone(),
                                output.original_output_field.is_nullable(),
                            )
                        })
                    })
                    .collect::<Vec<_>>(),
            ));
            return self.push_node(
                PlanNode::Aggregate(crate::plan::AggregateNode { input, spec }),
                PlanFacts::device(output_schema, SchemaBehavior::Aggregate),
            );
        }
        let new_facts = old_node.facts.clone();
        let new_kind = match &old_node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                SourceNode::LocalParquet(source) => {
                    let mut source = source.clone();
                    let signature = local_source_materialization_signature(&source);
                    let mut source_facts = None;
                    if let Some(columns) = matching_source_entry(&family.widened_source_columns, &signature)
                        && widen_local_parquet_source(&mut source, columns)
                    {
                        source_facts =
                            Some(PlanFacts::device(Arc::clone(&source.schema), SchemaBehavior::SourceOutput));
                    }
                    if let Some(predicate) = matching_source_entry(&family.widened_source_predicates, &signature) {
                        source.reader_predicate = Some(predicate.clone());
                        source.refresh_diagnostics();
                    }
                    let new_kind = PlanNode::Source(Box::new(SourceNode::LocalParquet(source)));
                    return self.push_node(new_kind, source_facts.unwrap_or_else(|| old_node.facts.clone()));
                },
                _ => PlanNode::Source(source.clone()),
            },
            _ => self.clone_kind_with_input_mapper(&old_node.kind, |this, input| {
                this.clone_materialized_subtree(input, family)
            }),
        };
        self.push_node(new_kind, new_facts)
    }

    fn clone_materialized_subtree_with_required_columns(
        &mut self,
        old_id: NodeId,
        family: &SharedAggregateFamilyRewrite,
        required_output_columns: &BTreeSet<String>,
    ) -> NodeId {
        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        match &old_node.kind {
            PlanNode::Source(source) => {
                let (new_kind, facts) = match source.as_ref() {
                    SourceNode::LocalParquet(source) => {
                        let mut source = source.clone();
                        let signature = local_source_materialization_signature(&source);
                        let mut facts = old_node.facts.clone();
                        if let Some(columns) = matching_source_entry(&family.widened_source_columns, &signature)
                            && widen_local_parquet_source(&mut source, columns)
                        {
                            facts = PlanFacts::device(Arc::clone(&source.schema), SchemaBehavior::SourceOutput);
                        }
                        if let Some(predicate) = matching_source_entry(&family.widened_source_predicates, &signature) {
                            source.reader_predicate = Some(predicate.clone());
                            source.refresh_diagnostics();
                        }
                        (PlanNode::Source(Box::new(SourceNode::LocalParquet(source))), facts)
                    },
                    _ => (PlanNode::Source(source.clone()), old_node.facts.clone()),
                };
                self.push_node(new_kind, facts)
            },
            PlanNode::Filter(filter) => {
                if source_reader_predicate_filter_is_elidable(self.original, filter) {
                    return self.clone_materialized_subtree_with_required_columns(
                        filter.input,
                        family,
                        required_output_columns,
                    );
                }
                let mut input_required = required_output_columns.clone();
                collect_expr_column_names(&filter.predicate, &mut input_required);
                let input =
                    self.clone_materialized_subtree_with_required_columns(filter.input, family, &input_required);
                let mut filter = filter.clone();
                filter.input = input;
                let input_schema = self.nodes[input.index()].output_schema().clone();
                self.push_node(
                    PlanNode::Filter(filter),
                    PlanFacts::device(input_schema, SchemaBehavior::PreservesInput),
                )
            },
            PlanNode::Projection(projection) => {
                let mut outputs = projection.outputs.clone();
                let mut output_names =
                    outputs.iter().map(|output| output.field.name().clone()).collect::<BTreeSet<_>>();
                for column in required_output_columns {
                    if output_names.insert(column.clone()) {
                        let input_field = field_with_name(
                            self.original.node(projection.input).expect("projection input must exist").output_schema(),
                            column,
                        )
                        .cloned()
                        .or_else(|| {
                            propagatable_materialized_field(
                                self.original,
                                projection.input,
                                column,
                                &family.widened_source_columns,
                            )
                        })
                        .expect("required projection column must exist on input");
                        outputs.push(ProjectionOutput::new(Expr::column(column.clone()), input_field));
                    }
                }
                let mut input_required = BTreeSet::new();
                for output in &outputs {
                    collect_expr_column_names(&output.expr, &mut input_required);
                }
                let input =
                    self.clone_materialized_subtree_with_required_columns(projection.input, family, &input_required);
                let projection = ProjectionNode::new(input, outputs);
                let facts = PlanFacts::device(Arc::clone(projection.output_schema()), SchemaBehavior::Projection);
                self.push_node(PlanNode::Projection(projection), facts)
            },
            PlanNode::Join(join) => {
                let mut output_projection = join.spec.output_projection.clone();
                let mut left_required = BTreeSet::new();
                let mut right_required = BTreeSet::new();
                for key in &join.spec.keys {
                    left_required.insert(key.left.name().to_string());
                    right_required.insert(key.right.name().to_string());
                }
                if let Some(predicate) = &join.spec.residual_predicate {
                    collect_join_expr_columns(predicate, &mut left_required, &mut right_required);
                }
                for output in &output_projection {
                    match output.side {
                        JoinSide::Left => {
                            left_required.insert(output.column.name().to_string());
                        },
                        JoinSide::Right => {
                            right_required.insert(output.column.name().to_string());
                        },
                    }
                }

                for column in required_output_columns {
                    if let Some(output) = output_projection.iter().find(|output| output.output_name == *column) {
                        match output.side {
                            JoinSide::Left => {
                                left_required.insert(output.column.name().to_string());
                            },
                            JoinSide::Right => {
                                right_required.insert(output.column.name().to_string());
                            },
                        }
                        continue;
                    }
                    let side = unique_join_input_that_can_propagate_column(
                        self.original,
                        join.inputs[0],
                        join.inputs[1],
                        column,
                    )
                    .or_else(|| {
                        unique_join_input_that_can_propagate_materialized_column(
                            self.original,
                            join.inputs[0],
                            join.inputs[1],
                            column,
                            &family.widened_source_columns,
                        )
                    })
                    .expect("required join column must have one input owner");
                    match side {
                        JoinSide::Left => {
                            left_required.insert(column.clone());
                            output_projection.push(JoinOutputColumn::left(
                                crate::expr::ColumnRef::input(0, column.clone(), column.clone()),
                                column.clone(),
                            ));
                        },
                        JoinSide::Right => {
                            right_required.insert(column.clone());
                            output_projection.push(JoinOutputColumn::right(
                                crate::expr::ColumnRef::input(1, column.clone(), column.clone()),
                                column.clone(),
                            ));
                        },
                    }
                }

                let left =
                    self.clone_materialized_subtree_with_required_columns(join.inputs[0], family, &left_required);
                let right =
                    self.clone_materialized_subtree_with_required_columns(join.inputs[1], family, &right_required);
                let mut join = join.clone();
                join.inputs = [left, right];
                join.spec.output_projection = output_projection;
                let output_schema = join_output_schema(
                    self.nodes[left.index()].output_schema(),
                    self.nodes[right.index()].output_schema(),
                    &join.spec.output_projection,
                );
                self.push_node(PlanNode::Join(join), PlanFacts::device(output_schema, SchemaBehavior::Join))
            },
            PlanNode::Sort(sort) => {
                let mut input_required = required_output_columns.clone();
                for sort_expr in &sort.spec.expressions {
                    collect_expr_column_names(&sort_expr.expr, &mut input_required);
                }
                let input = self.clone_materialized_subtree_with_required_columns(sort.input, family, &input_required);
                let mut sort = sort.clone();
                sort.input = input;
                let mut facts = old_node.facts.clone();
                facts.output_schema = self.nodes[input.index()].output_schema().clone();
                self.push_node(PlanNode::Sort(sort), facts)
            },
            PlanNode::Limit(limit) => {
                let input =
                    self.clone_materialized_subtree_with_required_columns(limit.input, family, required_output_columns);
                let mut limit = limit.clone();
                limit.input = input;
                let mut facts = old_node.facts.clone();
                facts.output_schema = self.nodes[input.index()].output_schema().clone();
                self.push_node(PlanNode::Limit(limit), facts)
            },
            PlanNode::Aggregate(_)
            | PlanNode::Window(_)
            | PlanNode::Union(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_) => {
                panic!("unsupported scalar filtered materialization subtree")
            },
        }
    }

    fn clone_kind_with_rewritten_inputs(&mut self, kind: &PlanNode) -> PlanNode {
        self.clone_kind_with_input_mapper(kind, |this, input| this.rewrite_subtree(input))
    }

    fn clone_kind_with_input_mapper(
        &mut self,
        kind: &PlanNode,
        mut rewrite_input: impl FnMut(&mut Self, NodeId) -> NodeId,
    ) -> PlanNode {
        rewrite::clone_plan_node_with_inputs(kind, |input| rewrite_input(self, input))
    }

    fn push_node(&mut self, kind: PlanNode, facts: PlanFacts) -> NodeId {
        let id = NodeId::from_index_for_internal_use(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}
