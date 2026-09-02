use super::*;

pub(in crate::optimizer::aggregate_join_pushdown) struct AggregateJoinPushdownRewriter<'a> {
    pub(in crate::optimizer::aggregate_join_pushdown) original: &'a QueryPlan,
    pub(in crate::optimizer::aggregate_join_pushdown) nodes: Vec<QueryNode>,
    pub(in crate::optimizer::aggregate_join_pushdown) memo: BTreeMap<NodeId, NodeId>,
    pub(in crate::optimizer::aggregate_join_pushdown) diagnostics: AggregateJoinPushdownDiagnostics,
    pub(in crate::optimizer::aggregate_join_pushdown) changed: bool,
}

impl<'a> AggregateJoinPushdownRewriter<'a> {
    pub(in crate::optimizer::aggregate_join_pushdown) fn new(original: &'a QueryPlan) -> Self {
        Self {
            original,
            nodes: Vec::new(),
            memo: BTreeMap::new(),
            diagnostics: AggregateJoinPushdownDiagnostics::default(),
            changed: false,
        }
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn rewrite_subtree(&mut self, old_id: NodeId) -> NodeId {
        if let Some(new_id) = self.memo.get(&old_id) {
            return *new_id;
        }

        let old_node = self.original.node(old_id).expect("optimizer input node must exist");
        let new_id = match &old_node.kind {
            PlanNode::Aggregate(aggregate) => self.rewrite_aggregate(old_id, aggregate, old_node.facts.clone()),
            _ => {
                let new_kind = self.clone_kind_with_rewritten_inputs(&old_node.kind);
                self.push_node(new_kind, old_node.facts.clone())
            },
        };
        self.memo.insert(old_id, new_id);
        new_id
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn rewrite_aggregate(
        &mut self,
        old_id: NodeId,
        aggregate: &AggregateNode,
        facts: PlanFacts,
    ) -> NodeId {
        let rewritten_input = self.rewrite_subtree(aggregate.input);
        let rewritten_aggregate = AggregateNode { input: rewritten_input, spec: aggregate.spec.clone() };

        let candidate = self.try_rewrite_aggregate_join(old_id, &rewritten_aggregate, &facts);
        match candidate {
            Ok(Some(new_id)) => {
                self.changed = true;
                self.diagnostics.considered_count += 1;
                self.diagnostics.applied_count += 1;
                self.diagnostics.candidates.push(AggregateJoinPushdownCandidateDiagnostics {
                    rule_name: AGGREGATE_JOIN_PUSHDOWN_RULE_NAME,
                    aggregate_node_id: old_id,
                    join_node_id: Some(rewritten_input),
                    applied: true,
                    rejected_reason: None,
                });
                new_id
            },
            Ok(None) => self.push_node(PlanNode::Aggregate(rewritten_aggregate), facts),
            Err(reason) => {
                self.diagnostics.considered_count += 1;
                self.diagnostics.candidates.push(AggregateJoinPushdownCandidateDiagnostics {
                    rule_name: AGGREGATE_JOIN_PUSHDOWN_RULE_NAME,
                    aggregate_node_id: old_id,
                    join_node_id: (reason != AggregateJoinPushdownRejectedReason::InputNotJoin)
                        .then_some(rewritten_input),
                    applied: false,
                    rejected_reason: Some(reason),
                });
                self.push_node(PlanNode::Aggregate(rewritten_aggregate), facts)
            },
        }
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn try_rewrite_aggregate_join(
        &mut self,
        _aggregate_id: NodeId,
        aggregate: &AggregateNode,
        aggregate_facts: &PlanFacts,
    ) -> Result<Option<NodeId>, AggregateJoinPushdownRejectedReason> {
        let (join, projection_chain) = self.resolve_aggregate_join_input(aggregate.input)?;
        if join.spec.residual_predicate.is_some() {
            return Err(AggregateJoinPushdownRejectedReason::JoinResidualUnsupported);
        }

        match join.spec.join_type {
            JoinType::Inner => self.rewrite_inner_join_aggregate(aggregate, aggregate_facts, &join, &projection_chain),
            JoinType::Left => {
                self.rewrite_left_join_count_aggregate(aggregate, aggregate_facts, &join, &projection_chain)
            },
            JoinType::LeftSemi => {
                self.rewrite_left_semi_join_aggregate(aggregate, aggregate_facts, &join, &projection_chain)
            },
            _ => Err(AggregateJoinPushdownRejectedReason::UnsupportedJoinType),
        }
        .map(Some)
    }

    /// Walks down from the aggregate's direct input through any [`PlanNode::Projection`]
    /// chain until reaching a [`PlanNode::Join`]. Returns the join and the chain
    /// of intervening projections with the outermost first (closest to the
    /// aggregate) and the innermost last (closest to the join).
    pub(in crate::optimizer::aggregate_join_pushdown) fn resolve_aggregate_join_input(
        &self,
        aggregate_input: NodeId,
    ) -> Result<(JoinNode, Vec<ProjectionNode>), AggregateJoinPushdownRejectedReason> {
        let mut chain = Vec::new();
        let mut current = aggregate_input;
        loop {
            let node = self
                .nodes
                .get(current.index())
                .filter(|node| node.id == current)
                .ok_or(AggregateJoinPushdownRejectedReason::InputNotJoin)?;
            match &node.kind {
                PlanNode::Join(join) => return Ok((join.clone(), chain)),
                PlanNode::Projection(projection) => {
                    let projection = projection.clone();
                    current = projection.input;
                    chain.push(projection);
                },
                _ => return Err(AggregateJoinPushdownRejectedReason::InputNotJoin),
            }
        }
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn rewrite_inner_join_aggregate(
        &mut self,
        aggregate: &AggregateNode,
        aggregate_facts: &PlanFacts,
        join: &JoinNode,
        projection_chain: &[ProjectionNode],
    ) -> Result<NodeId, AggregateJoinPushdownRejectedReason> {
        let lineage = AggregateJoinLineage::new(aggregate, join, projection_chain)?;
        let measure_side = lineage.single_measure_side()?;
        // For inner joins we allow group keys on either side or split across
        // both: the pre-aggregate only carries measure-side group keys, and the
        // final aggregate still sees the other-side keys through the rebuilt
        // join's output projection.
        let preagg_group_columns = lineage.group_columns_for_side(measure_side);
        self.rewrite_join_aggregate_with_preagg(
            aggregate,
            aggregate_facts,
            join,
            measure_side,
            preagg_group_columns,
            &lineage,
            JoinType::Inner,
            false,
        )
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn rewrite_left_join_count_aggregate(
        &mut self,
        aggregate: &AggregateNode,
        aggregate_facts: &PlanFacts,
        join: &JoinNode,
        projection_chain: &[ProjectionNode],
    ) -> Result<NodeId, AggregateJoinPushdownRejectedReason> {
        let lineage = AggregateJoinLineage::new(aggregate, join, projection_chain)?;
        if lineage.single_group_side()? != JoinSide::Left || lineage.single_measure_side()? != JoinSide::Right {
            return Err(AggregateJoinPushdownRejectedReason::UnsupportedSidePattern);
        }
        if !lineage.measure_columns.iter().all(|measure| {
            measure.side_input_aggregate.function == AggregateFunction::Count
                && matches!(measure.side_input_aggregate.expr, Expr::Column(_))
                && !measure.side_input_aggregate.distinct
                && measure.side_input_aggregate.filter.is_none()
        }) {
            return Err(AggregateJoinPushdownRejectedReason::UnsupportedAggregateExpression);
        }

        self.rewrite_join_aggregate_with_preagg(
            aggregate,
            aggregate_facts,
            join,
            JoinSide::Right,
            lineage.group_columns_for_side(JoinSide::Right),
            &lineage,
            JoinType::Left,
            true,
        )
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn rewrite_left_semi_join_aggregate(
        &mut self,
        aggregate: &AggregateNode,
        aggregate_facts: &PlanFacts,
        join: &JoinNode,
        projection_chain: &[ProjectionNode],
    ) -> Result<NodeId, AggregateJoinPushdownRejectedReason> {
        let lineage = AggregateJoinLineage::new(aggregate, join, projection_chain)?;
        // LeftSemi only emits left-side columns, so both measures and groups
        // must live on the left input.
        if lineage.single_measure_side()? != JoinSide::Left || lineage.single_group_side()? != JoinSide::Left {
            return Err(AggregateJoinPushdownRejectedReason::UnsupportedSidePattern);
        }
        self.rewrite_join_aggregate_with_preagg(
            aggregate,
            aggregate_facts,
            join,
            JoinSide::Left,
            lineage.group_columns_for_side(JoinSide::Left),
            &lineage,
            JoinType::LeftSemi,
            false,
        )
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn simple_inner_join_preaggregate_has_large_detail_key_risk(
        &self,
        aggregate: &AggregateNode,
        join: &JoinNode,
        preagg_side: JoinSide,
        preagg_group_columns: &[PreAggregateColumn],
    ) -> Result<bool, AggregateJoinPushdownRejectedReason> {
        if aggregate.spec.grouping_sets.is_some() || preagg_group_columns.is_empty() {
            return Ok(false);
        }
        let preagg_input = join.inputs[join_side_index(preagg_side)];
        let Some(input_rows) = self.estimated_rewritten_output_rows(preagg_input) else {
            return Ok(false);
        };
        if !AggregateJoinPushdownPolicy::default().simple_inner_detail_preaggregate_needs_reduction_evidence(input_rows)
        {
            return Ok(false);
        }
        let input_schema = self
            .nodes
            .get(preagg_input.index())
            .filter(|node| node.id == preagg_input)
            .map(|node| node.output_schema().clone())
            .ok_or(AggregateJoinPushdownRejectedReason::MissingInputField)?;
        for column in preagg_group_columns {
            let field = input_schema
                .field_with_name(column.source_name.as_str())
                .map_err(|_| AggregateJoinPushdownRejectedReason::MissingInputField)?;
            if arrow_fixed_width_byte_size(field.data_type()).is_none() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn rewrite_join_aggregate_with_preagg(
        &mut self,
        aggregate: &AggregateNode,
        aggregate_facts: &PlanFacts,
        join: &JoinNode,
        preagg_side: JoinSide,
        mut preagg_group_columns: Vec<PreAggregateColumn>,
        lineage: &AggregateJoinLineage,
        join_type: JoinType,
        coalesce_preagg_nulls_to_zero: bool,
    ) -> Result<NodeId, AggregateJoinPushdownRejectedReason> {
        let preagg_input = join.inputs[join_side_index(preagg_side)];
        let preagg_input_schema = self
            .nodes
            .get(preagg_input.index())
            .filter(|node| node.id == preagg_input)
            .map(|node| node.output_schema().clone())
            .ok_or(AggregateJoinPushdownRejectedReason::MissingInputField)?;
        for key in &join.spec.keys {
            let key_column = join_key_for_side(key, preagg_side);
            push_unique_preagg_group_column(
                &mut preagg_group_columns,
                key_column.name(),
                key_column.display_name(),
                &preagg_input_schema,
            )?;
        }
        if join_type == JoinType::Inner
            && self.simple_inner_join_preaggregate_has_large_detail_key_risk(
                aggregate,
                join,
                preagg_side,
                &preagg_group_columns,
            )?
        {
            return Err(AggregateJoinPushdownRejectedReason::DetailKeyPreAggregateReductionNotEvidenced);
        }

        let preagg_state_aliases =
            preaggregate_state_aliases(aggregate, aggregate_facts.output_schema.as_ref(), &preagg_group_columns);
        let reserved_preagg_aliases =
            preagg_state_aliases.iter().flat_map(|aliases| aliases.iter().cloned()).collect::<Vec<_>>();
        // The aggregate executor requires aggregate measure arguments to be
        // plain column references. When every measure already resolves to a
        // plain column from the measure-side input (the q70/q86 case with no
        // projection chain), skip the materializing projection and have the
        // pre-aggregate read those columns directly. When at least one measure
        // came from a computed projection-chain expression (the q67 case),
        // materialize all measures once under fresh aliases so the pre-aggregate
        // can still reference them as plain columns.
        let needs_materializing_projection =
            lineage.measure_columns.iter().any(|measure| !matches!(measure.side_input_aggregate.expr, Expr::Column(_)));

        let (preagg_input_id, preagg_input_schema_for_spec, measure_input_columns) = if needs_materializing_projection {
            let aggregate_input_schema = self
                .nodes
                .get(aggregate.input.index())
                .filter(|node| node.id == aggregate.input)
                .map(|node| node.output_schema().clone())
                .ok_or(AggregateJoinPushdownRejectedReason::MissingInputField)?;
            let measure_input_aliases = aggregate
                .spec
                .aggr_exprs
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    unique_measure_input_alias(
                        index,
                        aggregate_facts.output_schema.as_ref(),
                        &preagg_group_columns,
                        &reserved_preagg_aliases,
                    )
                })
                .collect::<Vec<_>>();
            let projection_outputs = measure_input_projection_outputs(
                &preagg_input_schema,
                &aggregate_input_schema,
                aggregate,
                &preagg_group_columns,
                lineage,
                &measure_input_aliases,
            )?;
            let projection_node = ProjectionNode::new(preagg_input, projection_outputs);
            let projection_schema = Arc::clone(projection_node.output_schema());
            let projection_id = self.push_node(
                PlanNode::Projection(projection_node),
                PlanFacts::device(Arc::clone(&projection_schema), SchemaBehavior::Projection),
            );
            (projection_id, projection_schema, measure_input_aliases)
        } else {
            // Plain-column measures: pre-aggregate reads each measure column
            // by the side-input name the lineage already resolved.
            let measure_input_columns = lineage
                .measure_columns
                .iter()
                .map(|measure| {
                    let Expr::Column(column) = &measure.side_input_aggregate.expr else {
                        unreachable!("plain-column branch only entered when all measures are plain columns")
                    };
                    column.name().to_string()
                })
                .collect::<Vec<_>>();
            (preagg_input, preagg_input_schema, measure_input_columns)
        };

        let preagg_measure_states = preaggregate_measure_states(
            &preagg_input_schema_for_spec,
            aggregate,
            aggregate_facts.output_schema.as_ref(),
            &preagg_state_aliases,
            &measure_input_columns,
        )?;
        let preagg_schema = preaggregate_schema(
            &preagg_input_schema_for_spec,
            aggregate,
            aggregate_facts.output_schema.as_ref(),
            &preagg_group_columns,
            &preagg_measure_states,
        )?;
        let preagg_spec =
            preaggregate_spec(aggregate, &preagg_group_columns, &preagg_measure_states, &measure_input_columns)?;
        let preagg_id = self
            .try_rewrite_preaggregate_through_join_union(preagg_input_id, &preagg_spec, &preagg_schema)
            .unwrap_or_else(|| {
                self.push_node(
                    PlanNode::Aggregate(AggregateNode { input: preagg_input_id, spec: preagg_spec }),
                    PlanFacts::device(Arc::clone(&preagg_schema), SchemaBehavior::Aggregate),
                )
            });

        let new_join =
            self.rewritten_join_after_preagg(lineage, join, preagg_side, preagg_id, &preagg_measure_states, join_type);
        let join_schema = new_join_output_schema(
            aggregate,
            aggregate_facts.output_schema.as_ref(),
            &preagg_schema,
            &preagg_measure_states,
            preagg_side,
            join_type,
        )?;
        let join_id = self.push_node(PlanNode::Join(new_join), PlanFacts::device(join_schema, SchemaBehavior::Join));
        let final_input = if coalesce_preagg_nulls_to_zero {
            let projection_outputs = coalesce_preagg_projection_outputs(
                aggregate,
                aggregate_facts.output_schema.as_ref(),
                &preagg_measure_states,
            )?;
            let projection_node = ProjectionNode::new(join_id, projection_outputs);
            let projection_schema = Arc::clone(projection_node.output_schema());
            self.push_node(
                PlanNode::Projection(projection_node),
                PlanFacts::device(projection_schema, SchemaBehavior::Projection),
            )
        } else {
            join_id
        };
        let final_rewrite = final_aggregate_rewrite_after_preagg(
            aggregate,
            aggregate_facts.output_schema.as_ref(),
            &preagg_measure_states,
        )?;
        let final_aggregate_id = self.push_node(
            PlanNode::Aggregate(AggregateNode { input: final_input, spec: final_rewrite.spec }),
            PlanFacts::device(Arc::clone(&final_rewrite.output_schema), SchemaBehavior::Aggregate),
        );
        if let Some(projection_outputs) = final_rewrite.projection_outputs {
            let projection_node = ProjectionNode::new(final_aggregate_id, projection_outputs);
            let projection_schema = Arc::clone(projection_node.output_schema());
            return Ok(self.push_node(
                PlanNode::Projection(projection_node),
                PlanFacts::device(projection_schema, SchemaBehavior::Projection),
            ));
        }
        Ok(final_aggregate_id)
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn rewritten_join_after_preagg(
        &self,
        lineage: &AggregateJoinLineage,
        join: &JoinNode,
        preagg_side: JoinSide,
        preagg_id: NodeId,
        preagg_measure_states: &[PreAggregateMeasureState],
        join_type: JoinType,
    ) -> JoinNode {
        let other_side = opposite_side(preagg_side);
        let inputs = match preagg_side {
            JoinSide::Left => [preagg_id, join.inputs[join_side_index(other_side)]],
            JoinSide::Right => [join.inputs[join_side_index(other_side)], preagg_id],
        };
        let keys =
            join.spec.keys.iter().map(|key| rewritten_join_key_after_preagg(key, preagg_side)).collect::<Vec<_>>();
        let state_count = preagg_measure_states.iter().map(|measure| measure.state_columns.len()).sum::<usize>();
        let mut output_projection = Vec::with_capacity(lineage.group_columns.len() + state_count);
        for group in &lineage.group_columns {
            let side = group.side;
            let join_input_index =
                if side == preagg_side { join_side_index(preagg_side) } else { join_side_index(other_side) };
            let source_name = if side == preagg_side { group.output_name.as_str() } else { group.source_name.as_str() };
            output_projection.push(JoinOutputColumn::new(
                side,
                ColumnRef::input(join_input_index, source_name, group.output_name.as_str()),
                group.output_name.as_str(),
            ));
        }
        for state in preagg_measure_states.iter().flat_map(|measure| &measure.state_columns) {
            output_projection.push(JoinOutputColumn::new(
                preagg_side,
                ColumnRef::input(join_side_index(preagg_side), state.alias.as_str(), state.alias.as_str()),
                state.alias.as_str(),
            ));
        }
        let spec = JoinSpec {
            join_type,
            keys,
            residual_predicate: None,
            output_projection,
            null_equality: join.spec.null_equality,
            build_side: join.spec.build_side,
            datafusion_order_preserved_side: join.spec.datafusion_order_preserved_side,
            build_artifact_id: None,
            mark_column_name: None,
            cardinality_contract: None,
            total_payload_bound_preserving_outputs: Vec::new(),
        };
        JoinNode { inputs, spec }
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn try_rewrite_preaggregate_through_join_union(
        &mut self,
        preagg_input: NodeId,
        preagg_spec: &AggregateSpec,
        preagg_schema: &SchemaRef,
    ) -> Option<NodeId> {
        if preagg_spec.grouping_sets.is_some()
            || preagg_spec.aggr_exprs.is_empty()
            || !preagg_spec.aggr_exprs.iter().all(decomposable_union_partial_aggregate_expr)
        {
            return None;
        }

        let input_node = self.nodes.get(preagg_input.index()).filter(|node| node.id == preagg_input)?;
        let PlanNode::Join(input_join) = &input_node.kind else {
            return None;
        };
        if input_join.spec.join_type != JoinType::Inner
            || input_join.spec.residual_predicate.is_some()
            || input_join.spec.build_artifact_id.is_some()
            || input_join.spec.mark_column_name.is_some()
        {
            return None;
        }

        let (union_side, union_inputs) = self.distributable_join_union_input(input_join)?;
        let other_side = opposite_side(union_side);
        let other_input = input_join.inputs[join_side_index(other_side)];
        let other_bytes = self.estimated_rewritten_output_bytes(other_input)?;
        let known_union_bytes = union_inputs
            .iter()
            .try_fold(0_u64, |sum, input| sum.checked_add(self.estimated_rewritten_output_bytes(*input)?))?;
        if !AggregateJoinPushdownPolicy::default().join_union_preaggregate_admitted(
            known_union_bytes,
            other_bytes,
            union_inputs.len(),
        ) {
            return None;
        }

        let input_join = input_join.clone();
        let join_facts = input_node.facts.clone();
        let branch_aggregates = union_inputs
            .into_iter()
            .map(|branch_input| {
                let branch_join = join_with_replaced_input(&input_join, union_side, branch_input);
                let branch_join_id = self.push_node(PlanNode::Join(branch_join), join_facts.clone());
                self.push_node(
                    PlanNode::Aggregate(AggregateNode { input: branch_join_id, spec: preagg_spec.clone() }),
                    PlanFacts::device(Arc::clone(preagg_schema), SchemaBehavior::Aggregate),
                )
            })
            .collect::<Vec<_>>();
        let partial_union = self.push_node(
            PlanNode::Union(crate::plan::UnionNode { inputs: branch_aggregates }),
            PlanFacts::device(Arc::clone(preagg_schema), SchemaBehavior::Union),
        );
        let final_spec = final_aggregate_spec_after_union_preagg(preagg_spec, preagg_schema)?;
        Some(self.push_node(
            PlanNode::Aggregate(AggregateNode { input: partial_union, spec: final_spec }),
            PlanFacts::device(Arc::clone(preagg_schema), SchemaBehavior::Aggregate),
        ))
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn distributable_join_union_input(
        &self,
        join: &JoinNode,
    ) -> Option<(JoinSide, Vec<NodeId>)> {
        [JoinSide::Left, JoinSide::Right].into_iter().find_map(|side| {
            let input = join.inputs[join_side_index(side)];
            let node = self.nodes.get(input.index()).filter(|node| node.id == input)?;
            let PlanNode::Union(union) = &node.kind else {
                return None;
            };
            (union.inputs.len() > 1).then(|| (side, union.inputs.clone()))
        })
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn estimated_rewritten_output_bytes(
        &self,
        node_id: NodeId,
    ) -> Option<u64> {
        let node = self.nodes.get(node_id.index()).filter(|node| node.id == node_id)?;
        match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                crate::source::SourceNode::LocalParquet(source) => source
                    .diagnostics
                    .selected_row_group_stats
                    .as_ref()
                    .and_then(|stats| stats.selected_source_bytes())
                    .or(source.diagnostics.known_bytes),
                crate::source::SourceNode::GpuResident(_) => None,
                crate::source::SourceNode::LiteralBatch(source) => source
                    .batches
                    .iter()
                    .try_fold(0_u64, |sum, batch| sum.checked_add(batch.get_array_memory_size() as u64)),
            },
            PlanNode::Projection(projection) => self.estimated_rewritten_output_bytes(projection.input),
            PlanNode::Filter(filter) => self.estimated_rewritten_output_bytes(filter.input),
            PlanNode::Sort(sort) => self.estimated_rewritten_output_bytes(sort.input),
            PlanNode::Union(union) => union
                .inputs
                .iter()
                .try_fold(0_u64, |total, input| total.checked_add(self.estimated_rewritten_output_bytes(*input)?)),
            PlanNode::Limit(_)
            | PlanNode::Aggregate(_)
            | PlanNode::Join(_)
            | PlanNode::Window(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_) => None,
        }
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn estimated_rewritten_output_rows(
        &self,
        node_id: NodeId,
    ) -> Option<u64> {
        let node = self.nodes.get(node_id.index()).filter(|node| node.id == node_id)?;
        match &node.kind {
            PlanNode::Source(source) => match source.as_ref() {
                crate::source::SourceNode::LocalParquet(source) => {
                    let rows = source
                        .files
                        .iter()
                        .map(|file| {
                            file.row_group_rows.as_ref()?.iter().try_fold(0_u64, |sum, rows| sum.checked_add(*rows))
                        })
                        .try_fold(0_u64, |sum, rows| sum.checked_add(rows?))?;
                    let rows = source.scan_limit.map_or(rows, |limit| rows.min(limit as u64));
                    Some(source.max_rows.map_or(rows, |max_rows| rows.min(max_rows as u64)))
                },
                crate::source::SourceNode::GpuResident(_) => None,
                crate::source::SourceNode::LiteralBatch(source) => Some(source.row_count() as u64),
            },
            PlanNode::Projection(projection) => self.estimated_rewritten_output_rows(projection.input),
            PlanNode::Filter(filter) => self.estimated_rewritten_output_rows(filter.input),
            PlanNode::Window(window) => self.estimated_rewritten_output_rows(window.input),
            PlanNode::Sort(sort) => self.estimated_rewritten_output_rows(sort.input),
            PlanNode::Limit(limit) => self.estimated_rewritten_output_rows(limit.input).map(|rows| {
                limit.spec.fetch.map_or(rows.saturating_sub(limit.spec.skip as u64), |fetch| {
                    rows.min(limit.spec.skip.saturating_add(fetch) as u64).saturating_sub(limit.spec.skip as u64)
                })
            }),
            PlanNode::Union(union) => union
                .inputs
                .iter()
                .try_fold(0_u64, |total, input| total.checked_add(self.estimated_rewritten_output_rows(*input)?)),
            PlanNode::Aggregate(aggregate) if aggregate.spec.group_exprs.is_empty() => Some(1),
            PlanNode::Aggregate(_)
            | PlanNode::Join(_)
            | PlanNode::EdgeNormalize(_)
            | PlanNode::GraphAlgorithm(_)
            | PlanNode::VectorAlgorithm(_) => None,
        }
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn clone_kind_with_rewritten_inputs(
        &mut self,
        kind: &PlanNode,
    ) -> PlanNode {
        rewrite::clone_plan_node_with_inputs(kind, |input| self.rewrite_subtree(input))
    }

    pub(in crate::optimizer::aggregate_join_pushdown) fn push_node(
        &mut self,
        kind: PlanNode,
        facts: PlanFacts,
    ) -> NodeId {
        if let Some(existing) = self.nodes.iter().find(|node| node.kind == kind && node.facts == facts) {
            return existing.id;
        }
        let id = NodeId::from_index_for_internal_use(self.nodes.len());
        self.nodes.push(QueryNode { id, kind, facts });
        id
    }
}
