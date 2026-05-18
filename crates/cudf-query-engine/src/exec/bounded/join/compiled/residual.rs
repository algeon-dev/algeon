use super::*;

pub(super) fn materialize_compiled_one_shot_residual_output(
    context: &ExecutionContext,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    indices: CompiledJoinIndices,
    candidate_rows: u64,
    gather_map_reservation: ActiveReservation,
    timer: Instant,
    index_bytes: u64,
    reservations: &mut ReservationTracker,
    metrics: &mut QueryMetrics,
) -> Result<ResidentGpuFrame> {
    let overrides = execution_overrides(context);
    let residual_candidate = one_shot_residual_candidate_projection(node)?;
    let gather_map_bytes = indices.device_allocated_size_bytes(node)?;
    let candidate_reservation_bytes =
        indices.output_reservation_size_bytes(&left.frame, &right.frame, &overrides, &residual_candidate.projection)?;
    metrics.record_one_shot_join();
    metrics.record_one_shot_join_index_bytes(index_bytes);
    metrics.record_one_shot_join_materialization_bytes(candidate_reservation_bytes);
    let candidate_reservation = reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_residual_candidate_output",
            candidate_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    )?;
    let candidate = indices.materialize_output(
        &left.frame,
        &right.frame,
        &overrides,
        &residual_candidate.projection,
        node.spec.join_type,
    )?;
    let candidate_output_rows = candidate.num_rows() as u64;
    debug_assert_eq!(candidate_output_rows, candidate_rows);
    drop(indices);
    reservations.release(gather_map_reservation, metrics);

    let mut filtered_reservation = match reservations.try_reserve(
        bounded_join_request(
            ReservationKind::JoinOutput,
            "join_residual_filter_output",
            candidate_reservation_bytes,
            reservations.budget(),
            node_id,
        ),
        metrics,
    ) {
        Ok(reservation) => reservation,
        Err(error) => {
            drop(candidate);
            reservations.release(candidate_reservation, metrics);
            return Err(error);
        },
    };
    let token = match execution_budget_token_for_boundary(
        context,
        reservations.available_bytes_including(&filtered_reservation),
        "join_residual_filter_output",
        "cudf_query_engine.bounded.join.one_shot_residual_filter.budget_token",
        "failed to create bounded native one-shot residual filter budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            drop(candidate);
            reservations.release(filtered_reservation, metrics);
            reservations.release(candidate_reservation, metrics);
            return Err(error);
        },
    };
    let predicate = lower_expr(&residual_candidate.predicate)?;
    let filter_overrides = token.execution_overrides();
    let filtered = match candidate.with_execution(&filter_overrides).filter_expr(predicate) {
        Ok(filtered) => filtered,
        Err(error) => {
            drop(candidate);
            reservations.release(filtered_reservation, metrics);
            reservations.release(candidate_reservation, metrics);
            return Err(attach_budget_token_failure_facts(
                QueryError::from_cudf(
                    "cudf_query_engine.bounded.join.one_shot_residual_filter",
                    "failed to filter compiled one-shot residual join candidates",
                    error,
                )
                .with_fact("join_type", node.spec.join_type)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("build_side", node.spec.build_side),
                &token,
            ));
        },
    };
    drop(candidate);
    reservations.release(candidate_reservation, metrics);

    let output =
        filtered.select(residual_candidate.final_output_names.iter().map(String::as_str)).map_err(|error| {
            QueryError::from_cudf(
                "cudf_query_engine.bounded.join.one_shot_residual_output_projection",
                "failed to project compiled one-shot residual join output columns",
                error,
            )
        })?;
    drop(filtered);
    metrics.operator_time += timer.elapsed();
    let output_rows = output.num_rows() as u64;
    let output_bytes = observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), metrics)?;
    metrics.record_one_shot_join_output_bytes(output_bytes);
    if output_bytes > filtered_reservation.bytes() {
        let grow_request = bounded_join_request(
            ReservationKind::JoinOutput,
            "join_residual_filter_output",
            output_bytes.saturating_sub(filtered_reservation.bytes()),
            reservations.budget(),
            node_id,
        );
        if let Err(error) = reservations.try_grow(&mut filtered_reservation, grow_request, metrics) {
            drop(output);
            reservations.release(filtered_reservation, metrics);
            drop(left);
            drop(right);
            return Err(error);
        }
    } else {
        reservations.shrink_to_observed(&mut filtered_reservation, output_bytes, metrics);
    }
    let final_output_reservation_bytes = filtered_reservation.bytes();
    let build_side = node.spec.build_side;
    let probe_side = probe_side(build_side);
    metrics.record_operator_metrics(crate::metrics::OperatorMetrics::join(
        node_id,
        crate::metrics::JoinOperatorMetrics {
            join_type: node.spec.join_type.to_string(),
            execution_strategy: "bounded_one_shot_equi_join".to_string(),
            key_count: node.spec.keys.len(),
            residual_predicate: true,
            null_equality: node.spec.null_equality.to_string(),
            build_side: build_side.to_string(),
            probe_side: probe_side.to_string(),
            build_rows: side_metric(build_side, left.rows, right.rows),
            build_bytes: side_metric(build_side, left.bytes, right.bytes),
            probe_rows: side_metric(probe_side, left.rows, right.rows),
            probe_bytes: side_metric(probe_side, left.bytes, right.bytes),
            hash_table_bytes: 0,
            filtered_probe_state_bytes: 0,
            mixed_size_data_bytes: 0,
            gather_map_bytes,
            output_reservation_bytes: final_output_reservation_bytes,
            output_rows,
            output_bytes,
        },
    ));
    drop(left);
    drop(right);
    Ok(ResidentGpuFrame { frame: GpuDataFrame::new(output), reservation: filtered_reservation })
}

pub(super) struct OneShotResidualCandidateProjection {
    projection: Vec<JoinOutputColumn>,
    predicate: Expr,
    final_output_names: Vec<String>,
}

pub(super) fn one_shot_residual_candidate_projection(node: &JoinNode) -> Result<OneShotResidualCandidateProjection> {
    let residual_columns = residual_input_columns(node)?;
    let mut projection = node.spec.output_projection.clone();
    let final_output_names = node.spec.output_projection.iter().map(|column| column.output_name.clone()).collect();
    let mut used_names = projection.iter().map(|column| column.output_name.clone()).collect::<BTreeSet<_>>();
    let mut residual_output_names = BTreeMap::new();

    for side in [JoinSide::Left, JoinSide::Right] {
        for (index, name) in residual_columns.names(side).iter().enumerate() {
            let output_name = residual_candidate_column_name(side, index, name, &mut used_names);
            residual_output_names.insert((residual_candidate_side_key(side), name.clone()), output_name.clone());
            let input_index = match side {
                JoinSide::Left => 0,
                JoinSide::Right => 1,
            };
            projection.push(JoinOutputColumn::new(
                side,
                ColumnRef::input(input_index, name.clone(), output_name.clone()),
                output_name,
            ));
        }
    }

    let predicate = rewrite_join_residual_for_candidate(
        node.spec.residual_predicate.as_ref().ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::InvalidPlan,
                "cudf_query_engine.bounded.join.one_shot_residual_projection",
                "one-shot residual projection requires a residual predicate",
            )
        })?,
        &residual_output_names,
    )?;

    Ok(OneShotResidualCandidateProjection { projection, predicate, final_output_names })
}

pub(super) fn residual_candidate_column_name(
    side: JoinSide,
    index: usize,
    name: &str,
    used_names: &mut BTreeSet<String>,
) -> String {
    let side_label = match side {
        JoinSide::Left => "left",
        JoinSide::Right => "right",
    };
    let mut sanitized = name.chars().map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' }).collect::<String>();
    if sanitized.is_empty() {
        sanitized.push_str("column");
    }
    let base = format!("__cudf_one_shot_residual_{side_label}_{index}_{sanitized}");
    let mut candidate = base.clone();
    let mut suffix = 0_usize;
    while !used_names.insert(candidate.clone()) {
        suffix += 1;
        candidate = format!("{base}_{suffix}");
    }
    candidate
}

pub(super) fn residual_candidate_side_key(side: JoinSide) -> u8 {
    match side {
        JoinSide::Left => 0,
        JoinSide::Right => 1,
    }
}

pub(super) fn rewrite_join_residual_for_candidate(
    expr: &Expr,
    residual_output_names: &BTreeMap<(u8, String), String>,
) -> Result<Expr> {
    match expr {
        Expr::Column(column) => {
            let side = match column.qualifier() {
                Some(ColumnQualifier::Input { index: 0 }) => JoinSide::Left,
                Some(ColumnQualifier::Input { index: 1 }) => JoinSide::Right,
                qualifier => {
                    return Err(QueryError::new(
                        QueryErrorCode::UnsupportedExpression,
                        "cudf_query_engine.bounded.join.one_shot_residual_projection",
                        "one-shot residual predicates require input-qualified column references",
                    )
                    .with_fact("column", column.name())
                    .with_fact("qualifier", qualifier.map(ToString::to_string).unwrap_or_else(|| "none".to_string())));
                },
            };
            let name = residual_output_names
                .get(&(residual_candidate_side_key(side), column.name().to_string()))
                .ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "cudf_query_engine.bounded.join.one_shot_residual_projection",
                        "one-shot residual predicate column was not materialized into the candidate frame",
                    )
                    .with_fact("side", side)
                    .with_fact("column", column.name())
                })?;
            Ok(Expr::column(name.clone()))
        },
        Expr::Literal(value) => Ok(Expr::Literal(value.clone())),
        Expr::Not(input) => Ok(Expr::logical_not(rewrite_join_residual_for_candidate(input, residual_output_names)?)),
        Expr::IsNull(input) => Ok(Expr::is_null(rewrite_join_residual_for_candidate(input, residual_output_names)?)),
        Expr::IsNotNull(input) => {
            Ok(Expr::is_not_null(rewrite_join_residual_for_candidate(input, residual_output_names)?))
        },
        Expr::Cast { expr, target_type } => {
            Ok(Expr::cast(rewrite_join_residual_for_candidate(expr, residual_output_names)?, target_type.clone()))
        },
        Expr::Binary { left, op, right, output_type } => {
            let left = rewrite_join_residual_for_candidate(left, residual_output_names)?;
            let right = rewrite_join_residual_for_candidate(right, residual_output_names)?;
            Ok(match output_type {
                Some(output_type) => Expr::binary_with_output_type(left, *op, right, output_type.clone()),
                None => Expr::binary(left, *op, right),
            })
        },
        Expr::Like { expr, pattern, escape, negated } => Ok(Expr::Like {
            expr: Box::new(rewrite_join_residual_for_candidate(expr, residual_output_names)?),
            pattern: pattern.clone(),
            escape: escape.clone(),
            negated: *negated,
        }),
        Expr::Substring { expr, start, length } => {
            Ok(Expr::substring(rewrite_join_residual_for_candidate(expr, residual_output_names)?, *start, *length))
        },
        Expr::Lower(expr) => Ok(Expr::lower(rewrite_join_residual_for_candidate(expr, residual_output_names)?)),
        Expr::Upper(expr) => Ok(Expr::upper(rewrite_join_residual_for_candidate(expr, residual_output_names)?)),
        Expr::Abs(expr) => Ok(Expr::abs(rewrite_join_residual_for_candidate(expr, residual_output_names)?)),
        Expr::Concat(exprs) => Ok(Expr::concat(
            exprs
                .iter()
                .map(|expr| rewrite_join_residual_for_candidate(expr, residual_output_names))
                .collect::<Result<Vec<_>>>()?,
        )),
        Expr::Round { expr, decimal_places, output_type } => Ok(Expr::round(
            rewrite_join_residual_for_candidate(expr, residual_output_names)?,
            *decimal_places,
            output_type.clone(),
        )),
        Expr::InList { expr, list, negated } => Ok(Expr::InList {
            expr: Box::new(rewrite_join_residual_for_candidate(expr, residual_output_names)?),
            list: list
                .iter()
                .map(|item| rewrite_join_residual_for_candidate(item, residual_output_names))
                .collect::<Result<Vec<_>>>()?,
            negated: *negated,
        }),
        Expr::Case { branches, else_expr } => Ok(Expr::case(
            branches
                .iter()
                .map(|branch| {
                    Ok(crate::expr::CaseWhen {
                        when: rewrite_join_residual_for_candidate(&branch.when, residual_output_names)?,
                        then: rewrite_join_residual_for_candidate(&branch.then, residual_output_names)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            rewrite_join_residual_for_candidate(else_expr, residual_output_names)?,
        )),
        Expr::Extract { field, expr } => {
            Ok(Expr::extract(*field, rewrite_join_residual_for_candidate(expr, residual_output_names)?))
        },
        Expr::Alias { expr, alias } => {
            Ok(rewrite_join_residual_for_candidate(expr, residual_output_names)?.alias(alias.clone()))
        },
    }
}
