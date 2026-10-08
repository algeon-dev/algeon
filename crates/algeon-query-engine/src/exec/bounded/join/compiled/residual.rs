//! Residual filtering for one-shot equi joins whose predicate cannot run inside the mixed kernel.
//!
//! An inner join with equality keys and a residual containing CASE, IS DISTINCT FROM, IS NOT
//! DISTINCT FROM or a decimal division is classified `KeyedEquiPostFilter` and always runs as
//! `OneShotEqui`. The equality gather maps pair every key match, and the residual removes pairs
//! afterwards. This module gathers a candidate frame that holds the output columns plus every
//! column the residual reads, under generated `__algeon_native_one_shot_residual_*` names. It
//! then filters the candidate with the residual rewritten to those names
//! (`rewrite_join_residual_for_candidate`) and selects the final output columns.

use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

use query_planning::expr::CaseWhen;

use super::{
    BoundedExecutionAttempt, BoundedJoinSideFrame, ColumnQualifier, ColumnRef, CompiledJoinIndices, Expr, GpuDataFrame,
    JoinMaterializationTxn, JoinNode, JoinOutputColumn, JoinSide, NodeId, OneShotJoinChunkMetrics, QueryError,
    QueryErrorCode, ResidentGpuFrame, Result, attach_budget_token_failure_facts, cudf_query_error,
    datafusion_order_columns, execution_budget_token_for_boundary, lower_expr, observe_device_memory_bytes, probe_side,
    residual_input_columns, side_metric,
};

/// Gathers the key-matched candidate rows, filters them with the residual, and returns the final
/// output with its chunk metrics.
///
/// Takes both side frames and the equality gather maps into a `JoinMaterializationTxn`. Every
/// failure path calls `rollback` and success calls `commit_frame`. The gather maps are released
/// with `drop_indices` once the candidate frame exists, before the filter allocates; the side
/// frames stay until commit because the metrics read their row counts. `candidate_rows` and
/// `expected_index_bytes` come from the preflight, and the gathered values are compared with them
/// in debug builds only. The filter runs under a budget token for `join_residual_filter_output`,
/// which shares the stream's allocator member and does not change the grant. Its failures carry the
/// token's allocation peak.
#[expect(
    clippy::too_many_arguments,
    reason = "bounded execution passes operator-occurrence inputs explicitly beside the attempt state"
)]
pub(super) fn materialize_compiled_one_shot_residual_output(
    execution: &mut BoundedExecutionAttempt<'_>,
    node_id: NodeId,
    node: &JoinNode,
    left: BoundedJoinSideFrame,
    right: BoundedJoinSideFrame,
    indices: CompiledJoinIndices,
    candidate_rows: u64,
    expected_index_bytes: u64,
    timer: Instant,
) -> Result<(ResidentGpuFrame, OneShotJoinChunkMetrics)> {
    let mut txn = JoinMaterializationTxn::with_sides(left, right);
    txn.set_indices(indices);
    let exec_runtime = execution.context().runtime()?;
    let overrides = exec_runtime.exec();
    let residual_candidate = match one_shot_residual_candidate_projection(node) {
        Ok(candidate) => candidate,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    let gather_map_bytes = match txn.indices().device_allocated_size_bytes(node) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    debug_assert_eq!(gather_map_bytes, expected_index_bytes);
    let (left, right) = txn.sides();
    let candidate_reservation_bytes = match txn.indices().output_upper_bound_size_bytes(
        execution,
        node_id,
        &left.frame,
        &right.frame,
        candidate_rows,
        &residual_candidate.projection,
    ) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    let (left, right) = txn.sides();
    let candidate = match txn.indices().materialize_output(
        &left.frame,
        &right.frame,
        overrides,
        &residual_candidate.projection,
        node.spec.join_type.clone(),
        node.spec.datafusion_order_preserved_side,
    ) {
        Ok(candidate) => candidate,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    let candidate_output_rows = candidate.num_rows() as u64;
    debug_assert_eq!(candidate_output_rows, candidate_rows);
    let candidate_output_bytes =
        match observe_device_memory_bytes(&GpuDataFrame::new(candidate.clone()), execution.metrics()) {
            Ok(bytes) => bytes,
            Err(error) => {
                txn.rollback();
                return Err(error);
            },
        };
    let _candidate_output_bytes = candidate_output_bytes;
    txn.drop_indices();
    let token = match execution_budget_token_for_boundary(
        execution.context(),
        execution.device_budget_bytes(),
        "join_residual_filter_output",
        "query_engine.bounded.join.one_shot_residual_filter.budget_token",
        "failed to create bounded native one-shot residual filter budget token",
    ) {
        Ok(token) => token,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    let predicate = match lower_expr(&residual_candidate.predicate) {
        Ok(predicate) => predicate,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    let filter_overrides = token.exec();
    let filtered = match candidate.on(filter_overrides).filter_expr(predicate) {
        Ok(filtered) => filtered,
        Err(error) => {
            txn.rollback();
            return Err(attach_budget_token_failure_facts(
                cudf_query_error(
                    "query_engine.bounded.join.one_shot_residual_filter",
                    "failed to filter compiled one-shot residual join candidates",
                    error,
                )
                .with_fact("join_type", &node.spec.join_type)
                .with_fact("key_count", node.spec.keys.len())
                .with_fact("build_side", node.spec.build_side),
                &token,
            ));
        },
    };
    let mut final_output_names = residual_candidate.final_output_names;
    final_output_names.extend(datafusion_order_columns(&filtered));
    let output = match filtered.select(final_output_names.iter().map(String::as_str)) {
        Ok(output) => output,
        Err(error) => {
            txn.rollback();
            return Err(cudf_query_error(
                "query_engine.bounded.join.one_shot_residual_output_projection",
                "failed to project compiled one-shot residual join output columns",
                error,
            ));
        },
    };
    execution.metrics().add_operator_time(timer.elapsed());
    let output_rows = output.num_rows() as u64;
    let output_bytes = match observe_device_memory_bytes(&GpuDataFrame::new(output.clone()), execution.metrics()) {
        Ok(bytes) => bytes,
        Err(error) => {
            txn.rollback();
            return Err(error);
        },
    };
    let build_side = node.spec.build_side;
    let probe_side = probe_side(build_side);
    let (left, right) = txn.sides();
    let chunk_metrics = OneShotJoinChunkMetrics {
        build_rows: side_metric(build_side, left.rows, right.rows),
        build_bytes: side_metric(build_side, left.bytes, right.bytes),
        probe_rows: side_metric(probe_side, left.rows, right.rows),
        probe_bytes: side_metric(probe_side, left.bytes, right.bytes),
        index_bytes: gather_map_bytes,
        materialization_bytes: candidate_reservation_bytes,
        output_reservation_bytes: Some(output_bytes),
        output_rows,
        output_bytes,
    };
    let output = txn.commit_frame(GpuDataFrame::new(output), output_bytes);
    Ok((output, chunk_metrics))
}

// Candidate output layout that preserves columns needed to evaluate a residual predicate.
pub(super) struct OneShotResidualCandidateProjection {
    projection: Vec<JoinOutputColumn>,
    predicate: Expr,
    final_output_names: Vec<String>,
}

/// Plans the candidate frame: the node's output projection followed by one generated column for
/// each input column the residual reads, the residual rewritten onto those generated names, and the
/// output names to keep after filtering. Fails with `UnsupportedExpression` when the residual
/// references a column that is not input-qualified.
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
                "query_engine.bounded.join.one_shot_residual_projection",
                "one-shot residual projection requires a residual predicate",
            )
        })?,
        &residual_output_names,
    )?;

    Ok(OneShotResidualCandidateProjection { projection, predicate, final_output_names })
}

/// Returns a generated candidate column name that is unique within `used_names`, which records it.
/// The name embeds the side, the column's position and a sanitized copy of the source name.
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
    let base = format!("__algeon_native_one_shot_residual_{side_label}_{index}_{sanitized}");
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

/// Rewrites input-qualified column references in `expr` to the generated candidate column names,
/// leaving the rest of the expression unchanged. A column absent from `residual_output_names` is an
/// invariant violation (`InvalidPlan`).
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
                        "query_engine.bounded.join.one_shot_residual_projection",
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
                        "query_engine.bounded.join.one_shot_residual_projection",
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
        Expr::TryCast { expr, target_type } => {
            Ok(Expr::try_cast(rewrite_join_residual_for_candidate(expr, residual_output_names)?, target_type.clone()))
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
        Expr::Trim { expr, trim_chars } => {
            Ok(Expr::trim(rewrite_join_residual_for_candidate(expr, residual_output_names)?, trim_chars.clone()))
        },
        Expr::LTrim { expr, trim_chars } => {
            Ok(Expr::ltrim(rewrite_join_residual_for_candidate(expr, residual_output_names)?, trim_chars.clone()))
        },
        Expr::RTrim { expr, trim_chars } => {
            Ok(Expr::rtrim(rewrite_join_residual_for_candidate(expr, residual_output_names)?, trim_chars.clone()))
        },
        Expr::StringContains { expr, pattern } => Ok(Expr::string_contains(
            rewrite_join_residual_for_candidate(expr, residual_output_names)?,
            pattern.clone(),
        )),
        Expr::StartsWith { expr, prefix } => {
            Ok(Expr::starts_with(rewrite_join_residual_for_candidate(expr, residual_output_names)?, prefix.clone()))
        },
        Expr::EndsWith { expr, suffix } => {
            Ok(Expr::ends_with(rewrite_join_residual_for_candidate(expr, residual_output_names)?, suffix.clone()))
        },
        Expr::StringLength { expr, output_type } => Ok(Expr::string_length(
            rewrite_join_residual_for_candidate(expr, residual_output_names)?,
            output_type.clone(),
        )),
        Expr::RegexpReplace { expr, pattern, replacement, output_type } => Ok(Expr::regexp_replace(
            rewrite_join_residual_for_candidate(expr, residual_output_names)?,
            pattern.clone(),
            replacement.clone(),
            output_type.clone(),
        )),
        Expr::Abs(expr) => Ok(Expr::abs(rewrite_join_residual_for_candidate(expr, residual_output_names)?)),
        Expr::UnaryMath { op, expr, output_type } => Ok(Expr::unary_math(
            *op,
            rewrite_join_residual_for_candidate(expr, residual_output_names)?,
            output_type.clone(),
        )),
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
        Expr::IsDistinctFrom { left, right } => Ok(Expr::is_distinct_from(
            rewrite_join_residual_for_candidate(left, residual_output_names)?,
            rewrite_join_residual_for_candidate(right, residual_output_names)?,
        )),
        Expr::IsNotDistinctFrom { left, right } => Ok(Expr::is_not_distinct_from(
            rewrite_join_residual_for_candidate(left, residual_output_names)?,
            rewrite_join_residual_for_candidate(right, residual_output_names)?,
        )),
        Expr::Case { branches, else_expr } => Ok(Expr::case(
            branches
                .iter()
                .map(|branch| {
                    Ok(CaseWhen {
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
        Expr::DateTrunc { field, expr, output_type } => Ok(Expr::date_trunc(
            *field,
            rewrite_join_residual_for_candidate(expr, residual_output_names)?,
            output_type.clone(),
        )),
        Expr::Alias { expr, alias } => {
            Ok(rewrite_join_residual_for_candidate(expr, residual_output_names)?.alias(alias.clone()))
        },
    }
}
