use std::sync::Arc;

use crate::{
    core::{
        column::Column,
        error::{Error, ErrorCode, FACT_KEYS_COUNT, FACT_LEFT_TYPE_ID, FACT_RIGHT_TYPE_ID, Result},
    },
    frame::{
        dataframe::{
            DataFrame, DataFrameConditionalFullJoinSizeData, DataFrameCrossJoinOptions, DataFrameFilteredJoin,
            DataFrameFilteredJoinOptions, DataFrameFilteredJoinProbeState, DataFrameHashJoin,
            DataFrameHashJoinMaterialized, DataFrameHashJoinOptions, DataFrameHashJoinPairIndices,
            DataFrameJoinIndices, DataFrameJoinOptions, DataFrameJoinPairIndices, DataFrameJoinSides,
            DataFrameMixedJoinOptions, DataFrameMixedJoinSizeData, DataFrameMixedSingleSideJoinSizeData,
            DataFramePredicateJoinOptions, DataFrameProjectedCrossJoinPlan, derive_frame_domain_from_columns,
        },
        series::Series,
    },
    internal::{
        hash::{FastHashSet, FastIndexMap, fast_index_map_with_capacity},
        resolve::{
            build_borrowed_table_view_from_columns, payload_indices, resolve_column_names, validate_no_duplicate_names,
        },
        temp_table::TempTableView,
    },
    ops::{ast::AstExpression, join, replace},
    runtime::{
        Runtime, execution_context::ExecutionOverrides, execution_domain::ExecutionDomain,
        execution_resolver::ResolvedExecution,
    },
};

const MATERIALIZE_JOIN_INDICES_RESERVATION_CONTRACT: &str = "materialize_join_indices_reservation_size_bytes";
const PROJECTED_CROSS_JOIN_RESERVATION_CONTRACT: &str = "projected_cross_join_reservation_size_bytes";

struct JoinSplitKeys<'a, 'key> {
    left_on: &'a [&'key str],
    right_on: &'a [&'key str],
}

struct JoinSplitCall {
    join_type: join::JoinType,
    null_equality: join::NullEquality,
    forced_output_domain: Option<Arc<ExecutionDomain>>,
    operation: &'static str,
}

mod conditional;
mod cross;
mod filter;
mod hash;
mod mixed;

struct MixedDataFrameJoinTables<'a> {
    left_equality: TempTableView<'a>,
    right_equality: TempTableView<'a>,
    left_conditional: TempTableView<'a>,
    right_conditional: TempTableView<'a>,
}

fn string_refs(values: &[String]) -> Vec<&str> {
    values.iter().map(|value| value.as_str()).collect()
}

fn predicate_tables<'a>(
    left: &'a DataFrame,
    right: &'a DataFrame,
    options: &DataFramePredicateJoinOptions,
    operation: &'static str,
) -> Result<(TempTableView<'a>, TempTableView<'a>)> {
    let left_names = options.left_column_names();
    let right_names = options.right_column_names();
    if left_names.is_empty() || right_names.is_empty() {
        return Err(merge_validation_error(
            operation,
            "predicate join requires at least one predicate column from each side",
            "empty_predicate_columns",
            "set the left and right predicate column lists used by the AST predicate",
        ));
    }
    let left_refs = string_refs(left_names);
    let right_refs = string_refs(right_names);
    let left_indices = resolve_column_names(left.columns_map(), &left_refs, operation)?;
    let right_indices = resolve_column_names(right.columns_map(), &right_refs, operation)?;
    Ok((
        build_borrowed_table_view_from_columns(left.columns_map(), &left_indices)?,
        build_borrowed_table_view_from_columns(right.columns_map(), &right_indices)?,
    ))
}

fn mixed_join_tables<'a>(
    left: &'a DataFrame,
    right: &'a DataFrame,
    options: &DataFrameMixedJoinOptions,
    operation: &'static str,
) -> Result<MixedDataFrameJoinTables<'a>> {
    let (left_on, right_on, _) =
        resolve_join_keys(options.on_keys(), options.left_on_keys(), options.right_on_keys(), operation)?;
    if options.left_predicate_column_names().is_empty() || options.right_predicate_column_names().is_empty() {
        return Err(merge_validation_error(
            operation,
            "mixed join requires at least one predicate column from each side",
            "empty_predicate_columns",
            "set the left and right predicate column lists used by the AST predicate",
        ));
    }

    let left_key_refs = string_refs(&left_on);
    let right_key_refs = string_refs(&right_on);
    let left_predicate_refs = string_refs(options.left_predicate_column_names());
    let right_predicate_refs = string_refs(options.right_predicate_column_names());

    let left_key_indices = resolve_column_names(left.columns_map(), &left_key_refs, operation)?;
    let right_key_indices = resolve_column_names(right.columns_map(), &right_key_refs, operation)?;
    let left_predicate_indices = resolve_column_names(left.columns_map(), &left_predicate_refs, operation)?;
    let right_predicate_indices = resolve_column_names(right.columns_map(), &right_predicate_refs, operation)?;

    Ok(MixedDataFrameJoinTables {
        left_equality: build_borrowed_table_view_from_columns(left.columns_map(), &left_key_indices)?,
        right_equality: build_borrowed_table_view_from_columns(right.columns_map(), &right_key_indices)?,
        left_conditional: build_borrowed_table_view_from_columns(left.columns_map(), &left_predicate_indices)?,
        right_conditional: build_borrowed_table_view_from_columns(right.columns_map(), &right_predicate_indices)?,
    })
}

fn pair_indices_from_raw(raw_indices: join::JoinIndices, operation: &'static str) -> Result<DataFrameJoinPairIndices> {
    let left = Series::from_column(raw_indices.left)
        .map_err(|e| e.reanchor(operation).with_detail("resource", "left_join_indices"))?;
    let right = Series::from_column(raw_indices.right)
        .map_err(|e| e.reanchor(operation).with_detail("resource", "right_join_indices"))?;
    Ok(DataFrameJoinPairIndices::new(left, right))
}

fn single_side_indices_from_raw(
    raw_indices: join::SingleSideJoinIndices,
    operation: &'static str,
) -> Result<DataFrameJoinIndices> {
    let indices = Series::from_column(raw_indices.indices)
        .map_err(|e| e.reanchor(operation).with_detail("resource", "join_indices"))?;
    Ok(DataFrameJoinIndices::new(indices))
}

/// Build the schema name list for a join result side:
/// key column names (in `on` order) followed by payload column names.
fn build_join_schema_names(
    columns: &FastIndexMap<String, Arc<Column>>,
    key_indices: &[usize],
    payload_indices: &[usize],
) -> Vec<String> {
    let keys_iter = columns.keys().collect::<Vec<_>>();

    let mut names = Vec::with_capacity(key_indices.len() + payload_indices.len());
    for &idx in key_indices {
        names.push(keys_iter[idx].clone());
    }
    for &idx in payload_indices {
        names.push(keys_iter[idx].clone());
    }
    names
}

/// Coalesce two columns by preferring non-null values from `left`; if null,
/// take from `right`. Runs entirely on GPU via `replace_nulls_column`.
fn coalesce_columns(left: &Column, right: &Column, operation: &'static str) -> Result<Column> {
    let output_domain = Arc::clone(left.execution_domain());
    let lcv = left.view()?;
    let rcv = right.view()?;

    let ldt = lcv.data_type()?;
    let rdt = rcv.data_type()?;
    if ldt != rdt {
        return Err(Error::new_with(
            ErrorCode::TypeMismatch,
            operation,
            "cannot coalesce columns with different types",
        )
        .with_fact(FACT_LEFT_TYPE_ID, ldt.id())
        .with_fact(FACT_RIGHT_TYPE_ID, rdt.id()));
    }

    let execution = ResolvedExecution::with_output_domain(
        Arc::clone(&output_domain),
        [lcv.execution_domain(), rcv.execution_domain()],
        operation,
    )?;
    replace::replace_nulls_column_with_exec_options(
        &lcv,
        &rcv,
        execution.output_domain(),
        execution.exec_options(),
        operation,
    )
}

// ---------------------------------------------------------------------------
// High-level merge helpers
// ---------------------------------------------------------------------------

/// Resolve the key names from `MergeOptions`, returning `(left_on, right_on, shared)`.
///
/// `shared` is `true` when `on()` was used (both sides share the same key names),
/// meaning the output should emit one shared key column per key.
fn merge_validation_error(
    operation: &'static str,
    message: &'static str,
    reason: &'static str,
    remedy: &'static str,
) -> Error {
    Error::new_with(ErrorCode::UnsupportedOperation, operation, message)
        .with_detail("reason", reason)
        .with_detail("remedy", remedy)
}

fn validate_hash_join_key_names(keys: &[String], operation: &'static str) -> Result<()> {
    if keys.is_empty() {
        return Err(merge_validation_error(
            operation,
            "no join keys specified",
            "missing_join_keys",
            "set at least one hash join key column",
        ));
    }
    validate_no_duplicate_names(keys, operation)
}

fn resolve_probe_key_table<'a, I, S>(
    probe: &'a DataFrame,
    probe_on: I,
    build_key_count: usize,
    operation: &'static str,
) -> Result<TempTableView<'a>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let probe_keys = probe_on.into_iter().map(|key| key.as_ref().to_string()).collect::<Vec<_>>();
    validate_hash_join_key_names(&probe_keys, operation)?;
    if probe_keys.len() != build_key_count {
        return Err(merge_key_count_mismatch_error(operation, probe_keys.len(), build_key_count));
    }

    let probe_key_refs = string_refs(&probe_keys);
    let probe_key_indices = resolve_column_names(probe.columns_map(), &probe_key_refs, operation)?;
    build_borrowed_table_view_from_columns(probe.columns_map(), &probe_key_indices)
}

fn merge_key_count_mismatch_error(operation: &'static str, left_count: usize, right_count: usize) -> Error {
    merge_validation_error(
        operation,
        "left_on and right_on must have the same number of key columns",
        "join_key_count_mismatch",
        "set left_on() and right_on() with the same number of key columns",
    )
    .with_detail("left_on_count", left_count.to_string())
    .with_detail("right_on_count", right_count.to_string())
}

pub(super) fn trigger_merge_reassembly_key_count_error_for_test_impl() -> Result<DataFrame> {
    let runtime = Runtime::new()?;
    let left = DataFrame::from_series([("id", runtime.series(&[1i32])?)])?;
    let right = DataFrame::from_series([("id", runtime.series(&[1i32])?)])?;

    merge_join_result_with_suffixes(
        DataFrameJoinSides { left, right },
        &["id", "unexpected_extra_key"],
        true,
        false,
        "_left",
        "_right",
        "dataframe::merge",
    )
}

pub(super) fn trigger_projected_cross_join_output_row_overflow_for_test_impl() -> Result<usize> {
    let runtime = Runtime::new()?;
    let left = DataFrame::from_series([("id", runtime.series(&[1i32])?)])?;
    let right = DataFrame::from_series([("id", runtime.series(&[1i32])?)])?;
    let left =
        DataFrame::from_raw_parts(left.columns_map().clone(), i32::MAX as usize, Arc::clone(left.execution_domain()));
    let right = DataFrame::from_raw_parts(right.columns_map().clone(), 2, Arc::clone(right.execution_domain()));

    left.projected_cross_join_plan(&right, DataFrameCrossJoinOptions::new().left_columns(["id"]))
        .map(|plan| plan.output_rows())
}

pub(super) fn trigger_projected_cross_join_input_row_limit_for_test_impl() -> Result<usize> {
    let runtime = Runtime::new()?;
    let left = DataFrame::from_series([("id", runtime.series(&[1i32])?)])?;
    let right = DataFrame::from_series([("id", runtime.series(&[] as &[i32])?)])?;
    let left = DataFrame::from_raw_parts(
        left.columns_map().clone(),
        i32::MAX as usize + 1,
        Arc::clone(left.execution_domain()),
    );

    left.projected_cross_join_plan(&right, DataFrameCrossJoinOptions::new().left_columns(["id"]))
        .map(|plan| plan.output_rows())
}

fn resolve_join_keys(
    on: Option<&[String]>,
    left_on: Option<&[String]>,
    right_on: Option<&[String]>,
    operation: &'static str,
) -> Result<(Vec<String>, Vec<String>, bool)> {
    let (left, right, shared) = match (on, left_on, right_on) {
        // on() mode: shared key names
        (Some(on), None, None) => {
            let keys: Vec<String> = on.to_vec();
            Ok((keys.clone(), keys, true))
        },
        // left_on() + right_on() mode: asymmetric keys
        (None, Some(lo), Some(ro)) => Ok((lo.to_vec(), ro.to_vec(), false)),
        // left_on without right_on
        (None, Some(_), None) => Err(merge_validation_error(
            operation,
            "left_on was specified without right_on; set both or use on()",
            "left_on_without_right_on",
            "set right_on() or use on()",
        )),
        // right_on without left_on
        (None, None, Some(_)) => Err(merge_validation_error(
            operation,
            "right_on was specified without left_on; set both or use on()",
            "right_on_without_left_on",
            "set left_on() or use on()",
        )),
        // on() together with left_on/right_on is ambiguous — treat as error
        (Some(_), _, _) => Err(merge_validation_error(
            operation,
            "on() cannot be combined with left_on()/right_on(); use one or the other",
            "ambiguous_join_key_configuration",
            "use on() or left_on() + right_on(), not both",
        )),
        // Nothing specified
        (None, None, None) => Err(merge_validation_error(
            operation,
            "no join keys specified; use on() or left_on() + right_on()",
            "missing_join_keys",
            "set on() or left_on() + right_on()",
        )),
    }?;

    if left.len() != right.len() {
        return Err(merge_key_count_mismatch_error(operation, left.len(), right.len()));
    }
    Ok((left, right, shared))
}

/// Reassemble a split join result into a single DataFrame using high-level suffix rules.
///
/// When `shared_keys` is true (from `on()`):
///   - Emit one shared key column per key (coalesced for full join).
///   - Conflicting payload names get `suffix_left` / `suffix_right`.
///
/// When `shared_keys` is false (from `left_on()`/`right_on()`):
///   - Both key columns are preserved (left key name, right key name).
///   - Conflicting payload names get suffixes.
fn merge_join_result_with_suffixes(
    split: DataFrameJoinSides,
    left_on: &[&str],
    shared_keys: bool,
    coalesce_keys: bool,
    suffix_left: &str,
    suffix_right: &str,
    operation: &'static str,
) -> Result<DataFrame> {
    let DataFrameJoinSides { left, right } = split;
    let key_count = left_on.len();
    let row_count = left.num_rows().max(right.num_rows());

    let left_names = left.column_names_owned();
    let right_names = right.column_names_owned();
    let left_cols: Vec<Arc<Column>> = left.columns_map().values().cloned().collect();
    let right_cols: Vec<Arc<Column>> = right.columns_map().values().cloned().collect();

    if key_count > left_cols.len() || key_count > right_cols.len() {
        return Err(merge_validation_error(
            operation,
            "join key count exceeds result column count",
            "join_result_key_count_exceeds_result_columns",
            "report this as a library bug; join result schema should contain at least one column per key on each side",
        )
        .with_fact(FACT_KEYS_COUNT, key_count)
        .with_detail("left_columns", left_cols.len().to_string())
        .with_detail("right_columns", right_cols.len().to_string()));
    }

    // Collect left and right payload names (after key columns).
    let left_payload_names: Vec<&str> = left_names[key_count..].iter().map(|s| s.as_str()).collect();
    let right_payload_names: Vec<&str> = right_names[key_count..].iter().map(|s| s.as_str()).collect();

    // Find which payload names collide.
    let left_payload_set: FastHashSet<&str> = left_payload_names.iter().copied().collect();
    let right_payload_set: FastHashSet<&str> = right_payload_names.iter().copied().collect();
    let conflicting: FastHashSet<&str> = left_payload_set.intersection(&right_payload_set).copied().collect();

    let mut out_cols: Vec<Arc<Column>> = Vec::new();
    let mut out_names: Vec<String> = Vec::new();

    // Only shared-key full joins should coalesce key columns. For left_on/right_on
    // joins, the contract is to preserve both key columns separately.
    for i in 0..key_count {
        let key_col = if coalesce_keys {
            Arc::new(coalesce_columns(&left_cols[i], &right_cols[i], operation)?)
        } else {
            Arc::clone(&left_cols[i])
        };
        out_cols.push(key_col);
        out_names.push(left_names[i].clone());
    }

    // Left payload.
    for (idx, name) in left_payload_names.iter().enumerate() {
        let col_idx = key_count + idx;
        out_cols.push(Arc::clone(&left_cols[col_idx]));
        if conflicting.contains(name) {
            out_names.push(format!("{}{}", name, suffix_left));
        } else {
            out_names.push(name.to_string());
        }
    }

    if !shared_keys {
        // Emit right key columns for asymmetric join.
        for i in 0..key_count {
            out_cols.push(Arc::clone(&right_cols[i]));
            out_names.push(right_names[i].clone());
        }
    }

    // Right payload.
    for (idx, name) in right_payload_names.iter().enumerate() {
        let col_idx = key_count + idx;
        out_cols.push(Arc::clone(&right_cols[col_idx]));
        if conflicting.contains(name) {
            out_names.push(format!("{}{}", name, suffix_right));
        } else {
            out_names.push(name.to_string());
        }
    }

    validate_no_duplicate_names(&out_names, operation)?;

    let mut columns = fast_index_map_with_capacity(out_names.len());
    for (name, col) in out_names.into_iter().zip(out_cols) {
        columns.insert(name, col);
    }

    let domain = derive_frame_domain_from_columns(&columns, operation)?;
    Ok(DataFrame::from_raw_parts(columns, row_count, domain))
}
