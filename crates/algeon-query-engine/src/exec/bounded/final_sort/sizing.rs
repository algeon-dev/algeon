use super::{ExecutionBudgetReport, QueryError, QueryErrorCode, Result};

pub(super) fn final_sort_retained_bytes(report: &ExecutionBudgetReport) -> Result<u64> {
    let bytes = report.retained_bytes().ok_or_else(|| {
        let error = QueryError::new(
            QueryErrorCode::Internal,
            "query_engine.final_sort.prepare",
            "prepared final sort budget report did not include retained sorted-order bytes",
        )
        .with_fact("budget_boundary", report.request().boundary())
        .with_fact("budget_enforcement", report.enforcement().as_str())
        .with_fact("budget_status", "reservation_enforced");
        match report.member_limit_bytes() {
            Some(member_limit_bytes) => error.with_fact("budget_bytes", member_limit_bytes),
            None => error,
        }
    })?;
    Ok(bytes as u64)
}

pub(super) fn estimated_bytes_per_row(bytes: u64, rows: usize) -> u64 {
    if rows == 0 {
        return bytes.max(1);
    }
    bytes.div_ceil(rows as u64).max(1)
}

pub(super) fn estimate_rows_bytes(bytes_per_row: u64, rows: usize) -> u64 {
    let bytes = u128::from(bytes_per_row) * rows as u128;
    bytes.min(u128::from(u64::MAX)) as u64
}

pub(super) fn final_sort_rows_per_chunk(input_rows: usize, bytes_per_row: u64, available_bytes: u64) -> usize {
    if input_rows == 0 {
        return 1;
    }
    let per_row_budget = bytes_per_row.saturating_mul(3).max(1);
    let rows = (available_bytes / per_row_budget).min(input_rows as u64) as usize;
    rows.max(1)
}
