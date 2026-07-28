use crate::{
    Error,
    core::{Result, Table},
    frame::dataframe::DataFrame,
};

/// Testing-only: build the canonical `DataFrame::new` column-count mismatch
/// error shape used by integration-test helpers.
pub fn new_column_length_mismatch_error_for_test(column_name_count: usize, table_column_count: usize) -> Error {
    crate::core::schema::column_name_count_mismatch_error(
        "dataframe::new",
        "number of names must equal number of columns",
        column_name_count,
        table_column_count,
    )
}

/// Trigger the internal DataFrame reassembly helper directly for contract
/// tests.
pub fn reassemble_dataframe_from_table_and_names_for_test(table: Table, names: Vec<String>) -> Result<DataFrame> {
    crate::internal::testing_hooks::reassemble_dataframe_from_table_and_names_for_test(table, names)
}

/// Trigger the high-level DataFrame dropna subset-count limit validation
/// contract.
pub fn trigger_dataframe_dropna_subset_count_limit_error_for_test() -> Result<()> {
    crate::frame::dataframe::trigger_dataframe_dropna_subset_count_limit_error_for_test()
}

/// Trigger the DataFrame row-range start limit path used by integration tests.
pub fn trigger_dataframe_row_range_start_limit_error_for_test() -> Result<DataFrame> {
    crate::frame::dataframe::trigger_dataframe_row_range_start_limit_error_for_test()
}

/// Trigger the DataFrame row-range end limit path used by integration tests.
pub fn trigger_dataframe_row_range_end_limit_error_for_test() -> Result<DataFrame> {
    crate::frame::dataframe::trigger_dataframe_row_range_end_limit_error_for_test()
}

/// Trigger the DataFrame row-range overflow path used by integration tests.
pub fn trigger_dataframe_row_range_overflow_error_for_test() -> Result<DataFrame> {
    crate::frame::dataframe::trigger_dataframe_row_range_overflow_error_for_test()
}

/// Trigger the concat-by-name missing planned dtype invariant path used by
/// integration tests.
pub fn trigger_concat_by_name_missing_planned_dtype_for_test() -> Result<()> {
    crate::frame::dataframe::trigger_concat_by_name_missing_planned_dtype_for_test()
}

/// Reset the thread-local count of successful FixedSizeList structural proof
/// builds performed by `GpuRecordBatch::new`.
pub fn reset_fixed_size_list_structural_proof_build_count_for_test() {
    crate::frame::dataframe::reset_fixed_size_list_structural_proof_build_count_for_test();
}

/// Return the thread-local number of successful FixedSizeList structural
/// proof builds since the last reset.
pub fn fixed_size_list_structural_proof_build_count_for_test() -> usize {
    crate::frame::dataframe::fixed_size_list_structural_proof_build_count_for_test()
}

/// Attempt to combine one batch's retained FixedSizeList proof with another
/// batch's producer column. The production gate must reject the mismatch.
pub fn trigger_fixed_size_list_proof_identity_mismatch_for_test(
    producer: &crate::frame::dataframe::GpuRecordBatch,
    proof_owner: &crate::frame::dataframe::GpuRecordBatch,
    field_index: usize,
) -> crate::Result<()> {
    producer.trigger_fixed_size_list_proof_identity_mismatch_for_test(proof_owner, field_index)
}

/// Attempt to combine one batch's uniform-list proof with another batch's
/// producer column. The production gate must reject the mismatch.
pub fn trigger_uniform_list_proof_identity_mismatch_for_test(
    producer: &crate::frame::dataframe::GpuRecordBatch,
    proof_owner: &crate::frame::dataframe::GpuRecordBatch,
    field_index: usize,
) -> crate::Result<()> {
    producer.trigger_uniform_list_proof_identity_mismatch_for_test(proof_owner, field_index)
}
