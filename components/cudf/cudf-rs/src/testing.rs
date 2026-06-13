//! Explicit unstable testing support.
//!
//! This module exists so contract and integration tests can keep validating
//! low-level execution, synchronization, and allocation behavior without
//! forcing those hooks onto the primary public API surface.
//!
//! Items in this module are intentionally **not** part of the stable,
//! user-facing API contract. They may change or disappear at any time.

use std::ffi::c_void;

use crate::{
    AggSpec, MergeOptions, SortOptions,
    core::{
        Column, DeviceMemoryResource, Result, Scalar, Table, column::ColumnView as CoreColumnView,
        table::TableView as CoreTableView,
    },
    frame::dataframe::DataFrame,
    interop::{
        arrow_device::testing_hooks as arrow_device_testing_hooks,
        projection::testing_hooks as projection_testing_hooks,
    },
    io::arrow,
    ops::groupby::GroupBy,
    runtime::{CudaEvent, CudaStream, ExecutionOverrides, Runtime},
};

/// Trait imports for the testing support surface.
pub mod prelude {
    pub use super::{
        AllocationCountTestingExt, BorrowedColumnSliceTestingExt, BorrowedViewTestingExt, DataFrameInContextTestingExt,
        ExecutionDomainMemoryResourceTestingExt, ExecutionDomainRebindTestingExt, ExecutionDomainTestingExt,
        GroupByTestingExt, InteropRawHandleTestingExt, RawHandleTestingExt, RuntimeTestingExt,
    };
}

pub use crate::core::{column::ColumnView, table::TableView};

/// Testing-only access to retained execution-domain metadata.
pub trait ExecutionDomainTestingExt {
    fn testing_execution_domain_kind(&self) -> &'static str;
    fn testing_execution_domain_id(&self) -> u64;
    fn testing_execution_domain_stream_value(&self) -> usize;
}

/// Testing-only access to the retained memory-resource handle value.
pub trait ExecutionDomainMemoryResourceTestingExt {
    fn testing_execution_domain_mr_value(&self) -> usize;
}

/// Testing-only explicit execution-domain rebinding.
pub trait ExecutionDomainRebindTestingExt: Sized {
    fn testing_rebind_execution_domain_explicit(self, identity: u64) -> Result<Self>;
}

/// Testing-only access to raw handle values.
pub trait RawHandleTestingExt {
    fn testing_raw_handle_value(&self) -> usize;
}

/// Testing-only access to interop raw handle values.
pub trait InteropRawHandleTestingExt {
    fn testing_interop_raw_handle_value(&self) -> usize;
}

/// Testing-only access to allocation counters on stats memory resources.
pub trait AllocationCountTestingExt {
    fn testing_allocation_count(&self) -> Result<i64>;
}

/// Testing-only access to runtime provenance identity.
pub trait RuntimeTestingExt {
    fn testing_execution_domain_id(&self) -> u64;
}

/// Testing-only borrowed view access for integration tests and contract checks.
pub trait BorrowedViewTestingExt {
    type BorrowedView<'a>
    where Self: 'a;

    fn testing_view(&self) -> Result<Self::BorrowedView<'_>>;
}

/// Testing-only sliced borrowed view access that preserves offset semantics.
pub trait BorrowedColumnSliceTestingExt {
    fn testing_slice_view(&self, offset: usize, length: usize) -> Result<ColumnView<'_>>;
}

/// Testing-only access to DataFrame operations with borrowed execution
/// overrides.
pub trait DataFrameInContextTestingExt {
    fn testing_gather_in_context(&self, indices: &crate::Series, ctx: &ExecutionOverrides<'_>) -> Result<DataFrame>;
    fn testing_filter_in_context(&self, mask: &crate::Series, ctx: &ExecutionOverrides<'_>) -> Result<DataFrame>;
    fn testing_sort_values_in_context<I, S>(
        &self,
        by: I,
        options: SortOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>;
    fn testing_merge_in_context(
        &self,
        right: &DataFrame,
        options: MergeOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrame>;
}

/// Testing-only access to groupby execution-context hooks.
pub trait GroupByTestingExt {
    fn testing_agg_in_context(&self, specs: &[AggSpec], ctx: &ExecutionOverrides<'_>) -> Result<DataFrame>;
}

macro_rules! impl_execution_domain_testing {
    ($ty:ty) => {
        impl ExecutionDomainTestingExt for $ty {
            fn testing_execution_domain_kind(&self) -> &'static str {
                self.__debug_execution_domain_kind()
            }

            fn testing_execution_domain_id(&self) -> u64 {
                self.__debug_execution_domain_id()
            }

            fn testing_execution_domain_stream_value(&self) -> usize {
                self.__debug_execution_domain_stream_value()
            }
        }
    };
}

macro_rules! impl_execution_domain_mr_testing {
    ($ty:ty) => {
        impl ExecutionDomainMemoryResourceTestingExt for $ty {
            fn testing_execution_domain_mr_value(&self) -> usize {
                self.__debug_execution_domain_mr_value()
            }
        }
    };
}

impl_execution_domain_testing!(Column);
impl_execution_domain_testing!(ColumnView<'_>);
impl_execution_domain_testing!(Table);
impl_execution_domain_testing!(TableView<'_>);
impl_execution_domain_testing!(Scalar);
impl_execution_domain_testing!(DataFrame);

impl_execution_domain_mr_testing!(Column);
impl_execution_domain_mr_testing!(ColumnView<'_>);
impl_execution_domain_mr_testing!(Table);
impl_execution_domain_mr_testing!(TableView<'_>);
impl_execution_domain_mr_testing!(DataFrame);

impl ExecutionDomainRebindTestingExt for Column {
    fn testing_rebind_execution_domain_explicit(self, identity: u64) -> Result<Self> {
        self.__debug_rebind_execution_domain_explicit(identity)
    }
}

impl ExecutionDomainRebindTestingExt for Scalar {
    fn testing_rebind_execution_domain_explicit(self, identity: u64) -> Result<Self> {
        self.__debug_rebind_execution_domain_explicit(identity)
    }
}

impl RawHandleTestingExt for DeviceMemoryResource {
    fn testing_raw_handle_value(&self) -> usize {
        self.__debug_raw_handle_value()
    }
}

impl RawHandleTestingExt for CudaStream {
    fn testing_raw_handle_value(&self) -> usize {
        self.__debug_raw_handle_value()
    }
}

impl RawHandleTestingExt for CudaEvent {
    fn testing_raw_handle_value(&self) -> usize {
        self.__debug_raw_handle_value()
    }
}

impl InteropRawHandleTestingExt for CudaEvent {
    fn testing_interop_raw_handle_value(&self) -> usize {
        self.__debug_interop_raw_handle_value()
    }
}

impl AllocationCountTestingExt for DeviceMemoryResource {
    fn testing_allocation_count(&self) -> Result<i64> {
        self.__test_allocation_count()
    }
}

impl RuntimeTestingExt for Runtime {
    fn testing_execution_domain_id(&self) -> u64 {
        self.__debug_execution_domain_id()
    }
}

impl BorrowedViewTestingExt for Column {
    type BorrowedView<'a> = CoreColumnView<'a>;

    fn testing_view(&self) -> Result<Self::BorrowedView<'_>> {
        Column::view(self)
    }
}

impl BorrowedViewTestingExt for crate::Series {
    type BorrowedView<'a> = CoreColumnView<'a>;

    fn testing_view(&self) -> Result<Self::BorrowedView<'_>> {
        crate::Series::view(self)
    }
}

impl BorrowedViewTestingExt for Table {
    type BorrowedView<'a> = CoreTableView<'a>;

    fn testing_view(&self) -> Result<Self::BorrowedView<'_>> {
        Table::view(self)
    }
}

impl BorrowedViewTestingExt for DataFrame {
    type BorrowedView<'a> = CoreTableView<'a>;

    fn testing_view(&self) -> Result<Self::BorrowedView<'_>> {
        self.build_table_view()
    }
}

impl BorrowedColumnSliceTestingExt for Column {
    fn testing_slice_view(&self, offset: usize, length: usize) -> Result<ColumnView<'_>> {
        crate::core::column::slice_view_for_test(self, offset, length)
    }
}

impl DataFrameInContextTestingExt for DataFrame {
    fn testing_gather_in_context(&self, indices: &crate::Series, ctx: &ExecutionOverrides<'_>) -> Result<DataFrame> {
        self.__debug_gather_in_context(indices, ctx)
    }

    fn testing_filter_in_context(&self, mask: &crate::Series, ctx: &ExecutionOverrides<'_>) -> Result<DataFrame> {
        self.__debug_filter_in_context(mask, ctx)
    }

    fn testing_sort_values_in_context<I, S>(
        &self,
        by: I,
        options: SortOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.__debug_sort_values_in_context(by, options, ctx)
    }

    fn testing_merge_in_context(
        &self,
        right: &DataFrame,
        options: MergeOptions,
        ctx: &ExecutionOverrides<'_>,
    ) -> Result<DataFrame> {
        self.__debug_merge_in_context(right, options, ctx)
    }
}

impl GroupByTestingExt for GroupBy<'_> {
    fn testing_agg_in_context(&self, specs: &[AggSpec], ctx: &ExecutionOverrides<'_>) -> Result<DataFrame> {
        self.__debug_agg_in_context(specs, ctx)
    }
}

/// Build a stats memory resource for allocation-accounting tests.
pub fn stats_memory_resource() -> Result<DeviceMemoryResource> {
    crate::core::testing_hooks::stats_memory_resource()
}

/// Construct a malformed non-empty STRING column view without an offsets
/// child so integration tests can lock down foreign-contract validation.
pub fn synthetic_string_view_without_children(size: usize) -> Result<ColumnView<'static>> {
    crate::core::column::synthetic_string_view_without_children(size)
}

/// Pack host validity bits into cuDF's allocation-sized null-mask layout.
pub fn validity_to_host_null_mask_bytes(validity: &[bool]) -> Vec<u8> {
    crate::core::testing_hooks::validity_to_host_null_mask_bytes(validity)
}

/// Trigger the internal raw-byte materialization helper on a borrowed column
/// view so integration tests can lock down its downstream error contract.
pub fn column_view_to_raw_bytes_for_test(view: &ColumnView<'_>) -> Result<Vec<u8>> {
    crate::core::column::column_view_to_raw_bytes_for_test(view)
}

/// Trigger the internal fixed-width host-byte materialization helper on a
/// borrowed column view so integration tests can lock down its downstream
/// error contract.
pub fn column_view_to_fixed_width_host_bytes_for_test(view: &ColumnView<'_>) -> Result<Vec<u8>> {
    crate::core::column::column_view_to_fixed_width_host_bytes_for_test(view)
}

/// Trigger the fixed-width host-byte materialization overflow contract used by
/// integration tests.
pub fn trigger_column_view_fixed_width_host_byte_range_overflow_for_test() -> Result<()> {
    crate::core::column::trigger_fixed_width_host_byte_range_overflow_for_test()
}

/// Trigger the numeric Arrow error mapping path used by integration tests.
pub fn trigger_numeric_to_arrow_error_for_test() -> Result<()> {
    arrow::testing_hooks::trigger_numeric_to_arrow_error_for_test()
}

/// Trigger the record-batch Arrow error mapping path used by integration
/// tests.
pub fn trigger_record_batch_error_for_test() -> Result<()> {
    arrow::testing_hooks::trigger_record_batch_error_for_test()
}

/// Trigger the Utf8 Arrow downcast-mismatch path used by integration tests.
pub fn trigger_arrow_string_array_downcast_mismatch_for_test() -> Result<()> {
    arrow::testing_hooks::trigger_string_array_downcast_mismatch_for_test()
}

/// Trigger the LargeUtf8 Arrow downcast-mismatch path used by integration
/// tests.
pub fn trigger_arrow_large_string_array_downcast_mismatch_for_test() -> Result<()> {
    arrow::testing_hooks::trigger_large_string_array_downcast_mismatch_for_test()
}

/// Reset host-copy Arrow interop observation counters used by integration
/// tests.
pub fn reset_arrow_host_interop_observation_for_test() {
    arrow::testing_hooks::reset_host_interop_observation_for_test();
}

/// Read how many times the host-copy Arrow export entrypoint ran.
pub fn arrow_host_export_call_count_for_test() -> usize {
    arrow::testing_hooks::host_export_call_count_for_test()
}

/// Read how many times the host-copy Arrow import entrypoint ran.
pub fn arrow_host_import_call_count_for_test() -> usize {
    arrow::testing_hooks::host_import_call_count_for_test()
}

/// Return the Arrow C Device ABI device_type code for CPU roots.
pub fn arrow_device_cpu_type_for_test() -> i32 {
    arrow_device_testing_hooks::arrow_device_cpu_type_for_test()
}

/// Return the Arrow C Device ABI device_type code for CUDA_HOST roots.
pub fn arrow_device_cuda_host_type_for_test() -> i32 {
    arrow_device_testing_hooks::arrow_device_cuda_host_type_for_test()
}

/// Return the Arrow C Device ABI device_type code for CUDA_MANAGED roots.
pub fn arrow_device_cuda_managed_type_for_test() -> i32 {
    arrow_device_testing_hooks::arrow_device_cuda_managed_type_for_test()
}

/// Trigger the internal DataFrame reassembly helper directly for contract
/// tests.
pub fn reassemble_dataframe_from_table_and_names_for_test(table: Table, names: Vec<String>) -> Result<DataFrame> {
    crate::internal::testing_hooks::reassemble_dataframe_from_table_and_names_for_test(table, names)
}

/// Trigger the merge-result reassembly invariant branch used by integration
/// tests to lock the downstream error contract.
pub fn trigger_merge_reassembly_key_count_error_for_test() -> Result<DataFrame> {
    crate::ops::join::testing_hooks::trigger_merge_reassembly_key_count_error_for_test()
}

/// Trigger the shared invariant-violation helper path used by integration
/// tests.
pub fn trigger_invariant_violation_for_test() -> Result<()> {
    crate::core::testing_hooks::trigger_invariant_violation_for_test()
}

/// Trigger the shared null-on-success foreign-handle contract path used by
/// integration tests.
pub fn trigger_null_handle_contract_error_for_test() -> Result<()> {
    crate::core::testing_hooks::trigger_null_handle_contract_error_for_test()
}

/// Trigger the shared range-limit helper path used by integration tests.
pub fn trigger_range_limit_error_for_test() -> Result<()> {
    crate::core::testing_hooks::trigger_range_limit_error_for_test()
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

/// Trigger the Series row-range start limit path used by integration tests.
pub fn trigger_series_row_range_start_limit_error_for_test() -> Result<crate::Series> {
    crate::frame::series::trigger_series_row_range_start_limit_error_for_test()
}

/// Trigger the Series row-range end limit path used by integration tests.
pub fn trigger_series_row_range_end_limit_error_for_test() -> Result<crate::Series> {
    crate::frame::series::trigger_series_row_range_end_limit_error_for_test()
}

/// Trigger the Series row-range overflow path used by integration tests.
pub fn trigger_series_row_range_overflow_error_for_test() -> Result<crate::Series> {
    crate::frame::series::trigger_series_row_range_overflow_error_for_test()
}

/// Trigger the `Series::nunique` negative-distinct-count foreign-contract path
/// used by integration tests.
pub fn trigger_nunique_negative_distinct_count_for_test() -> Result<()> {
    crate::frame::series::reduction::trigger_nunique_negative_distinct_count_for_test()
}

/// Trigger the string-scalar negative-length foreign metadata path used by
/// integration tests.
pub fn trigger_negative_string_scalar_length_for_test() -> Result<()> {
    crate::core::scalar::trigger_negative_string_length_error_for_test()
}

/// Trigger the `Series::value_counts` missing-key-columns result-shape path
/// used by integration tests.
pub fn trigger_value_counts_missing_key_columns_for_test() -> Result<DataFrame> {
    crate::ops::groupby::testing_hooks::trigger_value_counts_missing_key_columns_for_test()
}

/// Trigger the `Series::value_counts` missing-count-results result-shape path
/// used by integration tests.
pub fn trigger_value_counts_missing_count_results_for_test() -> Result<DataFrame> {
    crate::ops::groupby::testing_hooks::trigger_value_counts_missing_count_results_for_test()
}

/// Trigger the `Series::value_counts` empty-count-result result-shape path
/// used by integration tests.
pub fn trigger_value_counts_empty_count_result_for_test() -> Result<DataFrame> {
    crate::ops::groupby::testing_hooks::trigger_value_counts_empty_count_result_for_test()
}

/// Trigger the high-level groupby result-group-index invariant path used by
/// integration tests.
pub fn trigger_groupby_result_group_index_out_of_bounds_for_test() -> Result<()> {
    crate::ops::groupby::testing_hooks::trigger_groupby_result_group_index_out_of_bounds_for_test()
}

/// Trigger the low-level groupby request-count limit validation path used by
/// integration tests.
pub fn trigger_groupby_request_count_limit_error_for_test() -> Result<()> {
    crate::ops::groupby::testing_hooks::trigger_groupby_request_count_limit_error_for_test()
}

/// Trigger the low-level groupby per-request aggregation-count limit
/// validation path used by integration tests.
pub fn trigger_groupby_request_agg_count_limit_error_for_test() -> Result<()> {
    crate::ops::groupby::testing_hooks::trigger_groupby_request_agg_count_limit_error_for_test()
}

/// Trigger the low-level groupby total aggregation-handle count limit
/// validation path used by integration tests.
pub fn trigger_groupby_total_agg_handle_count_limit_error_for_test() -> Result<()> {
    crate::ops::groupby::testing_hooks::trigger_groupby_total_agg_handle_count_limit_error_for_test()
}

/// Trigger the low-level sorting key-count limit validation path used by
/// integration tests.
pub fn trigger_sort_key_count_limit_error_for_test() -> Result<()> {
    crate::ops::sorting::testing_hooks::trigger_sort_key_count_limit_error_for_test()
}

/// Trigger the stable owned-table column-count limit validation contract.
pub fn trigger_table_column_count_limit_error_for_test() -> Result<()> {
    crate::core::table::testing_hooks::trigger_table_column_count_limit_error_for_test()
}

/// Trigger the stable borrowed-table-view column-count limit validation contract.
pub fn trigger_table_view_column_count_limit_error_for_test() -> Result<()> {
    crate::core::table::testing_hooks::trigger_table_view_column_count_limit_error_for_test()
}

/// Trigger the high-level groupby result-column-index invariant path used by
/// integration tests.
pub fn trigger_groupby_result_column_index_out_of_bounds_for_test() -> Result<()> {
    crate::ops::groupby::testing_hooks::trigger_groupby_result_column_index_out_of_bounds_for_test()
}

/// Trigger the high-level groupby duplicate result-column consumption
/// invariant path used by integration tests.
pub fn trigger_groupby_duplicate_result_column_consumption_for_test() -> Result<()> {
    crate::ops::groupby::testing_hooks::trigger_groupby_duplicate_result_column_consumption_for_test()
}

/// Trigger the dictionary match result-count mismatch foreign-contract path
/// used by integration tests.
pub fn trigger_match_dictionaries_result_count_mismatch_for_test() -> Result<()> {
    crate::ops::dictionary::testing_hooks::trigger_match_dictionaries_result_count_mismatch_for_test()
}

/// Trigger the dictionary match null-result-array foreign-contract path used
/// by integration tests.
pub fn trigger_match_dictionaries_null_result_array_for_test() -> Result<()> {
    crate::ops::dictionary::testing_hooks::trigger_match_dictionaries_null_result_array_for_test()
}

/// Trigger the shared-dictionary input-count limit validation contract.
pub fn trigger_match_dictionaries_column_count_limit_error_for_test() -> Result<()> {
    crate::ops::dictionary::testing_hooks::trigger_match_dictionaries_column_count_limit_error_for_test()
}

/// Trigger the factorization empty-row-code dtype invariant path used by
/// integration tests.
pub fn trigger_factorize_empty_row_code_dtype_error_for_test() -> Result<()> {
    crate::ops::factorization::testing_hooks::trigger_factorize_empty_row_code_dtype_error_for_test()
}

/// Trigger the Codebook key-count limit validation path used by integration
/// tests.
pub fn trigger_codebook_key_count_limit_error_for_test() -> Result<()> {
    crate::frame::factorization::trigger_codebook_key_count_limit_error_for_test()
}

/// Trigger the concat-by-name missing planned dtype invariant path used by
/// integration tests.
pub fn trigger_concat_by_name_missing_planned_dtype_for_test() -> Result<()> {
    crate::frame::dataframe::trigger_concat_by_name_missing_planned_dtype_for_test()
}

/// Trigger the concatenate-column input-count limit validation contract.
pub fn trigger_concatenate_column_count_limit_error_for_test() -> Result<()> {
    crate::ops::concatenate::trigger_concatenate_column_count_limit_error_for_test()
}

/// Trigger the concatenate-table input-count limit validation contract.
pub fn trigger_concatenate_table_count_limit_error_for_test() -> Result<()> {
    crate::ops::concatenate::trigger_concatenate_table_count_limit_error_for_test()
}

/// Trigger the shared interop projection active-range overflow contract used
/// by integration tests.
pub fn trigger_interop_projection_active_range_overflow_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_testing_hooks::trigger_active_range_overflow_for_test()
}

/// Trigger the shared interop projection negative-offset validation contract.
pub fn trigger_interop_projection_negative_offset_for_test() -> std::result::Result<(), rapids_interop::InteropError> {
    projection_testing_hooks::trigger_negative_offset_for_test()
}

/// Trigger the shared interop projection missing-string-offsets-child contract.
pub fn trigger_interop_projection_missing_string_offsets_child_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_testing_hooks::trigger_missing_string_offsets_child_for_test()
}

/// Trigger the shared interop projection list-offsets-width contract.
pub fn trigger_interop_projection_list_offsets_width_mismatch_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_testing_hooks::trigger_list_offsets_width_mismatch_for_test()
}

/// Trigger the shared interop projection dictionary-indices-width contract.
pub fn trigger_interop_projection_dictionary_indices_width_mismatch_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_testing_hooks::trigger_dictionary_indices_width_mismatch_for_test()
}

/// Trigger the shared interop projection host-offsets-range contract.
pub fn trigger_interop_projection_host_offsets_too_small_for_test()
-> std::result::Result<(), rapids_interop::InteropError> {
    projection_testing_hooks::trigger_host_offsets_too_small_for_test()
}

/// Trigger the shared interop projection missing-null-mask contract.
pub fn trigger_interop_projection_missing_null_mask_for_test() -> std::result::Result<(), rapids_interop::InteropError>
{
    projection_testing_hooks::trigger_missing_null_mask_for_test()
}

/// Testing-only hook: overwrite one owned ArrowDeviceArray handle's
/// `device_type` field in place.
///
/// # Safety
///
/// `raw_device_array` must be a valid ArrowDeviceArray handle produced by this
/// crate's Arrow device bridge and still owned by a live Rust wrapper.
pub unsafe fn force_device_arrow_array_device_type_for_test(raw_device_array: *mut c_void, device_type: i32) {
    unsafe { arrow_device_testing_hooks::force_device_arrow_array_device_type_for_test(raw_device_array, device_type) }
}
