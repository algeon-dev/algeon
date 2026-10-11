//! Explicit unstable testing support.
//!
//! This module exists so contract and integration tests can keep validating
//! low-level execution, synchronization, and allocation behavior without
//! forcing those hooks onto the primary public API surface.
//!
//! Items in this module are intentionally **not** part of the stable,
//! user-facing API contract. They may change or disappear at any time.

pub mod arrow;
pub mod column;
pub mod core;
pub mod dataframe;
pub mod device_safety;
pub mod dictionary;
pub mod error;
pub mod interop;
mod named_cases;
pub mod series;

pub use named_cases::run_named_cases;

use crate::{
    AggSpec, MergeOptions, SortOptions,
    core::{
        Column, DeviceMemoryResource, Result, Scalar, Table,
        column::{ColumnView as CoreColumnView, slice_view_for_test},
        table::TableView as CoreTableView,
    },
    frame::dataframe::DataFrame,
    ops::groupby::GroupBy,
    runtime::{CudaEvent, CudaStream, Exec, Runtime},
};

/// Trait imports for the testing support surface.
pub mod prelude {
    pub use super::{
        BorrowedColumnSliceTestingExt, BorrowedViewTestingExt, DataFrameInContextTestingExt,
        ExecutionDomainMemoryResourceTestingExt, ExecutionDomainRebindTestingExt, ExecutionDomainTestingExt,
        GroupByTestingExt, InteropRawHandleTestingExt, RawHandleTestingExt, RuntimeTestingExt,
    };
}

pub use crate::core::{column::ColumnView, table::TableView};

pub mod device_copy {
    use crate::frame::testing_hooks;

    /// Read how many `Series::copy_into_runtime` and `DataFrame::copy_into_runtime`
    /// calls deep-copied device data into another retained provenance.
    ///
    /// The total is process-wide and monotonic. Callers compare values read
    /// before and after the call under test. A copy made concurrently by
    /// another test in the same process would inflate that delta, so exact
    /// delta assertions rely on each test running in its own process, which is
    /// the nextest default.
    pub fn device_to_device_copy_count_for_test() -> usize {
        testing_hooks::device_to_device_copy_count_for_test()
    }
}

pub mod event {
    use cudf_sys::{
        CUDF_SUCCESS,
        test_support::{
            cudf_test_event_allow_synchronize, cudf_test_event_fail_synchronize, cudf_test_event_reset_observation,
            cudf_test_event_synchronize_call_count,
        },
    };

    /// Fails every CUDA event synchronization in the process until dropped.
    ///
    /// Dropping restores synchronization even when the test panics, so the
    /// fault never reaches a later test in the same process.
    #[must_use = "the fault clears when the guard drops"]
    pub struct EventSynchronizeFault(());

    impl EventSynchronizeFault {
        pub fn inject() -> Self {
            // SAFETY: the hook only sets a process-wide test fault flag.
            unsafe { cudf_test_event_fail_synchronize() };
            Self(())
        }
    }

    impl Drop for EventSynchronizeFault {
        fn drop(&mut self) {
            // SAFETY: the hook only clears a process-wide test fault flag.
            unsafe { cudf_test_event_allow_synchronize() };
        }
    }

    /// Clears the process-wide CUDA event observation counters.
    pub fn reset_event_observation_for_test() {
        // SAFETY: the hook only resets process-wide test counters.
        unsafe { cudf_test_event_reset_observation() };
    }

    /// CUDA event synchronizations since the last reset. The count is
    /// process-wide, so an exact assertion relies on each test running in its
    /// own process, which is the nextest default.
    pub fn event_synchronize_count_for_test() -> i32 {
        let mut count = 0;
        // SAFETY: `count` is a valid output location for the hook.
        let status = unsafe { cudf_test_event_synchronize_call_count(&mut count) };
        assert_eq!(status, CUDF_SUCCESS, "the event observation hook reads its counter");
        count
    }
}

pub mod stream {
    use cudf_sys::test_support::{cudf_test_stream_block_synchronize, cudf_test_stream_unblock_synchronize};

    /// Blocks every CUDA stream synchronization in the process until dropped.
    ///
    /// Dropping releases blocked and later synchronizations even when the test
    /// panics, so no caller stays parked past the test.
    #[must_use = "the block releases when the guard drops"]
    pub struct StreamSynchronizeBlock(());

    impl StreamSynchronizeBlock {
        pub fn engage() -> Self {
            // SAFETY: the hook only sets a process-wide test gate.
            unsafe { cudf_test_stream_block_synchronize() };
            Self(())
        }
    }

    impl Drop for StreamSynchronizeBlock {
        fn drop(&mut self) {
            // SAFETY: the hook only opens a process-wide test gate.
            unsafe { cudf_test_stream_unblock_synchronize() };
        }
    }
}

pub mod concatenate {
    use crate::{core::Result, ops::concatenate};

    /// Trigger the concatenate-column input-count limit validation contract.
    pub fn trigger_concatenate_column_count_limit_error_for_test() -> Result<()> {
        concatenate::trigger_concatenate_column_count_limit_error_for_test()
    }

    /// Trigger the concatenate-table input-count limit validation contract.
    pub fn trigger_concatenate_table_count_limit_error_for_test() -> Result<()> {
        concatenate::trigger_concatenate_table_count_limit_error_for_test()
    }
}

pub mod factorization {
    pub use crate::ops::factorization::testing_hooks::trigger_factorize_empty_row_code_dtype_error_for_test;
    use crate::{core::Result, frame::factorization};

    /// Trigger the Codebook key-count limit validation path used by integration
    /// tests.
    pub fn trigger_codebook_key_count_limit_error_for_test() -> Result<()> {
        factorization::trigger_codebook_key_count_limit_error_for_test()
    }
}

pub mod groupby {
    pub use crate::ops::groupby::testing_hooks::{
        trigger_groupby_duplicate_result_column_consumption_for_test,
        trigger_groupby_request_agg_count_limit_error_for_test, trigger_groupby_request_count_limit_error_for_test,
        trigger_groupby_result_column_index_out_of_bounds_for_test,
        trigger_groupby_result_group_index_out_of_bounds_for_test,
        trigger_groupby_total_agg_handle_count_limit_error_for_test, trigger_value_counts_empty_count_result_for_test,
        trigger_value_counts_missing_count_results_for_test, trigger_value_counts_missing_key_columns_for_test,
    };
}

pub mod join {
    use crate::{core::Result, frame::dataframe::DataFrame, ops::join};

    /// Trigger the merge-result reassembly invariant branch used by integration
    /// tests to lock the downstream error contract.
    pub fn trigger_merge_reassembly_key_count_error_for_test() -> Result<DataFrame> {
        join::trigger_merge_reassembly_key_count_error_for_test()
    }

    /// Trigger projected cross-join output row-count overflow before reservation
    /// or materialization.
    pub fn trigger_projected_cross_join_output_row_overflow_for_test() -> Result<usize> {
        join::trigger_projected_cross_join_output_row_overflow_for_test()
    }

    /// Trigger projected cross-join input row-count rejection when output rows
    /// would otherwise be zero.
    pub fn trigger_projected_cross_join_input_row_limit_for_test() -> Result<usize> {
        join::trigger_projected_cross_join_input_row_limit_for_test()
    }
}

pub mod sorting {
    pub use crate::ops::sorting::testing_hooks::trigger_sort_key_count_limit_error_for_test;
}

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

/// Testing-only access to runtime provenance identity.
pub trait RuntimeTestingExt {
    fn testing_execution_domain_id(&self) -> u64;
    fn testing_retained_lifetime_count(&self) -> usize;
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
    fn testing_gather_in_context(&self, indices: &crate::Series, ctx: Exec<'_>) -> Result<DataFrame>;
    fn testing_filter_in_context(&self, mask: &crate::Series, ctx: Exec<'_>) -> Result<DataFrame>;
    fn testing_sort_values_in_context<I, S>(&self, by: I, options: SortOptions, ctx: Exec<'_>) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>;
    fn testing_merge_in_context(&self, right: &DataFrame, options: MergeOptions, ctx: Exec<'_>) -> Result<DataFrame>;
}

/// Testing-only access to groupby execution-context hooks.
pub trait GroupByTestingExt {
    fn testing_agg_in_context(&self, specs: &[AggSpec], ctx: Exec<'_>) -> Result<DataFrame>;
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

impl RuntimeTestingExt for Runtime {
    fn testing_execution_domain_id(&self) -> u64 {
        self.__debug_execution_domain_id()
    }

    fn testing_retained_lifetime_count(&self) -> usize {
        self.__debug_retained_lifetime_count()
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
        slice_view_for_test(self, offset, length)
    }
}

impl DataFrameInContextTestingExt for DataFrame {
    fn testing_gather_in_context(&self, indices: &crate::Series, ctx: Exec<'_>) -> Result<DataFrame> {
        self.__debug_gather_in_context(indices, ctx)
    }

    fn testing_filter_in_context(&self, mask: &crate::Series, ctx: Exec<'_>) -> Result<DataFrame> {
        self.__debug_filter_in_context(mask, ctx)
    }

    fn testing_sort_values_in_context<I, S>(&self, by: I, options: SortOptions, ctx: Exec<'_>) -> Result<DataFrame>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.__debug_sort_values_in_context(by, options, ctx)
    }

    fn testing_merge_in_context(&self, right: &DataFrame, options: MergeOptions, ctx: Exec<'_>) -> Result<DataFrame> {
        self.__debug_merge_in_context(right, options, ctx)
    }
}

impl GroupByTestingExt for GroupBy<'_> {
    fn testing_agg_in_context(&self, specs: &[AggSpec], ctx: Exec<'_>) -> Result<DataFrame> {
        self.__debug_agg_in_context(specs, ctx)
    }
}
