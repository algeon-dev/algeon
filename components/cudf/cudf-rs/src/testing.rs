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
pub mod concatenate;
pub mod core;
pub mod dataframe;
pub mod dictionary;
pub mod error;
pub mod factorization;
pub mod groupby;
pub mod interop;
pub mod join;
pub mod series;
pub mod sorting;

use crate::{
    AggSpec, MergeOptions, SortOptions,
    core::{
        Column, DeviceMemoryResource, Result, Scalar, Table, column::ColumnView as CoreColumnView,
        table::TableView as CoreTableView,
    },
    frame::dataframe::DataFrame,
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
