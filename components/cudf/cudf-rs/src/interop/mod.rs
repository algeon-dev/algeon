//! External interop boundaries.
//!
//! This module re-exports the shared borrowed GPU-view ABI from
//! [`rapids_interop`] and implements `cudf-rs` producer adapters for:
//!
//! - [`crate::ColumnView`]
//! - [`crate::TableView`]
//! - [`crate::Series`]
//! - [`crate::DataFrame`]

use std::sync::Arc;

pub use rapids_interop::{
    BorrowedColumnView as SharedBorrowedColumnView, BorrowedTableView as SharedBorrowedTableView,
    CudaStreamHandle as SharedCudaStreamHandle, DataType as SharedDataType,
    DeviceMemoryResourceHandle as SharedDeviceMemoryResourceHandle, ExecutionBinding as SharedExecutionBinding,
    InteropError as SharedInteropError, InteropErrorKind as SharedInteropErrorKind, NullMask as SharedNullMask,
    NullMaskKind as SharedNullMaskKind, TryBorrowColumnView as SharedTryBorrowColumnView,
    TryBorrowTableView as SharedTryBorrowTableView, TypeId as SharedTypeId,
};
use rapids_interop::{
    BorrowedColumnView, BorrowedTableView, CudaStreamHandle, DataType as InteropDataType, DeviceMemoryResourceHandle,
    ExecutionBinding, InteropError, InteropErrorKind, NullMask, TryBorrowColumnView, TryBorrowTableView,
    TypeId as InteropTypeId,
};

use crate::{
    core::{
        column::{Column, ColumnView},
        table::TableView,
        types::{DataType, TypeId},
    },
    frame::{dataframe::DataFrame, series::Series},
    runtime::execution_domain::ExecutionDomain,
};

fn map_type_id(type_id: TypeId) -> InteropTypeId {
    match type_id {
        TypeId::Empty => InteropTypeId::Empty,
        TypeId::Int8 => InteropTypeId::Int8,
        TypeId::Int16 => InteropTypeId::Int16,
        TypeId::Int32 => InteropTypeId::Int32,
        TypeId::Int64 => InteropTypeId::Int64,
        TypeId::UInt8 => InteropTypeId::UInt8,
        TypeId::UInt16 => InteropTypeId::UInt16,
        TypeId::UInt32 => InteropTypeId::UInt32,
        TypeId::UInt64 => InteropTypeId::UInt64,
        TypeId::Float32 => InteropTypeId::Float32,
        TypeId::Float64 => InteropTypeId::Float64,
        TypeId::Bool8 => InteropTypeId::Bool8,
        TypeId::TimestampDays => InteropTypeId::TimestampDays,
        TypeId::TimestampSeconds => InteropTypeId::TimestampSeconds,
        TypeId::TimestampMilliseconds => InteropTypeId::TimestampMilliseconds,
        TypeId::TimestampMicroseconds => InteropTypeId::TimestampMicroseconds,
        TypeId::TimestampNanoseconds => InteropTypeId::TimestampNanoseconds,
        TypeId::DurationDays => InteropTypeId::DurationDays,
        TypeId::DurationSeconds => InteropTypeId::DurationSeconds,
        TypeId::DurationMilliseconds => InteropTypeId::DurationMilliseconds,
        TypeId::DurationMicroseconds => InteropTypeId::DurationMicroseconds,
        TypeId::DurationNanoseconds => InteropTypeId::DurationNanoseconds,
        TypeId::Dictionary32 => InteropTypeId::Dictionary32,
        TypeId::String => InteropTypeId::String,
        TypeId::List => InteropTypeId::List,
        TypeId::Decimal32 => InteropTypeId::Decimal32,
        TypeId::Decimal64 => InteropTypeId::Decimal64,
        TypeId::Decimal128 => InteropTypeId::Decimal128,
        TypeId::Struct => InteropTypeId::Struct,
    }
}

fn map_data_type(data_type: DataType) -> InteropDataType {
    if data_type.id().is_decimal() {
        InteropDataType::fixed_point(map_type_id(data_type.id()), data_type.scale())
    } else {
        InteropDataType::new(map_type_id(data_type.id()))
    }
}

fn map_cudf_error(kind: InteropErrorKind, operation: &'static str, error: crate::Error) -> InteropError {
    let mut interop = InteropError::new(kind, error.to_string()).with_context("operation", operation);
    for (key, value) in error.context() {
        interop = interop.with_context(key, value.clone());
    }
    interop
}

fn execution_binding<'a>(domain: &Arc<ExecutionDomain>) -> Result<ExecutionBinding<'a>, InteropError> {
    match domain.explicit_exec_options_raw() {
        None => Ok(ExecutionBinding::AmbientDefault { device_snapshot: domain.device_snapshot() }),
        Some(options) => Ok(ExecutionBinding::Explicit {
            device: options.device,
            cuda_stream: CudaStreamHandle::from_raw(
                domain.explicit_stream().expect("explicit domain missing stream").interop_raw_handle(),
            )?,
            mr: DeviceMemoryResourceHandle::from_raw(
                domain
                    .explicit_memory_resource()
                    .expect("explicit domain missing memory resource")
                    .interop_raw_handle(),
            )?,
        }),
    }
}

fn borrow_from_materialized_view<'a>(
    operation: &'static str,
    data_type: DataType,
    num_children: i32,
    len: usize,
    offset: usize,
    data_ptr: *const std::ffi::c_void,
    null_mask: NullMask<'a>,
    execution: ExecutionBinding<'a>,
) -> Result<BorrowedColumnView<'a>, InteropError> {
    let interop_type = map_data_type(data_type);
    if interop_type.id().fixed_width_size_bytes().is_none() {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedType,
            "borrowed interop v1 supports fixed-width column types only",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase()));
    }
    if num_children != 0 {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedLayout,
            "borrowed interop v1 does not support child-bearing column layouts",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase())
        .with_context("num_children", num_children.to_string()));
    }

    BorrowedColumnView::try_new(interop_type, len, offset, data_ptr, null_mask, execution)
        .map_err(|error| error.with_context("operation", operation))
}

fn null_mask_for_view<'a>(
    operation: &'static str,
    nullable: bool,
    null_mask_ptr: Option<*const std::ffi::c_void>,
    offset: usize,
    len: usize,
    null_count: usize,
) -> Result<NullMask<'a>, InteropError> {
    if !nullable {
        return Ok(NullMask::absent());
    }

    let bit_len = offset.checked_add(len).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "null mask bit length overflowed")
            .with_context("operation", operation)
            .with_context("offset", offset.to_string())
            .with_context("len", len.to_string())
    })?;

    let ptr = null_mask_ptr.ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "nullable column did not expose a null-mask pointer")
            .with_context("operation", operation)
    })?;

    NullMask::present(ptr, bit_len, null_count).map_err(|error| error.with_context("operation", operation))
}

fn borrow_from_column_view<'a>(
    view: &'a ColumnView<'_>,
    operation: &'static str,
) -> Result<BorrowedColumnView<'a>, InteropError> {
    let data_type =
        view.data_type().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let num_children =
        view.num_children().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let len = view.len().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let offset = usize::try_from(
        view.offset().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "column view offset must be >= 0")
            .with_context("operation", operation)
    })?;
    let data_ptr =
        view.device_data_ptr().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let nullable = view.nullable().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_count = usize::try_from(
        view.null_count().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "null count must be >= 0").with_context("operation", operation)
    })?;
    let null_mask_ptr =
        view.null_mask_ptr().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let execution = execution_binding(view.execution_domain())?;
    let null_mask = null_mask_for_view(operation, nullable, null_mask_ptr, offset, len, null_count)?;

    borrow_from_materialized_view(operation, data_type, num_children, len, offset, data_ptr, null_mask, execution)
}

fn borrow_from_owned_column<'a>(
    column: &'a Column,
    operation: &'static str,
) -> Result<BorrowedColumnView<'a>, InteropError> {
    let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let data_type =
        view.data_type().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let num_children =
        view.num_children().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let len = view.len().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let offset = usize::try_from(
        view.offset().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "column view offset must be >= 0")
            .with_context("operation", operation)
    })?;
    let data_ptr =
        view.device_data_ptr().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let nullable = view.nullable().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_count = usize::try_from(
        view.null_count().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "null count must be >= 0").with_context("operation", operation)
    })?;
    let null_mask_ptr =
        view.null_mask_ptr().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let execution = execution_binding(column.execution_domain())?;
    let null_mask = null_mask_for_view(operation, nullable, null_mask_ptr, offset, len, null_count)?;

    borrow_from_materialized_view(operation, data_type, num_children, len, offset, data_ptr, null_mask, execution)
}

impl<'view> ColumnView<'view> {
    /// Project this low-level borrowed column view into the shared RAPIDS
    /// borrowed interop ABI.
    pub fn try_borrow_column_view(&self) -> Result<BorrowedColumnView<'_>, InteropError> {
        <Self as TryBorrowColumnView>::try_borrow_column_view(self)
    }
}

impl TryBorrowColumnView for ColumnView<'_> {
    fn try_borrow_column_view<'a>(&'a self) -> Result<BorrowedColumnView<'a>, InteropError> {
        borrow_from_column_view(self, "interop::ColumnView::try_borrow_column_view")
    }
}

impl TableView<'_> {
    /// Project this low-level borrowed table view into the shared RAPIDS
    /// borrowed interop ABI.
    pub fn try_borrow_table_view(&self) -> Result<BorrowedTableView<'_>, InteropError> {
        <Self as TryBorrowTableView>::try_borrow_table_view(self)
    }
}

impl TryBorrowTableView for TableView<'_> {
    fn try_borrow_table_view<'a>(&'a self) -> Result<BorrowedTableView<'a>, InteropError> {
        let num_rows = usize::try_from(self.num_rows().map_err(|error| {
            map_cudf_error(InteropErrorKind::InvalidView, "interop::TableView::try_borrow_table_view", error)
        })?)
        .map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "table row count must be >= 0")
                .with_context("operation", "interop::TableView::try_borrow_table_view")
        })?;
        let num_columns = usize::try_from(self.num_columns().map_err(|error| {
            map_cudf_error(InteropErrorKind::InvalidView, "interop::TableView::try_borrow_table_view", error)
        })?)
        .map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "table column count must be >= 0")
                .with_context("operation", "interop::TableView::try_borrow_table_view")
        })?;
        let execution = execution_binding(self.execution_domain())?;
        let mut columns = Vec::with_capacity(num_columns);
        for index in 0..num_columns {
            let view = self.column(index as i32).map_err(|error| {
                map_cudf_error(InteropErrorKind::InvalidView, "interop::TableView::try_borrow_table_view", error)
            })?;
            let borrowed = borrow_from_materialized_view(
                "interop::TableView::try_borrow_table_view",
                view.data_type().map_err(|error| {
                    map_cudf_error(InteropErrorKind::InvalidView, "interop::TableView::try_borrow_table_view", error)
                })?,
                view.num_children().map_err(|error| {
                    map_cudf_error(InteropErrorKind::InvalidView, "interop::TableView::try_borrow_table_view", error)
                })?,
                view.len().map_err(|error| {
                    map_cudf_error(InteropErrorKind::InvalidView, "interop::TableView::try_borrow_table_view", error)
                })?,
                usize::try_from(view.offset().map_err(|error| {
                    map_cudf_error(InteropErrorKind::InvalidView, "interop::TableView::try_borrow_table_view", error)
                })?)
                .map_err(|_| {
                    InteropError::new(InteropErrorKind::InvalidView, "column view offset must be >= 0")
                        .with_context("operation", "interop::TableView::try_borrow_table_view")
                })?,
                view.device_data_ptr().map_err(|error| {
                    map_cudf_error(InteropErrorKind::InvalidView, "interop::TableView::try_borrow_table_view", error)
                })?,
                null_mask_for_view(
                    "interop::TableView::try_borrow_table_view",
                    view.nullable().map_err(|error| {
                        map_cudf_error(
                            InteropErrorKind::InvalidView,
                            "interop::TableView::try_borrow_table_view",
                            error,
                        )
                    })?,
                    view.null_mask_ptr().map_err(|error| {
                        map_cudf_error(
                            InteropErrorKind::InvalidView,
                            "interop::TableView::try_borrow_table_view",
                            error,
                        )
                    })?,
                    usize::try_from(view.offset().map_err(|error| {
                        map_cudf_error(
                            InteropErrorKind::InvalidView,
                            "interop::TableView::try_borrow_table_view",
                            error,
                        )
                    })?)
                    .map_err(|_| {
                        InteropError::new(InteropErrorKind::InvalidView, "column view offset must be >= 0")
                            .with_context("operation", "interop::TableView::try_borrow_table_view")
                    })?,
                    view.len().map_err(|error| {
                        map_cudf_error(
                            InteropErrorKind::InvalidView,
                            "interop::TableView::try_borrow_table_view",
                            error,
                        )
                    })?,
                    usize::try_from(view.null_count().map_err(|error| {
                        map_cudf_error(
                            InteropErrorKind::InvalidView,
                            "interop::TableView::try_borrow_table_view",
                            error,
                        )
                    })?)
                    .map_err(|_| {
                        InteropError::new(InteropErrorKind::InvalidView, "null count must be >= 0")
                            .with_context("operation", "interop::TableView::try_borrow_table_view")
                    })?,
                )?,
                execution,
            )?;
            columns.push(borrowed);
        }
        BorrowedTableView::try_new(num_rows, columns, execution)
            .map_err(|error| error.with_context("operation", "interop::TableView::try_borrow_table_view"))
    }
}

impl Series {
    /// Convenience producer that lends this high-level `Series` as a shared
    /// borrowed interop column view without host copies on the input path.
    pub fn try_borrow_column_view(&self) -> Result<BorrowedColumnView<'_>, InteropError> {
        <Self as TryBorrowColumnView>::try_borrow_column_view(self)
    }
}

impl TryBorrowColumnView for Series {
    fn try_borrow_column_view<'a>(&'a self) -> Result<BorrowedColumnView<'a>, InteropError> {
        borrow_from_owned_column(self.as_column(), "interop::Series::try_borrow_column_view")
    }
}

impl DataFrame {
    /// Convenience producer that lends this high-level `DataFrame` as a shared
    /// borrowed interop table view without host copies on the input path.
    pub fn try_borrow_table_view(&self) -> Result<BorrowedTableView<'_>, InteropError> {
        <Self as TryBorrowTableView>::try_borrow_table_view(self)
    }
}

impl TryBorrowTableView for DataFrame {
    fn try_borrow_table_view<'a>(&'a self) -> Result<BorrowedTableView<'a>, InteropError> {
        let execution = execution_binding(self.execution_domain())?;
        let columns = self
            .columns_map()
            .values()
            .map(|column| borrow_from_owned_column(column.as_ref(), "interop::DataFrame::try_borrow_table_view"))
            .collect::<Result<Vec<_>, _>>()?;
        BorrowedTableView::try_new(self.num_rows(), columns, execution)
            .map_err(|error| error.with_context("operation", "interop::DataFrame::try_borrow_table_view"))
    }
}
