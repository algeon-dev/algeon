use core::{ffi::c_void, marker::PhantomData, ptr::NonNull};

use crate::{DataType, ExecutionBinding, InteropError, InteropErrorKind, Result, TypeId};

/// Distinguishes between the absence of a null mask and the presence of a null
/// mask whose null count may still be zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullMaskKind {
    Absent,
    Present { ptr: NonNull<c_void>, bit_len: usize, null_count: usize },
}

/// Lifetime-bound borrowed null-mask descriptor for a column view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NullMask<'a> {
    kind: NullMaskKind,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> NullMask<'a> {
    /// Construct the canonical "no null mask is present" descriptor.
    pub fn absent() -> Self {
        Self { kind: NullMaskKind::Absent, _lifetime: PhantomData }
    }

    /// Construct the canonical "a null mask is present" descriptor.
    ///
    /// `ptr` is accepted as `*const c_void` because this is a read-only
    /// borrowed descriptor. The internal `NonNull<c_void>` storage is only an
    /// implementation detail for carrying a non-null raw address.
    ///
    /// `null_count == 0` remains a valid present-mask state.
    pub fn present(ptr: *const c_void, bit_len: usize, null_count: usize) -> Result<Self> {
        let ptr = NonNull::new(ptr as *mut c_void).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "present null mask must use a non-null pointer")
                .with_context("field", "null_mask")
        })?;
        Ok(Self { kind: NullMaskKind::Present { ptr, bit_len, null_count }, _lifetime: PhantomData })
    }

    pub fn kind(&self) -> &NullMaskKind {
        &self.kind
    }
}

/// Borrowed descriptor for a single GPU column view.
///
/// This is a low-level fact record about producer-owned memory. It does not
/// transfer ownership, does not allocate output, and must not outlive the
/// producer borrow that created it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BorrowedColumnView<'a> {
    data_type: DataType,
    len: usize,
    offset: usize,
    data_ptr: *const c_void,
    null_mask: NullMask<'a>,
    execution: ExecutionBinding<'a>,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> BorrowedColumnView<'a> {
    /// Validate and construct a borrowed column descriptor.
    ///
    /// The current borrowed compatibility profile accepts only fixed-width
    /// types and rejects invalid physical view states such as null data
    /// pointers for non-empty views, overflowing active ranges, misaligned
    /// active addresses, and undersized null masks.
    pub fn try_new(
        data_type: DataType,
        len: usize,
        offset: usize,
        data_ptr: *const c_void,
        null_mask: NullMask<'a>,
        execution: ExecutionBinding<'a>,
    ) -> Result<Self> {
        let elem_size = data_type.id().fixed_width_size_bytes().ok_or_else(|| {
            InteropError::new(InteropErrorKind::UnsupportedType, "borrowed interop v1 supports fixed-width types only")
                .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase())
        })?;

        if len > 0 && data_ptr.is_null() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "non-empty view must use a non-null data pointer",
            )
            .with_context("len", len.to_string())
            .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase()));
        }

        let active_start = offset.checked_mul(elem_size).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "active byte range overflowed while computing offset")
                .with_context("offset", offset.to_string())
                .with_context("elem_size", elem_size.to_string())
        })?;
        let active_len = len.checked_mul(elem_size).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "active byte range overflowed while computing length")
                .with_context("len", len.to_string())
                .with_context("elem_size", elem_size.to_string())
        })?;
        let _active_end = active_start.checked_add(active_len).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "active byte range overflowed while computing end offset")
                .with_context("offset", offset.to_string())
                .with_context("len", len.to_string())
                .with_context("elem_size", elem_size.to_string())
        })?;

        if !data_ptr.is_null() {
            let active_addr = (data_ptr as usize).checked_add(active_start).ok_or_else(|| {
                InteropError::new(InteropErrorKind::InvalidView, "active data address overflowed")
                    .with_context("offset", offset.to_string())
                    .with_context("elem_size", elem_size.to_string())
            })?;
            if active_addr % elem_size != 0 {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "active data pointer is misaligned for declared type",
                )
                .with_context("address", active_addr.to_string())
                .with_context("alignment", elem_size.to_string())
                .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase()));
            }
        }

        match null_mask.kind() {
            NullMaskKind::Absent => {},
            NullMaskKind::Present { bit_len, null_count, .. } => {
                if *null_count > len {
                    return Err(InteropError::new(InteropErrorKind::InvalidView, "null count exceeds logical length")
                        .with_context("null_count", null_count.to_string())
                        .with_context("len", len.to_string()));
                }
                let required_bits = offset.checked_add(len).ok_or_else(|| {
                    InteropError::new(InteropErrorKind::InvalidView, "required null-mask range overflowed")
                        .with_context("offset", offset.to_string())
                        .with_context("len", len.to_string())
                })?;
                if *bit_len < required_bits {
                    return Err(InteropError::new(
                        InteropErrorKind::InvalidView,
                        "null mask is too small for the active logical range",
                    )
                    .with_context("bit_len", bit_len.to_string())
                    .with_context("required_bits", required_bits.to_string()));
                }
            },
        }

        Ok(Self { data_type, len, offset, data_ptr, null_mask, execution, _lifetime: PhantomData })
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub fn type_id(&self) -> TypeId {
        self.data_type.id()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn data_ptr(&self) -> *const c_void {
        self.data_ptr
    }

    pub fn null_mask(&self) -> &NullMask<'a> {
        &self.null_mask
    }

    pub fn execution(&self) -> ExecutionBinding<'a> {
        self.execution
    }

    pub fn element_size_bytes(&self) -> usize {
        self.type_id().fixed_width_size_bytes().expect("BorrowedColumnView always holds a fixed-width type")
    }
}

/// Borrowed descriptor for a GPU table view.
///
/// The table owns its descriptor collection so consumers do not need to keep
/// parallel temporary vectors alive just to preserve column descriptor
/// lifetimes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowedTableView<'a> {
    num_rows: usize,
    columns: Vec<BorrowedColumnView<'a>>,
    execution: ExecutionBinding<'a>,
}

impl<'a> BorrowedTableView<'a> {
    /// Validate and construct a borrowed table descriptor whose columns all
    /// share the same row count and execution binding.
    pub fn try_new(
        num_rows: usize,
        columns: Vec<BorrowedColumnView<'a>>,
        execution: ExecutionBinding<'a>,
    ) -> Result<Self> {
        for (index, column) in columns.iter().enumerate() {
            if column.len() != num_rows {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    "column length does not match borrowed table row count",
                )
                .with_context("column_index", index.to_string())
                .with_context("column_len", column.len().to_string())
                .with_context("num_rows", num_rows.to_string()));
            }
            if column.execution() != execution {
                return Err(InteropError::new(
                    InteropErrorKind::ExecutionBindingMismatch,
                    "all borrowed table columns must share the table execution binding",
                )
                .with_context("column_index", index.to_string()));
            }
        }

        Ok(Self { num_rows, columns, execution })
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn columns(&self) -> &[BorrowedColumnView<'a>] {
        &self.columns
    }

    pub fn execution(&self) -> ExecutionBinding<'a> {
        self.execution
    }
}
