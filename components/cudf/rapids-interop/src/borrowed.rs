use core::{ffi::c_void, marker::PhantomData, ptr::NonNull};
use std::collections::{HashMap, HashSet};

use crate::{
    BufferBinding, ColumnId, ColumnLayout, DataType, ExecutionBinding, InteropError, InteropErrorKind,
    PhysicalValidity, Result, SemanticColumn, TypeId,
    validation::{binding_map, semantic_column_map, validate_prepared_column_against_bindings},
};

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

impl NullMask<'_> {
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
#[derive(Debug, Clone, PartialEq, Eq)]
enum BorrowedColumnViewRepr<'a> {
    RawFixedWidth {
        data_type: DataType,
        len: usize,
        offset: usize,
        data_ptr: *const c_void,
        null_mask: NullMask<'a>,
    },
    Semantic {
        column: SemanticColumn,
        semantic_columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'a>>,
        data_ptr: *const c_void,
        null_mask: NullMask<'a>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowedColumnView<'a> {
    repr: BorrowedColumnViewRepr<'a>,
    execution: ExecutionBinding<'a>,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> BorrowedColumnView<'a> {
    fn validate_raw_fixed_width(
        data_type: DataType,
        len: usize,
        offset: usize,
        data_ptr: *const c_void,
        null_mask: NullMask<'a>,
    ) -> Result<()> {
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

        Ok(())
    }

    fn validate_supported_semantic_subset(
        column: &SemanticColumn,
        columns: &HashMap<ColumnId, &SemanticColumn>,
        visited: &mut HashSet<ColumnId>,
    ) -> Result<()> {
        if !visited.insert(column.id()) {
            return Ok(());
        }

        match column.layout() {
            ColumnLayout::FixedWidth { .. } | ColumnLayout::BinaryLike { .. } => Ok(()),
            ColumnLayout::List { child, .. } => {
                let child_column = columns.get(child).copied().ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::NotFound,
                        "borrowed compatibility list layout referenced a missing child semantic column",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("child_column_id", child.get().to_string())
                })?;
                Self::validate_supported_semantic_subset(child_column, columns, visited)
            },
            ColumnLayout::Struct { children } => {
                for child in children {
                    let child_column = columns.get(child).copied().ok_or_else(|| {
                        InteropError::new(
                            InteropErrorKind::NotFound,
                            "borrowed compatibility struct layout referenced a missing child semantic column",
                        )
                        .with_context("column_id", column.id().get().to_string())
                        .with_context("child_column_id", child.get().to_string())
                    })?;
                    Self::validate_supported_semantic_subset(child_column, columns, visited)?;
                }
                Ok(())
            },
            ColumnLayout::Dictionary { keys, .. } => {
                let keys_column = columns.get(keys).copied().ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::NotFound,
                        "borrowed compatibility dictionary layout referenced a missing keys semantic column",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("child_column_id", keys.get().to_string())
                })?;
                Self::validate_supported_semantic_subset(keys_column, columns, visited)
            },
        }
    }

    /// Validate and construct a borrowed column descriptor.
    ///
    /// This raw constructor is the legacy fixed-width-only entry point. It
    /// rejects invalid physical view states such as null data pointers for
    /// non-empty views, overflowing active ranges, misaligned active
    /// addresses, and undersized null masks.
    pub fn try_new(
        data_type: DataType,
        len: usize,
        offset: usize,
        data_ptr: *const c_void,
        null_mask: NullMask<'a>,
        execution: ExecutionBinding<'a>,
    ) -> Result<Self> {
        Self::validate_raw_fixed_width(data_type, len, offset, data_ptr, null_mask)?;
        Ok(Self {
            repr: BorrowedColumnViewRepr::RawFixedWidth { data_type, len, offset, data_ptr, null_mask },
            execution,
            _lifetime: PhantomData,
        })
    }

    /// Validate and construct a borrowed descriptor from a semantic column
    /// graph plus concrete device bindings.
    ///
    /// This is the borrowed compatibility constructor used for the current
    /// fixed-width, `STRING`, `LIST`, `STRUCT`, and `DICTIONARY32` slice. The
    /// graph must resolve to one CUDA device that matches the provided
    /// execution binding.
    pub fn try_new_with_semantic_columns(
        column: SemanticColumn,
        semantic_columns: Vec<SemanticColumn>,
        bindings: Vec<BufferBinding<'a>>,
        execution: ExecutionBinding<'a>,
    ) -> Result<Self> {
        let column_map = semantic_column_map(&semantic_columns, "column_id")?;
        let mapped_root = column_map.get(&column.id()).copied().ok_or_else(|| {
            InteropError::new(
                InteropErrorKind::NotFound,
                "borrowed column view root must be present in the semantic column graph",
            )
            .with_context("column_id", column.id().get().to_string())
        })?;
        if mapped_root != &column {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "borrowed column view root must match the semantic column graph entry for its column id",
            )
            .with_context("column_id", column.id().get().to_string()));
        }

        let binding_map = binding_map(&bindings)?;
        let mut validated_devices = HashMap::new();
        let producer_device = validate_prepared_column_against_bindings(
            &column,
            &column_map,
            &binding_map,
            InteropErrorKind::ExecutionBindingMismatch,
            &mut HashSet::new(),
            &mut validated_devices,
        )?;
        if validated_devices.len() != column_map.len() {
            let unreferenced = semantic_columns
                .iter()
                .find(|candidate| !validated_devices.contains_key(&candidate.id()))
                .expect("missing validated semantic column must exist");
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "borrowed column view included an unreferenced semantic column",
            )
            .with_context("column_id", unreferenced.id().get().to_string()));
        }
        if producer_device != execution.device() {
            return Err(InteropError::new(
                InteropErrorKind::ExecutionBindingMismatch,
                "borrowed column view semantic graph must live on the execution CUDA device",
            )
            .with_context("column_id", column.id().get().to_string())
            .with_context("producer_device", producer_device.to_string())
            .with_context("execution_device", execution.device().to_string()));
        }
        Self::validate_supported_semantic_subset(&column, &column_map, &mut HashSet::new())?;

        let data_ptr = match column.layout() {
            ColumnLayout::FixedWidth { values } => {
                binding_map.get(values).map(|binding| binding.bytes().base_ptr()).ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::NotFound,
                        "borrowed column view fixed-width layout referenced a missing value binding",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("buffer_id", values.get().to_string())
                })?
            },
            ColumnLayout::BinaryLike { .. }
            | ColumnLayout::Dictionary { .. }
            | ColumnLayout::List { .. }
            | ColumnLayout::Struct { .. } => core::ptr::null(),
        };

        let null_mask = match column.validity() {
            PhysicalValidity::NoValidityBuffer => NullMask::absent(),
            PhysicalValidity::ValidityMask { mask, null_count } => {
                let mask_binding = binding_map.get(mask).copied().ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::NotFound,
                        "borrowed column view referenced a missing validity binding",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("buffer_id", mask.get().to_string())
                })?;
                let null_count = null_count.ok_or_else(|| {
                    InteropError::new(
                        InteropErrorKind::InvalidView,
                        "borrowed column view root validity mask must declare a null count",
                    )
                    .with_context("column_id", column.id().get().to_string())
                    .with_context("buffer_id", mask.get().to_string())
                })?;
                NullMask::present(mask_binding.bytes().base_ptr(), mask_binding.bytes().byte_len() * 8, null_count)?
            },
        };

        Ok(Self {
            repr: BorrowedColumnViewRepr::Semantic { column, semantic_columns, bindings, data_ptr, null_mask },
            execution,
            _lifetime: PhantomData,
        })
    }

    pub fn data_type(&self) -> DataType {
        match &self.repr {
            BorrowedColumnViewRepr::RawFixedWidth { data_type, .. } => *data_type,
            BorrowedColumnViewRepr::Semantic { column, .. } => column.data_type(),
        }
    }

    pub fn type_id(&self) -> TypeId {
        self.data_type().id()
    }

    pub fn len(&self) -> usize {
        match &self.repr {
            BorrowedColumnViewRepr::RawFixedWidth { len, .. } => *len,
            BorrowedColumnViewRepr::Semantic { column, .. } => column.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn offset(&self) -> usize {
        match &self.repr {
            BorrowedColumnViewRepr::RawFixedWidth { offset, .. } => *offset,
            BorrowedColumnViewRepr::Semantic { column, .. } => column.offset(),
        }
    }

    pub fn data_ptr(&self) -> *const c_void {
        match &self.repr {
            BorrowedColumnViewRepr::RawFixedWidth { data_ptr, .. } => *data_ptr,
            BorrowedColumnViewRepr::Semantic { data_ptr, .. } => *data_ptr,
        }
    }

    pub fn null_mask(&self) -> &NullMask<'a> {
        match &self.repr {
            BorrowedColumnViewRepr::RawFixedWidth { null_mask, .. } => null_mask,
            BorrowedColumnViewRepr::Semantic { null_mask, .. } => null_mask,
        }
    }

    pub fn execution(&self) -> ExecutionBinding<'a> {
        self.execution
    }

    pub fn column(&self) -> Option<&SemanticColumn> {
        match &self.repr {
            BorrowedColumnViewRepr::RawFixedWidth { .. } => None,
            BorrowedColumnViewRepr::Semantic { column, .. } => Some(column),
        }
    }

    pub fn semantic_columns(&self) -> &[SemanticColumn] {
        match &self.repr {
            BorrowedColumnViewRepr::RawFixedWidth { .. } => &[],
            BorrowedColumnViewRepr::Semantic { semantic_columns, .. } => semantic_columns,
        }
    }

    pub fn bindings(&self) -> &[BufferBinding<'a>] {
        match &self.repr {
            BorrowedColumnViewRepr::RawFixedWidth { .. } => &[],
            BorrowedColumnViewRepr::Semantic { bindings, .. } => bindings,
        }
    }

    pub fn element_size_bytes(&self) -> Option<usize> {
        self.type_id().fixed_width_size_bytes()
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
