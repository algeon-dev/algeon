use super::*;

pub(super) fn map_interop_type_id(type_id: InteropTypeId) -> TypeId {
    match type_id {
        InteropTypeId::Empty => TypeId::Empty,
        InteropTypeId::Int8 => TypeId::Int8,
        InteropTypeId::Int16 => TypeId::Int16,
        InteropTypeId::Int32 => TypeId::Int32,
        InteropTypeId::Int64 => TypeId::Int64,
        InteropTypeId::UInt8 => TypeId::UInt8,
        InteropTypeId::UInt16 => TypeId::UInt16,
        InteropTypeId::UInt32 => TypeId::UInt32,
        InteropTypeId::UInt64 => TypeId::UInt64,
        InteropTypeId::Float32 => TypeId::Float32,
        InteropTypeId::Float64 => TypeId::Float64,
        InteropTypeId::Bool8 => TypeId::Bool8,
        InteropTypeId::TimestampDays => TypeId::TimestampDays,
        InteropTypeId::TimestampSeconds => TypeId::TimestampSeconds,
        InteropTypeId::TimestampMilliseconds => TypeId::TimestampMilliseconds,
        InteropTypeId::TimestampMicroseconds => TypeId::TimestampMicroseconds,
        InteropTypeId::TimestampNanoseconds => TypeId::TimestampNanoseconds,
        InteropTypeId::DurationDays => TypeId::DurationDays,
        InteropTypeId::DurationSeconds => TypeId::DurationSeconds,
        InteropTypeId::DurationMilliseconds => TypeId::DurationMilliseconds,
        InteropTypeId::DurationMicroseconds => TypeId::DurationMicroseconds,
        InteropTypeId::DurationNanoseconds => TypeId::DurationNanoseconds,
        InteropTypeId::Dictionary32 => TypeId::Dictionary32,
        InteropTypeId::String => TypeId::String,
        InteropTypeId::List => TypeId::List,
        InteropTypeId::Decimal32 => TypeId::Decimal32,
        InteropTypeId::Decimal64 => TypeId::Decimal64,
        InteropTypeId::Decimal128 => TypeId::Decimal128,
        InteropTypeId::Struct => TypeId::Struct,
    }
}

pub(in crate::interop) struct PreparedDeviceImportColumnGraph {
    pub(in crate::interop) ffi_columns: Vec<cudf_sys::cudf_interop_semantic_column_t>,
    pub(in crate::interop) ffi_bindings: Vec<cudf_sys::cudf_interop_buffer_binding_t>,
    pub(in crate::interop) _child_id_buffers: Vec<Box<[u32]>>,
}

pub(in crate::interop) struct PreparedDeviceImportTableGraph {
    pub(in crate::interop) ffi_table: cudf_sys::cudf_interop_semantic_table_t,
    pub(in crate::interop) ffi_columns: Vec<cudf_sys::cudf_interop_semantic_column_t>,
    pub(in crate::interop) ffi_bindings: Vec<cudf_sys::cudf_interop_buffer_binding_t>,
    pub(in crate::interop) _child_id_buffers: Vec<Box<[u32]>>,
    pub(in crate::interop) _table_column_ids: Box<[u32]>,
}

pub(super) fn checked_i32_from_usize(
    field: &'static str,
    value: usize,
    operation: &'static str,
) -> Result<i32, InteropError> {
    checked_invalid_view_i32_from_usize(field, value, operation)
}

fn interop_binding_from_prepared_binding(binding: &BufferBinding<'_>) -> cudf_sys::cudf_interop_buffer_binding_t {
    let (placement_kind, device) = match binding.placement() {
        Placement::Host => (cudf_sys::cudf_interop_placement_kind_t::CUDF_INTEROP_PLACEMENT_HOST, -1),
        Placement::CudaDevice { device } => {
            (cudf_sys::cudf_interop_placement_kind_t::CUDF_INTEROP_PLACEMENT_CUDA_DEVICE, *device)
        },
    };

    cudf_sys::cudf_interop_buffer_binding_t {
        buffer_id: binding.id().get(),
        data: binding.bytes().base_ptr() as *const _,
        byte_len: binding.bytes().byte_len(),
        placement_kind,
        device,
    }
}

fn interop_semantic_column_from_prepared_column(
    column: &SemanticColumn,
    operation: &'static str,
    child_id_buffers: &mut Vec<Box<[u32]>>,
) -> Result<cudf_sys::cudf_interop_semantic_column_t, InteropError> {
    let (validity_kind, validity_buffer_id, null_count) = match column.validity() {
        PhysicalValidity::NoValidityBuffer => {
            (cudf_sys::cudf_interop_validity_kind_t::CUDF_INTEROP_VALIDITY_NONE, 0, 0)
        },
        PhysicalValidity::ValidityMask { mask, null_count } => (
            cudf_sys::cudf_interop_validity_kind_t::CUDF_INTEROP_VALIDITY_MASK,
            mask.get(),
            match null_count {
                Some(count) => checked_i32_from_usize("null_count", *count, operation)?,
                None => -1,
            },
        ),
    };

    let mut ffi = cudf_sys::cudf_interop_semantic_column_t {
        column_id: column.id().get(),
        type_id: map_interop_type_id(column.data_type().id()).to_raw(),
        scale: column.data_type().scale(),
        len: checked_i32_from_usize("len", column.len(), operation)?,
        offset: checked_i32_from_usize("offset", column.offset(), operation)?,
        validity_kind,
        validity_buffer_id,
        null_count,
        layout_kind: cudf_sys::cudf_interop_layout_kind_t::CUDF_INTEROP_LAYOUT_FIXED_WIDTH,
        values_buffer_id: 0,
        offsets_buffer_id: 0,
        data_buffer_id: 0,
        indices_buffer_id: 0,
        child_column_id: 0,
        child_column_ids: std::ptr::null(),
        num_child_column_ids: 0,
        keys_column_id: 0,
    };

    match column.layout() {
        ColumnLayout::FixedWidth { values } => {
            ffi.layout_kind = cudf_sys::cudf_interop_layout_kind_t::CUDF_INTEROP_LAYOUT_FIXED_WIDTH;
            ffi.values_buffer_id = values.get();
        },
        ColumnLayout::BinaryLike { offsets, data } => {
            ffi.layout_kind = cudf_sys::cudf_interop_layout_kind_t::CUDF_INTEROP_LAYOUT_BINARY_LIKE;
            ffi.offsets_buffer_id = offsets.get();
            ffi.data_buffer_id = data.get();
        },
        ColumnLayout::List { offsets, child } => {
            ffi.layout_kind = cudf_sys::cudf_interop_layout_kind_t::CUDF_INTEROP_LAYOUT_LIST;
            ffi.offsets_buffer_id = offsets.get();
            ffi.child_column_id = child.get();
        },
        ColumnLayout::Struct { children } => {
            ffi.layout_kind = cudf_sys::cudf_interop_layout_kind_t::CUDF_INTEROP_LAYOUT_STRUCT;
            let child_ids: Box<[u32]> = children.iter().map(|child| child.get()).collect::<Vec<_>>().into_boxed_slice();
            ffi.num_child_column_ids = checked_i32_from_usize("num_child_column_ids", child_ids.len(), operation)?;
            ffi.child_column_ids = child_ids.as_ptr();
            child_id_buffers.push(child_ids);
        },
        ColumnLayout::Dictionary { indices, keys } => {
            ffi.layout_kind = cudf_sys::cudf_interop_layout_kind_t::CUDF_INTEROP_LAYOUT_DICTIONARY;
            ffi.indices_buffer_id = indices.get();
            ffi.keys_column_id = keys.get();
        },
    }

    Ok(ffi)
}

pub(super) fn prepared_device_import_column_graph_from_parts(
    semantic_columns: &[SemanticColumn],
    bindings: &[BufferBinding<'_>],
    operation: &'static str,
) -> Result<PreparedDeviceImportColumnGraph, InteropError> {
    let mut child_id_buffers = Vec::new();
    let ffi_columns = semantic_columns
        .iter()
        .map(|column| interop_semantic_column_from_prepared_column(column, operation, &mut child_id_buffers))
        .collect::<Result<Vec<_>, _>>()?;
    let ffi_bindings = bindings.iter().map(interop_binding_from_prepared_binding).collect();
    Ok(PreparedDeviceImportColumnGraph { ffi_columns, ffi_bindings, _child_id_buffers: child_id_buffers })
}

pub(super) fn prepared_device_import_column_graph(
    export: &PreparedColumnExport<'_>,
    operation: &'static str,
) -> Result<PreparedDeviceImportColumnGraph, InteropError> {
    prepared_device_import_column_graph_from_parts(export.semantic_columns(), export.bindings(), operation)
}

pub(super) fn prepared_device_import_table_graph_from_parts(
    num_rows: usize,
    table_columns: impl IntoIterator<Item = u32>,
    semantic_columns: &[SemanticColumn],
    bindings: &[BufferBinding<'_>],
    operation: &'static str,
) -> Result<PreparedDeviceImportTableGraph, InteropError> {
    let mut child_id_buffers = Vec::new();
    let ffi_columns = semantic_columns
        .iter()
        .map(|column| interop_semantic_column_from_prepared_column(column, operation, &mut child_id_buffers))
        .collect::<Result<Vec<_>, _>>()?;
    let ffi_bindings = bindings.iter().map(interop_binding_from_prepared_binding).collect::<Vec<_>>();
    let table_column_ids: Box<[u32]> = table_columns.into_iter().collect::<Vec<_>>().into();
    let ffi_table = cudf_sys::cudf_interop_semantic_table_t {
        num_rows: checked_i32_from_usize("num_rows", num_rows, operation)?,
        column_ids: table_column_ids.as_ptr(),
        num_columns: checked_i32_from_usize("num_columns", table_column_ids.len(), operation)?,
    };

    Ok(PreparedDeviceImportTableGraph {
        ffi_table,
        ffi_columns,
        ffi_bindings,
        _child_id_buffers: child_id_buffers,
        _table_column_ids: table_column_ids,
    })
}

pub(super) fn prepared_device_import_table_graph(
    export: &PreparedTableExport<'_>,
    operation: &'static str,
) -> Result<PreparedDeviceImportTableGraph, InteropError> {
    prepared_device_import_table_graph_from_parts(
        export.table().num_rows(),
        export.table().columns().iter().map(|column| column.get()),
        export.semantic_columns(),
        export.bindings(),
        operation,
    )
}

pub(super) fn owned_device_import_table_graph(
    transfer: &OwnedTableTransfer,
    operation: &'static str,
) -> Result<PreparedDeviceImportTableGraph, InteropError> {
    prepared_device_import_table_graph_from_parts(
        transfer.table().num_rows(),
        transfer.table().columns().iter().map(|column| column.get()),
        transfer.semantic_columns(),
        transfer.bindings(),
        operation,
    )
}
