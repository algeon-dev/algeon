use super::*;

struct MaterializedViewProjectionInput {
    data_type: DataType,
    num_children: usize,
    len: usize,
    offset: usize,
    data_ptr: *const std::ffi::c_void,
    nullable: bool,
    null_mask_ptr: Option<*const std::ffi::c_void>,
    null_count: usize,
}

struct MaterializedViewProjectionState {
    operation: &'static str,
    default_device: i32,
    column_id: ColumnId,
    next_column_id: u32,
    next_buffer_id: u32,
}

fn prepared_column_projection_from_materialized_view<'a>(
    input: MaterializedViewProjectionInput,
    state: MaterializedViewProjectionState,
) -> Result<PreparedColumnProjection<'a>, InteropError> {
    let operation = state.operation;
    let data_type = input.data_type;
    let interop_type = map_data_type(data_type);
    let active_elements = input
        .offset
        .checked_add(input.len)
        .ok_or_else(|| active_projection_range_overflow_error(operation, input.offset, input.len))?;
    let mut next_buffer_id = state.next_buffer_id;

    if input.num_children != 0 {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedLayout,
            "prepared device export currently supports fixed-width leaf, string, list, struct, and dictionary column layouts only",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase())
        .with_context("child_count", input.num_children.to_string()));
    }

    let values_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
    let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
    let elem_size = interop_type.id().fixed_width_size_bytes().ok_or_else(|| {
        InteropError::new(
            InteropErrorKind::UnsupportedLayout,
            "prepared device export currently supports fixed-width leaf column layouts on the non-nested fallback path only",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase())
    })?;
    let values_byte_len = active_elements.checked_mul(elem_size).ok_or_else(|| {
        projection_byte_range_overflow_error(
            operation,
            "active value byte range overflowed",
            "value_byte_range",
            input.offset,
            input.len,
        )
        .with_context("elem_size", elem_size.to_string())
    })?;

    let values_device = pointer_device(operation, input.data_ptr, state.default_device)?;
    let (validity, validity_binding) = validity_for_prepared_projection(
        operation,
        input.nullable,
        input.null_mask_ptr,
        input.null_count,
        active_elements,
        validity_id,
        state.default_device,
    )?;
    let mut bindings = vec![BufferBinding::new(
        values_id,
        bound_bytes(operation, input.data_ptr, values_byte_len)?,
        Placement::CudaDevice { device: values_device },
    )];
    if let Some(binding) = validity_binding {
        bindings.push(binding);
    }

    let column = SemanticColumn::new(
        state.column_id,
        interop_type,
        input.len,
        input.offset,
        validity,
        schema_nullability(input.nullable),
        ColumnLayout::FixedWidth { values: values_id },
    );
    Ok(PreparedColumnProjection {
        column: column.clone(),
        semantic_columns: vec![column],
        bindings,
        next_buffer_id,
        next_column_id: state.next_column_id,
    })
}

pub(super) fn prepared_column_projection_from_view<'a>(
    view: &ColumnView<'a>,
    operation: &'static str,
    default_device: i32,
    next_column_id: u32,
    next_buffer_id: u32,
) -> Result<PreparedColumnProjection<'a>, InteropError> {
    let data_type =
        view.data_type().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let num_children =
        view.num_children().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let len = view.len().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let offset_raw = view.offset().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let offset =
        usize::try_from(offset_raw).map_err(|_| negative_projection_offset_error(operation, offset_raw as i64))?;
    let data_ptr =
        view.device_data_ptr().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let nullable = view.nullable().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let schema_nullability = schema_nullability(nullable);
    let null_count =
        view.null_count().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_mask_ptr =
        view.null_mask_ptr().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let active_elements =
        offset.checked_add(len).ok_or_else(|| active_projection_range_overflow_error(operation, offset, len))?;
    let mut next_column_id = next_column_id;
    let column_id = allocate_column_id(operation, &mut next_column_id)?;

    if map_data_type(data_type).id() == InteropTypeId::String {
        let mut next_buffer_id = next_buffer_id;
        let offsets_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let data_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

        if num_children < 1 {
            return Err(projection_child_count_error(
                operation,
                "string column view must expose an offsets child",
                "string",
                num_children,
                1,
                "requires_offsets_child",
                "expose an offsets child before projecting string data",
            ));
        }

        let offsets_view =
            view.child(0).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_type = offsets_view
            .data_type()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_elem_size = map_data_type(offsets_type).id().fixed_width_size_bytes().ok_or_else(|| {
            projection_child_fixed_width_error(
                operation,
                "string offsets child must be fixed-width",
                "offsets_layout",
                format!("{:?}", offsets_type.id()).to_lowercase(),
                "materialize string offsets as a fixed-width Int32 child before projection",
            )
        })?;
        if offsets_elem_size != std::mem::size_of::<i32>() {
            return Err(projection_child_width_error(
                operation,
                "string offsets child must use 32-bit offsets",
                "offsets_layout",
                offsets_elem_size,
                std::mem::size_of::<i32>(),
                "materialize string offsets as 32-bit signed integers before projection",
            ));
        }
        let offsets_required_elements = offset
            .checked_add(len)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| offsets_range_overflow_error(operation, "string offsets range overflowed", offset, len))?;
        let offsets_byte_len = offsets_required_elements.checked_mul(offsets_elem_size).ok_or_else(|| {
            projection_byte_range_overflow_error(
                operation,
                "string offsets byte range overflowed",
                "offsets_byte_range",
                offset,
                len,
            )
        })?;
        let offsets_ptr = offsets_view
            .device_data_ptr()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_device = pointer_device(operation, offsets_ptr, default_device)?;

        let (chars_ptr, chars_byte_len) = view
            .string_chars_ptr_and_size()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let chars_device = pointer_device(operation, chars_ptr, default_device)?;

        let (validity, validity_binding) = validity_for_prepared_projection(
            operation,
            nullable,
            null_mask_ptr,
            null_count,
            active_elements,
            validity_id,
            default_device,
        )?;

        let mut bindings = vec![
            BufferBinding::new(
                offsets_id,
                bound_bytes(operation, offsets_ptr, offsets_byte_len)?,
                Placement::CudaDevice { device: offsets_device },
            ),
            BufferBinding::new(
                data_id,
                bound_bytes(operation, chars_ptr, chars_byte_len)?,
                Placement::CudaDevice { device: chars_device },
            ),
        ];
        if let Some(binding) = validity_binding {
            bindings.push(binding);
        }

        let column = SemanticColumn::new(
            column_id,
            map_data_type(data_type),
            len,
            offset,
            validity,
            schema_nullability,
            ColumnLayout::BinaryLike { offsets: offsets_id, data: data_id },
        );
        return Ok(PreparedColumnProjection {
            column: column.clone(),
            semantic_columns: vec![column],
            bindings,
            next_buffer_id,
            next_column_id,
        });
    }

    if map_data_type(data_type).id() == InteropTypeId::Dictionary32 {
        let mut next_buffer_id = next_buffer_id;
        let indices_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

        if num_children < 2 {
            return Err(projection_child_count_error(
                operation,
                "dictionary column view must expose indices and keys columns",
                "dictionary32",
                num_children,
                2,
                "requires_indices_and_keys",
                "expose both indices and keys children before projecting dictionary data",
            ));
        }

        let indices_view = dictionary::indices(view)
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let indices_type = indices_view
            .data_type()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let indices_elem_size = map_data_type(indices_type).id().fixed_width_size_bytes().ok_or_else(|| {
            projection_child_fixed_width_error(
                operation,
                "dictionary indices child must be fixed-width",
                "indices_layout",
                format!("{:?}", indices_type.id()).to_lowercase(),
                "materialize dictionary indices as a fixed-width Int32 child before projection",
            )
        })?;
        if indices_elem_size != std::mem::size_of::<i32>() {
            return Err(projection_child_width_error(
                operation,
                "dictionary indices child must use 32-bit indices",
                "indices_layout",
                indices_elem_size,
                std::mem::size_of::<i32>(),
                "materialize dictionary indices as 32-bit signed integers before projection",
            ));
        }
        let indices_byte_len = active_elements.checked_mul(indices_elem_size).ok_or_else(|| {
            projection_byte_range_overflow_error(
                operation,
                "dictionary indices byte range overflowed",
                "indices_byte_range",
                offset,
                len,
            )
        })?;
        let indices_ptr = indices_view
            .device_data_ptr()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let indices_device = pointer_device(operation, indices_ptr, default_device)?;

        let keys_view =
            dictionary::keys(view).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let keys_projection = prepared_column_projection_from_view(
            &keys_view,
            operation,
            default_device,
            next_column_id,
            next_buffer_id,
        )?;
        next_buffer_id = keys_projection.next_buffer_id;
        next_column_id = keys_projection.next_column_id;

        let (validity, validity_binding) = validity_for_prepared_projection(
            operation,
            nullable,
            null_mask_ptr,
            null_count,
            active_elements,
            validity_id,
            default_device,
        )?;

        let mut bindings = vec![BufferBinding::new(
            indices_id,
            bound_bytes(operation, indices_ptr, indices_byte_len)?,
            Placement::CudaDevice { device: indices_device },
        )];
        if let Some(binding) = validity_binding {
            bindings.push(binding);
        }
        bindings.extend(keys_projection.bindings);

        let column = SemanticColumn::new(
            column_id,
            map_data_type(data_type),
            len,
            offset,
            validity,
            schema_nullability,
            ColumnLayout::Dictionary { indices: indices_id, keys: keys_projection.column.id() },
        );
        let mut semantic_columns = vec![column.clone()];
        semantic_columns.extend(keys_projection.semantic_columns);
        return Ok(PreparedColumnProjection { column, semantic_columns, bindings, next_buffer_id, next_column_id });
    }

    if map_data_type(data_type).id() == InteropTypeId::List {
        let mut next_buffer_id = next_buffer_id;
        let offsets_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

        if num_children < 2 {
            return Err(projection_child_count_error(
                operation,
                "list column view must expose offsets and child columns",
                "list",
                num_children,
                2,
                "requires_offsets_and_child",
                "expose both offsets and child columns before projecting list data",
            ));
        }

        let offsets_view =
            view.child(0).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_type = offsets_view
            .data_type()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_elem_size = map_data_type(offsets_type).id().fixed_width_size_bytes().ok_or_else(|| {
            projection_child_fixed_width_error(
                operation,
                "list offsets child must be fixed-width",
                "offsets_layout",
                format!("{:?}", offsets_type.id()).to_lowercase(),
                "materialize list offsets as a fixed-width Int32 child before projection",
            )
        })?;
        if offsets_elem_size != std::mem::size_of::<i32>() {
            return Err(projection_child_width_error(
                operation,
                "list offsets child must use 32-bit offsets",
                "offsets_layout",
                offsets_elem_size,
                std::mem::size_of::<i32>(),
                "materialize list offsets as 32-bit signed integers before projection",
            ));
        }
        let offsets_required_elements = offset
            .checked_add(len)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| offsets_range_overflow_error(operation, "list offsets range overflowed", offset, len))?;
        let offsets_byte_len = offsets_required_elements.checked_mul(offsets_elem_size).ok_or_else(|| {
            projection_byte_range_overflow_error(
                operation,
                "list offsets byte range overflowed",
                "offsets_byte_range",
                offset,
                len,
            )
        })?;
        let offsets_ptr = offsets_view
            .device_data_ptr()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_device = pointer_device(operation, offsets_ptr, default_device)?;

        let child_view = view
            .list_sliced_child()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let child_projection = prepared_column_projection_from_view(
            &child_view,
            operation,
            default_device,
            next_column_id,
            next_buffer_id,
        )?;
        next_buffer_id = child_projection.next_buffer_id;
        next_column_id = child_projection.next_column_id;

        let (validity, validity_binding) = validity_for_prepared_projection(
            operation,
            nullable,
            null_mask_ptr,
            null_count,
            active_elements,
            validity_id,
            default_device,
        )?;

        let mut bindings = vec![BufferBinding::new(
            offsets_id,
            bound_bytes(operation, offsets_ptr, offsets_byte_len)?,
            Placement::CudaDevice { device: offsets_device },
        )];
        if let Some(binding) = validity_binding {
            bindings.push(binding);
        }
        bindings.extend(child_projection.bindings);

        let column = SemanticColumn::new(
            column_id,
            map_data_type(data_type),
            len,
            offset,
            validity,
            schema_nullability,
            ColumnLayout::List { offsets: offsets_id, child: child_projection.column.id() },
        );
        let mut semantic_columns = vec![column.clone()];
        semantic_columns.extend(child_projection.semantic_columns);
        return Ok(PreparedColumnProjection { column, semantic_columns, bindings, next_buffer_id, next_column_id });
    }

    if map_data_type(data_type).id() == InteropTypeId::Struct {
        let mut next_buffer_id = next_buffer_id;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let child_count = num_children;
        if child_count == 0 {
            return Err(projection_struct_requires_child_error(operation));
        }

        let (validity, validity_binding) = validity_for_prepared_projection(
            operation,
            nullable,
            null_mask_ptr,
            null_count,
            active_elements,
            validity_id,
            default_device,
        )?;

        let mut bindings = Vec::new();
        if let Some(binding) = validity_binding {
            bindings.push(binding);
        }
        let mut child_ids = Vec::with_capacity(child_count);
        let mut semantic_columns = Vec::new();

        for index in 0..child_count {
            let _ = checked_child_index(operation, index)?;
            let child_view =
                view.child(index).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let child_projection = prepared_column_projection_from_view(
                &child_view,
                operation,
                default_device,
                next_column_id,
                next_buffer_id,
            )?;
            next_buffer_id = child_projection.next_buffer_id;
            next_column_id = child_projection.next_column_id;
            child_ids.push(child_projection.column.id());
            bindings.extend(child_projection.bindings);
            semantic_columns.extend(child_projection.semantic_columns);
        }

        let column = SemanticColumn::new(
            column_id,
            map_data_type(data_type),
            len,
            offset,
            validity,
            schema_nullability,
            ColumnLayout::Struct { children: child_ids },
        );
        let mut all_columns = vec![column.clone()];
        all_columns.extend(semantic_columns);
        return Ok(PreparedColumnProjection {
            column,
            semantic_columns: all_columns,
            bindings,
            next_buffer_id,
            next_column_id,
        });
    }

    prepared_column_projection_from_materialized_view(
        MaterializedViewProjectionInput {
            data_type,
            num_children,
            len,
            offset,
            data_ptr,
            nullable,
            null_mask_ptr,
            null_count,
        },
        MaterializedViewProjectionState { operation, default_device, column_id, next_column_id, next_buffer_id },
    )
}
