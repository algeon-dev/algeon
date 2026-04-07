use super::*;

fn prepared_column_projection_from_materialized_view<'a>(
    operation: &'static str,
    data_type: DataType,
    num_children: i32,
    len: usize,
    offset: usize,
    data_ptr: *const std::ffi::c_void,
    nullable: bool,
    null_mask_ptr: Option<*const std::ffi::c_void>,
    null_count: usize,
    default_device: i32,
    next_column_id: u32,
    next_buffer_id: u32,
) -> Result<PreparedColumnProjection<'a>, InteropError> {
    let interop_type = map_data_type(data_type);
    let active_elements = offset.checked_add(len).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "active element range overflowed")
            .with_context("operation", operation)
            .with_context("offset", offset.to_string())
            .with_context("len", len.to_string())
    })?;
    let mut next_column_id = next_column_id;
    let mut next_buffer_id = next_buffer_id;
    let column_id = allocate_column_id(operation, &mut next_column_id)?;

    if num_children != 0 {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedLayout,
            "prepared device export currently supports fixed-width leaf, string, list, struct, and dictionary column layouts only",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase())
        .with_context("num_children", num_children.to_string()));
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
        InteropError::new(InteropErrorKind::InvalidView, "active value byte range overflowed")
            .with_context("operation", operation)
            .with_context("offset", offset.to_string())
            .with_context("len", len.to_string())
            .with_context("elem_size", elem_size.to_string())
    })?;

    let values_device = pointer_device(operation, data_ptr, default_device)?;
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
        values_id,
        bound_bytes(operation, data_ptr, values_byte_len)?,
        Placement::CudaDevice { device: values_device },
    )];
    if let Some(binding) = validity_binding {
        bindings.push(binding);
    }

    let column = SemanticColumn::new(
        column_id,
        interop_type,
        len,
        offset,
        validity,
        schema_nullability(nullable),
        ColumnLayout::FixedWidth { values: values_id },
    );
    Ok(PreparedColumnProjection {
        column: column.clone(),
        semantic_columns: vec![column],
        bindings,
        next_buffer_id,
        next_column_id,
    })
}

pub(super) fn prepared_column_projection_from_view<'a>(
    view: &ColumnView<'_>,
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
    let schema_nullability = schema_nullability(nullable);
    let null_count =
        view.null_count().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_mask_ptr =
        view.null_mask_ptr().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let active_elements = offset.checked_add(len).ok_or_else(|| {
        InteropError::new(InteropErrorKind::InvalidView, "active element range overflowed")
            .with_context("operation", operation)
            .with_context("offset", offset.to_string())
            .with_context("len", len.to_string())
    })?;
    let mut next_column_id = next_column_id;
    let column_id = allocate_column_id(operation, &mut next_column_id)?;

    if map_data_type(data_type).id() == InteropTypeId::String {
        let mut next_buffer_id = next_buffer_id;
        let offsets_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let data_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

        if num_children < 1 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "string column view must expose an offsets child",
            )
            .with_context("operation", operation)
            .with_context("type_id", "string")
            .with_context("num_children", num_children.to_string()));
        }

        let offsets_view =
            view.child(0).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_type = offsets_view
            .data_type()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_elem_size = map_data_type(offsets_type).id().fixed_width_size_bytes().ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "string offsets child must be fixed-width")
                .with_context("operation", operation)
                .with_context("offsets_type_id", format!("{:?}", offsets_type.id()).to_lowercase())
        })?;
        if offsets_elem_size != std::mem::size_of::<i32>() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "string offsets child must use 32-bit offsets",
            )
            .with_context("operation", operation)
            .with_context("offsets_elem_size", offsets_elem_size.to_string()));
        }
        let offsets_required_elements = offset.checked_add(len).and_then(|v| v.checked_add(1)).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "string offsets range overflowed")
                .with_context("operation", operation)
                .with_context("offset", offset.to_string())
                .with_context("len", len.to_string())
        })?;
        let offsets_byte_len = offsets_required_elements.checked_mul(offsets_elem_size).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "string offsets byte range overflowed")
                .with_context("operation", operation)
                .with_context("offset", offset.to_string())
                .with_context("len", len.to_string())
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
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "dictionary column view must expose indices and keys columns",
            )
            .with_context("operation", operation)
            .with_context("type_id", "dictionary32")
            .with_context("num_children", num_children.to_string()));
        }

        let indices_view = dictionary::indices(view)
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let indices_type = indices_view
            .data_type()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let indices_elem_size = map_data_type(indices_type).id().fixed_width_size_bytes().ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "dictionary indices child must be fixed-width")
                .with_context("operation", operation)
                .with_context("indices_type_id", format!("{:?}", indices_type.id()).to_lowercase())
        })?;
        if indices_elem_size != std::mem::size_of::<i32>() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "dictionary indices child must use 32-bit indices",
            )
            .with_context("operation", operation)
            .with_context("indices_elem_size", indices_elem_size.to_string()));
        }
        let indices_byte_len = active_elements.checked_mul(indices_elem_size).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "dictionary indices byte range overflowed")
                .with_context("operation", operation)
                .with_context("offset", offset.to_string())
                .with_context("len", len.to_string())
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
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "list column view must expose offsets and child columns",
            )
            .with_context("operation", operation)
            .with_context("type_id", "list")
            .with_context("num_children", num_children.to_string()));
        }

        let offsets_view =
            view.child(0).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_type = offsets_view
            .data_type()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let offsets_elem_size = map_data_type(offsets_type).id().fixed_width_size_bytes().ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "list offsets child must be fixed-width")
                .with_context("operation", operation)
                .with_context("offsets_type_id", format!("{:?}", offsets_type.id()).to_lowercase())
        })?;
        if offsets_elem_size != std::mem::size_of::<i32>() {
            return Err(InteropError::new(InteropErrorKind::InvalidView, "list offsets child must use 32-bit offsets")
                .with_context("operation", operation)
                .with_context("offsets_elem_size", offsets_elem_size.to_string()));
        }
        let offsets_required_elements = offset.checked_add(len).and_then(|v| v.checked_add(1)).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "list offsets range overflowed")
                .with_context("operation", operation)
                .with_context("offset", offset.to_string())
                .with_context("len", len.to_string())
        })?;
        let offsets_byte_len = offsets_required_elements.checked_mul(offsets_elem_size).ok_or_else(|| {
            InteropError::new(InteropErrorKind::InvalidView, "list offsets byte range overflowed")
                .with_context("operation", operation)
                .with_context("offset", offset.to_string())
                .with_context("len", len.to_string())
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
        let child_count = usize::try_from(num_children).map_err(|_| {
            InteropError::new(InteropErrorKind::InvalidView, "struct child count must be >= 0")
                .with_context("operation", operation)
                .with_context("type_id", "struct")
                .with_context("num_children", num_children.to_string())
        })?;
        if child_count == 0 {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                "struct column view must expose at least one child column",
            )
            .with_context("operation", operation)
            .with_context("type_id", "struct"));
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
            let child_index = checked_child_index(operation, index)?;
            let child_view = view
                .child(child_index)
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
        operation,
        data_type,
        num_children,
        len,
        offset,
        data_ptr,
        nullable,
        null_mask_ptr,
        null_count,
        default_device,
        column_id.get(),
        next_buffer_id,
    )
}
