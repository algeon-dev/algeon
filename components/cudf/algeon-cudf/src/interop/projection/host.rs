use std::sync::Arc;

use super::*;

fn prepared_host_leaf_or_string_column_projection_from_view(
    view: &ColumnView<'_>,
    operation: &'static str,
    next_column_id: u32,
    next_buffer_id: u32,
) -> Result<PreparedHostColumnProjection, InteropError> {
    let data_type =
        view.data_type().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let interop_type = map_data_type(data_type);
    let num_children =
        view.num_children().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let len = view.len().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let nullable = view.nullable().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let null_count =
        view.null_count().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

    if interop_type.id() == InteropTypeId::String {
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

        let mut next_column_id = next_column_id;
        let mut next_buffer_id = next_buffer_id;
        let column_id = allocate_column_id(operation, &mut next_column_id)?;
        let offsets_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let data_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

        let strings = view
            .to_optional_strings()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let validity = view
            .to_host_null_mask_bytes()
            .map(|mask| mask.map(Arc::<[u8]>::from))
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

        let string_values = strings.iter().map(|value| value.as_deref().unwrap_or("").to_string()).collect::<Vec<_>>();
        let offsets_bytes = encode_host_string_offsets(operation, &string_values)?;
        let data_bytes: Arc<[u8]> =
            Arc::from(string_values.iter().flat_map(|value| value.as_bytes().iter().copied()).collect::<Vec<u8>>());

        let offsets_binding = retained_host_binding(operation, offsets_id, &offsets_bytes)?;
        let data_binding = retained_host_binding(operation, data_id, &data_bytes)?;
        let host_validity = host_validity_for_prepared_projection(operation, validity, null_count, validity_id)?;

        let mut bindings = vec![offsets_binding, data_binding];
        let mut retained_host_buffers = vec![offsets_bytes, data_bytes];
        let validity = host_validity.retain_into(&mut bindings, &mut retained_host_buffers);

        let column = SemanticColumn::new(
            column_id,
            interop_type,
            len,
            0,
            validity,
            schema_nullability(nullable),
            ColumnLayout::BinaryLike { offsets: offsets_id, data: data_id },
        );
        return Ok(PreparedHostColumnProjection {
            column: column.clone(),
            semantic_columns: vec![column],
            bindings,
            retained_host_buffers,
            next_buffer_id,
            next_column_id,
        });
    }

    if interop_type.id().fixed_width_size_bytes().is_none() {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedType,
            "unsupported type for host export leaf projection; struct, list, and dictionary are handled by parent dispatch, and this leaf path supports only fixed-width and string columns",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase()));
    }
    if num_children != 0 {
        return Err(InteropError::new(
            InteropErrorKind::UnsupportedLayout,
            "prepared host export currently supports fixed-width leaf column layouts on the non-list fallback path only",
        )
        .with_context("operation", operation)
        .with_context("type_id", format!("{:?}", data_type.id()).to_lowercase())
        .with_context("child_count", num_children.to_string()));
    }

    let mut next_column_id = next_column_id;
    let mut next_buffer_id = next_buffer_id;
    let column_id = allocate_column_id(operation, &mut next_column_id)?;
    let values_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
    let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

    let values_bytes: Arc<[u8]> = Arc::from(
        view.to_fixed_width_host_bytes()
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    );
    let validity = view
        .to_host_null_mask_bytes()
        .map(|mask| mask.map(Arc::<[u8]>::from))
        .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let values_binding = retained_host_binding(operation, values_id, &values_bytes)?;
    let host_validity = host_validity_for_prepared_projection(operation, validity, null_count, validity_id)?;

    let mut bindings = vec![values_binding];
    let mut retained_host_buffers = vec![values_bytes];
    let validity = host_validity.retain_into(&mut bindings, &mut retained_host_buffers);

    let column = SemanticColumn::new(
        column_id,
        interop_type,
        len,
        0,
        validity,
        schema_nullability(nullable),
        ColumnLayout::FixedWidth { values: values_id },
    );
    Ok(PreparedHostColumnProjection {
        column: column.clone(),
        semantic_columns: vec![column],
        bindings,
        retained_host_buffers,
        next_buffer_id,
        next_column_id,
    })
}

fn prepared_host_column_projection_with_current_supported_child_projection_from_view(
    view: &ColumnView<'_>,
    operation: &'static str,
    next_column_id: u32,
    next_buffer_id: u32,
) -> Result<PreparedHostColumnProjection, InteropError> {
    let data_type =
        view.data_type().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let interop_type = map_data_type(data_type);
    let num_children =
        view.num_children().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let len = view.len().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let nullable = view.nullable().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let schema_nullability = schema_nullability(nullable);
    let null_count =
        view.null_count().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

    if interop_type.id() != InteropTypeId::List {
        return prepared_host_leaf_or_string_column_projection_from_view(
            view,
            operation,
            next_column_id,
            next_buffer_id,
        );
    }

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

    let root_offset_raw =
        view.offset().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let root_offset = usize::try_from(root_offset_raw)
        .map_err(|_| negative_projection_offset_error(operation, root_offset_raw as i64))?;
    let root_end = root_offset.checked_add(len).ok_or_else(|| {
        projection_overflow_error(
            operation,
            "list host export active root range overflowed",
            "row_range",
            "end_overflowed_usize",
            root_offset,
            len,
            "reduce offset + row_count so the active host export range fits within usize",
        )
    })?;

    let mut next_column_id = next_column_id;
    let mut next_buffer_id = next_buffer_id;
    let column_id = allocate_column_id(operation, &mut next_column_id)?;
    let offsets_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
    let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

    let offsets_view =
        view.child(0).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let offsets_type =
        offsets_view.data_type().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
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
    let host_offsets = offsets_view
        .to_vec::<i32>()
        .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    if host_offsets.len() <= root_end {
        return Err(projection_offsets_too_small_error(
            operation,
            "list offsets child is too small for the active logical range",
            root_offset,
            len,
            host_offsets.len(),
        ));
    }

    let base_child_offset = host_offsets[root_offset];
    if base_child_offset < 0 {
        return Err(invalid_view_contract_error(
            operation,
            "list offsets child must not expose negative offsets",
            "offsets_range",
            "negative_offset_value",
            "ensure projected list offsets are non-negative before host export",
        )
        .with_context("offset_index", root_offset.to_string())
        .with_context("offset_value", base_child_offset.to_string()));
    }
    let mut normalized_offsets = Vec::with_capacity(len + 1);
    let mut previous = base_child_offset;
    for (position, raw_offset) in host_offsets[root_offset..=root_end].iter().copied().enumerate() {
        if raw_offset < previous {
            return Err(invalid_view_contract_error(
                operation,
                "list offsets child must be monotonically non-decreasing",
                "offsets_range",
                "non_monotonic_offsets",
                "ensure projected list offsets are monotonically non-decreasing before host export",
            )
            .with_context("offset_index", (root_offset + position).to_string())
            .with_context("previous_offset", previous.to_string())
            .with_context("offset_value", raw_offset.to_string()));
        }
        normalized_offsets.push(raw_offset - base_child_offset);
        previous = raw_offset;
    }
    let offsets_bytes = encode_host_i32_values(&normalized_offsets);
    let offsets_binding = retained_host_binding(operation, offsets_id, &offsets_bytes)?;

    let child_view =
        view.list_sliced_child().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let child_projection =
        prepared_host_column_projection_from_view(&child_view, operation, next_column_id, next_buffer_id)?;
    next_buffer_id = child_projection.next_buffer_id;
    next_column_id = child_projection.next_column_id;

    let validity = view
        .to_host_null_mask_bytes()
        .map(|mask| mask.map(Arc::<[u8]>::from))
        .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let host_validity = host_validity_for_prepared_projection(operation, validity, null_count, validity_id)?;

    let mut bindings = vec![offsets_binding];
    let mut retained_host_buffers = vec![offsets_bytes];
    let validity = host_validity.retain_into(&mut bindings, &mut retained_host_buffers);
    bindings.extend(child_projection.bindings);
    retained_host_buffers.extend(child_projection.retained_host_buffers);

    let column = SemanticColumn::new(
        column_id,
        interop_type,
        len,
        0,
        validity,
        schema_nullability,
        ColumnLayout::List { offsets: offsets_id, child: child_projection.column.id() },
    );
    let mut semantic_columns = vec![column.clone()];
    semantic_columns.extend(child_projection.semantic_columns);
    Ok(PreparedHostColumnProjection {
        column,
        semantic_columns,
        bindings,
        retained_host_buffers,
        next_buffer_id,
        next_column_id,
    })
}

pub(super) fn prepared_host_column_projection_from_view(
    view: &ColumnView<'_>,
    operation: &'static str,
    next_column_id: u32,
    next_buffer_id: u32,
) -> Result<PreparedHostColumnProjection, InteropError> {
    let data_type =
        view.data_type().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let interop_type = map_data_type(data_type);
    let num_children =
        view.num_children().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let len = view.len().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let nullable = view.nullable().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let schema_nullability = schema_nullability(nullable);
    let null_count =
        view.null_count().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;

    if interop_type.id() == InteropTypeId::Struct {
        let root_offset_raw =
            view.offset().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let root_offset = usize::try_from(root_offset_raw)
            .map_err(|_| negative_projection_offset_error(operation, root_offset_raw as i64))?;
        if root_offset != 0 {
            let normalized = copy_view_to_owned(view)
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let normalized_view =
                normalized.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            return prepared_host_column_projection_from_view(
                &normalized_view,
                operation,
                next_column_id,
                next_buffer_id,
            );
        }

        let child_count = num_children;
        if child_count == 0 {
            return Err(projection_struct_requires_child_error(operation));
        }

        let mut next_column_id = next_column_id;
        let mut next_buffer_id = next_buffer_id;
        let column_id = allocate_column_id(operation, &mut next_column_id)?;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let validity = view
            .to_host_null_mask_bytes()
            .map(|mask| mask.map(Arc::<[u8]>::from))
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let host_validity = host_validity_for_prepared_projection(operation, validity, null_count, validity_id)?;

        let mut bindings = Vec::new();
        let mut retained_host_buffers = Vec::new();
        let validity = host_validity.retain_into(&mut bindings, &mut retained_host_buffers);

        let mut child_ids = Vec::with_capacity(child_count);
        let mut semantic_columns = Vec::new();
        for index in 0..child_count {
            let _ = checked_child_index(operation, index)?;
            let child_view =
                view.child(index).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
            let child_projection =
                prepared_host_column_projection_from_view(&child_view, operation, next_column_id, next_buffer_id)?;
            next_buffer_id = child_projection.next_buffer_id;
            next_column_id = child_projection.next_column_id;
            child_ids.push(child_projection.column.id());
            bindings.extend(child_projection.bindings);
            retained_host_buffers.extend(child_projection.retained_host_buffers);
            semantic_columns.extend(child_projection.semantic_columns);
        }

        let column = SemanticColumn::new(
            column_id,
            interop_type,
            len,
            0,
            validity,
            schema_nullability,
            ColumnLayout::Struct { children: child_ids },
        );
        let mut all_columns = vec![column.clone()];
        all_columns.extend(semantic_columns);
        return Ok(PreparedHostColumnProjection {
            column,
            semantic_columns: all_columns,
            bindings,
            retained_host_buffers,
            next_buffer_id,
            next_column_id,
        });
    }

    if interop_type.id() == InteropTypeId::Dictionary32 {
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

        let mut next_column_id = next_column_id;
        let mut next_buffer_id = next_buffer_id;
        let column_id = allocate_column_id(operation, &mut next_column_id)?;
        let indices_id = allocate_buffer_id(operation, &mut next_buffer_id)?;
        let validity_id = allocate_buffer_id(operation, &mut next_buffer_id)?;

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

        let indices_bytes: Arc<[u8]> = Arc::from(
            indices_view
                .to_fixed_width_host_bytes()
                .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
        );
        let indices_binding = retained_host_binding(operation, indices_id, &indices_bytes)?;

        let keys_view =
            dictionary::keys(view).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let keys_projection =
            prepared_host_column_projection_from_view(&keys_view, operation, next_column_id, next_buffer_id)?;
        next_buffer_id = keys_projection.next_buffer_id;
        next_column_id = keys_projection.next_column_id;

        let validity = view
            .to_host_null_mask_bytes()
            .map(|mask| mask.map(Arc::<[u8]>::from))
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let host_validity = host_validity_for_prepared_projection(operation, validity, null_count, validity_id)?;

        let mut bindings = vec![indices_binding];
        let mut retained_host_buffers = vec![indices_bytes];
        let validity = host_validity.retain_into(&mut bindings, &mut retained_host_buffers);
        bindings.extend(keys_projection.bindings);
        retained_host_buffers.extend(keys_projection.retained_host_buffers);

        let column = SemanticColumn::new(
            column_id,
            interop_type,
            len,
            0,
            validity,
            schema_nullability,
            ColumnLayout::Dictionary { indices: indices_id, keys: keys_projection.column.id() },
        );
        let mut semantic_columns = vec![column.clone()];
        semantic_columns.extend(keys_projection.semantic_columns);
        return Ok(PreparedHostColumnProjection {
            column,
            semantic_columns,
            bindings,
            retained_host_buffers,
            next_buffer_id,
            next_column_id,
        });
    }

    prepared_host_column_projection_with_current_supported_child_projection_from_view(
        view,
        operation,
        next_column_id,
        next_buffer_id,
    )
}
