use std::{any::Any, sync::Arc};

use cudf::interop::contract::{
    ColumnLayout, CompletionAuthority, InteropError, LeaseCompletionResult, PhysicalValidity, Placement,
    ReadStartAuthority,
};

use super::{
    errors::{map_prepared_lease_activation_interop_error, map_prepared_lease_completion_interop_error},
    prepared_support::{
        checked_active_byte_len, checked_column_offset_bytes, edge_list_dtype, offset_device_ptr,
        validate_edge_list_devices,
    },
};
use crate::{
    array::{DeviceArrayView, RawDeviceArrayView},
    error::{Error, ErrorKind, Result},
    graph::{DeferredInputCompletion, Graph, GraphBuilder, Transposition},
    handle::EnteredContext,
    similarity::VertexPairs,
    types::GraphProperties,
};

#[derive(Clone, Copy)]
pub(super) struct PreparedEdgeColumn<'a> {
    pub(super) column: &'a cudf::interop::contract::SemanticColumn,
    pub(super) bindings: &'a [cudf::interop::contract::BufferBinding<'a>],
}

pub(super) struct PreparedEdgeList<'a> {
    pub(super) src: PreparedEdgeColumn<'a>,
    pub(super) dst: PreparedEdgeColumn<'a>,
    pub(super) weights: Option<PreparedEdgeColumn<'a>>,
    pub(super) edge_ids: Option<PreparedEdgeColumn<'a>>,
}

/// Prepared explicit vertex-pair columns supplied to a pairwise cuGraph
/// algorithm.  These are zero-copy views; their device completion witnesses
/// are retained by the resulting [`VertexPairs`] object.
pub(super) struct PreparedVertexPairList<'a> {
    pub(super) first: PreparedEdgeColumn<'a>,
    pub(super) second: PreparedEdgeColumn<'a>,
}

pub(super) struct RetainedPreparedCompletion {
    owner: Arc<dyn Any + Send + Sync>,
    witness: cudf::interop::contract::CompletionWitness,
}

pub(super) fn prepared_profile_supported(
    profile: cudf::interop::contract::ExportProfile,
    operation: &'static str,
) -> Result<()> {
    if profile == cudf::interop::contract::ExportProfile::PreparedDeviceExport {
        return Ok(());
    }

    Err(Error::new(ErrorKind::UnsupportedType, operation)
        .with_message(format!("{operation} requires prepared device export inputs, got {:?}", profile)))
}

fn prepared_fixed_width_values_binding<'a>(
    column: PreparedEdgeColumn<'a>,
    name: &str,
    operation: &'static str,
) -> Result<&'a cudf::interop::contract::BufferBinding<'a>> {
    let values = match column.column.layout() {
        ColumnLayout::FixedWidth { values } => *values,
        layout => {
            return Err(Error::new(ErrorKind::UnsupportedType, operation)
                .with_message(format!(
                    "{} column uses layout {:?}; prepared cuGraph ingress requires fixed-width exports",
                    name, layout
                ))
                .with_context("column_name", name.to_string()));
        },
    };

    let binding = column.bindings.iter().find(|binding| binding.id() == values).ok_or_else(|| {
        Error::new(ErrorKind::InvalidState, operation)
            .with_message(format!(
                "{} column prepared export is missing the values buffer binding referenced by its semantic layout",
                name
            ))
            .with_context("column_name", name.to_string())
            .with_context("buffer_id", values.get().to_string())
    })?;

    Ok(binding)
}

fn prepared_device_binding<'a>(
    column: PreparedEdgeColumn<'a>,
    name: &str,
    operation: &'static str,
) -> Result<&'a cudf::interop::contract::BufferBinding<'a>> {
    let binding = prepared_fixed_width_values_binding(column, name, operation)?;

    let Placement::CudaDevice { .. } = binding.placement() else {
        return Err(Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!(
                "{} column prepared export is host-materialized; prepared cuGraph ingress requires device-resident inputs",
                name
            ))
            .with_context("column_name", name.to_string()));
    };

    Ok(binding)
}

fn prepared_column_device(column: PreparedEdgeColumn<'_>, name: &str, operation: &'static str) -> Result<i32> {
    let binding = prepared_device_binding(column, name, operation)?;
    let Placement::CudaDevice { device } = binding.placement() else {
        unreachable!("prepared_device_binding must return a device-resident binding")
    };
    Ok(*device)
}

fn prepared_column_to_raw<'a>(
    column: PreparedEdgeColumn<'a>,
    name: &str,
    operation: &'static str,
) -> Result<RawDeviceArrayView<'a>> {
    let cugraph_dtype = edge_list_dtype(column.column.data_type().id(), name, operation)?;

    match column.column.validity() {
        PhysicalValidity::NoValidityBuffer => {},
        PhysicalValidity::ValidityMask { null_count: Some(0), .. } => {},
        PhysicalValidity::ValidityMask { null_count: Some(null_count), .. } => {
            return Err(Error::new(ErrorKind::InvalidInput, operation)
                .with_message(format!(
                    "{} column contains {} null value(s); nulls are not supported for prepared cuGraph inputs",
                    name, null_count
                ))
                .with_context("column_name", name.to_string())
                .with_context("null_count", null_count.to_string()));
        },
        PhysicalValidity::ValidityMask { null_count: None, .. } => {
            return Err(Error::new(ErrorKind::InvalidState, operation)
                .with_message(format!(
                    "{} column prepared export does not make null count explicit; prepared cuGraph inputs must prove a null-free active range",
                    name
                ))
                .with_context("column_name", name.to_string()));
        },
    }

    let binding = prepared_device_binding(column, name, operation)?;
    let byte_offset = checked_column_offset_bytes(column.column.offset(), cugraph_dtype, name, operation)?;
    let active_len = checked_active_byte_len(column.column.len(), cugraph_dtype, name, operation)?;
    let required_end = byte_offset.checked_add(active_len).ok_or_else(|| {
        Error::new(ErrorKind::InvalidState, operation)
            .with_message(format!("{} column active byte range overflowed", name))
            .with_context("column_name", name.to_string())
    })?;

    if binding.bytes().byte_len() < required_end {
        return Err(Error::new(ErrorKind::InvalidState, operation)
            .with_message(format!(
                "{} column prepared export binding is smaller than the active fixed-width range",
                name
            ))
            .with_context("column_name", name.to_string())
            .with_context("binding_byte_len", binding.bytes().byte_len().to_string())
            .with_context("required_end", required_end.to_string()));
    }

    let ptr = offset_device_ptr(binding.bytes().base_ptr() as *const u8, byte_offset, active_len);

    // SAFETY: `ptr` points at the active fixed-width region of the prepared export.
    Ok(unsafe { RawDeviceArrayView::try_from_raw(ptr, column.column.len(), cugraph_dtype, operation)? })
}

fn validate_prepared_edge_list_policy(
    handle: &EnteredContext<'_>,
    edge_list: &PreparedEdgeList<'_>,
    operation: &'static str,
) -> Result<()> {
    let producer_device = prepared_column_device(edge_list.src, "src", operation)?;
    let mut columns = vec![("dst", prepared_column_device(edge_list.dst, "dst", operation)?)];
    if let Some(weights) = edge_list.weights {
        columns.push(("wgt", prepared_column_device(weights, "wgt", operation)?));
    }
    if let Some(edge_ids) = edge_list.edge_ids {
        columns.push(("edge_id", prepared_column_device(edge_ids, "edge_id", operation)?));
    }

    validate_edge_list_devices(handle, producer_device, columns, "prepared", operation)
}

fn validate_prepared_vertex_pair_policy(
    handle: &EnteredContext<'_>,
    pairs: &PreparedVertexPairList<'_>,
    operation: &'static str,
) -> Result<()> {
    let first_device = prepared_column_device(pairs.first, "first", operation)?;
    let second_device = prepared_column_device(pairs.second, "second", operation)?;

    validate_edge_list_devices(handle, first_device, [("second", second_device)], "prepared vertex-pair", operation)
}

fn validate_prepared_vertex_pair_shape(
    first: &RawDeviceArrayView<'_>,
    second: &RawDeviceArrayView<'_>,
    operation: &'static str,
) -> Result<crate::types::DataType> {
    let first_dtype = first.dtype()?;
    let second_dtype = second.dtype()?;

    if !first_dtype.is_supported_index_type() {
        return Err(Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!("first vertex-pair column must use Int32 or Int64 vertex ids, got {first_dtype}"))
            .with_context("column_role", "first")
            .with_context("column_dtype", first_dtype.to_string()));
    }

    if !second_dtype.is_supported_index_type() {
        return Err(Error::new(ErrorKind::UnsupportedType, operation)
            .with_message(format!("second vertex-pair column must use Int32 or Int64 vertex ids, got {second_dtype}"))
            .with_context("column_role", "second")
            .with_context("column_dtype", second_dtype.to_string()));
    }

    if first_dtype != second_dtype {
        return Err(Error::new(ErrorKind::InvalidInput, operation)
            .with_message("first and second vertex-pair columns must have the same dtype")
            .with_context("first_dtype", first_dtype.to_string())
            .with_context("second_dtype", second_dtype.to_string()));
    }

    if first.len() != second.len() {
        return Err(Error::new(ErrorKind::InvalidInput, operation)
            .with_message("first and second vertex-pair columns must have the same row count")
            .with_context("first_len", first.len().to_string())
            .with_context("second_len", second.len().to_string()));
    }

    Ok(first_dtype)
}

pub(super) fn prepared_lease_authorities<'a>(
    handle: &'a EnteredContext<'_>,
    operation: &'static str,
) -> Result<(ReadStartAuthority<'a>, CompletionAuthority<'a>)> {
    match handle.runtime() {
        Some(runtime) => {
            let stream = cudf::interop::cuda_stream_handle(runtime.stream())
                .map_err(|error| map_prepared_lease_activation_interop_error(error, operation))?;
            Ok((ReadStartAuthority::ConsumerStream { stream }, CompletionAuthority::SameDeviceStream { stream }))
        },
        None => Ok((ReadStartAuthority::SynchronousByReturn, CompletionAuthority::SynchronousByReturn)),
    }
}

pub(super) fn complete_prepared_lease(
    result: std::result::Result<LeaseCompletionResult, InteropError>,
    retained_owner: Option<Arc<dyn Any + Send + Sync>>,
    operation: &'static str,
) -> Result<Option<RetainedPreparedCompletion>> {
    match result.map_err(|error| map_prepared_lease_completion_interop_error(error, operation))? {
        LeaseCompletionResult::FallbackSynchronized => Ok(None),
        LeaseCompletionResult::Completed(witness) => match retained_owner {
            Some(owner) => Ok(Some(RetainedPreparedCompletion { owner, witness })),
            None => witness
                .synchronize()
                .map_err(|error| map_prepared_lease_completion_interop_error(error, operation))
                .map(|()| None),
        },
    }
}

pub(super) fn finish_prepared_graph_result<T: Transposition>(
    graph_result: Result<Graph<T>>,
    completions: impl IntoIterator<Item = Result<Option<RetainedPreparedCompletion>>>,
) -> Result<Graph<T>> {
    let mut completion_error = None;
    let mut retained_owners = Vec::new();
    let mut retained_witnesses = Vec::new();
    for completion in completions {
        match completion {
            Ok(Some(retained)) => {
                retained_owners.push(retained.owner);
                retained_witnesses.push(retained.witness);
            },
            Ok(None) => {},
            Err(error) => {
                if completion_error.is_none() {
                    completion_error = Some(error);
                }
            },
        }
    }
    let pending_input_completion = DeferredInputCompletion::new(retained_owners, retained_witnesses);

    match graph_result {
        Ok(graph) => {
            if let Some(error) = completion_error {
                if let Some(pending) = pending_input_completion {
                    pending.synchronize();
                }
                Err(error)
            } else if let Some(pending) = pending_input_completion {
                Ok(graph.with_pending_input_completion(pending))
            } else {
                Ok(graph)
            }
        },
        Err(error) => {
            if let Some(pending) = pending_input_completion {
                pending.synchronize();
            }
            Err(error)
        },
    }
}

/// Finish prepared pair ingress without introducing a host-side fence on the
/// successful runtime-bound path.  Completed read leases retain both their
/// producer owner and completion witness on `VertexPairs`; dropping the pairs
/// synchronizes that witness before cuGraph releases the native pair object.
pub(super) fn finish_prepared_vertex_pairs_result(
    pairs_result: Result<VertexPairs>,
    completions: impl IntoIterator<Item = Result<Option<RetainedPreparedCompletion>>>,
) -> Result<VertexPairs> {
    let mut completion_error = None;
    let mut retained_owners = Vec::new();
    let mut retained_witnesses = Vec::new();
    for completion in completions {
        match completion {
            Ok(Some(retained)) => {
                retained_owners.push(retained.owner);
                retained_witnesses.push(retained.witness);
            },
            Ok(None) => {},
            Err(error) => {
                if completion_error.is_none() {
                    completion_error = Some(error);
                }
            },
        }
    }
    let pending_input_completion = DeferredInputCompletion::new(retained_owners, retained_witnesses);

    match pairs_result {
        Ok(pairs) => {
            if let Some(error) = completion_error {
                if let Some(pending) = pending_input_completion {
                    pending.synchronize();
                }
                Err(error)
            } else if let Some(pending) = pending_input_completion {
                Ok(pairs.with_pending_input_completion(pending))
            } else {
                Ok(pairs)
            }
        },
        Err(error) => {
            if let Some(pending) = pending_input_completion {
                pending.synchronize();
            }
            Err(error)
        },
    }
}

pub(super) fn build_graph_from_prepared_edge_list_raw<T: Transposition>(
    handle: &EnteredContext<'_>,
    edge_list: &PreparedEdgeList<'_>,
    properties: GraphProperties,
    renumber: bool,
    operation: &'static str,
) -> Result<Graph<T>> {
    validate_prepared_edge_list_policy(handle, edge_list, operation)?;

    let src = prepared_column_to_raw(edge_list.src, "src", operation)?;
    let dst = prepared_column_to_raw(edge_list.dst, "dst", operation)?;
    let wgt = edge_list.weights.map(|column| prepared_column_to_raw(column, "wgt", operation)).transpose()?;
    let edge_ids = edge_list.edge_ids.map(|column| prepared_column_to_raw(column, "edge_id", operation)).transpose()?;

    let mut builder = GraphBuilder::from_raw_views(properties, &src, &dst)?.renumber_flag(renumber);
    if let Some(ref weights) = wgt {
        builder = builder.weights_raw(weights)?;
    }
    if let Some(ref edge_ids) = edge_ids {
        builder = builder.edge_ids_raw(edge_ids)?;
    }
    builder.build_inner(handle, T::store_transposed())
}

/// Create cuGraph vertex pairs directly from two prepared device columns.
///
/// This path deliberately dispatches through [`Graph::vertex_pairs`] after
/// checking the shared runtime dtype.  That keeps graph-domain validation and
/// native pair construction in the owning graph API while prepared ingress
/// owns the cuDF placement, active-range, and lease contracts.
pub(super) fn vertex_pairs_from_prepared_columns<T: Transposition>(
    handle: &EnteredContext<'_>,
    graph: &mut Graph<T>,
    pairs: &PreparedVertexPairList<'_>,
    operation: &'static str,
) -> Result<VertexPairs> {
    validate_prepared_vertex_pair_policy(handle, pairs, operation)?;

    let first = prepared_column_to_raw(pairs.first, "first", operation)?;
    let second = prepared_column_to_raw(pairs.second, "second", operation)?;
    let dtype = validate_prepared_vertex_pair_shape(&first, &second, operation)?;

    match dtype {
        crate::types::DataType::Int32 => {
            let first = DeviceArrayView::<i32>::from_raw_unchecked(first.raw_view());
            let second = DeviceArrayView::<i32>::from_raw_unchecked(second.raw_view());
            graph.vertex_pairs(handle, &first, &second)
        },
        crate::types::DataType::Int64 => {
            let first = DeviceArrayView::<i64>::from_raw_unchecked(first.raw_view());
            let second = DeviceArrayView::<i64>::from_raw_unchecked(second.raw_view());
            graph.vertex_pairs(handle, &first, &second)
        },
        _ => unreachable!("prepared vertex-pair shape validation only permits Int32 and Int64"),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use cudf::interop::contract::{ColumnLayout, PhysicalValidity, Placement, SemanticColumn, TypeId as InteropTypeId};

    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn prepared_fixed_width_values_binding_missing_values_binding_is_invalid_state() {
        let column = SemanticColumn::new(
            cudf::interop::contract::ColumnId::new(1),
            cudf::interop::contract::DataType::new(InteropTypeId::Int32),
            3,
            0,
            PhysicalValidity::NoValidityBuffer,
            cudf::interop::contract::SchemaNullability::NonNullable,
            ColumnLayout::FixedWidth { values: cudf::interop::contract::BufferId::new(42) },
        );
        let prepared = PreparedEdgeColumn { column: &column, bindings: &[] };

        let err = prepared_fixed_width_values_binding(prepared, "src", "test_op")
            .expect_err("missing prepared values binding should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "test_op");
    }

    #[test]
    fn prepared_column_without_explicit_null_count_is_invalid_state() {
        let bytes =
            cudf::interop::contract::BoundBytes::from_raw_parts(std::ptr::dangling_mut::<c_void>(), 12).expect("bytes");
        let binding = cudf::interop::contract::BufferBinding::new(
            cudf::interop::contract::BufferId::new(7),
            bytes,
            Placement::CudaDevice { device: 0 },
        );
        let column = SemanticColumn::new(
            cudf::interop::contract::ColumnId::new(1),
            cudf::interop::contract::DataType::new(InteropTypeId::Int32),
            3,
            0,
            PhysicalValidity::ValidityMask { mask: cudf::interop::contract::BufferId::new(99), null_count: None },
            cudf::interop::contract::SchemaNullability::Nullable,
            ColumnLayout::FixedWidth { values: cudf::interop::contract::BufferId::new(7) },
        );
        let bindings = [binding];
        let prepared = PreparedEdgeColumn { column: &column, bindings: &bindings };

        let err = prepared_column_to_raw(prepared, "src", "test_op")
            .expect_err("prepared exports must make null count explicit");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "test_op");
    }

    #[test]
    fn prepared_column_binding_shorter_than_active_range_is_invalid_state() {
        let bytes =
            cudf::interop::contract::BoundBytes::from_raw_parts(std::ptr::dangling_mut::<c_void>(), 8).expect("bytes");
        let binding = cudf::interop::contract::BufferBinding::new(
            cudf::interop::contract::BufferId::new(7),
            bytes,
            Placement::CudaDevice { device: 0 },
        );
        let column = SemanticColumn::new(
            cudf::interop::contract::ColumnId::new(1),
            cudf::interop::contract::DataType::new(InteropTypeId::Int32),
            3,
            0,
            PhysicalValidity::NoValidityBuffer,
            cudf::interop::contract::SchemaNullability::NonNullable,
            ColumnLayout::FixedWidth { values: cudf::interop::contract::BufferId::new(7) },
        );
        let bindings = [binding];
        let prepared = PreparedEdgeColumn { column: &column, bindings: &bindings };

        let err = prepared_column_to_raw(prepared, "src", "test_op")
            .expect_err("prepared bindings smaller than the active range should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "test_op");
    }
}
