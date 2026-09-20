use std::sync::Arc;

use super::*;
use crate::runtime::BoundaryObserverRef;

struct DeviceTableProjection<'a> {
    columns: Vec<SemanticColumn>,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'a>>,
}

struct HostTableProjection {
    columns: Vec<SemanticColumn>,
    semantic_columns: Vec<SemanticColumn>,
    bindings: Vec<BufferBinding<'static>>,
    retained_host_buffers: Vec<Arc<[u8]>>,
}

pub(super) struct DeviceTableExportContext<'a> {
    operation: &'static str,
    placement: ExecutionPlacement,
    observer: Option<BoundaryObserverRef>,
    default_device: i32,
    readiness: ProducerReadiness<'a>,
}

impl<'a> DeviceTableExportContext<'a> {
    fn from_domain(operation: &'static str, domain: &Arc<ExecutionDomain>) -> Result<Self, InteropError> {
        let (default_device, readiness) = prepared_device_readiness(operation, domain)?;
        Ok(Self::new(operation, domain, default_device, readiness))
    }

    pub(super) fn from_event(
        operation: &'static str,
        domain: &Arc<ExecutionDomain>,
        event: &'a CudaEvent,
    ) -> Result<Self, InteropError> {
        let (default_device, readiness) = prepared_device_event_readiness(operation, domain, event)?;
        Ok(Self::new(operation, domain, default_device, readiness))
    }

    fn new(
        operation: &'static str,
        domain: &Arc<ExecutionDomain>,
        default_device: i32,
        readiness: ProducerReadiness<'a>,
    ) -> Self {
        Self {
            operation,
            placement: domain.placement(),
            observer: domain.boundary_observer(),
            default_device,
            readiness,
        }
    }
}

fn table_dimensions(view: &TableView<'_>, operation: &'static str) -> Result<(usize, usize), InteropError> {
    let num_rows = view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let num_columns =
        view.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    Ok((num_rows, num_columns))
}

fn build_device_table_projection_from_view<'a>(
    view: &TableView<'a>,
    operation: &'static str,
    default_device: i32,
) -> Result<DeviceTableProjection<'a>, InteropError> {
    let (_, num_columns) = table_dimensions(view, operation)?;
    let mut columns = Vec::with_capacity(num_columns);
    let mut semantic_columns = Vec::new();
    let mut bindings = Vec::new();
    let mut next_buffer_id = 0u32;
    let mut next_column_id = 0u32;

    for index in 0..num_columns {
        let _ = checked_column_index(operation, index)?;
        let column_view =
            view.column(index).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let projection = prepared_column_projection_from_view(
            &column_view,
            operation,
            default_device,
            next_column_id,
            next_buffer_id,
        )?;
        next_buffer_id = projection.next_buffer_id;
        next_column_id = projection.next_column_id;
        bindings.extend(projection.bindings);
        columns.push(projection.column.clone());
        semantic_columns.extend(projection.semantic_columns);
    }

    Ok(DeviceTableProjection { columns, semantic_columns, bindings })
}

fn build_host_table_projection_from_view(
    view: &TableView<'_>,
    operation: &'static str,
) -> Result<HostTableProjection, InteropError> {
    let (_, num_columns) = table_dimensions(view, operation)?;
    let mut columns = Vec::with_capacity(num_columns);
    let mut semantic_columns = Vec::new();
    let mut bindings = Vec::new();
    let mut retained_host_buffers = Vec::new();
    let mut next_buffer_id = 0u32;
    let mut next_column_id = 0u32;

    for index in 0..num_columns {
        let _ = checked_column_index(operation, index)?;
        let column_view =
            view.column(index).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let projection =
            prepared_host_column_projection_from_view(&column_view, operation, next_column_id, next_buffer_id)?;
        next_buffer_id = projection.next_buffer_id;
        next_column_id = projection.next_column_id;
        bindings.extend(projection.bindings);
        retained_host_buffers.extend(projection.retained_host_buffers);
        columns.push(projection.column.clone());
        semantic_columns.extend(projection.semantic_columns);
    }

    Ok(HostTableProjection { columns, semantic_columns, bindings, retained_host_buffers })
}

fn build_device_table_projection_from_columns<'a, I>(
    columns: I,
    operation: &'static str,
    default_device: i32,
) -> Result<DeviceTableProjection<'a>, InteropError>
where
    I: IntoIterator<Item = &'a Column>,
{
    let columns = columns.into_iter();
    let mut projected_columns = Vec::with_capacity(columns.size_hint().0);
    let mut semantic_columns = Vec::new();
    let mut bindings = Vec::new();
    let mut next_buffer_id = 0u32;
    let mut next_column_id = 0u32;

    for (index, column) in columns.enumerate() {
        let _ = checked_column_index(operation, index)?;
        let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let projection =
            prepared_column_projection_from_view(&view, operation, default_device, next_column_id, next_buffer_id)?;
        next_buffer_id = projection.next_buffer_id;
        next_column_id = projection.next_column_id;
        bindings.extend(projection.bindings);
        projected_columns.push(projection.column.clone());
        semantic_columns.extend(projection.semantic_columns);
    }

    Ok(DeviceTableProjection { columns: projected_columns, semantic_columns, bindings })
}

fn build_host_table_projection_from_columns<'a, I>(
    columns: I,
    operation: &'static str,
) -> Result<HostTableProjection, InteropError>
where
    I: IntoIterator<Item = &'a Column>,
{
    let columns = columns.into_iter();
    let mut projected_columns = Vec::with_capacity(columns.size_hint().0);
    let mut semantic_columns = Vec::new();
    let mut bindings = Vec::new();
    let mut retained_host_buffers = Vec::new();
    let mut next_buffer_id = 0u32;
    let mut next_column_id = 0u32;

    for (index, column) in columns.enumerate() {
        let _ = checked_column_index(operation, index)?;
        let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let projection = prepared_host_column_projection_from_view(&view, operation, next_column_id, next_buffer_id)?;
        next_buffer_id = projection.next_buffer_id;
        next_column_id = projection.next_column_id;
        bindings.extend(projection.bindings);
        retained_host_buffers.extend(projection.retained_host_buffers);
        projected_columns.push(projection.column.clone());
        semantic_columns.extend(projection.semantic_columns);
    }

    Ok(HostTableProjection { columns: projected_columns, semantic_columns, bindings, retained_host_buffers })
}

pub(super) fn prepare_table_export_from_view<'a>(
    view: &TableView<'a>,
    domain: &Arc<ExecutionDomain>,
    names: Option<Vec<String>>,
    operation: &'static str,
) -> Result<PreparedTableExport<'a>, InteropError> {
    let (default_device, readiness) = prepared_device_readiness(operation, domain)?;
    prepare_table_export_from_view_with_readiness(view, names, operation, default_device, readiness)
}

pub(super) fn prepare_table_export_from_view_with_readiness<'a>(
    view: &TableView<'a>,
    names: Option<Vec<String>>,
    operation: &'static str,
    default_device: i32,
    readiness: ProducerReadiness<'a>,
) -> Result<PreparedTableExport<'a>, InteropError> {
    let (num_rows, _) = table_dimensions(view, operation)?;
    let observation = table_view_boundary_observation(BoundaryKind::InteropLease, view, operation)?;
    let observer = view.execution_domain().boundary_observer();
    let projection = build_device_table_projection_from_view(view, operation, default_device)?;
    let table = SemanticTable::new(num_rows, projection.columns.iter().map(|column| column.id()).collect(), names);
    PreparedTableExport::try_new(
        table,
        projection.semantic_columns,
        projection.bindings,
        PreparedAccess::Device {
            readiness,
            backend: runtime_lease::shared_cuda_lease_backend_with_observation(observation, observer),
        },
    )
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn prepare_host_table_export_from_view(
    view: &TableView<'_>,
    names: Option<Vec<String>>,
    operation: &'static str,
) -> Result<PreparedTableExport<'static>, InteropError> {
    let (num_rows, _) = table_dimensions(view, operation)?;
    let projection = build_host_table_projection_from_view(view, operation)?;
    let table = SemanticTable::new(num_rows, projection.columns.iter().map(|column| column.id()).collect(), names);
    PreparedTableExport::try_new(table, projection.semantic_columns, projection.bindings, PreparedAccess::Host)
        .map(|export| export.with_retained_owner(Arc::new(projection.retained_host_buffers)))
        .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn prepare_device_table_export_from_columns<'a, I>(
    columns: I,
    num_rows: usize,
    names: Option<Vec<String>>,
    domain: &Arc<ExecutionDomain>,
    operation: &'static str,
) -> Result<PreparedTableExport<'a>, InteropError>
where
    I: IntoIterator<Item = &'a Column>,
{
    let context = DeviceTableExportContext::from_domain(operation, domain)?;
    prepare_device_table_export_from_columns_with_readiness(columns, num_rows, names, context)
}

pub(super) fn prepare_device_table_export_from_columns_with_readiness<'a, I>(
    columns: I,
    num_rows: usize,
    names: Option<Vec<String>>,
    context: DeviceTableExportContext<'a>,
) -> Result<PreparedTableExport<'a>, InteropError>
where
    I: IntoIterator<Item = &'a Column>,
{
    let operation = context.operation;
    let projection = build_device_table_projection_from_columns(columns, operation, context.default_device)?;
    let observation =
        boundary_observation(BoundaryKind::InteropLease, context.placement, num_rows, projection.columns.len(), false);
    let readiness = context.readiness;
    let observer = context.observer;
    let table = SemanticTable::new(num_rows, projection.columns.iter().map(|column| column.id()).collect(), names);
    PreparedTableExport::try_new(
        table,
        projection.semantic_columns,
        projection.bindings,
        PreparedAccess::Device {
            readiness,
            backend: runtime_lease::shared_cuda_lease_backend_with_observation(observation, observer),
        },
    )
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn prepare_into_owned_table_root_with_readiness(
    table: Table,
    names: Option<Vec<String>>,
    operation: &'static str,
    readiness: ProducerReadiness<'_>,
    retained_event: Option<CudaEvent>,
) -> Result<PreparedTableExport<'static>, InteropError> {
    let retained_table = Arc::new(table);
    let view =
        retained_table.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let observation = table_view_boundary_observation(BoundaryKind::InteropLease, &view, operation)?;
    let observer = retained_table.execution_domain().boundary_observer();
    let num_rows = view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let default_device = retained_table.execution_domain().device_id().ordinal() as i32;
    let projection = build_device_table_projection_from_view(&view, operation, default_device)?;
    let table = SemanticTable::new(num_rows, projection.columns.iter().map(|column| column.id()).collect(), names);
    PreparedTableExport::try_new(
        table,
        projection.semantic_columns,
        to_static_bindings(operation, projection.bindings)?,
        PreparedAccess::Device {
            readiness: to_static_readiness(operation, readiness)?,
            backend: runtime_lease::shared_cuda_lease_backend_with_observation(observation, observer),
        },
    )
    .map(|export| {
        match retained_event {
            // `to_static_readiness` stores only the raw CUDA event pointer, so
            // the export must co-own the event for its whole lifetime.
            Some(event) => export.with_retained_owner(Arc::new((Arc::clone(&retained_table), event))),
            None => export.with_retained_owner(Arc::<Table>::clone(&retained_table)),
        }
    })
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn prepare_into_owned_table_root(
    table: Table,
    names: Option<Vec<String>>,
    operation: &'static str,
) -> Result<PreparedTableExport<'static>, InteropError> {
    let readiness = prepared_device_readiness(operation, table.execution_domain())?.1;
    prepare_into_owned_table_root_with_readiness(table, names, operation, readiness, None)
}

pub(super) fn prepare_host_table_export_from_columns<'a, I>(
    columns: I,
    num_rows: usize,
    names: Option<Vec<String>>,
    operation: &'static str,
) -> Result<PreparedTableExport<'static>, InteropError>
where
    I: IntoIterator<Item = &'a Column>,
{
    let projection = build_host_table_projection_from_columns(columns, operation)?;
    let table = SemanticTable::new(num_rows, projection.columns.iter().map(|column| column.id()).collect(), names);
    PreparedTableExport::try_new(table, projection.semantic_columns, projection.bindings, PreparedAccess::Host)
        .map(|export| export.with_retained_owner(Arc::new(projection.retained_host_buffers)))
        .map_err(|error| error.with_context("operation", operation))
}
