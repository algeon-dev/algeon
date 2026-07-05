use super::*;

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
    observer: Option<crate::runtime::BoundaryObserverRef>,
    default_device: i32,
    readiness: ProducerReadiness<'a>,
}

impl<'a> DeviceTableExportContext<'a> {
    fn from_domain(operation: &'static str, domain: &Arc<ExecutionDomain>) -> Result<Self, InteropError> {
        let (default_device, readiness) = prepared_device_readiness(operation, domain)?;
        Self::with_readiness(operation, domain, default_device, readiness)
    }

    pub(super) fn with_readiness(
        operation: &'static str,
        domain: &Arc<ExecutionDomain>,
        default_device: i32,
        readiness: ProducerReadiness<'a>,
    ) -> Result<Self, InteropError> {
        let domain_device = domain.device_snapshot();
        if default_device != domain_device {
            return Err(InteropError::new(
                InteropErrorKind::ReadinessMismatch,
                "prepared table export device must match producer execution domain",
            )
            .with_context("operation", operation)
            .with_context("left_device", default_device.to_string())
            .with_context("right_device", domain_device.to_string()));
        }

        match readiness {
            ProducerReadiness::DeviceStreamOrdered { stream } if stream.device() != default_device => {
                return Err(device_readiness_mismatch_error(operation, default_device, stream.device()));
            },
            ProducerReadiness::DeviceEventOrdered { event } if event.device() != default_device => {
                return Err(device_readiness_mismatch_error(operation, default_device, event.device()));
            },
            ProducerReadiness::HostVisible
            | ProducerReadiness::DeviceStreamOrdered { .. }
            | ProducerReadiness::DeviceEventOrdered { .. } => {},
        }

        Ok(Self {
            operation,
            placement: domain.placement(),
            observer: domain.boundary_observer(),
            default_device,
            readiness,
        })
    }
}

fn device_readiness_mismatch_error(
    operation: &'static str,
    default_device: i32,
    readiness_device: i32,
) -> InteropError {
    InteropError::new(
        InteropErrorKind::ReadinessMismatch,
        "prepared table export readiness must match producer CUDA device",
    )
    .with_context("operation", operation)
    .with_context("left_device", default_device.to_string())
    .with_context("right_device", readiness_device.to_string())
}

fn table_dimensions(view: &TableView<'_>, operation: &'static str) -> Result<(usize, usize), InteropError> {
    let num_rows = view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let num_columns =
        view.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    Ok((num_rows, num_columns))
}

fn build_device_table_projection_from_view<'a>(
    view: &TableView<'_>,
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
    view: &TableView<'_>,
    domain: &Arc<ExecutionDomain>,
    names: Option<Vec<String>>,
    operation: &'static str,
) -> Result<PreparedTableExport<'a>, InteropError> {
    let (default_device, readiness) = prepared_device_readiness(operation, domain)?;
    prepare_table_export_from_view_with_readiness(view, names, operation, default_device, readiness)
}

pub(super) fn prepare_table_export_from_view_with_readiness<'a>(
    view: &TableView<'_>,
    names: Option<Vec<String>>,
    operation: &'static str,
    default_device: i32,
    readiness: ProducerReadiness<'a>,
) -> Result<PreparedTableExport<'a>, InteropError> {
    let (num_rows, _) = table_dimensions(view, operation)?;
    let observation = table_view_boundary_observation(BoundaryKind::InteropLease, view, operation)?;
    let observer = view.execution_domain().boundary_observer();
    let projection = build_device_table_projection_from_view(view, operation, default_device)?;
    let table = SemanticTable::new(
        TableId::new(0),
        num_rows,
        projection.columns.iter().map(|column| column.id()).collect(),
        names,
    );
    PreparedTableExport::try_new_with_semantic_columns(
        table,
        projection.columns,
        projection.semantic_columns,
        projection.bindings,
        readiness,
        export_diagnostics(),
    )
    .map(|export| {
        export.with_lease_backend(runtime_lease::shared_cuda_lease_backend_with_observation(observation, observer))
    })
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn prepare_host_table_export_from_view(
    view: &TableView<'_>,
    names: Option<Vec<String>>,
    operation: &'static str,
) -> Result<PreparedTableExport<'static>, InteropError> {
    let (num_rows, _) = table_dimensions(view, operation)?;
    let projection = build_host_table_projection_from_view(view, operation)?;
    let table = SemanticTable::new(
        TableId::new(0),
        num_rows,
        projection.columns.iter().map(|column| column.id()).collect(),
        names,
    );
    PreparedTableExport::try_new_with_semantic_columns(
        table,
        projection.columns,
        projection.semantic_columns,
        projection.bindings,
        ProducerReadiness::HostVisible,
        host_export_diagnostics(),
    )
    .map(|export| export.with_retained_host_buffers(projection.retained_host_buffers))
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
    let table = SemanticTable::new(
        TableId::new(0),
        num_rows,
        projection.columns.iter().map(|column| column.id()).collect(),
        names,
    );
    PreparedTableExport::try_new_with_semantic_columns(
        table,
        projection.columns,
        projection.semantic_columns,
        projection.bindings,
        readiness,
        export_diagnostics(),
    )
    .map(|export| {
        export.with_lease_backend(runtime_lease::shared_cuda_lease_backend_with_observation(observation, observer))
    })
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn prepare_into_owned_table_root_with_readiness(
    table: Table,
    names: Option<Vec<String>>,
    operation: &'static str,
    readiness: ProducerReadiness<'_>,
) -> Result<PreparedTableExport<'static>, InteropError> {
    let retained_table = Arc::new(table);
    let view =
        retained_table.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let observation = table_view_boundary_observation(BoundaryKind::InteropLease, &view, operation)?;
    let observer = retained_table.execution_domain().boundary_observer();
    let num_rows = view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let default_device = retained_table.execution_domain().device_id().ordinal() as i32;
    let projection = build_device_table_projection_from_view(&view, operation, default_device)?;
    let table = SemanticTable::new(
        TableId::new(0),
        num_rows,
        projection.columns.iter().map(|column| column.id()).collect(),
        names,
    );
    PreparedTableExport::try_new_with_semantic_columns(
        table,
        projection.columns,
        projection.semantic_columns,
        to_static_bindings(operation, projection.bindings)?,
        to_static_readiness(operation, readiness)?,
        export_diagnostics(),
    )
    .map(|export| {
        export
            .with_lease_backend(runtime_lease::shared_cuda_lease_backend_with_observation(observation, observer))
            .with_retained_owner_arc(Arc::clone(&retained_table))
    })
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn prepare_into_owned_table_root(
    table: Table,
    names: Option<Vec<String>>,
    operation: &'static str,
) -> Result<PreparedTableExport<'static>, InteropError> {
    let readiness = prepared_device_readiness(operation, table.execution_domain())?.1;
    prepare_into_owned_table_root_with_readiness(table, names, operation, readiness)
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
    let table = SemanticTable::new(
        TableId::new(0),
        num_rows,
        projection.columns.iter().map(|column| column.id()).collect(),
        names,
    );
    PreparedTableExport::try_new_with_semantic_columns(
        table,
        projection.columns,
        projection.semantic_columns,
        projection.bindings,
        ProducerReadiness::HostVisible,
        host_export_diagnostics(),
    )
    .map(|export| export.with_retained_host_buffers(projection.retained_host_buffers))
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn transfer_from_owned_table(
    table: Table,
    operation: &'static str,
) -> Result<OwnedTableTransfer, InteropError> {
    transfer_from_owned_table_with_names(table, None, operation)
}

pub(super) fn transfer_from_owned_table_with_names(
    table: Table,
    names: Option<Vec<String>>,
    operation: &'static str,
) -> Result<OwnedTableTransfer, InteropError> {
    let observation = table_boundary_observation(BoundaryKind::OwnedTransfer, &table, operation)?;
    let domain = Arc::clone(table.execution_domain());
    let (default_device, readiness) = prepared_device_readiness(operation, table.execution_domain())?;
    let (num_rows, columns, semantic_columns, bindings) = {
        let view = table.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let (num_rows, _) = table_dimensions(&view, operation)?;
        let projection = build_device_table_projection_from_view(&view, operation, default_device)?;
        (num_rows, projection.columns, projection.semantic_columns, projection.bindings)
    };

    let table_semantic =
        SemanticTable::new(TableId::new(0), num_rows, columns.iter().map(|column| column.id()).collect(), names);
    let transfer = OwnedTableTransfer::try_new_with_semantic_columns(OwnedTableTransferParts {
        table: table_semantic,
        columns,
        semantic_columns,
        // SAFETY: `to_static_bindings` depends on this retained owner to keep
        // the source table alive for the transfer lifetime.
        bindings: to_static_bindings(operation, bindings)?,
        readiness: to_static_readiness(operation, readiness)?,
        diagnostics: owned_transfer_diagnostics(),
        release_contract: owned_transfer_release_contract(default_device),
        owner: OwnedTableTransferState::new(table),
    })
    .map_err(|error| error.with_context("operation", operation))?;
    record_observation(&domain, observation);
    Ok(transfer)
}

pub(super) fn transfer_from_owned_table_with_runtime_release(
    table: Table,
    operation: &'static str,
) -> Result<(OwnedTableTransfer, OwnedTableTransferReleaser), InteropError> {
    transfer_from_owned_table_with_runtime_release_synchronization(
        table,
        ReleaseSynchronization::Asynchronous,
        operation,
    )
}

pub(super) fn transfer_from_owned_table_with_runtime_release_synchronization(
    table: Table,
    synchronization: ReleaseSynchronization,
    operation: &'static str,
) -> Result<(OwnedTableTransfer, OwnedTableTransferReleaser), InteropError> {
    transfer_from_owned_table_with_names_and_runtime_release_synchronization(table, None, synchronization, operation)
}

pub(super) fn transfer_from_owned_table_with_names_and_runtime_release(
    table: Table,
    names: Option<Vec<String>>,
    operation: &'static str,
) -> Result<(OwnedTableTransfer, OwnedTableTransferReleaser), InteropError> {
    transfer_from_owned_table_with_names_and_runtime_release_synchronization(
        table,
        names,
        ReleaseSynchronization::Asynchronous,
        operation,
    )
}

pub(super) fn transfer_from_owned_table_with_names_and_runtime_release_synchronization(
    table: Table,
    names: Option<Vec<String>>,
    synchronization: ReleaseSynchronization,
    operation: &'static str,
) -> Result<(OwnedTableTransfer, OwnedTableTransferReleaser), InteropError> {
    let observation = table_boundary_observation(BoundaryKind::OwnedTransfer, &table, operation)?;
    let domain = Arc::clone(table.execution_domain());
    let (default_device, readiness) = prepared_device_readiness(operation, table.execution_domain())?;
    let (num_rows, columns, semantic_columns, bindings) = {
        let view = table.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let (num_rows, _) = table_dimensions(&view, operation)?;
        let projection = build_device_table_projection_from_view(&view, operation, default_device)?;
        (num_rows, projection.columns, projection.semantic_columns, projection.bindings)
    };

    let table_semantic =
        SemanticTable::new(TableId::new(0), num_rows, columns.iter().map(|column| column.id()).collect(), names);
    let (owner, releaser) = runtime_owned_table_owner_and_releaser(table, default_device, synchronization);
    let transfer = OwnedTableTransfer::try_new_with_semantic_columns(OwnedTableTransferParts {
        table: table_semantic,
        columns,
        semantic_columns,
        bindings: to_static_bindings(operation, bindings)?,
        readiness: to_static_readiness(operation, readiness)?,
        diagnostics: owned_transfer_diagnostics(),
        release_contract: owned_transfer_runtime_release_contract_with_synchronization(
            default_device,
            synchronization,
        )?,
        owner,
    })
    .map_err(|error| error.with_context("operation", operation))?;

    record_observation(&domain, observation);
    Ok((transfer, releaser))
}

pub(super) fn prepare_partitioned_device_table_export_from_columns<'a>(
    producer: &'a PartitionedDataFrameExport,
    operation: &'static str,
) -> Result<PartitionedPreparedTableExport<'a>, InteropError> {
    let mut partitions = Vec::new();

    for group in producer.grouped_borrowed_columns() {
        let export = prepare_device_table_export_from_columns(
            group.columns,
            producer.num_rows(),
            None,
            &group.domain,
            operation,
        )?;
        partitions.push(PartitionedPreparedTablePartition::new(group.top_level_column_indices, export));
    }

    PartitionedPreparedTableExport::try_new(Some(producer.column_names_owned()), producer.num_rows(), partitions)
        .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn prepare_partitioned_host_table_export_from_columns(
    producer: &PartitionedDataFrameExport,
    operation: &'static str,
) -> Result<PreparedTableExport<'static>, InteropError> {
    prepare_host_table_export_from_columns(
        producer.columns(),
        producer.num_rows(),
        Some(producer.column_names_owned()),
        operation,
    )
}

pub(super) fn transfer_partitioned_owned_table_from_columns(
    producer: PartitionedDataFrameExport,
    operation: &'static str,
) -> Result<PartitionedOwnedTableTransfer, InteropError> {
    let schema_names = producer.column_names_owned();
    let row_count = producer.num_rows();
    let mut partitions = Vec::new();

    for group in producer.grouped_owned_columns() {
        let table = Table::from_columns(group.columns)
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let transfer = transfer_from_owned_table(table, operation)?;
        partitions.push(PartitionedOwnedTableTransferPartition::new(group.top_level_column_indices, transfer));
    }

    PartitionedOwnedTableTransfer::try_new(Some(schema_names), row_count, partitions)
        .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn transfer_partitioned_owned_table_from_columns_with_runtime_release(
    producer: PartitionedDataFrameExport,
    operation: &'static str,
) -> Result<(PartitionedOwnedTableTransfer, PartitionedOwnedTableTransferReleaser), InteropError> {
    transfer_partitioned_owned_table_from_columns_with_runtime_release_synchronization(
        producer,
        ReleaseSynchronization::Asynchronous,
        operation,
    )
}

pub(super) fn transfer_partitioned_owned_table_from_columns_with_runtime_release_synchronization(
    producer: PartitionedDataFrameExport,
    synchronization: ReleaseSynchronization,
    operation: &'static str,
) -> Result<(PartitionedOwnedTableTransfer, PartitionedOwnedTableTransferReleaser), InteropError> {
    let schema_names = producer.column_names_owned();
    let row_count = producer.num_rows();
    let mut partitions = Vec::new();
    let mut releasers = Vec::new();

    for group in producer.grouped_owned_columns() {
        let top_level_column_indices = group.top_level_column_indices;
        let table = Table::from_columns(group.columns)
            .map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let (transfer, releaser) = transfer_from_owned_table_with_names_and_runtime_release_synchronization(
            table,
            None,
            synchronization,
            operation,
        )?;
        partitions.push(PartitionedOwnedTableTransferPartition::new(top_level_column_indices.clone(), transfer));
        releasers.push(PartitionedOwnedTableTransferReleaserPartition::new(top_level_column_indices, releaser));
    }

    let transfer = PartitionedOwnedTableTransfer::try_new(Some(schema_names), row_count, partitions)
        .map_err(|error| error.with_context("operation", operation))?;
    Ok((transfer, PartitionedOwnedTableTransferReleaser::new(releasers)))
}
