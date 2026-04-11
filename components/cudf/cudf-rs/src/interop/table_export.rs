use super::{table_support::checked_column_index, *};

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
) -> Result<(Vec<SemanticColumn>, Vec<SemanticColumn>, Vec<BufferBinding<'a>>), InteropError> {
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

    Ok((columns, semantic_columns, bindings))
}

fn build_host_table_projection_from_view(
    view: &TableView<'_>,
    operation: &'static str,
) -> Result<(Vec<SemanticColumn>, Vec<SemanticColumn>, Vec<BufferBinding<'static>>, Vec<Arc<[u8]>>), InteropError> {
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

    Ok((columns, semantic_columns, bindings, retained_host_buffers))
}

fn build_device_table_projection_from_columns<'a, I>(
    columns: I,
    operation: &'static str,
    default_device: i32,
) -> Result<(Vec<SemanticColumn>, Vec<SemanticColumn>, Vec<BufferBinding<'a>>), InteropError>
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

    Ok((projected_columns, semantic_columns, bindings))
}

fn build_host_table_projection_from_columns<'a, I>(
    columns: I,
    operation: &'static str,
) -> Result<(Vec<SemanticColumn>, Vec<SemanticColumn>, Vec<BufferBinding<'static>>, Vec<Arc<[u8]>>), InteropError>
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

    Ok((projected_columns, semantic_columns, bindings, retained_host_buffers))
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
    let (columns, semantic_columns, bindings) =
        build_device_table_projection_from_view(view, operation, default_device)?;
    let table =
        SemanticTable::new(TableId::new(0), num_rows, columns.iter().map(|column| column.id()).collect(), names);
    PreparedTableExport::try_new_with_semantic_columns(
        table,
        columns,
        semantic_columns,
        bindings,
        readiness,
        export_diagnostics(),
    )
    .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn prepare_host_table_export_from_view(
    view: &TableView<'_>,
    names: Option<Vec<String>>,
    operation: &'static str,
) -> Result<PreparedTableExport<'static>, InteropError> {
    let (num_rows, _) = table_dimensions(view, operation)?;
    let (columns, semantic_columns, bindings, retained_host_buffers) =
        build_host_table_projection_from_view(view, operation)?;
    let table =
        SemanticTable::new(TableId::new(0), num_rows, columns.iter().map(|column| column.id()).collect(), names);
    PreparedTableExport::try_new_with_semantic_columns(
        table,
        columns,
        semantic_columns,
        bindings,
        ProducerReadiness::HostVisible,
        host_export_diagnostics(),
    )
    .map(|export| export.with_retained_host_buffers(retained_host_buffers))
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
    let (default_device, readiness) = prepared_device_readiness(operation, domain)?;
    prepare_device_table_export_from_columns_with_readiness(
        columns,
        num_rows,
        names,
        operation,
        default_device,
        readiness,
    )
}

pub(super) fn prepare_device_table_export_from_columns_with_readiness<'a, I>(
    columns: I,
    num_rows: usize,
    names: Option<Vec<String>>,
    operation: &'static str,
    default_device: i32,
    readiness: ProducerReadiness<'a>,
) -> Result<PreparedTableExport<'a>, InteropError>
where
    I: IntoIterator<Item = &'a Column>,
{
    let (columns, semantic_columns, bindings) =
        build_device_table_projection_from_columns(columns, operation, default_device)?;
    let table =
        SemanticTable::new(TableId::new(0), num_rows, columns.iter().map(|column| column.id()).collect(), names);
    PreparedTableExport::try_new_with_semantic_columns(
        table,
        columns,
        semantic_columns,
        bindings,
        readiness,
        export_diagnostics(),
    )
    .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
    .map_err(|error| error.with_context("operation", operation))
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
    let (columns, semantic_columns, bindings, retained_host_buffers) =
        build_host_table_projection_from_columns(columns, operation)?;
    let table =
        SemanticTable::new(TableId::new(0), num_rows, columns.iter().map(|column| column.id()).collect(), names);
    PreparedTableExport::try_new_with_semantic_columns(
        table,
        columns,
        semantic_columns,
        bindings,
        ProducerReadiness::HostVisible,
        host_export_diagnostics(),
    )
    .map(|export| export.with_retained_host_buffers(retained_host_buffers))
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
    let (default_device, readiness) = prepared_device_readiness(operation, table.execution_domain())?;
    let (num_rows, columns, semantic_columns, bindings) = {
        let view = table.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let (num_rows, _) = table_dimensions(&view, operation)?;
        let (columns, semantic_columns, bindings) =
            build_device_table_projection_from_view(&view, operation, default_device)?;
        (num_rows, columns, semantic_columns, bindings)
    };

    let table_semantic =
        SemanticTable::new(TableId::new(0), num_rows, columns.iter().map(|column| column.id()).collect(), names);
    OwnedTableTransfer::try_new_with_semantic_columns(
        table_semantic,
        columns,
        semantic_columns,
        // SAFETY: `to_static_bindings` depends on this retained owner to keep
        // the source table alive for the transfer lifetime.
        to_static_bindings(operation, bindings)?,
        to_static_readiness(readiness),
        owned_transfer_diagnostics(),
        owned_transfer_release_contract(default_device),
        OwnedTableTransferState::new(table),
    )
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn transfer_from_owned_table_with_runtime_release(
    table: Table,
    operation: &'static str,
) -> Result<(OwnedTableTransfer, RuntimeOwnedTableTransferReleaser), InteropError> {
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
) -> Result<(OwnedTableTransfer, RuntimeOwnedTableTransferReleaser), InteropError> {
    transfer_from_owned_table_with_names_and_runtime_release_synchronization(table, None, synchronization, operation)
}

pub(super) fn transfer_from_owned_table_with_names_and_runtime_release(
    table: Table,
    names: Option<Vec<String>>,
    operation: &'static str,
) -> Result<(OwnedTableTransfer, RuntimeOwnedTableTransferReleaser), InteropError> {
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
) -> Result<(OwnedTableTransfer, RuntimeOwnedTableTransferReleaser), InteropError> {
    let (default_device, readiness) = prepared_device_readiness(operation, table.execution_domain())?;
    let (num_rows, columns, semantic_columns, bindings) = {
        let view = table.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let (num_rows, _) = table_dimensions(&view, operation)?;
        let (columns, semantic_columns, bindings) =
            build_device_table_projection_from_view(&view, operation, default_device)?;
        (num_rows, columns, semantic_columns, bindings)
    };

    let table_semantic =
        SemanticTable::new(TableId::new(0), num_rows, columns.iter().map(|column| column.id()).collect(), names);
    let (owner, releaser) = runtime_owned_table_owner_and_releaser(table, default_device, synchronization);
    let transfer = OwnedTableTransfer::try_new_with_semantic_columns(
        table_semantic,
        columns,
        semantic_columns,
        to_static_bindings(operation, bindings)?,
        to_static_readiness(readiness),
        owned_transfer_diagnostics(),
        owned_transfer_runtime_release_contract_with_synchronization(default_device, synchronization)?,
        owner,
    )
    .map_err(|error| error.with_context("operation", operation))?;

    Ok((transfer, releaser))
}

pub(super) fn prepare_partitioned_device_table_export_from_columns<'a>(
    producer: &'a PartitionedDataFrameExport,
    operation: &'static str,
) -> Result<PartitionedPreparedTableExport<'a>, InteropError> {
    let mut partitions = Vec::new();

    for group in producer.grouped_borrowed_columns() {
        let export = prepare_device_table_export_from_columns(
            group.columns.into_iter(),
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
) -> Result<(PartitionedOwnedTableTransfer, PartitionedRuntimeOwnedTableTransferReleaser), InteropError> {
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
) -> Result<(PartitionedOwnedTableTransfer, PartitionedRuntimeOwnedTableTransferReleaser), InteropError> {
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
        releasers.push(PartitionedRuntimeOwnedTableTransferReleaserPartition::new(top_level_column_indices, releaser));
    }

    let transfer = PartitionedOwnedTableTransfer::try_new(Some(schema_names), row_count, partitions)
        .map_err(|error| error.with_context("operation", operation))?;
    Ok((transfer, PartitionedRuntimeOwnedTableTransferReleaser::new(releasers)))
}
