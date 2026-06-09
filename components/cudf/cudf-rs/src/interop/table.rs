use super::{
    table_export::{
        transfer_from_owned_table_with_runtime_release, transfer_from_owned_table_with_runtime_release_synchronization,
    },
    table_support::checked_column_index,
    *,
};

/// One partition-local table result consumed from a partitioned prepared table
/// export.
///
/// This preserves the original top-level logical schema coverage for the
/// partition and carries the imported local `Table` produced on the matching
/// destination runtime.
pub struct PartitionedLocalTableResult {
    top_level_column_indices: Vec<usize>,
    table: Table,
}

impl std::fmt::Debug for PartitionedLocalTableResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let num_columns = self.table.view().ok().and_then(|view| view.num_columns().ok());
        f.debug_struct("PartitionedLocalTableResult")
            .field("top_level_column_indices", &self.top_level_column_indices)
            .field("num_columns", &num_columns)
            .field("execution_domain_id", &self.table.__debug_execution_domain_id())
            .finish()
    }
}

impl PartitionedLocalTableResult {
    pub(crate) fn new(top_level_column_indices: Vec<usize>, table: Table) -> Self {
        Self { top_level_column_indices, table }
    }

    /// Return the original top-level schema indices covered by this local
    /// partition result.
    pub fn top_level_column_indices(&self) -> &[usize] {
        &self.top_level_column_indices
    }

    /// Borrow the imported local table for this partition.
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Consume this local partition result and return the imported table.
    pub fn into_table(self) -> Table {
        self.table
    }
}

/// One logical partitioned prepared-table consumption result made explicit as
/// per-partition local tables rather than a synthetic aggregate multi-GPU
/// table.
pub struct PartitionedLocalTableResults {
    schema_names: Option<Vec<String>>,
    num_rows: usize,
    partitions: Vec<PartitionedLocalTableResult>,
}

impl std::fmt::Debug for PartitionedLocalTableResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PartitionedLocalTableResults")
            .field("schema_names", &self.schema_names)
            .field("num_rows", &self.num_rows)
            .field("partitions", &self.partitions)
            .finish()
    }
}

impl PartitionedLocalTableResults {
    pub(crate) fn new(
        schema_names: Option<Vec<String>>,
        num_rows: usize,
        partitions: Vec<PartitionedLocalTableResult>,
    ) -> Self {
        Self { schema_names, num_rows, partitions }
    }

    /// Borrow the logical schema names carried by the partitioned root, when
    /// present.
    pub fn schema_names(&self) -> Option<&[String]> {
        self.schema_names.as_deref()
    }

    /// Return the logical row count shared by every partition result.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// Borrow the local per-partition table results in partition order.
    pub fn partitions(&self) -> &[PartitionedLocalTableResult] {
        &self.partitions
    }

    /// Consume this wrapper and return the local per-partition table results.
    pub fn into_partitions(self) -> Vec<PartitionedLocalTableResult> {
        self.partitions
    }
}

fn borrow_table_view_from_table_view<'a>(
    view: &'a TableView<'_>,
    operation: &'static str,
) -> Result<BorrowedTableView<'a>, InteropError> {
    let num_rows = view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let num_columns =
        view.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let execution = execution_binding(operation, view.execution_domain())?;
    let mut columns = Vec::with_capacity(num_columns);
    for index in 0..num_columns {
        let _ = checked_column_index(operation, index)?;
        let column_view =
            view.column(index).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let borrowed = borrow_from_view_with_execution(&column_view, execution, operation)?;
        columns.push(borrowed);
    }
    BorrowedTableView::try_new(num_rows, columns, execution).map_err(|error| error.with_context("operation", operation))
}

fn borrow_table_view_from_table<'a>(
    table: &'a Table,
    operation: &'static str,
) -> Result<BorrowedTableView<'a>, InteropError> {
    let view = table.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let num_rows = view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let num_columns =
        view.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let execution = execution_binding(operation, table.execution_domain())?;
    let mut columns = Vec::with_capacity(num_columns);
    for index in 0..num_columns {
        let _ = checked_column_index(operation, index)?;
        let column_view =
            view.column(index).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let borrowed = borrow_from_view_with_execution(&column_view, execution, operation)?;
        columns.push(borrowed);
    }
    BorrowedTableView::try_new(num_rows, columns, execution).map_err(|error| error.with_context("operation", operation))
}

impl TableView<'_> {
    /// Project this low-level borrowed table view into the shared RAPIDS
    /// borrowed interop ABI.
    ///
    /// This is a borrowed-only projection: it does not transfer ownership and
    /// avoids host copies on the input path. The current borrowed compatibility
    /// profile accepts the current fixed-width, `STRING`, `LIST`, `STRUCT`,
    /// and `DICTIONARY32` semantic slice and preserves the table execution
    /// binding.
    pub fn try_borrow_table_view(&self) -> Result<BorrowedTableView<'_>, InteropError> {
        <Self as TryBorrowTableView>::try_borrow_table_view(self)
    }

    /// Project this low-level borrowed table view into the shared RAPIDS
    /// prepared device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedTableExport::try_activate_read_lease`].
    pub fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        <Self as TryPrepareDeviceTableExport>::try_prepare_device_table_export(self)
    }

    /// Project this low-level borrowed table view into the shared RAPIDS
    /// prepared host export profile.
    ///
    /// This materializes each supported top-level column to owned host-visible
    /// bytes and therefore is not zero-copy.
    pub fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        <Self as TryPrepareHostTableExport>::try_prepare_host_table_export(self)
    }

    /// Project this low-level borrowed table view into the shared RAPIDS
    /// prepared device export profile with producer event-ordered readiness.
    ///
    /// This still returns a producer-rooted export; the supplied event only
    /// refines the read-start fact carried by the export's shared readiness
    /// contract.
    pub fn try_prepare_device_table_export_with_event<'a>(
        &'a self,
        event: &'a CudaEvent,
    ) -> Result<PreparedTableExport<'a>, InteropError> {
        let operation = "interop::TableView::try_prepare_device_table_export_with_event";
        let (default_device, readiness) = prepared_device_event_readiness(operation, self.execution_domain(), event)?;
        prepare_table_export_from_view_with_readiness(self, None, operation, default_device, readiness)
    }
}

impl TryBorrowTableView for TableView<'_> {
    fn try_borrow_table_view<'a>(&'a self) -> Result<BorrowedTableView<'a>, InteropError> {
        borrow_table_view_from_table_view(self, "interop::TableView::try_borrow_table_view")
    }
}

impl TryPrepareDeviceTableExport for TableView<'_> {
    fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        prepare_table_export_from_view(
            self,
            self.execution_domain(),
            None,
            "interop::TableView::try_prepare_device_table_export",
        )
    }
}

impl TryPrepareHostTableExport for TableView<'_> {
    fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        prepare_host_table_export_from_view(self, None, "interop::TableView::try_prepare_host_table_export")
    }
}

impl Table {
    /// Deep-copy a shared borrowed interop table descriptor into an owned
    /// `cudf-rs` table inside the provided explicit runtime.
    ///
    /// This imports each borrowed column through a device-to-device copy on
    /// the descriptor's CUDA device, then rebuilds one owned table in the
    /// destination runtime's retained provenance. It is not zero-copy, and the
    /// destination runtime must be pinned to the same CUDA device as the
    /// borrowed execution binding.
    pub fn try_import_from_borrowed_table_view_into_runtime(
        borrowed: &BorrowedTableView<'_>,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        import_table_from_borrowed_view(
            borrowed,
            runtime,
            "interop::Table::try_import_from_borrowed_table_view_into_runtime",
        )
    }

    /// Consume a shared owned device transfer root back into an owned
    /// `cudf-rs` table inside the provided explicit runtime.
    ///
    /// This is a zero-copy ownership import. The destination runtime must be
    /// pinned to the same CUDA device as the transfer readiness contract, and
    /// the imported table is rebound to that runtime's retained provenance.
    pub fn try_import_from_owned_device_table_transfer_into_runtime(
        transfer: OwnedTableTransfer,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        import_table_from_owned_transfer(
            transfer,
            runtime,
            "interop::Table::try_import_from_owned_device_table_transfer_into_runtime",
        )
    }

    /// Deep-copy a shared prepared export root into an owned
    /// `cudf-rs` table inside the provided explicit runtime.
    ///
    /// Prepared device exports activate on the destination runtime's CUDA
    /// stream and perform same-device device-to-device copies for the current
    /// prepared-device layout set. Prepared host exports activate
    /// synchronously by return and upload their host-visible bytes into new
    /// device ownership inside the destination runtime. Both paths
    /// synchronize completion before returning so producer liveness can end
    /// honestly.
    pub fn try_import_from_prepared_table_export_into_runtime(
        export: &PreparedTableExport<'_>,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        import_table_from_prepared_export(
            export,
            runtime,
            "interop::Table::try_import_from_prepared_table_export_into_runtime",
        )
    }

    /// Consume one partitioned prepared-table export into one local table per
    /// partition runtime.
    ///
    /// This keeps multi-device truth explicit at the boundary: each partition
    /// activates and completes its own read lease, imports on its matching
    /// destination runtime, and returns one local `Table` annotated with the
    /// original top-level schema indices it covers. This method does not
    /// synthesize one aggregate multi-GPU table.
    pub fn try_import_partitioned_local_tables_from_prepared_table_export_into_runtimes(
        export: &PartitionedPreparedTableExport<'_>,
        runtimes: &[&Runtime],
    ) -> Result<PartitionedLocalTableResults, InteropError> {
        let operation = "interop::Table::try_import_partitioned_local_tables_from_prepared_table_export_into_runtimes";
        let partitions = import_partitioned_local_tables_from_prepared_export(export, runtimes, operation)?
            .into_iter()
            .map(|(top_level_column_indices, table)| PartitionedLocalTableResult::new(top_level_column_indices, table))
            .collect();
        Ok(PartitionedLocalTableResults::new(
            export.schema_names().map(|names| names.to_vec()),
            export.num_rows(),
            partitions,
        ))
    }

    /// Lend this owned low-level table as a shared borrowed interop table
    /// view without host copies on the input path.
    ///
    /// `Table` is a convenience adapter, not the ABI itself. The returned
    /// descriptor remains borrowed-only, does not transfer ownership, and is
    /// limited to the current borrowed compatibility profile's fixed-width,
    /// `STRING`, `LIST`, `STRUCT`, and `DICTIONARY32` support.
    pub fn try_borrow_table_view(&self) -> Result<BorrowedTableView<'_>, InteropError> {
        <Self as TryBorrowTableView>::try_borrow_table_view(self)
    }

    /// Project this owned low-level table into the shared RAPIDS prepared
    /// device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedTableExport::try_activate_read_lease`].
    ///
    /// Borrowed interop is also available directly on [`crate::core::Table`]
    /// via [`crate::core::Table::try_borrow_table_view`] when you need the
    /// borrowed compatibility descriptor instead of a prepared export.
    pub fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        <Self as TryPrepareDeviceTableExport>::try_prepare_device_table_export(self)
    }

    /// Project this owned low-level table into the shared RAPIDS prepared host
    /// export profile.
    ///
    /// This materializes each supported top-level column to owned host-visible
    /// bytes and therefore is not zero-copy.
    pub fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        <Self as TryPrepareHostTableExport>::try_prepare_host_table_export(self)
    }

    /// Move this owned table into the shared RAPIDS owned device transfer
    /// profile.
    pub fn try_into_owned_transfer(self) -> Result<OwnedTableTransfer, InteropError> {
        <Self as TryIntoOwnedDeviceTableTransfer>::try_into_owned_device_table_transfer(self)
    }

    /// Move this owned table into the shared RAPIDS owned device transfer
    /// profile plus a CUDA-backed releaser for signal-driven release.
    ///
    /// The returned releaser accepts release and schedules any required
    /// consumer-signal wait onto a background runtime job instead of blocking
    /// the caller thread.
    pub fn try_into_owned_transfer_with_release(
        self,
    ) -> Result<(OwnedTableTransfer, RuntimeOwnedTableTransferReleaser), InteropError> {
        transfer_from_owned_table_with_runtime_release(self, "interop::Table::try_into_owned_transfer_with_release")
    }

    /// Move this owned table into the shared RAPIDS owned device transfer
    /// profile plus a CUDA-backed releaser for signal-driven release with an
    /// explicit release-synchronization contract.
    ///
    /// The returned transfer always declares
    /// [`TransferCompletionRule::ReleaseWaitsOnConsumerSignal`]. Pass
    /// [`ReleaseSynchronization::Asynchronous`] when release must return after
    /// acceptance and background scheduling, or
    /// [`ReleaseSynchronization::MaySynchronizeInternally`] when a blocking
    /// release call is acceptable. Caller-managed synchronization remains
    /// unsupported for this runtime-backed producer path.
    pub fn try_into_owned_transfer_with_release_synchronization(
        self,
        synchronization: ReleaseSynchronization,
    ) -> Result<(OwnedTableTransfer, RuntimeOwnedTableTransferReleaser), InteropError> {
        transfer_from_owned_table_with_runtime_release_synchronization(
            self,
            synchronization,
            "interop::Table::try_into_owned_transfer_with_release_synchronization",
        )
    }

    /// Project this owned low-level table into the shared RAPIDS prepared
    /// device export profile with producer event-ordered readiness.
    pub fn try_prepare_device_table_export_with_event<'a>(
        &'a self,
        event: &'a CudaEvent,
    ) -> Result<PreparedTableExport<'a>, InteropError> {
        let operation = "interop::Table::try_prepare_device_table_export_with_event";
        let view = self.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let (default_device, readiness) = prepared_device_event_readiness(operation, self.execution_domain(), event)?;
        prepare_table_export_from_view_with_readiness(&view, None, operation, default_device, readiness)
    }
}

impl TryBorrowTableView for Table {
    fn try_borrow_table_view<'a>(&'a self) -> Result<BorrowedTableView<'a>, InteropError> {
        borrow_table_view_from_table(self, "interop::Table::try_borrow_table_view")
    }
}

impl TryPrepareDeviceTableExport for Table {
    fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        let operation = "interop::Table::try_prepare_device_table_export";
        let view = self.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        prepare_table_export_from_view(&view, self.execution_domain(), None, operation)
    }
}

impl TryPrepareHostTableExport for Table {
    fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        let operation = "interop::Table::try_prepare_host_table_export";
        let view = self.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        prepare_host_table_export_from_view(&view, None, operation)
    }
}

impl TryIntoOwnedDeviceTableTransfer for Table {
    fn try_into_owned_device_table_transfer(self) -> Result<OwnedTableTransfer, InteropError> {
        transfer_from_owned_table(self, "interop::Table::try_into_owned_device_table_transfer")
    }
}
