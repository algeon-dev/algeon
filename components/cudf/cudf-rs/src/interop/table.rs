use super::{table_export::transfer_from_owned_table_with_runtime_release, *};

fn borrow_table_view_from_table_view<'a>(
    view: &'a TableView<'_>,
    operation: &'static str,
) -> Result<BorrowedTableView<'a>, InteropError> {
    let num_rows = usize::try_from(
        view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "table row count must be >= 0")
            .with_context("operation", operation)
    })?;
    let num_columns = usize::try_from(
        view.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "table column count must be >= 0")
            .with_context("operation", operation)
    })?;
    let execution = execution_binding(operation, view.execution_domain())?;
    let mut columns = Vec::with_capacity(num_columns);
    for index in 0..num_columns {
        let index_i32 = checked_column_index(operation, index)?;
        let column_view =
            view.column(index_i32).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
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
    let num_rows = usize::try_from(
        view.num_rows().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "table row count must be >= 0")
            .with_context("operation", operation)
    })?;
    let num_columns = usize::try_from(
        view.num_columns().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?,
    )
    .map_err(|_| {
        InteropError::new(InteropErrorKind::InvalidView, "table column count must be >= 0")
            .with_context("operation", operation)
    })?;
    let execution = execution_binding(operation, table.execution_domain())?;
    let mut columns = Vec::with_capacity(num_columns);
    for index in 0..num_columns {
        let index_i32 = checked_column_index(operation, index)?;
        let column_view =
            view.column(index_i32).map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
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
    /// profile accepts only fixed-width columns and preserves the table
    /// execution binding.
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
    /// `cudf-rs` table.
    ///
    /// This is a zero-copy ownership import. `cudf-rs` round-trips preserve
    /// the original retained execution provenance, while foreign owner roots
    /// in the current consumer slice are accepted for the supported foreign
    /// layout set
    /// without deep-copying into a destination runtime.
    pub fn try_from_owned_device_table_transfer(transfer: OwnedTableTransfer) -> Result<Self, InteropError> {
        import_table_from_owned_transfer(transfer, "interop::Table::try_from_owned_device_table_transfer")
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

    /// Lend this owned low-level table as a shared borrowed interop table
    /// view without host copies on the input path.
    ///
    /// `Table` is a convenience adapter, not the ABI itself. The returned
    /// descriptor remains borrowed-only, does not transfer ownership, and is
    /// limited to the current borrowed compatibility profile's fixed-width,
    /// `STRING`, and `DICTIONARY32` support.
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
    pub fn try_into_owned_device_table_transfer(self) -> Result<OwnedTableTransfer, InteropError> {
        <Self as TryIntoOwnedDeviceTableTransfer>::try_into_owned_device_table_transfer(self)
    }

    /// Move this owned table into the shared RAPIDS owned device transfer
    /// profile plus a CUDA-backed releaser for signal-driven release.
    pub fn try_into_owned_device_table_transfer_with_runtime_release(
        self,
    ) -> Result<(OwnedTableTransfer, RuntimeOwnedTableTransferReleaser), InteropError> {
        transfer_from_owned_table_with_runtime_release(
            self,
            "interop::Table::try_into_owned_device_table_transfer_with_runtime_release",
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
