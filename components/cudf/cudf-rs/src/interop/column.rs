use super::*;

fn prepare_from_column_view<'a>(
    view: &'a ColumnView<'_>,
    operation: &'static str,
) -> Result<PreparedColumnExport<'a>, InteropError> {
    let (default_device, readiness) = prepared_device_readiness(operation, view.execution_domain())?;
    prepare_from_column_view_with_readiness(view, operation, default_device, readiness)
}

fn prepare_from_column_view_with_readiness<'a>(
    view: &'a ColumnView<'_>,
    operation: &'static str,
    default_device: i32,
    readiness: ProducerReadiness<'a>,
) -> Result<PreparedColumnExport<'a>, InteropError> {
    let projection = prepared_column_projection_from_view(view, operation, default_device, 0, 0)?;
    PreparedColumnExport::try_new_with_semantic_columns(
        projection.column,
        projection.semantic_columns,
        projection.bindings,
        readiness,
        export_diagnostics(),
    )
    .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
    .map_err(|error| error.with_context("operation", operation))
}

fn prepare_host_from_column_view(
    view: &ColumnView<'_>,
    operation: &'static str,
) -> Result<PreparedColumnExport<'static>, InteropError> {
    let projection = prepared_host_column_projection_from_view(view, operation, 0, 0)?;
    PreparedColumnExport::try_new_with_semantic_columns(
        projection.column,
        projection.semantic_columns,
        projection.bindings,
        ProducerReadiness::HostVisible,
        host_export_diagnostics(),
    )
    .map(|export| export.with_retained_host_buffers(projection.retained_host_buffers))
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn prepare_from_owned_column<'a>(
    column: &'a Column,
    operation: &'static str,
) -> Result<PreparedColumnExport<'a>, InteropError> {
    let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let (default_device, readiness) = prepared_device_readiness(operation, column.execution_domain())?;
    let projection = prepared_column_projection_from_view(&view, operation, default_device, 0, 0)?;
    PreparedColumnExport::try_new_with_semantic_columns(
        projection.column,
        projection.semantic_columns,
        projection.bindings,
        readiness,
        export_diagnostics(),
    )
    .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn prepare_host_from_owned_column(
    column: &Column,
    operation: &'static str,
) -> Result<PreparedColumnExport<'static>, InteropError> {
    let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let projection = prepared_host_column_projection_from_view(&view, operation, 0, 0)?;
    PreparedColumnExport::try_new_with_semantic_columns(
        projection.column,
        projection.semantic_columns,
        projection.bindings,
        ProducerReadiness::HostVisible,
        host_export_diagnostics(),
    )
    .map(|export| export.with_retained_host_buffers(projection.retained_host_buffers))
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn transfer_from_owned_column(
    column: Column,
    operation: &'static str,
) -> Result<OwnedColumnTransfer, InteropError> {
    let (default_device, readiness) = prepared_device_readiness(operation, column.execution_domain())?;
    let projection = {
        let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        prepared_column_projection_from_view(&view, operation, default_device, 0, 0)?
    };
    OwnedColumnTransfer::try_new_with_semantic_columns(
        projection.column,
        projection.semantic_columns,
        // SAFETY: `to_static_bindings` depends on this retained owner to keep
        // the source column alive for the transfer lifetime.
        to_static_bindings(operation, projection.bindings)?,
        to_static_readiness(readiness),
        owned_transfer_diagnostics(),
        owned_transfer_release_contract(default_device),
        OwnedColumnTransferState::new(column),
    )
    .map_err(|error| error.with_context("operation", operation))
}

impl<'view> ColumnView<'view> {
    /// Project this low-level borrowed column view into the shared RAPIDS
    /// borrowed interop ABI.
    ///
    /// This is a borrowed-only projection: it does not transfer ownership and
    /// avoids host copies on the input path. The current borrowed compatibility
    /// profile accepts fixed-width columns plus the current `STRING` and
    /// `DICTIONARY32` slice, and preserves the column's execution binding.
    pub fn try_borrow_column_view(&self) -> Result<BorrowedColumnView<'_>, InteropError> {
        <Self as TryBorrowColumnView>::try_borrow_column_view(self)
    }

    /// Project this low-level borrowed column view into the shared RAPIDS
    /// prepared device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedColumnExport::try_activate_read_lease`].
    pub fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>, InteropError> {
        <Self as TryPrepareDeviceColumnExport>::try_prepare_device_column_export(self)
    }

    /// Project this low-level borrowed column view into the shared RAPIDS
    /// prepared host export profile.
    ///
    /// This materializes the active logical range to owned host-visible bytes
    /// and therefore is not zero-copy.
    pub fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>, InteropError> {
        <Self as TryPrepareHostColumnExport>::try_prepare_host_column_export(self)
    }

    /// Project this low-level borrowed column view into the shared RAPIDS
    /// prepared device export profile with producer event-ordered readiness.
    ///
    /// This still returns a producer-rooted export; the supplied event only
    /// refines the read-start fact carried by the export's shared readiness
    /// contract.
    pub fn try_prepare_device_column_export_with_event<'a>(
        &'a self,
        event: &'a CudaEvent,
    ) -> Result<PreparedColumnExport<'a>, InteropError> {
        let operation = "interop::ColumnView::try_prepare_device_column_export_with_event";
        let (default_device, readiness) = prepared_device_event_readiness(operation, self.execution_domain(), event)?;
        prepare_from_column_view_with_readiness(self, operation, default_device, readiness)
    }
}

impl TryBorrowColumnView for ColumnView<'_> {
    fn try_borrow_column_view<'a>(&'a self) -> Result<BorrowedColumnView<'a>, InteropError> {
        borrow_from_column_view(self, "interop::ColumnView::try_borrow_column_view")
    }
}

impl TryPrepareDeviceColumnExport for ColumnView<'_> {
    fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>, InteropError> {
        prepare_from_column_view(self, "interop::ColumnView::try_prepare_device_column_export")
    }
}

impl TryPrepareHostColumnExport for ColumnView<'_> {
    fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>, InteropError> {
        prepare_host_from_column_view(self, "interop::ColumnView::try_prepare_host_column_export")
    }
}

impl Column {
    /// Deep-copy a shared borrowed interop column descriptor into an owned
    /// `cudf-rs` column inside the provided explicit runtime.
    ///
    /// This is a device-to-device copy on the descriptor's CUDA device. It is
    /// not zero-copy, and the destination runtime must be pinned to the same
    /// CUDA device as the borrowed execution binding.
    pub fn try_import_from_borrowed_column_view_into_runtime(
        borrowed: &BorrowedColumnView<'_>,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        import_column_from_borrowed_view(
            borrowed,
            runtime,
            "interop::Column::try_import_from_borrowed_column_view_into_runtime",
        )
    }

    /// Consume a shared owned device transfer root back into an owned
    /// `cudf-rs` column.
    ///
    /// This is a zero-copy ownership import. `cudf-rs` round-trips preserve
    /// the original retained execution provenance, while foreign owner roots
    /// in the current consumer slice are accepted for the supported foreign
    /// layout set
    /// without deep-copying into a destination runtime.
    pub fn try_from_owned_device_column_transfer(transfer: OwnedColumnTransfer) -> Result<Self, InteropError> {
        import_column_from_owned_transfer(transfer, "interop::Column::try_from_owned_device_column_transfer")
    }

    /// Deep-copy a shared prepared export root into an owned
    /// `cudf-rs` column inside the provided explicit runtime.
    ///
    /// Prepared device exports activate on the destination runtime's CUDA
    /// stream and perform a same-device device-to-device copy. Prepared host
    /// exports activate synchronously by return and upload their host-visible
    /// bytes into new device ownership inside the destination runtime. Both
    /// paths synchronize completion before returning so producer liveness can
    /// end honestly.
    pub fn try_import_from_prepared_column_export_into_runtime(
        export: &PreparedColumnExport<'_>,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        import_column_from_prepared_export(
            export,
            runtime,
            "interop::Column::try_import_from_prepared_column_export_into_runtime",
        )
    }

    /// Lend this owned low-level column as a shared borrowed interop column
    /// view without host copies on the input path.
    ///
    /// `Column` is a convenience adapter, not the ABI itself. The returned
    /// descriptor remains borrowed-only, does not transfer ownership, and is
    /// limited to the current borrowed compatibility profile's fixed-width,
    /// `STRING`, and `DICTIONARY32` support.
    pub fn try_borrow_column_view(&self) -> Result<BorrowedColumnView<'_>, InteropError> {
        <Self as TryBorrowColumnView>::try_borrow_column_view(self)
    }

    /// Project this owned low-level column into the shared RAPIDS prepared
    /// device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedColumnExport::try_activate_read_lease`].
    ///
    /// Borrowed interop is also available directly on
    /// [`crate::core::Column`] via
    /// [`crate::core::Column::try_borrow_column_view`] when you need the
    /// borrowed compatibility descriptor instead of a prepared export.
    pub fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>, InteropError> {
        <Self as TryPrepareDeviceColumnExport>::try_prepare_device_column_export(self)
    }

    /// Project this owned low-level column into the shared RAPIDS prepared
    /// host export profile.
    ///
    /// This materializes the active logical range to owned host-visible bytes
    /// and therefore is not zero-copy.
    pub fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>, InteropError> {
        <Self as TryPrepareHostColumnExport>::try_prepare_host_column_export(self)
    }

    /// Move this owned column into the shared RAPIDS owned device transfer
    /// profile.
    pub fn try_into_owned_device_column_transfer(self) -> Result<OwnedColumnTransfer, InteropError> {
        <Self as TryIntoOwnedDeviceColumnTransfer>::try_into_owned_device_column_transfer(self)
    }

    /// Project this owned low-level column into the shared RAPIDS prepared
    /// device export profile with producer event-ordered readiness.
    pub fn try_prepare_device_column_export_with_event<'a>(
        &'a self,
        event: &'a CudaEvent,
    ) -> Result<PreparedColumnExport<'a>, InteropError> {
        let operation = "interop::Column::try_prepare_device_column_export_with_event";
        let view = self.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let (default_device, readiness) = prepared_device_event_readiness(operation, self.execution_domain(), event)?;
        let projection = prepared_column_projection_from_view(&view, operation, default_device, 0, 0)?;
        PreparedColumnExport::try_new_with_semantic_columns(
            projection.column,
            projection.semantic_columns,
            projection.bindings,
            readiness,
            export_diagnostics(),
        )
        .map(|export| export.with_lease_backend(runtime_lease::shared_cuda_lease_backend()))
        .map_err(|error| error.with_context("operation", operation))
    }
}

impl TryBorrowColumnView for Column {
    fn try_borrow_column_view<'a>(&'a self) -> Result<BorrowedColumnView<'a>, InteropError> {
        borrow_from_owned_column(self, "interop::Column::try_borrow_column_view")
    }
}

impl TryPrepareDeviceColumnExport for Column {
    fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>, InteropError> {
        prepare_from_owned_column(self, "interop::Column::try_prepare_device_column_export")
    }
}

impl TryPrepareHostColumnExport for Column {
    fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>, InteropError> {
        prepare_host_from_owned_column(self, "interop::Column::try_prepare_host_column_export")
    }
}

impl TryIntoOwnedDeviceColumnTransfer for Column {
    fn try_into_owned_device_column_transfer(self) -> Result<OwnedColumnTransfer, InteropError> {
        transfer_from_owned_column(self, "interop::Column::try_into_owned_device_column_transfer")
    }
}
