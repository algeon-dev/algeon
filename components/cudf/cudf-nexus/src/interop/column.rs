use std::sync::Arc;

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
    let observation = column_view_boundary_observation(BoundaryKind::InteropLease, view, operation)?;
    let observer = view.execution_domain().boundary_observer();
    let projection = prepared_column_projection_from_view(view, operation, default_device, 0, 0)?;
    PreparedColumnExport::try_new_with_semantic_columns(
        projection.column,
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

fn prepare_from_owned_column_root_with_readiness(
    column: Column,
    operation: &'static str,
    readiness: ProducerReadiness<'_>,
    retained_event: Option<CudaEvent>,
) -> Result<PreparedColumnExport<'static>, InteropError> {
    let retained_column = Arc::new(column);
    let view =
        retained_column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let observation = column_view_boundary_observation(BoundaryKind::InteropLease, &view, operation)?;
    let observer = retained_column.execution_domain().boundary_observer();
    let default_device = retained_column.execution_domain().device_id().ordinal() as i32;
    let projection = prepared_column_projection_from_view(&view, operation, default_device, 0, 0)?;
    PreparedColumnExport::try_new_with_semantic_columns(
        projection.column,
        projection.semantic_columns,
        to_static_bindings(operation, projection.bindings)?,
        to_static_readiness(operation, readiness)?,
        export_diagnostics(),
    )
    .map(|export| {
        let export =
            export.with_lease_backend(runtime_lease::shared_cuda_lease_backend_with_observation(observation, observer));
        match retained_event {
            // `to_static_readiness` stores only the raw CUDA event pointer, so
            // the export must co-own the event for its whole lifetime.
            Some(event) => export.with_retained_owner_arc(Arc::new((Arc::clone(&retained_column), event))),
            None => export.with_retained_owner_arc(Arc::clone(&retained_column)),
        }
    })
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn prepare_from_owned_column<'a>(
    column: &'a Column,
    operation: &'static str,
) -> Result<PreparedColumnExport<'a>, InteropError> {
    let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let observation = column_view_boundary_observation(BoundaryKind::InteropLease, &view, operation)?;
    let observer = column.execution_domain().boundary_observer();
    let (default_device, readiness) = prepared_device_readiness(operation, column.execution_domain())?;
    let projection = prepared_column_projection_from_view(&view, operation, default_device, 0, 0)?;
    PreparedColumnExport::try_new_with_semantic_columns(
        projection.column,
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

pub(super) fn prepare_into_owned_column_root(
    column: Column,
    operation: &'static str,
) -> Result<PreparedColumnExport<'static>, InteropError> {
    let readiness = prepared_device_readiness(operation, column.execution_domain())?.1;
    prepare_from_owned_column_root_with_readiness(column, operation, readiness, None)
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
    let observation = column_boundary_observation(BoundaryKind::OwnedTransfer, &column, operation)?;
    let domain = Arc::clone(column.execution_domain());
    let (default_device, readiness) = prepared_device_readiness(operation, column.execution_domain())?;
    let projection = {
        let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        prepared_column_projection_from_view(&view, operation, default_device, 0, 0)?
    };
    let transfer = OwnedColumnTransfer::try_new_with_semantic_columns(
        projection.column,
        projection.semantic_columns,
        // SAFETY: `to_static_bindings` depends on this retained owner to keep
        // the source column alive for the transfer lifetime.
        to_static_bindings(operation, projection.bindings)?,
        to_static_readiness(operation, readiness)?,
        owned_transfer_diagnostics(),
        owned_transfer_release_contract(default_device),
        OwnedColumnTransferState::new(column),
    )
    .map_err(|error| error.with_context("operation", operation))?;
    record_observation(&domain, observation);
    Ok(transfer)
}

pub(super) fn transfer_from_owned_column_with_runtime_release(
    column: Column,
    operation: &'static str,
) -> Result<(OwnedColumnTransfer, OwnedColumnTransferReleaser), InteropError> {
    transfer_from_owned_column_with_runtime_release_synchronization(
        column,
        ReleaseSynchronization::Asynchronous,
        operation,
    )
}

pub(super) fn transfer_from_owned_column_with_runtime_release_synchronization(
    column: Column,
    synchronization: ReleaseSynchronization,
    operation: &'static str,
) -> Result<(OwnedColumnTransfer, OwnedColumnTransferReleaser), InteropError> {
    let observation = column_boundary_observation(BoundaryKind::OwnedTransfer, &column, operation)?;
    let domain = Arc::clone(column.execution_domain());
    let (default_device, readiness) = prepared_device_readiness(operation, column.execution_domain())?;
    let (column_projection, semantic_columns, bindings) = {
        let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let projection = prepared_column_projection_from_view(&view, operation, default_device, 0, 0)?;
        (projection.column, projection.semantic_columns, to_static_bindings(operation, projection.bindings)?)
    };
    let readiness = to_static_readiness(operation, readiness)?;
    let (owner, releaser) = runtime_owned_column_owner_and_releaser(column, default_device, synchronization);
    let transfer = OwnedColumnTransfer::try_new_with_semantic_columns(
        column_projection,
        semantic_columns,
        bindings,
        readiness,
        owned_transfer_diagnostics(),
        owned_transfer_runtime_release_contract_with_synchronization(default_device, synchronization)?,
        owner,
    )
    .map_err(|error| error.with_context("operation", operation))?;

    record_observation(&domain, observation);
    Ok((transfer, releaser))
}

impl ColumnView<'_> {
    /// Project this low-level borrowed column view into the shared RAPIDS
    /// borrowed interop ABI.
    ///
    /// This is a borrowed-only projection: it does not transfer ownership and
    /// avoids host copies on the input path. The current borrowed compatibility
    /// profile accepts fixed-width columns plus the current `STRING`, `LIST`,
    /// `STRUCT`, and `DICTIONARY32` slice, and preserves the column's
    /// execution binding.
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
    fn try_borrow_column_view(&self) -> Result<BorrowedColumnView<'_>, InteropError> {
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
    /// `cudf` column inside the provided explicit runtime.
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
    /// `cudf` column inside the provided explicit runtime.
    ///
    /// This is a zero-copy ownership import. The destination runtime must be
    /// pinned to the same CUDA device as the transfer readiness contract, and
    /// the imported column is rebound to that runtime's retained provenance.
    pub fn try_import_from_owned_device_column_transfer_into_runtime(
        transfer: OwnedColumnTransfer,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        import_column_from_owned_transfer(
            transfer,
            runtime,
            "interop::Column::try_import_from_owned_device_column_transfer_into_runtime",
        )
    }

    /// Deep-copy a shared prepared export root into an owned
    /// `cudf` column inside the provided explicit runtime.
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
    /// `STRING`, `LIST`, `STRUCT`, and `DICTIONARY32` support.
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

    /// Move this owned low-level column into a producer-rooted shared prepared
    /// device export.
    ///
    /// Unlike the borrow-preserving [`Column::try_prepare_device_column_export`]
    /// adapter, this consuming variant upgrades the column itself into the
    /// prepared-export root. The returned export carries retained owner
    /// state, so downstream consumers can defer completion honestly without
    /// widening lifetimes through borrowed aliases.
    pub fn try_into_prepared_device_column_export(self) -> Result<PreparedColumnExport<'static>, InteropError> {
        prepare_into_owned_column_root(self, "interop::Column::try_into_prepared_device_column_export")
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
    pub fn try_into_owned_transfer(self) -> Result<OwnedColumnTransfer, InteropError> {
        <Self as TryIntoOwnedDeviceColumnTransfer>::try_into_owned_device_column_transfer(self)
    }

    /// Move this owned column into the shared RAPIDS owned device transfer
    /// profile plus a CUDA-backed releaser for signal-driven release.
    ///
    /// The returned transfer declares
    /// [`TransferCompletionRule::ReleaseWaitsOnConsumerSignal`] with
    /// [`ReleaseSynchronization::Asynchronous`]. Calling the returned releaser
    /// accepts release and schedules the consumer-signal wait onto a
    /// background runtime job instead of blocking the caller thread. The
    /// caller must eventually drive release through that releaser when a
    /// foreign consumer keeps the shared transfer root. If the transfer is
    /// instead imported back into a native `cudf` owner, that native owner
    /// claims the underlying ownership and the releaser becomes invalid.
    pub fn try_into_owned_transfer_with_release(
        self,
    ) -> Result<(OwnedColumnTransfer, OwnedColumnTransferReleaser), InteropError> {
        transfer_from_owned_column_with_runtime_release(self, "interop::Column::try_into_owned_transfer_with_release")
    }

    /// Move this owned column into the shared RAPIDS owned device transfer
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
    ) -> Result<(OwnedColumnTransfer, OwnedColumnTransferReleaser), InteropError> {
        transfer_from_owned_column_with_runtime_release_synchronization(
            self,
            synchronization,
            "interop::Column::try_into_owned_transfer_with_release_synchronization",
        )
    }

    /// Project this owned low-level column into the shared RAPIDS prepared
    /// device export profile with producer event-ordered readiness.
    pub fn try_prepare_device_column_export_with_event<'a>(
        &'a self,
        event: &'a CudaEvent,
    ) -> Result<PreparedColumnExport<'a>, InteropError> {
        let operation = "interop::Column::try_prepare_device_column_export_with_event";
        let view = self.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        let observation = column_view_boundary_observation(BoundaryKind::InteropLease, &view, operation)?;
        let observer = self.execution_domain().boundary_observer();
        let (default_device, readiness) = prepared_device_event_readiness(operation, self.execution_domain(), event)?;
        let projection = prepared_column_projection_from_view(&view, operation, default_device, 0, 0)?;
        PreparedColumnExport::try_new_with_semantic_columns(
            projection.column,
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

    /// Move this owned low-level column into a producer-rooted shared prepared
    /// device export with explicit event-ordered readiness.
    pub fn try_into_prepared_device_column_export_with_event(
        self,
        event: &CudaEvent,
    ) -> Result<PreparedColumnExport<'static>, InteropError> {
        let operation = "interop::Column::try_into_prepared_device_column_export_with_event";
        let readiness = prepared_device_event_readiness(operation, self.execution_domain(), event)?.1;
        prepare_from_owned_column_root_with_readiness(self, operation, readiness, Some(event.clone()))
    }
}

impl TryBorrowColumnView for Column {
    fn try_borrow_column_view(&self) -> Result<BorrowedColumnView<'_>, InteropError> {
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
