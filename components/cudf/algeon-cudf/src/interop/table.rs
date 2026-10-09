use super::{
    table_export::{prepare_into_owned_table_root, prepare_into_owned_table_root_with_readiness},
    *,
};

impl TableView<'_> {
    /// Project this low-level borrowed table view into the shared RAPIDS
    /// prepared device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedTableExport::try_activate_read_lease`].
    ///
    /// The export describes `self`'s device buffers without taking ownership, so
    /// it cannot outlive that borrow. Returns an [`InteropError`] when a column's
    /// layout falls outside the prepared device profile, its buffers are not on a
    /// CUDA device, or the execution binding or producer readiness of `self`'s
    /// runtime domain is unavailable.
    pub fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        prepare_table_export_from_view(
            self,
            self.execution_domain(),
            None,
            "interop::TableView::try_prepare_device_table_export",
        )
    }

    /// Project this low-level borrowed table view into the shared RAPIDS
    /// prepared host export profile.
    ///
    /// This materializes each supported top-level column to owned host-visible
    /// bytes and therefore is not zero-copy.
    ///
    /// The `'static` export owns retained host copies and does not borrow `self`.
    /// Returns an [`InteropError`] when a column's type or layout is not
    /// supported by host export or its data cannot be copied to host memory.
    pub fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        prepare_host_table_export_from_view(self, None, "interop::TableView::try_prepare_host_table_export")
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

impl Table {
    /// Deep-copy a shared prepared export root into an owned
    /// `cudf` table inside the provided explicit runtime.
    ///
    /// Prepared device exports activate on the destination runtime's CUDA
    /// stream and perform same-device device-to-device copies for the current
    /// prepared-device layout set. Prepared host exports activate
    /// synchronously by return and upload their host-visible bytes into new
    /// device ownership inside the destination runtime. Both paths
    /// synchronize completion before returning. The producer's liveness can
    /// end once the call returns.
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

    /// Project this owned low-level table into the shared RAPIDS prepared
    /// device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedTableExport::try_activate_read_lease`].
    ///
    /// The export describes `self`'s device buffers without taking ownership, so
    /// it cannot outlive that borrow. Returns an [`InteropError`] when a column's
    /// layout falls outside the prepared device profile, its buffers are not on a
    /// CUDA device, or the execution binding or producer readiness of `self`'s
    /// runtime domain is unavailable.
    pub fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        let operation = "interop::Table::try_prepare_device_table_export";
        let view = self.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        prepare_table_export_from_view(&view, self.execution_domain(), None, operation)
    }

    /// Move this owned low-level table into a producer-rooted shared prepared
    /// device export.
    ///
    /// This consuming variant upgrades the table itself into the prepared
    /// export root. The returned export retains producer ownership. Callers
    /// that need a non-owning projection use
    /// [`Table::try_prepare_device_table_export`].
    pub fn try_into_prepared_device_table_export(self) -> Result<PreparedTableExport<'static>, InteropError> {
        prepare_into_owned_table_root(self, None, "interop::Table::try_into_prepared_device_table_export")
    }

    /// Project this owned low-level table into the shared RAPIDS prepared host
    /// export profile.
    ///
    /// This materializes each supported top-level column to owned host-visible
    /// bytes and therefore is not zero-copy.
    ///
    /// The `'static` export owns retained host copies and does not borrow `self`.
    /// Returns an [`InteropError`] when a column's type or layout is not
    /// supported by host export or its data cannot be copied to host memory.
    pub fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        let operation = "interop::Table::try_prepare_host_table_export";
        let view = self.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
        prepare_host_table_export_from_view(&view, None, operation)
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

    /// Move this owned low-level table into a producer-rooted shared prepared
    /// device export with explicit event-ordered readiness.
    pub fn try_into_prepared_device_table_export_with_event(
        self,
        event: &CudaEvent,
    ) -> Result<PreparedTableExport<'static>, InteropError> {
        let operation = "interop::Table::try_into_prepared_device_table_export_with_event";
        let readiness = prepared_device_event_readiness(operation, self.execution_domain(), event)?.1;
        prepare_into_owned_table_root_with_readiness(self, None, operation, readiness, Some(event.clone()))
    }
}
