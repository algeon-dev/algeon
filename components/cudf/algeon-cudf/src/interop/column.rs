use std::sync::Arc;

use super::{
    Column, ColumnView, CudaEvent, InteropError, InteropErrorKind, PreparedAccess, PreparedColumnExport,
    ProducerReadiness, Runtime, import_column_from_prepared_export, map_cudf_error,
    prepared_column_projection_from_view, prepared_device_event_readiness, prepared_device_readiness,
    prepared_host_column_projection_from_view, runtime_lease, to_static_bindings, to_static_readiness,
};

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
    PreparedColumnExport::try_new(
        projection.column.id(),
        projection.semantic_columns,
        projection.bindings,
        PreparedAccess::Device { readiness, backend: runtime_lease::shared_cuda_lease_backend() },
    )
    .map_err(|error| error.with_context("operation", operation))
}

fn prepare_host_from_column_view(
    view: &ColumnView<'_>,
    operation: &'static str,
) -> Result<PreparedColumnExport<'static>, InteropError> {
    let projection = prepared_host_column_projection_from_view(view, operation, 0, 0)?;
    PreparedColumnExport::try_new(
        projection.column.id(),
        projection.semantic_columns,
        projection.bindings,
        PreparedAccess::Host,
    )
    .map(|export| export.with_retained_owner(Arc::new(projection.retained_host_buffers)))
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
    let default_device = retained_column.execution_domain().device_id().ordinal() as i32;
    let projection = prepared_column_projection_from_view(&view, operation, default_device, 0, 0)?;
    PreparedColumnExport::try_new(
        projection.column.id(),
        projection.semantic_columns,
        to_static_bindings(operation, projection.bindings)?,
        PreparedAccess::Device {
            readiness: to_static_readiness(operation, readiness)?,
            backend: runtime_lease::shared_cuda_lease_backend(),
        },
    )
    .map(|export| {
        match retained_event {
            // `to_static_readiness` stores only the raw CUDA event pointer, so
            // the export must co-own the event for its whole lifetime.
            Some(event) => export.with_retained_owner(Arc::new((Arc::clone(&retained_column), event))),
            None => export.with_retained_owner(Arc::<Column>::clone(&retained_column)),
        }
    })
    .map_err(|error| error.with_context("operation", operation))
}

pub(super) fn prepare_from_owned_column<'a>(
    column: &'a Column,
    operation: &'static str,
) -> Result<PreparedColumnExport<'a>, InteropError> {
    let view = column.view().map_err(|error| map_cudf_error(InteropErrorKind::InvalidView, operation, error))?;
    let (default_device, readiness) = prepared_device_readiness(operation, column.execution_domain())?;
    let projection = prepared_column_projection_from_view(&view, operation, default_device, 0, 0)?;
    PreparedColumnExport::try_new(
        projection.column.id(),
        projection.semantic_columns,
        projection.bindings,
        PreparedAccess::Device { readiness, backend: runtime_lease::shared_cuda_lease_backend() },
    )
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
    PreparedColumnExport::try_new(
        projection.column.id(),
        projection.semantic_columns,
        projection.bindings,
        PreparedAccess::Host,
    )
    .map(|export| export.with_retained_owner(Arc::new(projection.retained_host_buffers)))
    .map_err(|error| error.with_context("operation", operation))
}

impl ColumnView<'_> {
    /// Project this low-level borrowed column view into the shared RAPIDS
    /// prepared device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedColumnExport::try_activate_read_lease`].
    ///
    /// The export describes `self`'s device buffers without taking ownership, so
    /// it cannot outlive that borrow. Returns an [`InteropError`] when a column's
    /// layout falls outside the prepared device profile, its buffers are not on a
    /// CUDA device, or the execution binding or producer readiness of `self`'s
    /// runtime domain is unavailable.
    pub fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>, InteropError> {
        prepare_from_column_view(self, "interop::ColumnView::try_prepare_device_column_export")
    }

    /// Project this low-level borrowed column view into the shared RAPIDS
    /// prepared host export profile.
    ///
    /// This materializes the active logical range to owned host-visible bytes
    /// and therefore is not zero-copy.
    ///
    /// The `'static` export owns retained host copies and does not borrow `self`.
    /// Returns an [`InteropError`] when a column's type or layout is not
    /// supported by host export or its data cannot be copied to host memory.
    pub fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>, InteropError> {
        prepare_host_from_column_view(self, "interop::ColumnView::try_prepare_host_column_export")
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

impl Column {
    /// Deep-copy a shared prepared export root into an owned
    /// `cudf` column inside the provided explicit runtime.
    ///
    /// Prepared device exports activate on the destination runtime's CUDA
    /// stream and perform a same-device device-to-device copy. Prepared host
    /// exports activate synchronously by return and upload their host-visible
    /// bytes into new device ownership inside the destination runtime. Both
    /// paths synchronize completion before returning. The producer's liveness
    /// can end once the call returns.
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

    /// Project this owned low-level column into the shared RAPIDS prepared
    /// device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedColumnExport::try_activate_read_lease`].
    ///
    /// The export describes `self`'s device buffers without taking ownership, so
    /// it cannot outlive that borrow. Returns an [`InteropError`] when a column's
    /// layout falls outside the prepared device profile, its buffers are not on a
    /// CUDA device, or the execution binding or producer readiness of `self`'s
    /// runtime domain is unavailable.
    pub fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>, InteropError> {
        prepare_from_owned_column(self, "interop::Column::try_prepare_device_column_export")
    }

    /// Move this owned low-level column into a producer-rooted shared prepared
    /// device export.
    ///
    /// Unlike the borrow-preserving [`Column::try_prepare_device_column_export`]
    /// adapter, this consuming variant upgrades the column itself into the
    /// prepared-export root. The returned export carries retained owner
    /// state that keeps the column alive. Downstream consumers can defer
    /// completion without widening lifetimes through borrowed aliases.
    pub fn try_into_prepared_device_column_export(self) -> Result<PreparedColumnExport<'static>, InteropError> {
        prepare_into_owned_column_root(self, "interop::Column::try_into_prepared_device_column_export")
    }

    /// Project this owned low-level column into the shared RAPIDS prepared
    /// host export profile.
    ///
    /// This materializes the active logical range to owned host-visible bytes
    /// and therefore is not zero-copy.
    ///
    /// The `'static` export owns retained host copies and does not borrow `self`.
    /// Returns an [`InteropError`] when a column's type or layout is not
    /// supported by host export or its data cannot be copied to host memory.
    pub fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>, InteropError> {
        prepare_host_from_owned_column(self, "interop::Column::try_prepare_host_column_export")
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
        PreparedColumnExport::try_new(
            projection.column.id(),
            projection.semantic_columns,
            projection.bindings,
            PreparedAccess::Device { readiness, backend: runtime_lease::shared_cuda_lease_backend() },
        )
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
