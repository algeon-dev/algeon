use std::sync::Arc;

use super::{
    Column, CudaEvent, DataFrame, InteropError, PreparedColumnExport, PreparedTableExport, Runtime, Series, Table,
    dataframe_import_names, prepare_device_table_export_from_columns,
    prepare_device_table_export_from_columns_with_readiness, prepare_from_owned_column, prepare_host_from_owned_column,
    prepare_host_table_export_from_columns, wrap_imported_dataframe, wrap_imported_series,
};

impl Series {
    /// Convenience consumer that imports a shared prepared export root into an
    /// unnamed high-level `Series`.
    pub fn try_import_from_prepared_column_export_into_runtime(
        export: &PreparedColumnExport<'_>,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        let operation = "interop::Series::try_import_from_prepared_column_export_into_runtime";
        let column = Column::try_import_from_prepared_column_export_into_runtime(export, runtime)?;
        wrap_imported_series(column, operation)
    }

    /// Convenience producer that projects this high-level `Series` into the
    /// shared prepared device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedColumnExport::try_activate_read_lease`].
    ///
    /// The export describes `self`'s device buffers and holds a shared
    /// `Arc<Column>` for its lifetime. The returned type still borrows `self`,
    /// so it cannot outlive that borrow. Returns an [`InteropError`] when a
    /// column's layout falls outside the prepared device profile, its buffers
    /// are not on a CUDA device, or the execution binding or producer readiness
    /// of `self`'s runtime domain is unavailable.
    pub fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>, InteropError> {
        prepare_from_owned_column(self.as_column(), "interop::Series::try_prepare_device_column_export")
            // SAFETY: the Arc owns the projected storage and readiness domain.
            .map(|export| unsafe { export.with_retained_owner(self.arc()) })
    }

    /// Convenience producer that projects this high-level `Series` into the
    /// shared prepared host export profile.
    ///
    /// This materializes the active logical range to owned host-visible bytes
    /// and therefore is not zero-copy.
    ///
    /// The `'static` export owns retained host copies and does not borrow `self`.
    /// Returns an [`InteropError`] when a column's type or layout is not
    /// supported by host export or its data cannot be copied to host memory.
    pub fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>, InteropError> {
        prepare_host_from_owned_column(self.as_column(), "interop::Series::try_prepare_host_column_export")
    }

    /// Convenience producer that projects this high-level `Series` into the
    /// shared prepared device export profile with producer event-ordered
    /// readiness.
    pub fn try_prepare_device_column_export_with_event<'a>(
        &'a self,
        event: &'a CudaEvent,
    ) -> Result<PreparedColumnExport<'a>, InteropError> {
        self.as_column()
            .try_prepare_device_column_export_with_event(event)
            // SAFETY: retain the column storage and the supplied readiness event.
            .map(|export| unsafe { export.with_retained_owner(Arc::new((self.arc(), event.clone()))) })
    }
}

impl DataFrame {
    /// Imports a shared prepared table export root into a high-level
    /// `DataFrame`, using `names` when supplied or schema names carried on the
    /// shared root otherwise.
    pub fn try_import_from_prepared_table_export_into_runtime(
        export: &PreparedTableExport<'_>,
        names: Option<Vec<String>>,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        let operation = "interop::DataFrame::try_import_from_prepared_table_export_into_runtime";
        let names = dataframe_import_names(names, export.table().names(), operation)?;
        let table = Table::try_import_from_prepared_table_export_into_runtime(export, runtime)?;
        wrap_imported_dataframe(table, names, operation)
    }

    /// Convenience producer that projects this high-level `DataFrame` into the
    /// shared prepared device export profile.
    ///
    /// The returned export is producer-rooted. Consumers activate read access
    /// later via [`PreparedTableExport::try_activate_read_lease`].
    ///
    /// The export describes `self`'s device buffers and holds a clone of `self`
    /// for its lifetime. The returned type still borrows `self`, so it cannot
    /// outlive that borrow. Returns an [`InteropError`] when a column's layout
    /// falls outside the prepared device profile, its buffers are not on a CUDA
    /// device, or the execution binding or producer readiness of `self`'s
    /// runtime domain is unavailable.
    pub fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>, InteropError> {
        prepare_device_table_export_from_columns(
            self.columns_map().values().map(AsRef::as_ref),
            self.num_rows(),
            Some(self.column_names_owned()),
            self.execution_domain(),
            "interop::DataFrame::try_prepare_device_table_export",
        )
        // SAFETY: the clone retains each column and its readiness domain.
        .map(|export| unsafe { export.with_retained_owner(Arc::new(self.clone())) })
    }

    /// Convenience producer that projects this high-level `DataFrame` into the
    /// shared prepared host export profile.
    ///
    /// This materializes each supported top-level column to owned host-visible
    /// bytes and therefore is not zero-copy.
    ///
    /// The `'static` export owns retained host copies and does not borrow `self`.
    /// Returns an [`InteropError`] when a column's type or layout is not
    /// supported by host export or its data cannot be copied to host memory.
    pub fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        prepare_host_table_export_from_columns(
            self.columns_map().values().map(AsRef::as_ref),
            self.num_rows(),
            Some(self.column_names_owned()),
            "interop::DataFrame::try_prepare_host_table_export",
        )
    }

    /// Convenience producer that projects this high-level `DataFrame` into the
    /// shared prepared device export profile with producer event-ordered
    /// readiness.
    pub fn try_prepare_device_table_export_with_event<'a>(
        &'a self,
        event: &'a CudaEvent,
    ) -> Result<PreparedTableExport<'a>, InteropError> {
        let operation = "interop::DataFrame::try_prepare_device_table_export_with_event";
        let context =
            super::table_export::DeviceTableExportContext::from_event(operation, self.execution_domain(), event)?;
        prepare_device_table_export_from_columns_with_readiness(
            self.columns_map().values().map(AsRef::as_ref),
            self.num_rows(),
            Some(self.column_names_owned()),
            context,
        )
        // SAFETY: retain all frame columns and the supplied readiness event.
        .map(|export| unsafe { export.with_retained_owner(Arc::new((self.clone(), event.clone()))) })
    }
}
