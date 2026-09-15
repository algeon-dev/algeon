use std::sync::Arc;

use super::{
    column::{
        transfer_from_owned_column_with_runtime_release,
        transfer_from_owned_column_with_runtime_release_synchronization,
    },
    table_export::{
        transfer_from_owned_table_with_names_and_runtime_release,
        transfer_from_owned_table_with_names_and_runtime_release_synchronization,
    },
    *,
};

pub(crate) struct BorrowedPartitionColumnGroup<'a> {
    pub(crate) top_level_column_indices: Vec<usize>,
    pub(crate) columns: Vec<&'a Column>,
    pub(crate) domain: Arc<ExecutionDomain>,
}

pub(crate) struct OwnedPartitionColumnGroup {
    pub(crate) top_level_column_indices: Vec<usize>,
    pub(crate) columns: Vec<Column>,
    pub(crate) domain: Arc<ExecutionDomain>,
}

/// Interop-only high-level producer for heterogeneous retained-provenance
/// column sets that still share one row-count relationship.
///
/// Provenance here means the CUDA device and execution-domain lineage on
/// which each column's buffers were produced and remain valid.
///
/// Unlike [`crate::DataFrame`], this type does not claim one shared retained
/// provenance across all columns and does not participate in general dataframe
/// algebra. Its role is to drive partitioned device export, heterogeneous
/// host-visible export, and partitioned owned transfer honestly.
pub struct PartitionedDataFrameExport {
    columns: FastIndexMap<String, Column>,
    row_count: usize,
}

impl PartitionedDataFrameExport {
    /// Create a new heterogeneous interop producer from ordered column names
    /// and owned columns.
    pub fn new(names: Vec<String>, columns: Vec<Column>) -> crate::Result<Self> {
        if names.len() != columns.len() {
            return Err(crate::core::schema::column_name_count_mismatch_error(
                "interop::PartitionedDataFrameExport::new",
                "number of names must equal number of columns",
                names.len(),
                columns.len(),
            ));
        }

        validate_no_duplicate_names(&names, "interop::PartitionedDataFrameExport::new")?;

        if columns.is_empty() {
            return Ok(Self { columns: fast_index_map(), row_count: 0 });
        }

        let first_rows = columns[0].view()?.len()?;
        for (index, column) in columns.iter().enumerate().skip(1) {
            let rows = column.view()?.len()?;
            if rows != first_rows {
                return Err(crate::core::error::row_count_mismatch_error(
                    "interop::PartitionedDataFrameExport::new",
                    index,
                    rows,
                    first_rows,
                ));
            }
        }

        let mut map: FastIndexMap<String, Column> = fast_index_map_with_capacity(names.len());
        for (name, column) in names.into_iter().zip(columns) {
            map.insert(name, column);
        }

        Ok(Self { columns: map, row_count: first_rows })
    }

    /// Convenience constructor from a fixed-size array of `(name, Column)`
    /// pairs.
    pub fn from_columns<const N: usize>(pairs: [(&str, Column); N]) -> crate::Result<Self> {
        let (names, columns): (Vec<_>, Vec<_>) =
            pairs.into_iter().map(|(name, column)| (name.to_string(), column)).unzip();
        Self::new(names, columns)
    }

    pub fn num_rows(&self) -> usize {
        self.row_count
    }

    pub(crate) fn columns(&self) -> impl ExactSizeIterator<Item = &Column> {
        self.columns.values()
    }

    pub(crate) fn column_names_owned(&self) -> Vec<String> {
        self.columns.keys().cloned().collect()
    }

    pub(crate) fn grouped_borrowed_columns(&self) -> Vec<BorrowedPartitionColumnGroup<'_>> {
        let mut groups: Vec<BorrowedPartitionColumnGroup<'_>> = Vec::new();

        for (index, column) in self.columns.values().enumerate() {
            if let Some(group) = groups
                .iter_mut()
                .find(|group| ExecutionDomain::same_provenance(&group.domain, column.execution_domain()))
            {
                group.top_level_column_indices.push(index);
                group.columns.push(column);
            } else {
                groups.push(BorrowedPartitionColumnGroup {
                    top_level_column_indices: vec![index],
                    columns: vec![column],
                    domain: Arc::clone(column.execution_domain()),
                });
            }
        }

        groups
    }

    pub(crate) fn grouped_owned_columns(self) -> Vec<OwnedPartitionColumnGroup> {
        let mut groups: Vec<OwnedPartitionColumnGroup> = Vec::new();

        for (index, (_, column)) in self.columns.into_iter().enumerate() {
            if let Some(group) = groups
                .iter_mut()
                .find(|group| ExecutionDomain::same_provenance(&group.domain, column.execution_domain()))
            {
                group.top_level_column_indices.push(index);
                group.columns.push(column);
            } else {
                groups.push(OwnedPartitionColumnGroup {
                    top_level_column_indices: vec![index],
                    domain: Arc::clone(column.execution_domain()),
                    columns: vec![column],
                });
            }
        }

        groups
    }

    /// Project this heterogeneous producer into the shared RAPIDS partitioned
    /// prepared device export profile.
    ///
    /// The partitioned export borrows `self` and describes each partition's
    /// device buffers without taking ownership, so it cannot outlive that
    /// borrow. Columns are grouped into one partition per distinct CUDA
    /// execution-domain provenance, so mixed-provenance input produces multiple
    /// partition roots rather than being rejected. Returns an [`InteropError`]
    /// when a column's layout falls outside the prepared device profile, its
    /// buffers are not on a CUDA device, or the execution binding or producer
    /// readiness of any partition's runtime domain is unavailable.
    pub fn try_prepare_device_partitioned_table_export(
        &self,
    ) -> Result<PartitionedPreparedTableExport<'_>, InteropError> {
        prepare_partitioned_device_table_export_from_columns(
            self,
            "interop::PartitionedDataFrameExport::try_prepare_device_partitioned_table_export",
        )
    }

    /// Project this heterogeneous producer into one host-visible prepared
    /// table export root by materializing supported top-level columns to owned
    /// host bytes.
    ///
    /// The `'static` export owns retained host copies and does not borrow `self`.
    /// Returns an [`InteropError`] when a column's type or layout is not
    /// supported by host export or its data cannot be copied to host memory.
    pub fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>, InteropError> {
        prepare_partitioned_host_table_export_from_columns(
            self,
            "interop::PartitionedDataFrameExport::try_prepare_host_table_export",
        )
    }

    /// Move this heterogeneous producer into the shared RAPIDS partitioned
    /// owned device transfer profile.
    ///
    /// This preserves one owned-transfer root per retained-provenance
    /// partition. Each partition still follows the conservative default
    /// release contract unless you opt into the runtime-backed signal-release
    /// variant below.
    ///
    /// Consumes `self`: on success each partition's transfer root owns that
    /// partition's device buffers under the default owned-transfer release
    /// contract, and on error `self` has already been consumed. Returns an
    /// [`InteropError`] when a column's layout falls outside the owned device
    /// transfer profile, its buffers are not on a CUDA device, or the execution
    /// binding or producer readiness of any partition's runtime domain is
    /// unavailable.
    pub fn try_into_owned_transfer(self) -> Result<PartitionedOwnedTableTransfer, InteropError> {
        transfer_partitioned_owned_table_from_columns(
            self,
            "interop::PartitionedDataFrameExport::try_into_owned_device_partitioned_table_transfer",
        )
    }

    /// Move this heterogeneous producer into the shared RAPIDS partitioned
    /// owned device transfer profile plus one CUDA-backed releaser per
    /// retained-provenance partition for signal-driven release.
    ///
    /// Each partition retains its own release contract and runtime-backed
    /// releaser. Calling a partition releaser accepts release and schedules
    /// any required consumer-signal wait onto a background runtime job
    /// instead of blocking the caller thread.
    ///
    /// Ownership remains exclusive per partition: importing one partition back
    /// into a native `cudf` `Table` claims that partition's ownership and
    /// invalidates its matching releaser, while releasing one partition makes
    /// later native import of that same partition fail.
    pub fn try_into_owned_transfer_with_release(
        self,
    ) -> Result<(PartitionedOwnedTableTransfer, PartitionedOwnedTableTransferReleaser), InteropError> {
        transfer_partitioned_owned_table_from_columns_with_runtime_release(
            self,
            "interop::PartitionedDataFrameExport::try_into_owned_transfer_with_release",
        )
    }

    /// Move this heterogeneous producer into the shared RAPIDS partitioned
    /// owned device transfer profile plus one CUDA-backed releaser per
    /// retained-provenance partition with an explicit release-synchronization
    /// contract.
    ///
    /// Each partition always declares
    /// [`TransferCompletionRule::ReleaseWaitsOnConsumerSignal`]. Pass
    /// [`ReleaseSynchronization::Asynchronous`] when release should return
    /// after acceptance and background scheduling, or
    /// [`ReleaseSynchronization::MaySynchronizeInternally`] when blocking the
    /// caller thread is acceptable. Caller-managed synchronization remains
    /// unsupported for this runtime-backed producer path.
    pub fn try_into_owned_transfer_with_release_synchronization(
        self,
        synchronization: ReleaseSynchronization,
    ) -> Result<(PartitionedOwnedTableTransfer, PartitionedOwnedTableTransferReleaser), InteropError> {
        transfer_partitioned_owned_table_from_columns_with_runtime_release_synchronization(
            self,
            synchronization,
            "interop::PartitionedDataFrameExport::try_into_owned_transfer_with_release_synchronization",
        )
    }
}

impl Series {
    /// Convenience consumer that deep-copies a shared borrowed interop column
    /// descriptor into an unnamed high-level `Series`.
    pub fn try_import_from_borrowed_column_view_into_runtime(
        borrowed: &BorrowedColumnView<'_>,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        let operation = "interop::Series::try_import_from_borrowed_column_view_into_runtime";
        let column = Column::try_import_from_borrowed_column_view_into_runtime(borrowed, runtime)?;
        wrap_imported_series(column, operation)
    }

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

    /// Convenience consumer that zero-copy imports a shared owned device
    /// transfer root into an unnamed high-level `Series` inside the provided
    /// explicit runtime.
    ///
    /// This inherits the low-level `Column` import contract: the destination
    /// runtime must be pinned to the transfer device, and the imported series
    /// is rebound to that runtime's retained provenance without a deep copy.
    pub fn try_import_from_owned_device_column_transfer_into_runtime(
        transfer: OwnedColumnTransfer,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        let operation = "interop::Series::try_import_from_owned_device_column_transfer_into_runtime";
        let column = Column::try_import_from_owned_device_column_transfer_into_runtime(transfer, runtime)?;
        wrap_imported_series(column, operation)
    }

    /// Convenience producer that lends this high-level `Series` as a shared
    /// borrowed interop column view without host copies on the input path.
    ///
    /// `Series` is a convenience adapter, not the ABI itself. The returned
    /// descriptor remains borrowed-only, does not transfer ownership, and is
    /// limited to the current borrowed compatibility profile's fixed-width,
    /// `STRING`, `LIST`, `STRUCT`, and `DICTIONARY32` support.
    ///
    /// The descriptor borrows `self` and must not outlive that borrow. Returns an
    /// [`InteropError`] when the execution binding of `self`'s runtime domain is
    /// unavailable or when a column's type or layout falls outside the borrowed
    /// compatibility profile.
    pub fn try_borrow_column_view(&self) -> Result<BorrowedColumnView<'_>, InteropError> {
        borrow_from_owned_column(self.as_column(), "interop::Series::try_borrow_column_view")
    }

    /// Convenience producer that projects this high-level `Series` into the
    /// shared prepared device export profile.
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
        prepare_from_owned_column(self.as_column(), "interop::Series::try_prepare_device_column_export")
            .map(|export| export.with_retained_owner_arc(self.arc()))
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

    /// Convenience producer that moves this high-level `Series` into the
    /// shared owned device transfer profile.
    ///
    /// This is zero-copy when the inner `Arc<Column>` is uniquely owned. When
    /// the column is still shared by other `Series` or `DataFrame` values, the
    /// producer first deep-copies into one independent owned `Column` inside
    /// the same retained provenance, then transfers that ownership.
    ///
    /// Shared owned-transfer roots do not carry a schema-name fact, so the
    /// `Series` name is not exported.
    ///
    /// Consumes `self`: on success the transfer root owns the device buffers
    /// under the default owned-transfer release contract, and on error `self`
    /// has already been consumed. Returns an [`InteropError`] when a column's
    /// layout falls outside the owned device transfer profile, its buffers are
    /// not on a CUDA device, or the execution binding or producer readiness of
    /// `self`'s runtime domain is unavailable.
    pub fn try_into_owned_transfer(self) -> Result<OwnedColumnTransfer, InteropError> {
        let operation = "interop::Series::try_into_owned_device_column_transfer";
        let column = into_owned_column_for_transfer(self, operation)?;
        transfer_from_owned_column(column, operation)
    }

    /// Convenience producer that moves this high-level `Series` into the
    /// shared owned device transfer profile plus a CUDA-backed releaser for
    /// signal-driven release.
    ///
    /// The returned releaser accepts release and schedules any required
    /// consumer-signal wait onto a background runtime job instead of blocking
    /// the caller thread.
    pub fn try_into_owned_transfer_with_release(
        self,
    ) -> Result<(OwnedColumnTransfer, OwnedColumnTransferReleaser), InteropError> {
        let operation = "interop::Series::try_into_owned_transfer_with_release";
        let column = into_owned_column_for_transfer(self, operation)?;
        transfer_from_owned_column_with_runtime_release(column, operation)
    }

    /// Convenience producer that moves this high-level `Series` into the
    /// shared owned device transfer profile plus a CUDA-backed releaser for
    /// signal-driven release with an explicit release-synchronization
    /// contract.
    pub fn try_into_owned_transfer_with_release_synchronization(
        self,
        synchronization: ReleaseSynchronization,
    ) -> Result<(OwnedColumnTransfer, OwnedColumnTransferReleaser), InteropError> {
        let operation = "interop::Series::try_into_owned_transfer_with_release_synchronization";
        let column = into_owned_column_for_transfer(self, operation)?;
        transfer_from_owned_column_with_runtime_release_synchronization(column, synchronization, operation)
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
            .map(|export| export.with_retained_owner_arc(self.arc()))
    }
}

impl DataFrame {
    /// Imports a borrowed table view into a high-level `DataFrame`.
    ///
    /// Borrowed table views do not carry schema names, so `names` must be `Some`.
    pub fn try_import_from_borrowed_table_view_into_runtime(
        borrowed: &BorrowedTableView<'_>,
        names: Option<Vec<String>>,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        let operation = "interop::DataFrame::try_import_from_borrowed_table_view_into_runtime";
        let names = dataframe_import_names(names, None, operation)?;
        let table = Table::try_import_from_borrowed_table_view_into_runtime(borrowed, runtime)?;
        wrap_imported_dataframe(table, names, operation)
    }

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

    /// Zero-copy imports a shared owned device transfer root into a high-level
    /// `DataFrame`, using `names` when supplied or schema names carried on the
    /// shared root otherwise.
    ///
    /// This inherits the low-level `Table` import contract: the destination
    /// runtime must be pinned to the transfer device, and the imported frame
    /// is rebound to that runtime's retained provenance without a deep copy.
    pub fn try_import_from_owned_device_table_transfer_into_runtime(
        transfer: OwnedTableTransfer,
        names: Option<Vec<String>>,
        runtime: &Runtime,
    ) -> Result<Self, InteropError> {
        let operation = "interop::DataFrame::try_import_from_owned_device_table_transfer_into_runtime";
        let names = dataframe_import_names(names, transfer.table().names(), operation)?;
        let table = Table::try_import_from_owned_device_table_transfer_into_runtime(transfer, runtime)?;
        wrap_imported_dataframe(table, names, operation)
    }

    /// Convenience producer that lends this high-level `DataFrame` as a shared
    /// borrowed interop table view without host copies on the input path.
    ///
    /// `DataFrame` is a convenience adapter, not the ABI itself. The returned
    /// descriptor remains borrowed-only, does not transfer ownership, and does
    /// not promise zero-copy output for downstream consumer algorithms.
    ///
    /// The descriptor borrows `self` and must not outlive that borrow. Returns an
    /// [`InteropError`] when the execution binding of `self`'s runtime domain is
    /// unavailable or when a column's type or layout falls outside the borrowed
    /// compatibility profile.
    pub fn try_borrow_table_view(&self) -> Result<BorrowedTableView<'_>, InteropError> {
        let execution = execution_binding("interop::DataFrame::try_borrow_table_view", self.execution_domain())?;
        let columns = self
            .columns_map()
            .values()
            .map(|column| borrow_from_owned_column(column.as_ref(), "interop::DataFrame::try_borrow_table_view"))
            .collect::<Result<Vec<_>, _>>()?;
        BorrowedTableView::try_new(self.num_rows(), columns, execution)
            .map_err(|error| error.with_context("operation", "interop::DataFrame::try_borrow_table_view"))
    }

    /// Convenience producer that projects this high-level `DataFrame` into the
    /// shared prepared device export profile.
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
        prepare_device_table_export_from_columns(
            self.columns_map().values().map(AsRef::as_ref),
            self.num_rows(),
            Some(self.column_names_owned()),
            self.execution_domain(),
            "interop::DataFrame::try_prepare_device_table_export",
        )
        .map(|export| export.with_retained_owner_arc(Arc::new(self.clone())))
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

    /// Convenience producer that moves this high-level `DataFrame` into the
    /// shared owned device transfer profile.
    ///
    /// This is zero-copy when every inner `Arc<Column>` is uniquely owned.
    /// Shared columns are first deep-copied into one independent owned
    /// `Table` inside the same retained provenance so the transfer root has a
    /// real ownership boundary.
    ///
    /// Unlike low-level `Table` owned transfer, this high-level producer
    /// preserves the `DataFrame` schema names on the shared root.
    ///
    /// Consumes `self`: on success the transfer root owns the device buffers
    /// under the default owned-transfer release contract, and on error `self`
    /// has already been consumed. Returns an [`InteropError`] when a column's
    /// layout falls outside the owned device transfer profile, its buffers are
    /// not on a CUDA device, or the execution binding or producer readiness of
    /// `self`'s runtime domain is unavailable.
    pub fn try_into_owned_transfer(self) -> Result<OwnedTableTransfer, InteropError> {
        let operation = "interop::DataFrame::try_into_owned_device_table_transfer";
        let (table, names) = into_owned_table_for_transfer(self, operation)?;
        transfer_from_owned_table_with_names(table, Some(names), operation)
    }

    /// Convenience producer that moves this high-level `DataFrame` into the
    /// shared owned device transfer profile plus a CUDA-backed releaser for
    /// signal-driven release.
    ///
    /// The returned releaser accepts release and schedules any required
    /// consumer-signal wait onto a background runtime job instead of blocking
    /// the caller thread.
    pub fn try_into_owned_transfer_with_release(
        self,
    ) -> Result<(OwnedTableTransfer, OwnedTableTransferReleaser), InteropError> {
        let operation = "interop::DataFrame::try_into_owned_transfer_with_release";
        let (table, names) = into_owned_table_for_transfer(self, operation)?;
        transfer_from_owned_table_with_names_and_runtime_release(table, Some(names), operation)
    }

    /// Convenience producer that moves this high-level `DataFrame` into the
    /// shared owned device transfer profile plus a CUDA-backed releaser for
    /// signal-driven release with an explicit release-synchronization
    /// contract.
    pub fn try_into_owned_transfer_with_release_synchronization(
        self,
        synchronization: ReleaseSynchronization,
    ) -> Result<(OwnedTableTransfer, OwnedTableTransferReleaser), InteropError> {
        let operation = "interop::DataFrame::try_into_owned_transfer_with_release_synchronization";
        let (table, names) = into_owned_table_for_transfer(self, operation)?;
        transfer_from_owned_table_with_names_and_runtime_release_synchronization(
            table,
            Some(names),
            synchronization,
            operation,
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
        .map(|export| export.with_retained_owner_arc(Arc::new(self.clone())))
    }
}
