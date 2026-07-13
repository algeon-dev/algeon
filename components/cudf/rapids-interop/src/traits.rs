use crate::{
    BorrowedColumnView, BorrowedTableView, OwnedColumnTransfer, OwnedTableTransfer, PartitionedOwnedTableTransfer,
    PartitionedPreparedTableExport, PreparedColumnExport, PreparedExportProfile, PreparedTableExport, Result,
    validation::prepared_profile_error,
};

/// Fallible producer adapter for a borrowed column descriptor.
///
/// The returned descriptor lifetime is intentionally tied to the specific
/// borrow of `self`.
pub trait TryBorrowColumnView {
    fn try_borrow_column_view(&self) -> Result<BorrowedColumnView<'_>>;
}

/// Fallible producer adapter for a borrowed table descriptor.
///
/// The returned descriptor lifetime is intentionally tied to the specific
/// borrow of `self`.
pub trait TryBorrowTableView {
    fn try_borrow_table_view(&self) -> Result<BorrowedTableView<'_>>;
}

/// Fallible producer adapter for a prepared device column export.
pub trait TryPrepareDeviceColumnExport {
    fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>>;
}

/// Fallible producer adapter for a prepared device table export.
pub trait TryPrepareDeviceTableExport {
    fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>>;
}

/// Fallible producer adapter for a partitioned prepared device table export.
pub trait TryPrepareDevicePartitionedTableExport {
    fn try_prepare_device_partitioned_table_export(&self) -> Result<PartitionedPreparedTableExport<'_>>;
}

/// Fallible producer adapter for a prepared host table export.
pub trait TryPrepareHostTableExport {
    fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>>;
}

/// Fallible producer adapter for a prepared host column export.
pub trait TryPrepareHostColumnExport {
    fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>>;
}

/// Fallible producer adapter for an owned device column transfer.
pub trait TryIntoOwnedDeviceColumnTransfer: Sized {
    fn try_into_owned_device_column_transfer(self) -> Result<OwnedColumnTransfer>;
}

/// Fallible producer adapter for an owned device table transfer.
pub trait TryIntoOwnedDeviceTableTransfer: Sized {
    fn try_into_owned_device_table_transfer(self) -> Result<OwnedTableTransfer>;
}

/// Fallible producer adapter for a partitioned owned device table transfer.
pub trait TryIntoOwnedDevicePartitionedTableTransfer: Sized {
    fn try_into_owned_device_partitioned_table_transfer(self) -> Result<PartitionedOwnedTableTransfer>;
}

impl TryBorrowColumnView for BorrowedColumnView<'_> {
    fn try_borrow_column_view(&self) -> Result<BorrowedColumnView<'_>> {
        match self.column() {
            Some(column) => BorrowedColumnView::try_new_with_semantic_columns(
                column.clone(),
                self.semantic_columns().to_vec(),
                self.bindings().to_vec(),
                self.execution(),
            ),
            None => BorrowedColumnView::try_new(
                self.data_type(),
                self.len(),
                self.offset(),
                self.data_ptr(),
                *self.null_mask(),
                self.execution(),
            ),
        }
    }
}

impl TryBorrowTableView for BorrowedTableView<'_> {
    fn try_borrow_table_view(&self) -> Result<BorrowedTableView<'_>> {
        let columns =
            self.columns().iter().map(|column| column.try_borrow_column_view()).collect::<Result<Vec<_>>>()?;
        BorrowedTableView::try_new(self.num_rows(), columns, self.execution())
    }
}

impl TryPrepareDeviceColumnExport for PreparedColumnExport<'_> {
    fn try_prepare_device_column_export(&self) -> Result<PreparedColumnExport<'_>> {
        Ok(self.clone())
    }
}

impl TryPrepareDeviceTableExport for PreparedTableExport<'_> {
    fn try_prepare_device_table_export(&self) -> Result<PreparedTableExport<'_>> {
        Ok(self.clone())
    }
}

impl TryPrepareDevicePartitionedTableExport for PartitionedPreparedTableExport<'_> {
    fn try_prepare_device_partitioned_table_export(&self) -> Result<PartitionedPreparedTableExport<'_>> {
        Ok(self.clone())
    }
}

impl TryPrepareHostColumnExport for PreparedColumnExport<'static> {
    fn try_prepare_host_column_export(&self) -> Result<PreparedColumnExport<'static>> {
        if self.profile() != PreparedExportProfile::Host {
            return Err(prepared_profile_error(
                self.diagnostics().profile(),
                "prepared host export adapter requires prepared host export diagnostics",
            ));
        }
        Ok(self.clone())
    }
}

impl TryPrepareHostTableExport for PreparedTableExport<'static> {
    fn try_prepare_host_table_export(&self) -> Result<PreparedTableExport<'static>> {
        if self.profile() != PreparedExportProfile::Host {
            return Err(prepared_profile_error(
                self.diagnostics().profile(),
                "prepared host export adapter requires prepared host export diagnostics",
            ));
        }
        Ok(self.clone())
    }
}

impl TryIntoOwnedDeviceColumnTransfer for OwnedColumnTransfer {
    fn try_into_owned_device_column_transfer(self) -> Result<OwnedColumnTransfer> {
        Ok(self)
    }
}

impl TryIntoOwnedDeviceTableTransfer for OwnedTableTransfer {
    fn try_into_owned_device_table_transfer(self) -> Result<OwnedTableTransfer> {
        Ok(self)
    }
}

impl TryIntoOwnedDevicePartitionedTableTransfer for PartitionedOwnedTableTransfer {
    fn try_into_owned_device_partitioned_table_transfer(self) -> Result<PartitionedOwnedTableTransfer> {
        Ok(self)
    }
}
