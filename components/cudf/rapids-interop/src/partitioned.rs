use crate::{
    ActiveTableReadLease, CompletionAuthority, ExportProfile, IncompleteDropBehavior, InteropError, InteropErrorKind,
    LeaseCompletionResult, OwnedTableTransfer, PreparedTableExport, ReadStartAuthority, Result,
};

/// Consumer-activated read lease over one partition inside a partitioned
/// prepared table export.
#[derive(Debug)]
#[must_use = "lease should be completed explicitly via .complete(); dropping without completion follows the backend-defined incomplete-drop behavior"]
pub struct ActivePartitionedTableReadLease<'export, 'producer, 'consumer> {
    partition: &'export PartitionedPreparedTablePartition<'producer>,
    inner: ActiveTableReadLease<'export, 'producer, 'consumer>,
}

impl<'export, 'producer, 'consumer> ActivePartitionedTableReadLease<'export, 'producer, 'consumer> {
    /// Return the original top-level schema indices covered by this partition.
    pub fn top_level_column_indices(&self) -> &[usize] {
        self.partition.top_level_column_indices()
    }

    /// Borrow the single-device prepared export behind this partition lease.
    pub fn export(&self) -> &'export PreparedTableExport<'producer> {
        self.inner.export()
    }

    /// Return the read-start authority used to activate this partition lease.
    pub fn read_start(&self) -> ReadStartAuthority<'consumer> {
        self.inner.read_start()
    }

    /// Return the completion authority used to activate this partition lease.
    pub fn completion(&self) -> CompletionAuthority<'consumer> {
        self.inner.completion()
    }

    /// Return the incomplete-drop behavior for this partition lease.
    pub fn incomplete_drop_behavior(&self) -> IncompleteDropBehavior {
        self.inner.incomplete_drop_behavior()
    }

    /// Complete this partition lease and return the shared completion result.
    pub fn complete(self) -> Result<LeaseCompletionResult> {
        self.inner.complete()
    }
}

/// One partition inside a partitioned prepared table export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionedPreparedTablePartition<'a> {
    top_level_column_indices: Vec<usize>,
    export: PreparedTableExport<'a>,
}

impl<'a> PartitionedPreparedTablePartition<'a> {
    pub fn new(top_level_column_indices: Vec<usize>, export: PreparedTableExport<'a>) -> Self {
        Self { top_level_column_indices, export }
    }

    pub fn top_level_column_indices(&self) -> &[usize] {
        &self.top_level_column_indices
    }

    pub fn export(&self) -> &PreparedTableExport<'a> {
        &self.export
    }

    /// Activate a read lease for this partition without dropping down to the
    /// nested prepared export first.
    pub fn try_activate_read_lease<'export, 'consumer>(
        &'export self,
        read_start: ReadStartAuthority<'consumer>,
        completion: CompletionAuthority<'consumer>,
    ) -> Result<ActivePartitionedTableReadLease<'export, 'a, 'consumer>> {
        let inner = self.export.try_activate_read_lease(read_start, completion)?;
        Ok(ActivePartitionedTableReadLease { partition: self, inner })
    }
}

/// Partitioned multi-device prepared device export for one logical table
/// schema split across multiple single-device prepared table roots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionedPreparedTableExport<'a> {
    schema_names: Option<Vec<String>>,
    num_rows: usize,
    partitions: Vec<PartitionedPreparedTablePartition<'a>>,
}

impl<'a> PartitionedPreparedTableExport<'a> {
    pub fn try_new(
        schema_names: Option<Vec<String>>,
        num_rows: usize,
        partitions: Vec<PartitionedPreparedTablePartition<'a>>,
    ) -> Result<Self> {
        validate_partitioned_table_partitions(
            "partitioned prepared table export",
            ExportProfile::PreparedDeviceExport,
            schema_names.as_deref(),
            num_rows,
            &partitions,
            |partition| partition.export().diagnostics().profile(),
            |partition| partition.export().table().num_rows(),
            |partition| partition.export().table().columns().len(),
            |partition| partition.top_level_column_indices(),
        )?;

        Ok(Self { schema_names, num_rows, partitions })
    }

    pub fn schema_names(&self) -> Option<&[String]> {
        self.schema_names.as_deref()
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn partitions(&self) -> &[PartitionedPreparedTablePartition<'a>] {
        &self.partitions
    }
}

/// One partition inside a partitioned owned table transfer.
#[derive(Debug)]
pub struct PartitionedOwnedTableTransferPartition {
    top_level_column_indices: Vec<usize>,
    transfer: OwnedTableTransfer,
}

impl PartitionedOwnedTableTransferPartition {
    pub fn new(top_level_column_indices: Vec<usize>, transfer: OwnedTableTransfer) -> Self {
        Self { top_level_column_indices, transfer }
    }

    pub fn top_level_column_indices(&self) -> &[usize] {
        &self.top_level_column_indices
    }

    pub fn transfer(&self) -> &OwnedTableTransfer {
        &self.transfer
    }

    pub fn into_transfer(self) -> OwnedTableTransfer {
        self.transfer
    }
}

/// Partitioned multi-device owned device transfer for one logical table schema
/// split across multiple single-device ownership-transfer roots.
#[derive(Debug)]
pub struct PartitionedOwnedTableTransfer {
    schema_names: Option<Vec<String>>,
    num_rows: usize,
    partitions: Vec<PartitionedOwnedTableTransferPartition>,
}

impl PartitionedOwnedTableTransfer {
    pub fn try_new(
        schema_names: Option<Vec<String>>,
        num_rows: usize,
        partitions: Vec<PartitionedOwnedTableTransferPartition>,
    ) -> Result<Self> {
        validate_partitioned_table_partitions(
            "partitioned owned table transfer",
            ExportProfile::OwnedDeviceTransfer,
            schema_names.as_deref(),
            num_rows,
            &partitions,
            |partition| partition.transfer().diagnostics().profile(),
            |partition| partition.transfer().table().num_rows(),
            |partition| partition.transfer().table().columns().len(),
            |partition| partition.top_level_column_indices(),
        )?;

        Ok(Self { schema_names, num_rows, partitions })
    }

    pub fn schema_names(&self) -> Option<&[String]> {
        self.schema_names.as_deref()
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    pub fn partitions(&self) -> &[PartitionedOwnedTableTransferPartition] {
        &self.partitions
    }

    pub fn into_partitions(self) -> Vec<PartitionedOwnedTableTransferPartition> {
        self.partitions
    }
}

fn validate_partitioned_table_partitions<P>(
    kind: &'static str,
    expected_profile: ExportProfile,
    schema_names: Option<&[String]>,
    num_rows: usize,
    partitions: &[P],
    profile_of: impl Fn(&P) -> ExportProfile,
    num_rows_of: impl Fn(&P) -> usize,
    top_level_column_count_of: impl Fn(&P) -> usize,
    top_level_column_indices_of: impl Fn(&P) -> &[usize],
) -> Result<()> {
    if partitions.is_empty() {
        return Err(InteropError::new(
            InteropErrorKind::InvalidView,
            format!("{kind} must contain at least one partition"),
        ));
    }

    let partition_top_level_columns = partitions.iter().map(&top_level_column_count_of).sum::<usize>();
    let total_top_level_columns = schema_names.map_or(partition_top_level_columns, |names| names.len());
    if let Some(names) = schema_names
        && names.len() < partition_top_level_columns
    {
        return Err(InteropError::new(
            InteropErrorKind::InvalidView,
            format!("{kind} names must cover every partitioned top-level column"),
        )
        .with_context("names_len", names.len().to_string())
        .with_context("partition_top_level_columns", partition_top_level_columns.to_string()));
    }

    let mut seen = vec![false; total_top_level_columns];
    for (partition_index, partition) in partitions.iter().enumerate() {
        let profile = profile_of(partition);
        if profile != expected_profile {
            return Err(InteropError::new(
                InteropErrorKind::ProfileUnsupported,
                format!("{kind} requires every partition to use one shared export profile"),
            )
            .with_context("partition_index", partition_index.to_string())
            .with_context("expected_profile", format!("{expected_profile:?}"))
            .with_context("actual_profile", format!("{profile:?}")));
        }

        let partition_num_rows = num_rows_of(partition);
        if partition_num_rows != num_rows {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                format!("{kind} requires every partition to have the same row count"),
            )
            .with_context("partition_index", partition_index.to_string())
            .with_context("partition_num_rows", partition_num_rows.to_string())
            .with_context("num_rows", num_rows.to_string()));
        }

        let top_level_column_indices = top_level_column_indices_of(partition);
        if top_level_column_indices.is_empty() {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                format!("{kind} partitions must not be empty"),
            )
            .with_context("partition_index", partition_index.to_string()));
        }

        let expected_top_level_columns = top_level_column_count_of(partition);
        if top_level_column_indices.len() != expected_top_level_columns {
            return Err(InteropError::new(
                InteropErrorKind::InvalidView,
                format!("{kind} partition metadata must match the partition root column count"),
            )
            .with_context("partition_index", partition_index.to_string())
            .with_context("segment_columns", top_level_column_indices.len().to_string())
            .with_context("partition_columns", expected_top_level_columns.to_string()));
        }

        for &top_level_column_index in top_level_column_indices {
            if top_level_column_index >= total_top_level_columns {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    format!("{kind} top-level column index is out of range"),
                )
                .with_context("partition_index", partition_index.to_string())
                .with_context("top_level_column_index", top_level_column_index.to_string())
                .with_context("num_top_level_columns", total_top_level_columns.to_string()));
            }

            if seen[top_level_column_index] {
                return Err(InteropError::new(
                    InteropErrorKind::InvalidView,
                    format!("{kind} must not cover one top-level column more than once"),
                )
                .with_context("partition_index", partition_index.to_string())
                .with_context("top_level_column_index", top_level_column_index.to_string()));
            }

            seen[top_level_column_index] = true;
        }
    }

    if let Some((missing_index, _)) = seen.iter().enumerate().find(|(_, covered)| !**covered) {
        return Err(InteropError::new(
            InteropErrorKind::NotFound,
            format!("{kind} must cover every original top-level column exactly once"),
        )
        .with_context("missing_top_level_column_index", missing_index.to_string()));
    }

    Ok(())
}
