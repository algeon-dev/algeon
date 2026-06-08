use std::collections::BTreeSet;

use parquet::file::metadata::ParquetMetaData;

use crate::source::{LocalParquetFile, LocalParquetSource};

#[derive(Debug, Clone)]
pub(super) struct RowGroupMetadata {
    row_group_bytes: Option<Vec<u64>>,
    projected_compressed_bytes: Option<Vec<u64>>,
    projected_uncompressed_bytes: Option<Vec<u64>>,
    pure_filter_compressed_bytes: Option<Vec<u64>>,
    pure_filter_uncompressed_bytes: Option<Vec<u64>>,
    row_group_rows: Option<Vec<u64>>,
}

impl RowGroupMetadata {
    pub(super) fn row_group_count(&self) -> usize {
        self.row_group_bytes
            .as_ref()
            .or(self.projected_uncompressed_bytes.as_ref())
            .or(self.projected_compressed_bytes.as_ref())
            .or(self.pure_filter_uncompressed_bytes.as_ref())
            .or(self.pure_filter_compressed_bytes.as_ref())
            .or(self.row_group_rows.as_ref())
            .map_or(0, Vec::len)
    }

    pub(super) fn row_group_bytes(&self, row_group: usize) -> Option<u64> {
        self.row_group_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    pub(super) fn projected_compressed_bytes(&self, row_group: usize) -> Option<u64> {
        self.projected_compressed_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    pub(super) fn projected_uncompressed_bytes(&self, row_group: usize) -> Option<u64> {
        self.projected_uncompressed_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    pub(super) fn pure_filter_compressed_bytes(&self, row_group: usize) -> Option<u64> {
        self.pure_filter_compressed_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    pub(super) fn pure_filter_uncompressed_bytes(&self, row_group: usize) -> Option<u64> {
        self.pure_filter_uncompressed_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    pub(super) fn row_group_rows(&self, row_group: usize) -> Option<u64> {
        self.row_group_rows.as_ref().and_then(|rows| rows.get(row_group).copied())
    }

    pub(super) fn row_group_bytes_vec(&self) -> Option<Vec<u64>> {
        self.row_group_bytes.clone()
    }

    pub(super) fn projected_compressed_bytes_vec(&self) -> Option<Vec<u64>> {
        self.projected_compressed_bytes.clone()
    }

    pub(super) fn projected_uncompressed_bytes_vec(&self) -> Option<Vec<u64>> {
        self.projected_uncompressed_bytes.clone()
    }

    pub(super) fn pure_filter_compressed_bytes_vec(&self) -> Option<Vec<u64>> {
        self.pure_filter_compressed_bytes.clone()
    }

    pub(super) fn pure_filter_uncompressed_bytes_vec(&self) -> Option<Vec<u64>> {
        self.pure_filter_uncompressed_bytes.clone()
    }

    pub(super) fn row_group_rows_vec(&self) -> Option<Vec<u64>> {
        self.row_group_rows.clone()
    }
}

pub(super) fn row_group_metadata_from_footer(
    metadata: &ParquetMetaData,
    source: &LocalParquetSource,
) -> RowGroupMetadata {
    let read_column_indexes = source.read_column_file_indices.iter().copied().collect::<BTreeSet<_>>();
    let projected_column_indexes = source
        .projected_output_column_source_refs
        .iter()
        .filter_map(|source_ref| match source_ref {
            crate::source::SourceColumnRef::File(index) => Some(*index),
            crate::source::SourceColumnRef::Partition(_) | crate::source::SourceColumnRef::MissingNull(_) => None,
        })
        .collect::<BTreeSet<_>>();
    let pure_filter_column_indexes = source
        .pure_filter_column_source_refs
        .iter()
        .filter_map(|source_ref| match source_ref {
            crate::source::SourceColumnRef::File(index) if !projected_column_indexes.contains(index) => Some(*index),
            crate::source::SourceColumnRef::File(_)
            | crate::source::SourceColumnRef::Partition(_)
            | crate::source::SourceColumnRef::MissingNull(_) => None,
        })
        .collect::<BTreeSet<_>>();

    RowGroupMetadata {
        row_group_bytes: metadata
            .row_groups()
            .iter()
            .map(|row_group| row_group_column_bytes(row_group, &read_column_indexes, ColumnByteKind::Compressed))
            .collect::<Option<Vec<_>>>(),
        projected_compressed_bytes: metadata
            .row_groups()
            .iter()
            .map(|row_group| row_group_column_bytes(row_group, &projected_column_indexes, ColumnByteKind::Compressed))
            .collect::<Option<Vec<_>>>(),
        projected_uncompressed_bytes: metadata
            .row_groups()
            .iter()
            .map(|row_group| row_group_column_bytes(row_group, &projected_column_indexes, ColumnByteKind::Uncompressed))
            .collect::<Option<Vec<_>>>(),
        pure_filter_compressed_bytes: metadata
            .row_groups()
            .iter()
            .map(|row_group| row_group_column_bytes(row_group, &pure_filter_column_indexes, ColumnByteKind::Compressed))
            .collect::<Option<Vec<_>>>(),
        pure_filter_uncompressed_bytes: metadata
            .row_groups()
            .iter()
            .map(|row_group| {
                row_group_column_bytes(row_group, &pure_filter_column_indexes, ColumnByteKind::Uncompressed)
            })
            .collect::<Option<Vec<_>>>(),
        row_group_rows: metadata
            .row_groups()
            .iter()
            .map(|row_group| u64::try_from(row_group.num_rows()).ok())
            .collect::<Option<Vec<_>>>(),
    }
}

pub(super) fn source_row_group_metadata(
    file: &LocalParquetFile,
    footer_metadata: Option<RowGroupMetadata>,
) -> Option<RowGroupMetadata> {
    let row_group_bytes = file
        .row_group_bytes
        .clone()
        .or_else(|| footer_metadata.as_ref().and_then(RowGroupMetadata::row_group_bytes_vec));
    let projected_compressed_bytes = file
        .projected_row_group_compressed_bytes
        .clone()
        .or_else(|| footer_metadata.as_ref().and_then(RowGroupMetadata::projected_compressed_bytes_vec));
    let projected_uncompressed_bytes = file
        .projected_row_group_uncompressed_bytes
        .clone()
        .or_else(|| footer_metadata.as_ref().and_then(RowGroupMetadata::projected_uncompressed_bytes_vec));
    let pure_filter_compressed_bytes = file
        .pure_filter_row_group_compressed_bytes
        .clone()
        .or_else(|| footer_metadata.as_ref().and_then(RowGroupMetadata::pure_filter_compressed_bytes_vec));
    let pure_filter_uncompressed_bytes = file
        .pure_filter_row_group_uncompressed_bytes
        .clone()
        .or_else(|| footer_metadata.as_ref().and_then(RowGroupMetadata::pure_filter_uncompressed_bytes_vec));
    let row_group_rows =
        file.row_group_rows.clone().or_else(|| footer_metadata.as_ref().and_then(RowGroupMetadata::row_group_rows_vec));

    if row_group_bytes.is_some()
        || projected_compressed_bytes.is_some()
        || projected_uncompressed_bytes.is_some()
        || pure_filter_compressed_bytes.is_some()
        || pure_filter_uncompressed_bytes.is_some()
        || row_group_rows.is_some()
    {
        Some(RowGroupMetadata {
            row_group_bytes,
            projected_compressed_bytes,
            projected_uncompressed_bytes,
            pure_filter_compressed_bytes,
            pure_filter_uncompressed_bytes,
            row_group_rows,
        })
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy)]
enum ColumnByteKind {
    Compressed,
    Uncompressed,
}

fn row_group_column_bytes(
    row_group: &parquet::file::metadata::RowGroupMetaData,
    column_indexes: &BTreeSet<usize>,
    kind: ColumnByteKind,
) -> Option<u64> {
    column_indexes.iter().try_fold(0u64, |total, column_index| {
        let column = row_group.columns().get(*column_index)?;
        let bytes = match kind {
            ColumnByteKind::Compressed => column.compressed_size(),
            ColumnByteKind::Uncompressed => column.uncompressed_size(),
        };
        let bytes = u64::try_from(bytes).ok()?;
        total.checked_add(bytes)
    })
}
