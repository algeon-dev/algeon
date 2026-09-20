use std::collections::BTreeSet;

use parquet::file::metadata::ParquetMetaData;

use crate::{
    parquet_stats::{ColumnByteKind, row_group_column_bytes},
    source::{LocalParquetFile, LocalParquetSource, SourceColumnRef},
};

// Per-row-group sizing evidence merged from source facts and optional footer reads.
#[derive(Debug, Clone)]
#[doc(hidden)]
pub struct RowGroupMetadata {
    row_group_bytes: Option<Vec<u64>>,
    projected_compressed_bytes: Option<Vec<u64>>,
    projected_uncompressed_bytes: Option<Vec<u64>>,
    pure_filter_compressed_bytes: Option<Vec<u64>>,
    pure_filter_uncompressed_bytes: Option<Vec<u64>>,
    row_group_rows: Option<Vec<u64>>,
}

impl RowGroupMetadata {
    pub(crate) fn row_group_count(&self) -> usize {
        self.row_group_bytes
            .as_ref()
            .or(self.projected_uncompressed_bytes.as_ref())
            .or(self.projected_compressed_bytes.as_ref())
            .or(self.pure_filter_uncompressed_bytes.as_ref())
            .or(self.pure_filter_compressed_bytes.as_ref())
            .or(self.row_group_rows.as_ref())
            .map_or(0, Vec::len)
    }

    pub(crate) fn row_group_bytes(&self, row_group: usize) -> Option<u64> {
        self.row_group_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    pub(crate) fn projected_compressed_bytes(&self, row_group: usize) -> Option<u64> {
        self.projected_compressed_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    pub(crate) fn projected_uncompressed_bytes(&self, row_group: usize) -> Option<u64> {
        self.projected_uncompressed_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    pub(crate) fn pure_filter_compressed_bytes(&self, row_group: usize) -> Option<u64> {
        self.pure_filter_compressed_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    pub(crate) fn pure_filter_uncompressed_bytes(&self, row_group: usize) -> Option<u64> {
        self.pure_filter_uncompressed_bytes.as_ref().and_then(|bytes| bytes.get(row_group).copied())
    }

    pub(crate) fn row_group_rows(&self, row_group: usize) -> Option<u64> {
        self.row_group_rows.as_ref().and_then(|rows| rows.get(row_group).copied())
    }

    pub(crate) fn row_group_bytes_vec(&self) -> Option<Vec<u64>> {
        self.row_group_bytes.clone()
    }

    pub(crate) fn projected_compressed_bytes_vec(&self) -> Option<Vec<u64>> {
        self.projected_compressed_bytes.clone()
    }

    pub(crate) fn projected_uncompressed_bytes_vec(&self) -> Option<Vec<u64>> {
        self.projected_uncompressed_bytes.clone()
    }

    pub(crate) fn pure_filter_compressed_bytes_vec(&self) -> Option<Vec<u64>> {
        self.pure_filter_compressed_bytes.clone()
    }

    pub(crate) fn pure_filter_uncompressed_bytes_vec(&self) -> Option<Vec<u64>> {
        self.pure_filter_uncompressed_bytes.clone()
    }

    pub(crate) fn row_group_rows_vec(&self) -> Option<Vec<u64>> {
        self.row_group_rows.clone()
    }
}

#[doc(hidden)]
pub fn row_group_metadata_from_footer(metadata: &ParquetMetaData, source: &LocalParquetSource) -> RowGroupMetadata {
    let read_column_indexes = source.read_column_file_indices.iter().copied().collect::<BTreeSet<_>>();
    let projected_column_indexes = source
        .projected_output_column_source_refs
        .iter()
        .filter_map(|source_ref| match source_ref {
            SourceColumnRef::File(index) => Some(*index),
            SourceColumnRef::Partition(_) | SourceColumnRef::MissingNull(_) => None,
        })
        .collect::<BTreeSet<_>>();
    let pure_filter_column_indexes = source
        .pure_filter_column_source_refs
        .iter()
        .filter_map(|source_ref| match source_ref {
            SourceColumnRef::File(index) if !projected_column_indexes.contains(index) => Some(*index),
            SourceColumnRef::File(_) | SourceColumnRef::Partition(_) | SourceColumnRef::MissingNull(_) => None,
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

#[doc(hidden)]
pub fn source_row_group_metadata(
    file: &LocalParquetFile,
    footer_metadata: Option<RowGroupMetadata>,
) -> Option<RowGroupMetadata> {
    let footer_row_group_count =
        footer_metadata.as_ref().map(RowGroupMetadata::row_group_count).filter(|count| *count > 0);
    let row_group_bytes = merge_row_group_values(
        file.row_group_bytes.clone(),
        footer_metadata.as_ref().and_then(RowGroupMetadata::row_group_bytes_vec),
        footer_row_group_count,
    );
    let projected_compressed_bytes = merge_row_group_values(
        file.projected_row_group_compressed_bytes.clone(),
        footer_metadata.as_ref().and_then(RowGroupMetadata::projected_compressed_bytes_vec),
        footer_row_group_count,
    );
    let projected_uncompressed_bytes = merge_row_group_values(
        file.projected_row_group_uncompressed_bytes.clone(),
        footer_metadata.as_ref().and_then(RowGroupMetadata::projected_uncompressed_bytes_vec),
        footer_row_group_count,
    );
    let pure_filter_compressed_bytes = merge_row_group_values(
        file.pure_filter_row_group_compressed_bytes.clone(),
        footer_metadata.as_ref().and_then(RowGroupMetadata::pure_filter_compressed_bytes_vec),
        footer_row_group_count,
    );
    let pure_filter_uncompressed_bytes = merge_row_group_values(
        file.pure_filter_row_group_uncompressed_bytes.clone(),
        footer_metadata.as_ref().and_then(RowGroupMetadata::pure_filter_uncompressed_bytes_vec),
        footer_row_group_count,
    );
    let row_group_rows = merge_row_group_values(
        file.row_group_rows.clone(),
        footer_metadata.as_ref().and_then(RowGroupMetadata::row_group_rows_vec),
        footer_row_group_count,
    );

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

fn merge_row_group_values(
    file_values: Option<Vec<u64>>,
    footer_values: Option<Vec<u64>>,
    footer_row_group_count: Option<usize>,
) -> Option<Vec<u64>> {
    match footer_row_group_count {
        Some(count) => match file_values {
            Some(values) if values.len() == count => Some(values),
            Some(_) => footer_values,
            None => footer_values,
        },
        None => file_values.or(footer_values),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::LocalParquetFile;

    fn footer_metadata_with_row_groups(rows: Vec<u64>) -> RowGroupMetadata {
        let bytes = vec![0; rows.len()];
        RowGroupMetadata {
            row_group_bytes: Some(bytes.clone()),
            projected_compressed_bytes: Some(bytes.clone()),
            projected_uncompressed_bytes: Some(bytes.clone()),
            pure_filter_compressed_bytes: Some(bytes.clone()),
            pure_filter_uncompressed_bytes: Some(bytes),
            row_group_rows: Some(rows),
        }
    }

    #[test]
    fn source_row_group_metadata_prefers_footer_rows_when_file_count_is_whole_file() {
        let file = LocalParquetFile::object_store_uri("s3://bucket/table/part.parquet")
            .expect("valid object-store URI")
            .with_row_group_rows(vec![35]);
        let footer = footer_metadata_with_row_groups(vec![10, 20, 5]);

        let metadata = source_row_group_metadata(&file, Some(footer)).expect("merged metadata");

        assert_eq!(metadata.row_group_count(), 3);
        assert_eq!(metadata.row_group_rows_vec().as_deref(), Some(&[10, 20, 5][..]));
    }

    #[test]
    fn source_row_group_metadata_keeps_file_rows_without_footer_layout() {
        let file = LocalParquetFile::object_store_uri("s3://bucket/table/part.parquet")
            .expect("valid object-store URI")
            .with_row_group_rows(vec![35]);

        let metadata = source_row_group_metadata(&file, None).expect("synthetic metadata");

        assert_eq!(metadata.row_group_count(), 1);
        assert_eq!(metadata.row_group_rows_vec().as_deref(), Some(&[35][..]));
    }
}
