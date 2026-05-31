//! Footer-first ranged Parquet fetch used when a source carries a `scan_limit`.
//!
//! For `SELECT ... LIMIT N` over remote S3 Iceberg/Parquet, the legacy fetch
//! path issues a single full-object `GET`, transfers the entire file, and only
//! then trims to `N` rows on the device. For multi-GiB single-file Parquet
//! tables (e.g. SF100 `customer` ≈ 1.2 GiB) that pulls the whole file just to
//! return a handful of rows.
//!
//! This module fetches only the selected row groups, at Parquet row-group
//! granularity:
//!
//! 1. Two `ObjectStore::get_range` calls pull the tail of the file: the last
//!    `FOOTER_SIZE` bytes give the metadata length, then the metadata range
//!    itself is fetched.
//! 2. The footer is parsed with the `parquet` crate to obtain row-group
//!    counts, column-chunk byte ranges, and the file schema.
//! 3. The minimum prefix of row groups that covers `scan_limit` rows, or an
//!    explicit row-group subset selected by bounded chunk planning, is selected.
//! 4. The byte ranges spanning those row groups are fetched with
//!    `ObjectStore::get_ranges`.
//! 5. A compact, legal Parquet file is rewritten in memory: magic header,
//!    selected row-group bytes laid out back-to-back, and a regenerated footer
//!    whose column-chunk offsets point at the new buffer layout. Bloom filter,
//!    column index, and offset index references are cleared because the
//!    payload bytes for those structures are not fetched.
//!
//! libcudf's host-buffer reader requires a complete legal Parquet file
//! (`cudf-rs/cudf-rs/src/frame/dataframe/io.rs` docstring + the libcudf C++
//! reader parses the footer in-buffer to locate row-group offsets). Splicing
//! "footer + sparse row-group bytes" into a zero-padded buffer does not work:
//! libcudf would either read garbage or crash. The mini-Parquet rewrite is
//! what makes the ranged fetch safe to feed back into the existing reader
//! contract.

use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use object_store::{ObjectStore, ObjectStoreExt, path::Path as ObjectStorePath};
use parquet::file::{
    FOOTER_SIZE,
    metadata::{
        ColumnChunkMetaData, FooterTail, ParquetMetaDataBuilder, ParquetMetaDataReader, ParquetMetaDataWriter,
        RowGroupMetaData,
    },
};

use crate::error::{QueryError, QueryErrorCode, Result};

const PARQUET_MAGIC: &[u8; 4] = b"PAR1";
/// Speculative tail size for the first ranged GET. Large enough to cover the
/// footer of any practical Parquet file in a single round trip while still
/// being a tiny fraction of even a small file.
const FOOTER_PREFETCH_BYTES: u64 = 64 * 1024;

/// Selector for the row-group subset to fetch.
#[derive(Debug, Clone)]
pub(super) enum RangedRowGroupSelector {
    /// Fetch the minimal row-group prefix that covers a row scan limit.
    ScanLimit(u64),
    /// Fetch a caller-chosen set of row groups, in the order given.
    Explicit(Vec<usize>),
}

/// A compact, legal Parquet payload derived from one object-store Parquet file.
///
/// `mini_*` metadata describes row groups inside `bytes`.
/// `selected_original_row_groups` is diagnostic/mapping metadata only and must
/// not be passed to libcudf as row-group indices for the mini-Parquet buffer.
#[derive(Debug, Clone)]
pub(super) struct RangedParquetFetch {
    pub(super) bytes: Bytes,
    pub(super) original_file_bytes: u64,
    pub(super) selected_original_row_groups: Vec<usize>,
    pub(super) mini_row_group_rows: Vec<u64>,
    pub(super) mini_row_group_bytes: Vec<u64>,
    pub(super) footer_bytes: u64,
}

impl RangedParquetFetch {
    pub(super) fn mini_payload_bytes(&self) -> u64 {
        u64::try_from(self.bytes.len()).expect("usize payload length always fits in u64")
    }

    fn validate(self, location: &ObjectStorePath) -> Result<Self> {
        if self.mini_row_group_bytes.len() != self.mini_row_group_rows.len() {
            return Err(invalid_rewrite(location, "mini row-group byte and row-count metadata lengths differ")
                .with_fact("mini_row_group_bytes", self.mini_row_group_bytes.len())
                .with_fact("mini_row_group_rows", self.mini_row_group_rows.len()));
        }
        if self.selected_original_row_groups.len() != self.mini_row_group_rows.len() {
            return Err(invalid_rewrite(
                location,
                "original-to-mini row-group mapping length differs from mini metadata",
            )
            .with_fact("selected_original_row_groups", self.selected_original_row_groups.len())
            .with_fact("mini_row_group_rows", self.mini_row_group_rows.len()));
        }
        let mini_payload_bytes = self.mini_payload_bytes();
        let mini_row_group_bytes = self
            .mini_row_group_bytes
            .iter()
            .try_fold(0u64, |total, bytes| total.checked_add(*bytes))
            .ok_or_else(|| invalid_rewrite(location, "mini row-group byte metadata overflows u64"))?;
        if mini_row_group_bytes > mini_payload_bytes {
            return Err(invalid_rewrite(location, "mini row-group bytes exceed mini-Parquet payload length")
                .with_fact("mini_row_group_bytes", mini_row_group_bytes)
                .with_fact("mini_payload_bytes", mini_payload_bytes));
        }
        if !self.bytes.starts_with(PARQUET_MAGIC) || !self.bytes.ends_with(PARQUET_MAGIC) {
            return Err(invalid_rewrite(location, "rewritten mini-Parquet payload is missing PAR1 magic bytes"));
        }
        Ok(self)
    }
}

/// Fetches only the row groups needed to satisfy a scan limit or an explicit
/// row-group selection from an object-store Parquet file and returns a
/// compact mini-Parquet buffer that libcudf can decode directly.
pub(super) async fn fetch_ranged_parquet(
    store: Arc<dyn ObjectStore>,
    location: ObjectStorePath,
    selector: RangedRowGroupSelector,
) -> Result<RangedParquetFetch> {
    match &selector {
        RangedRowGroupSelector::ScanLimit(0) => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "ranged Parquet fetch requires a positive scan limit",
            ));
        },
        RangedRowGroupSelector::Explicit(indexes) if indexes.is_empty() => {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "ranged Parquet fetch requires a non-empty row-group selection",
            ));
        },
        _ => {},
    }

    let head = store
        .head(&location)
        .await
        .map_err(|error| ranged_fetch_error("failed to HEAD object-store Parquet file", error, &location))?;
    let file_size = head.size;
    if file_size < FOOTER_SIZE as u64 {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "object-store Parquet file is smaller than the Parquet footer",
        )
        .with_fact("path", location.as_ref())
        .with_fact("file_size", file_size));
    }

    let prefetch = FOOTER_PREFETCH_BYTES.min(file_size);
    let tail_start = file_size - prefetch;
    let tail = store
        .get_range(&location, tail_start..file_size)
        .await
        .map_err(|error| ranged_fetch_error("failed to GET Parquet footer range", error, &location))?;
    if tail.len() < FOOTER_SIZE {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "object-store Parquet tail fetch returned fewer bytes than the footer size",
        )
        .with_fact("path", location.as_ref())
        .with_fact("file_size", file_size)
        .with_fact("tail_bytes", tail.len()));
    }

    let footer_tail_bytes: [u8; FOOTER_SIZE] =
        tail[tail.len() - FOOTER_SIZE..].try_into().expect("footer tail slice was length-checked");
    let footer_tail = FooterTail::try_from(footer_tail_bytes).map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "failed to decode Parquet footer tail",
        )
        .with_fact("path", location.as_ref())
        .with_source(error)
    })?;
    if footer_tail.is_encrypted_footer() {
        return Err(QueryError::new(
            QueryErrorCode::UnsupportedPlanShape,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "encrypted Parquet footers are not supported by ranged scan-limit fetch",
        )
        .with_fact("path", location.as_ref()));
    }

    let metadata_len = footer_tail.metadata_length();
    let total_footer_len = metadata_len + FOOTER_SIZE;
    if (total_footer_len as u64) > file_size {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "Parquet footer metadata length exceeds file size",
        )
        .with_fact("path", location.as_ref())
        .with_fact("file_size", file_size)
        .with_fact("metadata_length", metadata_len));
    }

    let metadata_bytes = if (total_footer_len as u64) <= prefetch {
        let start = tail.len() - total_footer_len;
        let end = tail.len() - FOOTER_SIZE;
        tail.slice(start..end)
    } else {
        let metadata_start = file_size - total_footer_len as u64;
        let metadata_end = file_size - FOOTER_SIZE as u64;
        store
            .get_range(&location, metadata_start..metadata_end)
            .await
            .map_err(|error| ranged_fetch_error("failed to GET Parquet footer metadata range", error, &location))?
    };

    let mut reader = ParquetMetaDataReader::new();
    let mut footer_with_tail = BytesMut::with_capacity(total_footer_len);
    footer_with_tail.extend_from_slice(&metadata_bytes);
    footer_with_tail.extend_from_slice(&footer_tail_bytes);
    reader.try_parse_sized(&footer_with_tail.freeze(), file_size).map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "failed to decode Parquet footer metadata",
        )
        .with_fact("path", location.as_ref())
        .with_source(error)
    })?;
    let metadata = reader.finish().map_err(|error| {
        QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "failed to finalize Parquet footer metadata",
        )
        .with_fact("path", location.as_ref())
        .with_source(error)
    })?;

    let row_groups = metadata.row_groups();
    if row_groups.is_empty() {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "Parquet footer reports zero row groups",
        )
        .with_fact("path", location.as_ref()));
    }

    let selected_indices: Vec<usize> = match &selector {
        RangedRowGroupSelector::ScanLimit(scan_limit) => {
            let mut needed_rows = *scan_limit;
            let mut selected = Vec::new();
            for (idx, row_group) in row_groups.iter().enumerate() {
                selected.push(idx);
                let rg_rows = u64::try_from(row_group.num_rows().max(0)).unwrap_or(0);
                if rg_rows >= needed_rows {
                    break;
                }
                needed_rows -= rg_rows;
            }
            selected
        },
        RangedRowGroupSelector::Explicit(indexes) => {
            for &idx in indexes {
                if idx >= row_groups.len() {
                    return Err(QueryError::new(
                        QueryErrorCode::InvalidPlan,
                        "nexus_query_engine.source_fetch.ranged_parquet",
                        "explicit row-group selection references an unknown row group",
                    )
                    .with_fact("path", location.as_ref())
                    .with_fact("requested_row_group", idx)
                    .with_fact("row_groups_in_file", row_groups.len()));
                }
            }
            indexes.clone()
        },
    };
    let selected_count = selected_indices.len();
    let selected_row_groups: Vec<&RowGroupMetaData> = selected_indices.iter().map(|&idx| &row_groups[idx]).collect();
    let mini_row_group_rows = selected_row_groups
        .iter()
        .map(|row_group| u64::try_from(row_group.num_rows().max(0)).unwrap_or(0))
        .collect::<Vec<_>>();

    // Compute the byte ranges to fetch (one per column chunk in each selected
    // row group), and prepare a parallel list of new offsets after rewrite.
    let mut fetch_ranges: Vec<std::ops::Range<u64>> = Vec::new();
    let mut new_offsets: Vec<u64> = Vec::new();
    let mut row_group_start_offsets: Vec<u64> = Vec::with_capacity(selected_row_groups.len());
    let mut mini_row_group_bytes: Vec<u64> = Vec::with_capacity(selected_row_groups.len());
    let mut next_offset: u64 = PARQUET_MAGIC.len() as u64;
    for row_group in &selected_row_groups {
        row_group_start_offsets.push(next_offset);
        let row_group_start = next_offset;
        for column in row_group.columns() {
            let (start, length) = column.byte_range();
            fetch_ranges.push(
                start..start.checked_add(length).ok_or_else(|| {
                    QueryError::new(
                        QueryErrorCode::CudfDependencyFailure,
                        "nexus_query_engine.source_fetch.ranged_parquet",
                        "Parquet column-chunk byte range overflows file size",
                    )
                    .with_fact("path", location.as_ref())
                    .with_fact("column_start", start)
                    .with_fact("column_length", length)
                })?,
            );
            new_offsets.push(next_offset);
            next_offset = next_offset.checked_add(length).ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "nexus_query_engine.source_fetch.ranged_parquet",
                    "rewritten Parquet column-chunk offset overflows u64",
                )
                .with_fact("path", location.as_ref())
            })?;
        }
        mini_row_group_bytes.push(next_offset - row_group_start);
    }

    let column_data_total_bytes = next_offset - PARQUET_MAGIC.len() as u64;
    let column_chunks = if fetch_ranges.is_empty() {
        Vec::new()
    } else {
        store
            .get_ranges(&location, &fetch_ranges)
            .await
            .map_err(|error| ranged_fetch_error("failed to GET Parquet column-chunk ranges", error, &location))?
    };
    if column_chunks.len() != fetch_ranges.len() {
        return Err(QueryError::new(
            QueryErrorCode::CudfDependencyFailure,
            "nexus_query_engine.source_fetch.ranged_parquet",
            "object-store ranged GET returned an unexpected chunk count",
        )
        .with_fact("path", location.as_ref())
        .with_fact("requested_ranges", fetch_ranges.len())
        .with_fact("returned_chunks", column_chunks.len()));
    }

    // Rebuild metadata with adjusted column-chunk offsets and stripped
    // bloom-filter / column-index / offset-index references.
    let mut rewritten_row_groups: Vec<RowGroupMetaData> = Vec::with_capacity(selected_row_groups.len());
    let mut column_iter = new_offsets.into_iter();
    for (row_group, &rg_start) in selected_row_groups.iter().zip(row_group_start_offsets.iter()) {
        let mut rg_builder = (*row_group).clone().into_builder();
        let columns: Vec<ColumnChunkMetaData> = rg_builder.take_columns();
        let mut rewritten_columns = Vec::with_capacity(columns.len());
        for column in columns {
            let new_start = column_iter.next().ok_or_else(|| {
                QueryError::new(
                    QueryErrorCode::Internal,
                    "nexus_query_engine.source_fetch.ranged_parquet",
                    "internal error: rewritten column offset list shorter than column count",
                )
            })?;
            let dict_offset = column.dictionary_page_offset();
            let data_offset = column.data_page_offset();
            let new_chunk_start = i64::try_from(new_start).map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "nexus_query_engine.source_fetch.ranged_parquet",
                    "rewritten Parquet column-chunk offset exceeds i64 range",
                )
            })?;
            let (new_data_page_offset, new_dictionary_page_offset) = match dict_offset {
                Some(dict) => (new_chunk_start + (data_offset - dict), Some(new_chunk_start)),
                None => (new_chunk_start, None),
            };
            let rewritten = column
                .into_builder()
                .set_data_page_offset(new_data_page_offset)
                .set_dictionary_page_offset(new_dictionary_page_offset)
                .set_index_page_offset(None)
                .set_bloom_filter_offset(None)
                .set_bloom_filter_length(None)
                .set_offset_index_offset(None)
                .set_offset_index_length(None)
                .set_column_index_offset(None)
                .set_column_index_length(None)
                .build()
                .map_err(|error| {
                    QueryError::new(
                        QueryErrorCode::CudfDependencyFailure,
                        "nexus_query_engine.source_fetch.ranged_parquet",
                        "failed to rebuild rewritten Parquet column-chunk metadata",
                    )
                    .with_source(error)
                })?;
            rewritten_columns.push(rewritten);
        }
        let rg_file_offset = i64::try_from(rg_start).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "rewritten Parquet row-group file offset exceeds i64 range",
            )
        })?;
        let rg_meta =
            rg_builder.set_column_metadata(rewritten_columns).set_file_offset(rg_file_offset).build().map_err(
                |error| {
                    QueryError::new(
                        QueryErrorCode::CudfDependencyFailure,
                        "nexus_query_engine.source_fetch.ranged_parquet",
                        "failed to rebuild rewritten Parquet row-group metadata",
                    )
                    .with_source(error)
                },
            )?;
        rewritten_row_groups.push(rg_meta);
    }

    let mini_total_rows = selected_row_groups.iter().try_fold(0i64, |total, row_group| {
        total.checked_add(row_group.num_rows()).ok_or_else(|| {
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "rewritten Parquet file row count overflows i64",
            )
            .with_fact("path", location.as_ref())
        })
    })?;
    let source_file_metadata = metadata.file_metadata();
    let rewritten_file_metadata = parquet::file::metadata::FileMetaData::new(
        source_file_metadata.version(),
        mini_total_rows,
        source_file_metadata.created_by().map(ToOwned::to_owned),
        source_file_metadata.key_value_metadata().cloned(),
        source_file_metadata.schema_descr_ptr(),
        source_file_metadata.column_orders().cloned(),
    );
    let rewritten_metadata = ParquetMetaDataBuilder::new(rewritten_file_metadata)
        .set_row_groups(rewritten_row_groups)
        .set_column_index(None)
        .set_offset_index(None)
        .build();

    // Emit: PAR1 + column-chunk payload + thrift footer + footer-length + PAR1.
    // ParquetMetaDataWriter assumes its output starts at offset 0; the offsets
    // already written into rewritten_metadata point at the layout we are about
    // to produce.
    let mut output: Vec<u8> =
        Vec::with_capacity(PARQUET_MAGIC.len() + column_data_total_bytes as usize + metadata_len + FOOTER_SIZE);
    output.extend_from_slice(PARQUET_MAGIC);
    for chunk in &column_chunks {
        output.extend_from_slice(chunk);
    }

    let footer_start = output.len();
    {
        let mut footer_buf: Vec<u8> = Vec::new();
        ParquetMetaDataWriter::new(&mut footer_buf, &rewritten_metadata).finish().map_err(|error| {
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "failed to serialize rewritten Parquet footer metadata",
            )
            .with_source(error)
        })?;
        let footer_len = u32::try_from(footer_buf.len()).map_err(|_error| {
            QueryError::new(
                QueryErrorCode::CudfDependencyFailure,
                "nexus_query_engine.source_fetch.ranged_parquet",
                "rewritten Parquet footer length exceeds u32 range",
            )
        })?;
        output.extend_from_slice(&footer_buf);
        output.extend_from_slice(&footer_len.to_le_bytes());
        output.extend_from_slice(PARQUET_MAGIC);
        debug_assert!(output.len() > footer_start);
    }

    let footer_bytes = u64::try_from(output.len() - footer_start).map_err(|_error| {
        invalid_rewrite(&location, "rewritten Parquet footer length exceeds u64 range")
            .with_fact("selected_row_groups", selected_count)
    })?;
    RangedParquetFetch {
        bytes: Bytes::from(output),
        original_file_bytes: file_size,
        selected_original_row_groups: selected_indices,
        mini_row_group_rows,
        mini_row_group_bytes,
        footer_bytes,
    }
    .validate(&location)
}

fn ranged_fetch_error(message: &'static str, error: object_store::Error, location: &ObjectStorePath) -> QueryError {
    QueryError::new(QueryErrorCode::CudfDependencyFailure, "nexus_query_engine.source_fetch.ranged_parquet", message)
        .with_fact("path", location.as_ref())
        .with_source(error)
}

fn invalid_rewrite(location: &ObjectStorePath, message: &'static str) -> QueryError {
    QueryError::new(QueryErrorCode::CudfDependencyFailure, "nexus_query_engine.source_fetch.ranged_parquet", message)
        .with_fact("path", location.as_ref())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::{
        array::{Int64Array, RecordBatch, StringArray},
        datatypes::{DataType, Field, Schema},
    };
    use bytes::Bytes;
    use object_store::{ObjectStoreExt, memory::InMemory, path::Path as ObjectStorePath};
    use parquet::{
        arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
        file::properties::WriterProperties,
    };

    use super::*;

    fn write_test_parquet(num_row_groups: usize, rows_per_group: usize) -> Bytes {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let total_rows = num_row_groups * rows_per_group;
        let ids: Vec<i64> = (0..total_rows as i64).collect();
        let names: Vec<String> = (0..total_rows).map(|i| format!("row-{i}")).collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(ids)), Arc::new(StringArray::from(names))],
        )
        .unwrap();

        let mut buf: Vec<u8> = Vec::new();
        let props = WriterProperties::builder().set_max_row_group_row_count(Some(rows_per_group)).build();
        let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        Bytes::from(buf)
    }

    fn collect_int64_column(parquet_bytes: Bytes, column: usize) -> Vec<i64> {
        let reader = ParquetRecordBatchReaderBuilder::try_new(parquet_bytes).unwrap().build().unwrap();
        let mut values = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let array = batch.column(column).as_any().downcast_ref::<Int64Array>().unwrap();
            values.extend(array.iter().map(|v| v.unwrap()));
        }
        values
    }

    fn collect_string_column(parquet_bytes: Bytes, column: usize) -> Vec<String> {
        let reader = ParquetRecordBatchReaderBuilder::try_new(parquet_bytes).unwrap().build().unwrap();
        let mut values = Vec::new();
        for batch in reader {
            let batch = batch.unwrap();
            let array = batch.column(column).as_any().downcast_ref::<StringArray>().unwrap();
            values.extend(array.iter().map(|v| v.unwrap().to_string()));
        }
        values
    }

    fn parquet_file_num_rows(parquet_bytes: Bytes) -> i64 {
        ParquetRecordBatchReaderBuilder::try_new(parquet_bytes).unwrap().metadata().file_metadata().num_rows()
    }

    #[tokio::test]
    async fn fetches_only_first_row_group_when_limit_fits() {
        let parquet_bytes = write_test_parquet(4, 25);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectStorePath::from("table/part-0.parquet");
        store.put(&location, parquet_bytes.clone().into()).await.unwrap();

        let mini = fetch_ranged_parquet(store, location, RangedRowGroupSelector::ScanLimit(10)).await.unwrap();

        assert!(mini.bytes.len() < parquet_bytes.len(), "mini-Parquet should be smaller than original");
        assert_eq!(mini.original_file_bytes, parquet_bytes.len() as u64);
        assert_eq!(mini.selected_original_row_groups, vec![0]);
        assert_eq!(mini.mini_row_group_rows, vec![25]);
        assert_eq!(mini.mini_row_group_bytes.len(), 1);
        assert!(mini.footer_bytes > FOOTER_SIZE as u64);
        let ids = collect_int64_column(mini.bytes.clone(), 0);
        let names = collect_string_column(mini.bytes, 1);
        assert_eq!(ids.len(), 25);
        assert_eq!(ids, (0..25).collect::<Vec<_>>());
        assert_eq!(names[0], "row-0");
        assert_eq!(names[24], "row-24");
    }

    #[tokio::test]
    async fn fetches_minimum_row_group_prefix_to_cover_limit() {
        let parquet_bytes = write_test_parquet(4, 25);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectStorePath::from("table/part-0.parquet");
        store.put(&location, parquet_bytes.clone().into()).await.unwrap();

        let mini = fetch_ranged_parquet(store, location, RangedRowGroupSelector::ScanLimit(30)).await.unwrap();

        assert_eq!(mini.selected_original_row_groups, vec![0, 1]);
        assert_eq!(mini.mini_row_group_rows, vec![25, 25]);
        assert_eq!(parquet_file_num_rows(mini.bytes.clone()), 50);
        let ids = collect_int64_column(mini.bytes, 0);
        assert_eq!(ids.len(), 50, "30-row limit straddles 25-row groups; expect first two row groups");
        assert_eq!(ids, (0..50).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn small_files_fit_in_footer_prefetch() {
        let parquet_bytes = write_test_parquet(1, 8);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectStorePath::from("tiny.parquet");
        store.put(&location, parquet_bytes.clone().into()).await.unwrap();

        let mini = fetch_ranged_parquet(store, location, RangedRowGroupSelector::ScanLimit(4)).await.unwrap();
        let ids = collect_int64_column(mini.bytes, 0);
        assert_eq!(ids, (0..8).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn explicit_selection_keeps_original_mapping_but_mini_groups_are_contiguous() {
        let parquet_bytes = write_test_parquet(4, 25);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectStorePath::from("table/part-1.parquet");
        store.put(&location, parquet_bytes.into()).await.unwrap();

        let mini = fetch_ranged_parquet(store, location, RangedRowGroupSelector::Explicit(vec![2, 3])).await.unwrap();

        assert_eq!(mini.selected_original_row_groups, vec![2, 3]);
        assert_eq!(mini.mini_row_group_rows, vec![25, 25]);
        assert_eq!(mini.mini_row_group_bytes.len(), 2);
        let ids = collect_int64_column(mini.bytes, 0);
        assert_eq!(ids, (50..100).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn zero_scan_limit_is_rejected() {
        let parquet_bytes = write_test_parquet(1, 8);
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let location = ObjectStorePath::from("tiny.parquet");
        store.put(&location, parquet_bytes.into()).await.unwrap();

        let error = fetch_ranged_parquet(store, location, RangedRowGroupSelector::ScanLimit(0)).await.unwrap_err();
        assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    }
}
