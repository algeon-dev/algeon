//! Shared Parquet footer/statistics helpers used by native pruning and sizing.

use std::collections::{BTreeMap, BTreeSet};

use parquet::{
    basic::{ConvertedType, Encoding, LogicalType, Type as ParquetPhysicalType},
    file::{
        metadata::{ParquetMetaData, RowGroupMetaData},
        statistics::Statistics,
    },
    schema::types::SchemaDescriptor,
};

/// Footer proof for the value encoding family used by a variable-width column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParquetVariableWidthEncodingProof {
    /// No row group is selected, so the payload is exactly zero bytes.
    Empty,
    /// Every selected chunk has a PLAIN value encoding.
    Plain,
    /// Every selected chunk has a dictionary-family value encoding.
    Dictionary,
    /// The selected chunks use both PLAIN and dictionary-family encodings.
    PlainAndDictionary,
}

impl ParquetVariableWidthEncodingProof {
    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (Self::Plain, Self::Plain) => Self::Plain,
            (Self::Dictionary, Self::Dictionary) => Self::Dictionary,
            _ => Self::PlainAndDictionary,
        }
    }
}

/// Source-coupled upper bound for one variable-width Parquet file column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParquetVariableWidthPayloadEvidence {
    /// Zero-based physical root-column index in the Parquet file schema.
    pub physical_column_index: usize,
    /// Exact physical row-group IDs included in this bound.
    pub selected_row_groups: Vec<usize>,
    /// Checked sum of selected column-chunk uncompressed sizes.
    pub payload_upper_bound_bytes: u64,
    /// Encoding-family proof established from every selected column chunk.
    pub encoding_proof: ParquetVariableWidthEncodingProof,
}

impl ParquetVariableWidthPayloadEvidence {
    /// Creates one source-coupled payload fact.
    pub fn new(
        physical_column_index: usize,
        selected_row_groups: Vec<usize>,
        payload_upper_bound_bytes: u64,
        encoding_proof: ParquetVariableWidthEncodingProof,
    ) -> Self {
        Self { physical_column_index, selected_row_groups, payload_upper_bound_bytes, encoding_proof }
    }
}

/// Returns the only primitive leaf column for a top-level root column.
///
/// Nested roots and roots that map to zero or multiple leaves return `None`.
pub fn single_leaf_primitive_root_column_index(schema: &SchemaDescriptor, root_index: usize) -> Option<usize> {
    if !schema.root_schema().get_fields().get(root_index)?.is_primitive() {
        return None;
    }
    let mut leaf_indices =
        (0..schema.num_columns()).filter(|leaf_index| schema.get_column_root_idx(*leaf_index) == root_index);
    let leaf_index = leaf_indices.next()?;
    leaf_indices.next().is_none().then_some(leaf_index)
}

/// Returns statistics for a top-level primitive root column represented by one leaf.
pub fn leaf_column_statistics<'a>(
    row_group: &'a RowGroupMetaData,
    schema: &SchemaDescriptor,
    root_column_index: usize,
) -> Option<&'a Statistics> {
    let leaf_column_index = single_leaf_primitive_root_column_index(schema, root_column_index)?;
    row_group.columns().get(leaf_column_index)?.statistics()
}

/// Which encoded byte count to read from a Parquet column chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnByteKind {
    /// Compressed bytes reported by the Parquet column metadata.
    Compressed,
    /// Uncompressed bytes reported by the Parquet column metadata.
    Uncompressed,
}

/// Sums compressed or uncompressed bytes for the requested leaf column indexes.
pub fn row_group_column_bytes(
    row_group: &RowGroupMetaData,
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

/// Derives variable-width payload evidence for one top-level primitive BYTE_ARRAY root
/// carrying a UTF8/String logical or converted annotation.
pub fn variable_width_payload_evidence_for_root_column(
    metadata: &ParquetMetaData,
    root_column_index: usize,
    selected_row_groups: Option<&[usize]>,
) -> Option<ParquetVariableWidthPayloadEvidence> {
    let schema = metadata.file_metadata().schema_descr();
    let leaf_column_index = single_leaf_primitive_root_column_index(schema, root_column_index)?;
    let descriptor = schema.columns().get(leaf_column_index)?;
    let is_utf8 = matches!(descriptor.logical_type_ref(), Some(LogicalType::String))
        || descriptor.converted_type() == ConvertedType::UTF8;
    if descriptor.physical_type() != ParquetPhysicalType::BYTE_ARRAY || !is_utf8 {
        return None;
    }
    let selected_row_groups = selected_row_groups
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| (0..metadata.row_groups().len()).collect::<Vec<_>>());
    if selected_row_groups.is_empty() {
        return Some(ParquetVariableWidthPayloadEvidence::new(
            root_column_index,
            selected_row_groups,
            0,
            ParquetVariableWidthEncodingProof::Empty,
        ));
    }
    let mut encoding_proof: Option<ParquetVariableWidthEncodingProof> = None;
    let mut payload_upper_bound_bytes = 0_u64;
    for row_group_index in &selected_row_groups {
        let column = metadata.row_groups().get(*row_group_index)?.columns().get(leaf_column_index)?;
        let chunk_proof = variable_width_encoding_proof(column.encodings())?;
        encoding_proof = Some(match encoding_proof {
            Some(proof) => proof.merge(chunk_proof),
            None => chunk_proof,
        });
        let uncompressed_size = u64::try_from(column.uncompressed_size()).ok()?;
        payload_upper_bound_bytes = payload_upper_bound_bytes.checked_add(uncompressed_size)?;
    }
    Some(ParquetVariableWidthPayloadEvidence::new(
        root_column_index,
        selected_row_groups,
        payload_upper_bound_bytes,
        encoding_proof?,
    ))
}

/// Derives variable-width payload evidence for the requested physical root columns.
pub fn variable_width_payload_evidence_by_root_column(
    metadata: &ParquetMetaData,
    root_column_indexes: &BTreeSet<usize>,
    selected_row_groups: Option<&[usize]>,
) -> BTreeMap<usize, ParquetVariableWidthPayloadEvidence> {
    root_column_indexes
        .iter()
        .filter_map(|root_column_index| {
            variable_width_payload_evidence_for_root_column(metadata, *root_column_index, selected_row_groups)
                .map(|evidence| (*root_column_index, evidence))
        })
        .collect()
}

#[expect(
    deprecated,
    reason = "Parquet BIT_PACKED encoding is deprecated upstream but supported for Parquet metadata compatibility"
)]
fn variable_width_encoding_proof(
    encodings: impl IntoIterator<Item = Encoding>,
) -> Option<ParquetVariableWidthEncodingProof> {
    let mut has_plain = false;
    let mut has_dictionary = false;
    for encoding in encodings {
        match encoding {
            Encoding::PLAIN => has_plain = true,
            Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY => has_dictionary = true,
            Encoding::RLE | Encoding::BIT_PACKED => {},
            _ => return None,
        }
    }
    match (has_plain, has_dictionary) {
        (true, false) => Some(ParquetVariableWidthEncodingProof::Plain),
        (false, true) => Some(ParquetVariableWidthEncodingProof::Dictionary),
        (true, true) => Some(ParquetVariableWidthEncodingProof::PlainAndDictionary),
        (false, false) => None,
    }
}

/// Returns per-row-group null counts for a primitive root represented by one leaf.
pub fn row_group_null_counts_for_root_column(
    metadata: &ParquetMetaData,
    root_column_index: usize,
) -> Option<Vec<Option<u64>>> {
    let schema = metadata.file_metadata().schema_descr();
    let leaf_index = single_leaf_primitive_root_column_index(schema, root_column_index)?;
    Some(
        metadata
            .row_groups()
            .iter()
            .map(|row_group| {
                row_group
                    .columns()
                    .get(leaf_index)
                    .and_then(|column| column.statistics())
                    .and_then(|statistics| statistics.null_count_opt())
            })
            .collect(),
    )
}

/// Returns per-row-group null counts for all primitive roots represented by one leaf.
pub fn row_group_null_counts_by_root_column(metadata: &ParquetMetaData) -> BTreeMap<usize, Vec<Option<u64>>> {
    let schema = metadata.file_metadata().schema_descr();
    let root_field_count = schema.root_schema().get_fields().len();
    (0..root_field_count)
        .filter_map(|root_index| Some((root_index, row_group_null_counts_for_root_column(metadata, root_index)?)))
        .collect()
}

/// Inclusive `i64` range decoded from integer Parquet statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I64StatisticsRange {
    /// Minimum integer value recorded by Parquet statistics.
    pub min: i64,
    /// Maximum integer value recorded by Parquet statistics.
    pub max: i64,
}

/// Decodes Int32/Int64 Parquet statistics into an inclusive `i64` range.
pub fn statistics_i64_range(statistics: &Statistics) -> Option<I64StatisticsRange> {
    match statistics {
        Statistics::Int32(stats) => {
            Some(I64StatisticsRange { min: i64::from(*stats.min_opt()?), max: i64::from(*stats.max_opt()?) })
        },
        Statistics::Int64(stats) => Some(I64StatisticsRange { min: *stats.min_opt()?, max: *stats.max_opt()? }),
        _ => None,
    }
}

/// Decodes Int32/Int64 statistics for a leaf column in a row group.
pub fn row_group_i64_range(row_group: &RowGroupMetaData, leaf_column_index: usize) -> Option<I64StatisticsRange> {
    statistics_i64_range(row_group.columns().get(leaf_column_index)?.statistics()?)
}

/// Decodes Int32/Int64 statistics for a primitive root represented by one leaf.
pub fn row_group_i64_range_for_root_column(
    row_group: &RowGroupMetaData,
    schema: &SchemaDescriptor,
    root_column_index: usize,
) -> Option<I64StatisticsRange> {
    let leaf_column_index = single_leaf_primitive_root_column_index(schema, root_column_index)?;
    row_group_i64_range(row_group, leaf_column_index)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use parquet::{
        basic::Encoding,
        file::metadata::{ColumnChunkMetaData, FileMetaData, ParquetMetaData, RowGroupMetaData},
        schema::{parser::parse_message_type, types::SchemaDescriptor},
    };

    use super::{
        ParquetVariableWidthEncodingProof, single_leaf_primitive_root_column_index,
        variable_width_payload_evidence_for_root_column,
    };

    fn variable_width_metadata(
        schema_body: &str,
        encodings: Vec<Vec<Encoding>>,
        uncompressed_sizes: Vec<i64>,
    ) -> ParquetMetaData {
        let parquet_type = parse_message_type(schema_body).expect("test Parquet schema should parse");
        let schema = Arc::new(SchemaDescriptor::new(Arc::new(parquet_type)));
        let num_rows = i64::try_from(encodings.len()).expect("test row-group count should fit i64");
        let row_groups = encodings
            .into_iter()
            .zip(uncompressed_sizes)
            .map(|(encodings, uncompressed_size)| {
                let column = ColumnChunkMetaData::builder(schema.columns()[0].clone())
                    .set_encodings(encodings)
                    .set_total_uncompressed_size(uncompressed_size)
                    .build()
                    .expect("test column metadata should build");
                RowGroupMetaData::builder(Arc::clone(&schema))
                    .set_num_rows(1)
                    .add_column_metadata(column)
                    .build()
                    .expect("test row-group metadata should build")
            })
            .collect();
        ParquetMetaData::new(FileMetaData::new(1, num_rows, None, None, schema, None), row_groups)
    }

    #[test]
    fn single_leaf_mapping_refuses_nested_roots_and_maps_later_primitives() {
        let parquet_type = parse_message_type(
            "
            message schema {
              optional group payload {
                optional int32 x;
                optional int32 y;
              }
              optional int32 fact_key;
            }
            ",
        )
        .expect("test Parquet schema should parse");
        let schema = SchemaDescriptor::new(Arc::new(parquet_type));

        assert_eq!(single_leaf_primitive_root_column_index(&schema, 0), None);
        assert_eq!(single_leaf_primitive_root_column_index(&schema, 1), Some(2));
    }

    #[test]
    fn single_leaf_mapping_uses_parquet_root_indexes_not_names() {
        let parquet_type = parse_message_type(
            "
            message schema {
              optional int32 duplicate_name;
              optional int64 duplicate_name;
            }
            ",
        )
        .expect("test Parquet schema should parse");
        let schema = SchemaDescriptor::new(Arc::new(parquet_type));

        assert_eq!(single_leaf_primitive_root_column_index(&schema, 0), Some(0));
        assert_eq!(single_leaf_primitive_root_column_index(&schema, 1), Some(1));
    }

    #[test]
    #[expect(
        deprecated,
        reason = "Parquet BIT_PACKED encoding is deprecated upstream but tested for Parquet metadata compatibility"
    )]
    fn variable_width_payload_evidence_sums_exact_selected_row_groups_and_proves_encoding_family() {
        let metadata = variable_width_metadata(
            "
            message schema {
              optional binary payload (UTF8);
            }
            ",
            vec![vec![Encoding::PLAIN, Encoding::RLE], vec![Encoding::RLE_DICTIONARY, Encoding::BIT_PACKED]],
            vec![11, 13],
        );

        let selected = variable_width_payload_evidence_for_root_column(&metadata, 0, Some(&[1]))
            .expect("dictionary-encoded selected row group should produce evidence");
        assert_eq!(selected.physical_column_index, 0);
        assert_eq!(selected.selected_row_groups, [1]);
        assert_eq!(selected.payload_upper_bound_bytes, 13);
        assert_eq!(selected.encoding_proof, ParquetVariableWidthEncodingProof::Dictionary);

        let complete = variable_width_payload_evidence_for_root_column(&metadata, 0, None)
            .expect("all physical row groups should produce evidence");
        assert_eq!(complete.selected_row_groups, [0, 1]);
        assert_eq!(complete.payload_upper_bound_bytes, 24);
        assert_eq!(complete.encoding_proof, ParquetVariableWidthEncodingProof::PlainAndDictionary);
    }

    #[test]
    fn variable_width_payload_evidence_proves_empty_selection_without_an_encoding() {
        let empty_file = variable_width_metadata(
            "
            message schema {
              optional binary payload (UTF8);
            }
            ",
            Vec::new(),
            Vec::new(),
        );
        let empty_file_evidence = variable_width_payload_evidence_for_root_column(&empty_file, 0, None)
            .expect("an empty Parquet file has an exact zero-byte payload bound");
        assert!(empty_file_evidence.selected_row_groups.is_empty());
        assert_eq!(empty_file_evidence.payload_upper_bound_bytes, 0);
        assert_eq!(empty_file_evidence.encoding_proof, ParquetVariableWidthEncodingProof::Empty);

        let pruned_file = variable_width_metadata(
            "
            message schema {
              optional binary payload (UTF8);
            }
            ",
            vec![vec![Encoding::DELTA_BYTE_ARRAY]],
            vec![11],
        );
        let pruned_evidence = variable_width_payload_evidence_for_root_column(&pruned_file, 0, Some(&[]))
            .expect("an empty row-group selection has no encoded payload");
        assert!(pruned_evidence.selected_row_groups.is_empty());
        assert_eq!(pruned_evidence.payload_upper_bound_bytes, 0);
        assert_eq!(pruned_evidence.encoding_proof, ParquetVariableWidthEncodingProof::Empty);
    }

    #[test]
    fn variable_width_payload_evidence_rejects_unsafe_or_incomplete_encodings() {
        for encodings in [
            vec![Encoding::DELTA_BINARY_PACKED],
            vec![Encoding::DELTA_LENGTH_BYTE_ARRAY],
            vec![Encoding::DELTA_BYTE_ARRAY],
            vec![Encoding::BYTE_STREAM_SPLIT],
            vec![Encoding::RLE],
        ] {
            let metadata = variable_width_metadata(
                "
                message schema {
                  optional binary payload (UTF8);
                }
                ",
                vec![encodings],
                vec![11],
            );
            assert!(variable_width_payload_evidence_for_root_column(&metadata, 0, None).is_none());
        }
    }

    #[test]
    fn variable_width_payload_evidence_rejects_negative_chunk_size() {
        let metadata = variable_width_metadata(
            "
            message schema {
              optional binary payload (UTF8);
            }
            ",
            vec![vec![Encoding::PLAIN]],
            vec![-1],
        );
        assert!(variable_width_payload_evidence_for_root_column(&metadata, 0, None).is_none());
    }

    #[test]
    fn variable_width_payload_evidence_rejects_overflowing_chunk_size_sum() {
        let metadata = variable_width_metadata(
            "
            message schema {
              optional binary payload (UTF8);
            }
            ",
            vec![vec![Encoding::PLAIN], vec![Encoding::PLAIN], vec![Encoding::PLAIN]],
            vec![i64::MAX, i64::MAX, i64::MAX],
        );
        assert!(variable_width_payload_evidence_for_root_column(&metadata, 0, None).is_none());
    }

    #[test]
    fn variable_width_payload_evidence_rejects_unannotated_binary_columns() {
        let metadata = variable_width_metadata(
            "
            message schema {
              optional binary payload;
            }
            ",
            vec![vec![Encoding::PLAIN]],
            vec![11],
        );
        assert!(variable_width_payload_evidence_for_root_column(&metadata, 0, None).is_none());
    }
}
