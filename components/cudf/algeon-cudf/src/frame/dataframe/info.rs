use std::{fmt, sync::Arc};

use super::owned::DataFrame;
use crate::{
    core::{
        column::Column,
        error::{Error, ErrorCode, Result},
        types::DataType,
    },
    hash::{FastHashMap, fast_hash_map_with_capacity},
};

bitflags::bitflags! {
    /// Controls how much metadata [`DataFrame::info_with`] computes.
    ///
    /// Flags compose via `|`. `NULL_COUNTS` implies `COLUMNS` via its bit
    /// pattern, making "null counts without columns" unrepresentable.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use cudf::frame::dataframe::InfoDetail;
    ///
    /// // Columns with null counts, no memory accounting.
    /// let detail = InfoDetail::NULL_COUNTS;
    /// assert!(detail.contains(InfoDetail::COLUMNS));
    ///
    /// // Both memory views, no per-column breakdown.
    /// let detail = InfoDetail::MEMORY;
    /// ```
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct InfoDetail: u8 {
        /// Include one metadata entry per schema column.
        const COLUMNS          = 0b0000_0001;
        /// Include per-column null and non-null counts (implies `COLUMNS`).
        const NULL_COUNTS      = 0b0000_0011;
        /// Include payload bytes.
        ///
        /// Reports bytes required to represent the logical payload, not the
        /// allocator-rounded device footprint. For explicit-runtime objects,
        /// this accounting path honors retained provenance.
        const PAYLOAD_BYTES    = 0b0000_0100;
        /// Include total device-allocated bytes.
        ///
        /// Reports the actual owned device allocation footprint, including
        /// allocator rounding and padding.
        const DEVICE_ALLOCATED = 0b0000_1000;
        /// Both `PAYLOAD_BYTES` and `DEVICE_ALLOCATED`.
        const MEMORY           = 0b0000_1100;
        /// All detail: `NULL_COUNTS` | `MEMORY`.
        const ALL              = 0b0000_1111;
    }
}

impl InfoDetail {
    fn includes_columns(self) -> bool {
        self.contains(Self::COLUMNS)
    }

    fn includes_null_counts(self) -> bool {
        self.contains(Self::NULL_COUNTS)
    }

    fn includes_payload(self) -> bool {
        self.contains(Self::PAYLOAD_BYTES)
    }

    fn includes_device_allocated(self) -> bool {
        self.contains(Self::DEVICE_ALLOCATED)
    }
}

/// Programmatic metadata snapshot of a [`DataFrame`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataFrameInfo {
    /// Logical row count.
    pub row_count: usize,
    /// Schema column count.
    pub column_count: usize,
    /// Counts of columns by exact data type.
    pub dtypes: Vec<DataTypeCount>,
    /// Optional per-column metadata in schema order.
    pub columns: Option<Vec<DataFrameColumnInfo>>,
    /// Optional top-level memory accounting totals.
    pub memory_accounting: Option<DataFrameMemoryAccounting>,
}

/// Count of columns with one exact [`DataType`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataTypeCount {
    /// The exact data type.
    pub data_type: DataType,
    /// Number of columns with that type.
    pub column_count: usize,
}

/// Programmatic metadata for one DataFrame column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataFrameColumnInfo {
    /// Zero-based schema position.
    pub position: usize,
    /// Column name.
    pub name: String,
    /// Exact data type.
    pub data_type: DataType,
    /// Whether the column has an allocated null mask.
    pub nullable: bool,
    /// Null count when requested.
    pub null_count: Option<usize>,
    /// Non-null count when requested.
    pub non_null_count: Option<usize>,
    /// Optional per-column memory accounting.
    pub memory_accounting: Option<ColumnMemoryAccounting>,
}

/// Top-level DataFrame memory accounting totals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataFrameMemoryAccounting {
    /// Payload bytes across all columns.
    ///
    /// This is the sum of payload-sized buffers required to represent the
    /// data, including nested child payloads and payload-sized null masks.
    pub payload_bytes: Option<usize>,
    /// Total device-allocated bytes across all columns.
    ///
    /// This is the actual device allocation footprint owned by the DataFrame's
    /// columns and may exceed `payload_bytes`.
    pub device_allocated_bytes: Option<usize>,
}

/// Per-column memory accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnMemoryAccounting {
    /// Payload bytes used by the column.
    ///
    /// This excludes allocator padding and reports only the bytes required to
    /// represent the column payload.
    pub payload_bytes: Option<usize>,
    /// Total device-allocated bytes for the column.
    ///
    /// This is the column's actual owned device allocation footprint and may
    /// exceed `payload_bytes`.
    pub device_allocated_bytes: Option<usize>,
}

/// Opaque retained-owner identity for a DataFrame column.
///
/// This identity is stable while the retained column owner is alive and is
/// intended for alias/ownership accounting. It is deliberately not an
/// allocation pointer contract.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ColumnMemoryIdentity(usize);

impl ColumnMemoryIdentity {
    pub(crate) fn from_column_arc(column: &Arc<Column>) -> Self {
        Self(Arc::as_ptr(column) as usize)
    }
}

impl fmt::Debug for ColumnMemoryIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ColumnMemoryIdentity(<opaque>)")
    }
}

/// Retained-owner identity for one DataFrame column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataFrameColumnMemoryIdentity {
    /// Zero-based schema position in the observed DataFrame.
    pub position: usize,
    /// Column name at that position.
    pub name: String,
    /// Opaque retained-owner identity for the column.
    pub identity: ColumnMemoryIdentity,
}

/// Retained-owner identity snapshot for a DataFrame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataFrameMemoryIdentity {
    /// Logical row count.
    pub row_count: usize,
    /// Schema column identities in order.
    pub columns: Vec<DataFrameColumnMemoryIdentity>,
}

impl DataFrameMemoryIdentity {
    /// Schema column count.
    pub fn column_count(&self) -> usize {
        self.columns.len()
    }
}

impl DataFrame {
    /// Return the deduplicated base allocations retained by this frame.
    pub fn backing_allocation_pointers(&self) -> Result<Vec<usize>> {
        let mut pointers = Vec::new();
        for column in self.columns.values() {
            pointers.extend(column.backing_allocation_pointers()?);
        }
        // Shared columns and nested views can expose the same base allocation
        // more than once; ownership attestation counts each allocation once.
        pointers.sort_unstable();
        pointers.dedup();
        Ok(pointers)
    }
}

/// Mapping proving that one output column aliases one input column owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameColumnMemoryAlias {
    /// Input column position.
    pub input_position: usize,
    /// Input column name.
    pub input_name: String,
    /// Output column position.
    pub output_position: usize,
    /// Output column name.
    pub output_name: String,
    /// Shared retained-owner identity.
    pub identity: ColumnMemoryIdentity,
}

/// Operation-level DataFrame memory ownership effect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameMemoryEffect {
    /// Output columns are metadata aliases of input columns and retain the
    /// same column owners.
    AliasOfInput {
        /// Retained-owner identity of the input frame.
        input: DataFrameMemoryIdentity,
        /// Column-level alias proof in output schema order.
        output_columns: Vec<FrameColumnMemoryAlias>,
    },
    /// The operation returns newly owned output allocations unless a more
    /// specific lower-level contract says otherwise.
    NewAllocation {
        /// Retained-owner identity of the output frame.
        output: DataFrameMemoryIdentity,
    },
}

impl FrameMemoryEffect {
    /// Returns true when the effect proves that no output column owner is new.
    pub fn is_alias_of_input(&self) -> bool {
        matches!(self, Self::AliasOfInput { .. })
    }
}

fn data_type_sort_key(dtype: DataType) -> (i32, i32) {
    (dtype.id() as i32, dtype.scale())
}

impl DataFrame {
    /// Return a cheap programmatic metadata snapshot of this DataFrame.
    ///
    /// This is a convenience wrapper over [`info_with`](Self::info_with)
    /// with no detail flags. The default snapshot includes only row count,
    /// column count, and dtype counts.
    ///
    /// `info()` is metadata-only. It does not materialize column values to host
    /// memory and it does not include memory accounting unless requested via
    /// [`InfoDetail`] flags.
    pub fn info(&self) -> Result<DataFrameInfo> {
        self.info_with(InfoDetail::empty())
    }

    /// Return a programmatic metadata snapshot of this DataFrame.
    ///
    /// This method reports schema and optional memory-accounting metadata
    /// without materializing column values to host memory.
    ///
    /// The memory-accounting views have different semantics:
    ///
    /// - [`InfoDetail::PAYLOAD_BYTES`] reports bytes required to represent
    ///   the column payload, including nested child payloads and payload-sized
    ///   null-mask bytes.
    /// - [`InfoDetail::DEVICE_ALLOCATED`] reports actual owned device
    ///   allocation footprint, including allocator rounding and padding.
    ///
    /// Payload bytes are not allocation bytes. For explicit-runtime objects,
    /// payload accounting honors retained provenance rather than
    /// silently falling back to the default stream.
    pub fn info_with(&self, detail: InfoDetail) -> Result<DataFrameInfo> {
        let include_columns = detail.includes_columns();
        let include_null_counts = detail.includes_null_counts();
        let include_payload_bytes = detail.includes_payload();
        let include_device_allocated_bytes = detail.includes_device_allocated();

        let mut dtype_counts: FastHashMap<DataType, usize> = fast_hash_map_with_capacity(self.columns.len());
        let mut columns_info = include_columns.then(|| Vec::<DataFrameColumnInfo>::with_capacity(self.columns.len()));
        let mut total_payload_bytes = 0usize;
        let mut total_device_allocated_bytes = 0usize;

        for (position, (name, column)) in self.columns.iter().enumerate() {
            let view = column.view().map_err(|e| e.reanchor("dataframe::info_with").with_column(name.clone()))?;
            let data_type =
                view.data_type().map_err(|e| e.reanchor("dataframe::info_with").with_column(name.clone()))?;
            *dtype_counts.entry(data_type).or_insert(0) += 1;

            let nullable = if include_columns {
                Some(view.nullable().map_err(|e| e.reanchor("dataframe::info_with").with_column(name.clone()))?)
            } else {
                None
            };

            let null_count = if include_columns && include_null_counts {
                Some(view.null_count().map_err(|e| e.reanchor("dataframe::info_with").with_column(name.clone()))?)
            } else {
                None
            };
            let non_null_count = null_count.map(|count| self.row_count.saturating_sub(count));

            let payload_bytes = if include_payload_bytes {
                let bytes = column
                    .payload_size_bytes()
                    .map_err(|e| e.reanchor("dataframe::info_with").with_column(name.clone()))?;
                total_payload_bytes = checked_dataframe_memory_total(total_payload_bytes, bytes, name, "payload")?;
                Some(bytes)
            } else {
                None
            };

            let device_allocated_bytes = if include_device_allocated_bytes {
                let bytes = column
                    .device_allocated_size_bytes()
                    .map_err(|e| e.reanchor("dataframe::info_with").with_column(name.clone()))?;
                total_device_allocated_bytes =
                    checked_dataframe_memory_total(total_device_allocated_bytes, bytes, name, "device_allocated")?;
                Some(bytes)
            } else {
                None
            };

            if let Some(columns_info) = columns_info.as_mut() {
                columns_info.push(DataFrameColumnInfo {
                    position,
                    name: name.clone(),
                    data_type,
                    nullable: nullable.expect("column details always populate nullability"),
                    null_count,
                    non_null_count,
                    memory_accounting: if payload_bytes.is_some() || device_allocated_bytes.is_some() {
                        Some(ColumnMemoryAccounting { payload_bytes, device_allocated_bytes })
                    } else {
                        None
                    },
                });
            }
        }

        let mut dtypes: Vec<DataTypeCount> = dtype_counts
            .into_iter()
            .map(|(data_type, column_count)| DataTypeCount { data_type, column_count })
            .collect();
        dtypes.sort_by_key(|entry| data_type_sort_key(entry.data_type));

        Ok(DataFrameInfo {
            row_count: self.row_count,
            column_count: self.columns.len(),
            dtypes,
            columns: columns_info,
            memory_accounting: if include_payload_bytes || include_device_allocated_bytes {
                Some(DataFrameMemoryAccounting {
                    payload_bytes: include_payload_bytes.then_some(total_payload_bytes),
                    device_allocated_bytes: include_device_allocated_bytes.then_some(total_device_allocated_bytes),
                })
            } else {
                None
            },
        })
    }

    pub(super) fn device_allocated_size_bytes(&self) -> Result<usize> {
        let info = self.info_with(InfoDetail::DEVICE_ALLOCATED)?;
        Ok(info
            .memory_accounting
            .expect("DEVICE_ALLOCATED always populates memory accounting")
            .device_allocated_bytes
            .expect("DEVICE_ALLOCATED always populates device-allocated bytes"))
    }

    /// Return retained-owner identities for this DataFrame's columns.
    ///
    /// This is a metadata-only ownership snapshot for query engines that need
    /// to distinguish schema aliases from newly materialized output columns.
    /// The identity is based on retained column ownership, not byte counts or
    /// raw allocation addresses.
    pub fn memory_identity(&self) -> DataFrameMemoryIdentity {
        let columns = self
            .columns
            .iter()
            .enumerate()
            .map(|(position, (name, column))| DataFrameColumnMemoryIdentity {
                position,
                name: name.clone(),
                identity: ColumnMemoryIdentity::from_column_arc(column),
            })
            .collect();
        DataFrameMemoryIdentity { row_count: self.row_count, columns }
    }
}

fn checked_dataframe_memory_total(
    accumulated: usize,
    column_bytes: usize,
    column_name: &str,
    accounting_kind: &'static str,
) -> Result<usize> {
    accumulated.checked_add(column_bytes).ok_or_else(|| {
        Error::new_with(
            ErrorCode::SizeOverflow,
            "dataframe::info_with",
            "DataFrame memory accounting total overflowed usize",
        )
        .with_column(column_name.to_string())
        .with_detail("accounting_kind", accounting_kind)
        .with_detail("accumulated_bytes", accumulated.to_string())
        .with_detail("column_bytes", column_bytes.to_string())
    })
}

#[cfg(test)]
mod size_accounting_tests {
    use super::*;
    use crate::core::error::ErrorStatus;

    #[test]
    fn dataframe_memory_total_overflow_is_typed_and_does_not_wrap() {
        for accounting_kind in ["payload", "device_allocated"] {
            let error = checked_dataframe_memory_total(usize::MAX, 1, "oversized", accounting_kind)
                .expect_err("usize overflow must be rejected");
            assert_eq!(error.code(), ErrorCode::SizeOverflow);
            assert_eq!(error.status(), ErrorStatus::Permanent);
            assert_eq!(error.detail_value("accounting_kind"), Some(accounting_kind));
            assert_eq!(error.detail_value("accumulated_bytes"), Some(usize::MAX.to_string().as_str()));
            assert_eq!(error.detail_value("column_bytes"), Some("1"));
        }
    }
}
