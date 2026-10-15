//! Parquet table reader option types.

use std::{
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
};

use typed_builder::TypedBuilder;

use super::remote::RemoteParquetSource;
use crate::{
    core::error::{self, Error, ErrorCode, Result},
    ops::ast::AstExpression,
    runtime::Runtime,
};

/// Options for reading Parquet into a low-level [`crate::core::table::Table`].
///
/// The source must be exactly one of:
///
/// - `filepath`: a path-based libcudf/KvikIO read.
/// - `filebytes`: a borrowed host-memory Parquet payload.
///
/// Host-buffer reads borrow `filebytes` only for the duration of the reader
/// call. The returned table owns device-resident decoded columns and does not
/// retain a pointer into the host buffer.
///
/// For DataFrame reads, use
/// [`DataFrame::read_parquet`](crate::DataFrame::read_parquet).
#[derive(Debug, Clone, TypedBuilder)]
pub struct ParquetTableReadOptions<'a> {
    #[builder(default, setter(transform = |path: impl AsRef<Path>| Some(path.as_ref().to_path_buf())))]
    pub(super) filepath: Option<PathBuf>,
    #[builder(default, setter(strip_option))]
    pub(super) remote_source: Option<RemoteParquetSource>,
    #[builder(default, setter(transform = |bytes: &'a [u8]| Some(bytes)))]
    pub(super) filebytes: Option<&'a [u8]>,
    #[builder(default, setter(transform = |names: &[&str]| Some(names.iter().map(|s| (*s).to_string()).collect())))]
    pub(super) columns: Option<Vec<String>>,
    #[builder(default)]
    pub(super) skip_rows: usize,
    #[builder(default, setter(transform = |num_rows: usize| Some(num_rows)))]
    pub(super) num_rows: Option<usize>,
    #[builder(default, setter(transform = |groups: &[i32]| if groups.is_empty() { None } else { Some(groups.to_vec()) }))]
    pub(super) row_groups: Option<Vec<i32>>,
    #[builder(default, setter(strip_option, into))]
    pub(super) predicate: Option<Arc<AstExpression>>,
    #[builder(setter(prefix = "with_", transform = |runtime: &Runtime| runtime.clone()))]
    pub(super) runtime: Runtime,
    #[builder(default, setter(skip))]
    pub(super) pass_read_limit: Option<u64>,
    #[builder(default, setter(skip))]
    _lifetime: PhantomData<&'a [u8]>,
}

/// Options for reading multiple Parquet paths into one low-level [`crate::core::table::Table`].
///
/// This is the multi-source counterpart to [`ParquetTableReadOptions`] for
/// path-compatible reads. File order is preserved by libcudf. When row groups are
/// supplied, each entry in `row_groups_per_file` applies to the corresponding
/// `inputs` entry.
#[derive(Debug, Clone)]
pub(crate) struct ParquetMultiTableReadOptions {
    pub(crate) inputs: Vec<ParquetMultiTableInput>,
    pub(crate) columns: Option<Vec<String>>,
    pub(crate) row_groups_per_file: Option<Vec<Vec<i32>>>,
    pub(crate) predicate: Option<Arc<AstExpression>>,
    pub(crate) runtime: Runtime,
    pub(crate) pass_read_limit: Option<u64>,
}

#[derive(Debug, Clone)]
pub(crate) enum ParquetMultiTableInput {
    LocalPath(PathBuf),
    RemoteSource(RemoteParquetSource),
}

impl ParquetMultiTableInput {
    pub(super) fn display_path(&self) -> String {
        match self {
            Self::LocalPath(path) => path.display().to_string(),
            Self::RemoteSource(source) => source.diagnostic_location(),
        }
    }

    pub(super) fn remote_source(&self) -> Option<&RemoteParquetSource> {
        match self {
            Self::LocalPath(_) => None,
            Self::RemoteSource(source) => Some(source),
        }
    }
}

impl ParquetMultiTableReadOptions {
    pub(crate) fn from_inputs(
        inputs: Vec<ParquetMultiTableInput>,
        columns: Option<Vec<String>>,
        row_groups_per_file: Option<Vec<Vec<i32>>>,
        predicate: Option<Arc<AstExpression>>,
        runtime: Runtime,
    ) -> Self {
        Self { inputs, columns, row_groups_per_file, predicate, runtime, pass_read_limit: None }
    }
}

impl<'a> ParquetTableReadOptions<'a> {
    /// Bound the read's Parquet read and decompression scratch to `limit`
    /// bytes; `None` leaves it unbounded.
    pub(crate) fn with_pass_read_limit(mut self, limit: Option<u64>) -> Self {
        self.pass_read_limit = limit;
        self
    }

    pub(crate) fn from_parts(
        filepath: PathBuf,
        columns: Option<Vec<String>>,
        skip_rows: usize,
        num_rows: Option<usize>,
        row_groups: Option<Vec<i32>>,
        predicate: Option<Arc<AstExpression>>,
        runtime: Runtime,
    ) -> Self {
        Self {
            filepath: Some(filepath),
            remote_source: None,
            filebytes: None,
            columns,
            skip_rows,
            num_rows,
            row_groups,
            predicate,
            runtime,
            pass_read_limit: None,
            _lifetime: PhantomData,
        }
    }

    pub(crate) fn from_host_buffer_parts(
        filebytes: &'a [u8],
        columns: Option<Vec<String>>,
        skip_rows: usize,
        num_rows: Option<usize>,
        row_groups: Option<Vec<i32>>,
        predicate: Option<Arc<AstExpression>>,
        runtime: Runtime,
    ) -> Self {
        Self {
            filepath: None,
            remote_source: None,
            filebytes: Some(filebytes),
            columns,
            skip_rows,
            num_rows,
            row_groups,
            predicate,
            runtime,
            pass_read_limit: None,
            _lifetime: PhantomData,
        }
    }

    pub(crate) fn from_remote_parts(
        remote_source: RemoteParquetSource,
        columns: Option<Vec<String>>,
        skip_rows: usize,
        num_rows: Option<usize>,
        row_groups: Option<Vec<i32>>,
        predicate: Option<Arc<AstExpression>>,
        runtime: Runtime,
    ) -> Self {
        Self {
            filepath: None,
            remote_source: Some(remote_source),
            filebytes: None,
            columns,
            skip_rows,
            num_rows,
            row_groups,
            predicate,
            runtime,
            pass_read_limit: None,
            _lifetime: PhantomData,
        }
    }
}

pub(super) fn checked_i64_from_usize(value: usize, field: &'static str, operation: &'static str) -> Result<i64> {
    i64::try_from(value).map_err(|_| error::range_limit_error(operation, field, "i64::MAX", value))
}

pub(super) fn row_group_combination_error(
    row_group_count: usize,
    skip_rows: usize,
    num_rows_context: String,
    operation: &'static str,
) -> Error {
    Error::new_with(
        ErrorCode::UnsupportedOperation,
        operation,
        "row_groups cannot be combined with skip_rows or num_rows",
    )
    .with_detail("row_group_count", row_group_count.to_string())
    .with_detail("skip_rows", skip_rows.to_string())
    .with_detail("num_rows", num_rows_context)
    .with_detail("resource", "row_groups")
    .with_detail("reason", "incompatible_option_combination")
    .with_detail("remedy", "use row_groups by itself, or remove row_groups and use skip_rows/num_rows")
}
