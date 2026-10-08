//! cuDF source-contract lowering for local and remote Parquet reads.
//!
//! The planner describes a Parquet source with `LocalParquetSource`, `LocalParquetFile` and
//! `SourceChunk`. cuDF reads from its own `LocalParquetSource`, `LocalParquetFile` and
//! `SourceChunk` values, which carry the same facts in the shape the reader validates. This module
//! converts one to the other and owns no execution state; every function is a pure conversion
//! over its inputs, except where it reads a Parquet footer or resolves S3 credentials.
//!
//! - `local` builds the source, file and chunk descriptors and attaches the lowered reader
//!   predicate.
//! - `schema` converts the logical schema, output and pure-filter columns, partition columns and
//!   reader decimal types.
//! - `remote` attaches the S3 KvikIO contract to object-store files.
//! - `error` decides when a cuDF error may keep its source chain.
//!
//! Design owner: `website/docs/design/data-path.mdx`.

mod error;
mod local;
mod remote;
mod schema;

pub(super) use local::{
    cudf_local_parquet_source_chunk, cudf_local_parquet_source_chunk_batch, cudf_local_parquet_source_file_selection,
};
#[cfg(test)]
mod tests;
