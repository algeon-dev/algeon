//! cuDF source-contract lowering for local and remote Parquet reads.

mod error;
mod local;
mod remote;
mod schema;

pub(super) use local::{
    cudf_local_parquet_source_chunk, cudf_local_parquet_source_chunk_batch, cudf_local_parquet_source_file_selection,
};
pub use remote::{ResolvedRemoteParquetDatasourceProbe, probe_resolved_remote_parquet_datasource};

#[cfg(test)]
mod tests;
