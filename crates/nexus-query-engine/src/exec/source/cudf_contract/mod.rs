//! cuDF source-contract lowering for local and remote Parquet reads.

#[cfg(test)]
use super::{ExecutionContext, LocalParquetFile, LocalParquetSource, QueryError, QueryErrorCode, SourceChunk};

mod error;
mod local;
mod remote;
mod schema;

// Re-export internals for unit tests via `use super::*`. The parent module only
// names them through the tests child, so unused-import lint does not apply.
#[cfg(test)]
#[allow(unused_imports)]
pub(super) use error::{
    cudf_error_for_chunk, cudf_error_for_file, cudf_error_for_location, cudf_error_for_source, path_fact,
    source_has_object_store_location,
};
#[cfg(test)]
#[allow(unused_imports)]
pub(super) use local::{
    attach_cudf_source_reader_predicate, chunk_with_reader_decimal_types, cudf_local_parquet_file,
    cudf_local_parquet_file_for_chunk, cudf_local_parquet_source_for_chunk, cudf_local_parquet_source_for_file,
    cudf_local_parquet_source_from_file, cudf_local_parquet_source_from_files,
    footer_verified_reader_decimal_types_for_chunk, footer_verified_reader_decimal_types_for_chunk_with_cache,
};
pub(super) use local::{
    cudf_local_parquet_source_chunk, cudf_local_parquet_source_chunk_batch, cudf_local_parquet_source_file_selection,
};
pub use remote::{ResolvedRemoteParquetDatasourceProbe, probe_resolved_remote_parquet_datasource};
#[cfg(test)]
#[allow(unused_imports)]
pub(super) use remote::{
    attach_cudf_remote_parquet_contract, cudf_remote_parquet_profile, cudf_remote_parquet_profile_from_storage,
    cudf_remote_parquet_source_for_resolved_source, remote_parquet_capability_disposition_label,
    resolve_remote_direct_default_chain_credential, resolved_remote_kvikio_access, resolved_remote_uri_scheme,
    validate_remote_kvikio_capability, validate_remote_parquet_datasource_probe_request,
};
#[cfg(test)]
#[allow(unused_imports)]
pub(super) use schema::{
    attach_cudf_reader_decimal_types, cudf_partition_columns, cudf_partition_value, cudf_reader_decimal_type,
    cudf_source_column_ref, cudf_source_output_columns, cudf_source_required_columns, cudf_source_schema,
};

#[cfg(test)]
mod tests;
