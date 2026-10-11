//! Parquet file I/O.

mod evidence;
mod options;
mod read;
mod remote;
mod write;

#[cfg(test)]
mod tests;

pub use evidence::{ParquetPruningEvidence, ParquetReadEvidence};
pub use options::ParquetTableReadOptions;
pub(crate) use options::{ParquetMultiTableInput, ParquetMultiTableReadOptions};
pub use read::read_parquet_table;
pub(crate) use read::{read_parquet_multi_table, read_parquet_table_with_evidence};
pub use remote::{
    RemoteCredentialMode, RemoteParquetBackendBuildStatus, RemoteParquetCapability, RemoteParquetCapabilityDisposition,
    RemoteParquetDatasourceProbe, RemoteParquetProfile, RemoteParquetSource, RemoteReadMode,
    probe_remote_parquet_datasource, remote_parquet_backend_build_status, remote_parquet_capability,
};
pub(crate) use write::write_parquet_with_exec;
pub use write::{ParquetDictionaryPolicy, ParquetTableWriteOptions, write_parquet};
