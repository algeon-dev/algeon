//! Stable source identities used by runtime-owned source residency.

mod fingerprint;
mod local_parquet;

pub(crate) use local_parquet::GpuLocalParquetSourceCacheKey;
