//! Stable source identities that remain meaningful without cache construction.

mod fingerprint;
mod local_parquet;

pub(crate) use local_parquet::GpuLocalParquetSourceCacheKey;
