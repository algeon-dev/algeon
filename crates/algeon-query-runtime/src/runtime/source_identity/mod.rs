//! Stable source identities used by runtime-owned source residency.

mod fingerprint;
mod local_parquet;

#[doc(hidden)]
pub use local_parquet::GpuLocalParquetSourceCacheKey;
