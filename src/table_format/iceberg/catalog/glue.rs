//! Leaf-owned AWS Glue catalog contracts.
//!
//! Catalog connection, FileIO construction, and the immutable
//! [`SourceAccessPolicy`](datafusion_nexus_iceberg::SourceAccessPolicy) belong
//! to the DataFusion-free Iceberg leaf. The root catalog module only adapts a
//! resolved leaf backend to DataFusion's catalog provider traits.

pub use datafusion_nexus_iceberg::{GlueIcebergCatalogBackend, GlueIcebergCatalogConfig};
