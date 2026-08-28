//! Leaf-owned Iceberg REST catalog contracts.
//!
//! Catalog connection, FileIO construction, and the immutable
//! [`SourceAccessPolicy`](nexus_iceberg::SourceAccessPolicy) belong
//! to the DataFusion-free Iceberg leaf. The root catalog module only adapts a
//! resolved leaf backend to DataFusion's catalog provider traits.

pub use nexus_iceberg::{RestCatalogBackend, RestCatalogConfig};
