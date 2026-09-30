//! The DataFusion side of the configured S3 storage.
//!
//! Native reads resolve `s3://` sources through the runtime storage profile
//! registered under [`S3_STORAGE_PROFILE_ID`]. A session installed by the
//! backend resolves the same URLs for DataFusion scans and footer pruning
//! from that profile, so both read with one endpoint, region, and credential
//! source.

use std::sync::Arc;

use datafusion::{
    common::{DataFusionError, Result as DataFusionResult},
    execution::{SessionStateBuilder, object_store::ObjectStoreRegistry, runtime_env::RuntimeEnvBuilder},
    object_store::ObjectStore,
};
use query_engine::runtime::{S3SourceStorageProfile, SourceStorageProfile};
use url::Url;

use crate::{
    error::{Error, ErrorCode, Result},
    native::S3_STORAGE_PROFILE_ID,
};

/// The region a bucket store uses when the storage names none: the ambient
/// AWS region, then `us-east-1`, as the native source reader resolves it.
fn fallback_region(storage: &S3SourceStorageProfile) -> String {
    storage
        .region
        .iter()
        .cloned()
        .chain(["AWS_REGION", "AWS_DEFAULT_REGION"].into_iter().filter_map(|name| std::env::var(name).ok()))
        .find(|region| !region.trim().is_empty())
        .unwrap_or_else(|| "us-east-1".to_string())
}

/// Replaces the builder's object store registry with one that serves
/// `s3://<bucket>` URLs from `storage` when no store is registered for the
/// bucket. Stores the caller registered keep precedence.
pub(crate) fn install_s3_storage(
    mut builder: SessionStateBuilder,
    storage: &S3SourceStorageProfile,
) -> Result<SessionStateBuilder> {
    let runtime_env = builder.runtime_env().clone().unwrap_or_default();
    let registry = Arc::new(S3StorageObjectStoreRegistry {
        inner: Arc::clone(&runtime_env.object_store_registry),
        fallback_region: fallback_region(storage),
        storage: SourceStorageProfile::s3(S3_STORAGE_PROFILE_ID, storage.clone()),
    });
    let runtime_env = RuntimeEnvBuilder::from_runtime_env(&runtime_env)
        .with_object_store_registry(registry)
        .build_arc()
        .map_err(|error| {
            Error::new(
                ErrorCode::InvalidRequest,
                "algeon_datafusion.s3_storage.install",
                "failed to install the S3 storage object store registry",
            )
            .with_source(error)
        })?;
    Ok(builder.with_runtime_env(runtime_env))
}

#[derive(Debug)]
struct S3StorageObjectStoreRegistry {
    inner: Arc<dyn ObjectStoreRegistry>,
    storage: SourceStorageProfile,
    fallback_region: String,
}

impl ObjectStoreRegistry for S3StorageObjectStoreRegistry {
    fn register_store(&self, url: &Url, store: Arc<dyn ObjectStore>) -> Option<Arc<dyn ObjectStore>> {
        self.inner.register_store(url, store)
    }

    fn deregister_store(&self, url: &Url) -> DataFusionResult<Arc<dyn ObjectStore>> {
        self.inner.deregister_store(url)
    }

    fn get_store(&self, url: &Url) -> DataFusionResult<Arc<dyn ObjectStore>> {
        match self.inner.get_store(url) {
            Err(_) if url.scheme() == "s3" && url.host_str().is_some_and(|bucket| !bucket.is_empty()) => {
                let bucket = url.host_str().unwrap_or_default();
                let store =
                    self.storage.build_s3_object_store(bucket, self.fallback_region.clone()).map_err(|error| {
                        DataFusionError::from(Error::from_native(
                            "algeon_datafusion.s3_storage.object_store",
                            "failed to build the S3 object store for a bucket",
                            error,
                        ))
                    })?;
                // Later lookups of the bucket reuse this store.
                self.inner.register_store(url, Arc::clone(&store));
                Ok(store)
            },
            result => result,
        }
    }
}
