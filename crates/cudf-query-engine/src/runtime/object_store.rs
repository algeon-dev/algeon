//! Object-store source fetch handles and shared runtime registry.

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use bytes::Bytes;

use crate::error::{QueryError, QueryErrorCode, Result};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct S3ObjectStoreKey {
    pub(crate) bucket: String,
    pub(crate) region: String,
}

pub(crate) struct ObjectStoreFetchResult {
    pub(crate) bytes: Result<Bytes>,
    pub(crate) elapsed: Duration,
}

pub(crate) struct ObjectStoreCustomFetchResult<T> {
    pub(crate) value: Result<T>,
    pub(crate) elapsed: Duration,
}

pub(super) struct SourceObjectStoreRegistry {
    stores: Mutex<HashMap<S3ObjectStoreKey, Arc<dyn object_store::ObjectStore>>>,
}

impl SourceObjectStoreRegistry {
    pub(super) fn new() -> Self {
        Self { stores: Mutex::new(HashMap::new()) }
    }

    pub(super) fn s3_store(&self, key: S3ObjectStoreKey) -> Result<Arc<dyn object_store::ObjectStore>> {
        let mut stores = self.stores.lock().map_err(|_error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "cudf_query_engine.source_object_store.registry",
                "source object-store registry mutex was poisoned",
            )
        })?;
        if let Some(store) = stores.get(&key) {
            return Ok(Arc::clone(store));
        }
        let store = object_store::aws::AmazonS3Builder::from_env()
            .with_bucket_name(key.bucket.as_str())
            .with_region(key.region.as_str())
            .build()
            .map_err(|error| {
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "cudf_query_engine.source_object_store.s3",
                    "failed to build S3 object-store client for native source fetch",
                )
                .with_fact("bucket", key.bucket.as_str())
                .with_fact("region", key.region.as_str())
                .with_source(error)
            })?;
        let store: Arc<dyn object_store::ObjectStore> = Arc::new(store);
        stores.insert(key, Arc::clone(&store));
        Ok(store)
    }
}

impl fmt::Debug for SourceObjectStoreRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cached_store_count = self.stores.lock().map_or(0, |stores| stores.len());
        f.debug_struct("SourceObjectStoreRegistry").field("cached_store_count", &cached_store_count).finish()
    }
}

pub(super) fn default_source_fetch_runtime() -> Arc<tokio::runtime::Runtime> {
    static RUNTIME: OnceLock<Arc<tokio::runtime::Runtime>> = OnceLock::new();
    Arc::clone(RUNTIME.get_or_init(|| {
        Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .thread_name("cudf-source-fetch")
                .enable_all()
                .build()
                .expect("source fetch runtime should initialize"),
        )
    }))
}
