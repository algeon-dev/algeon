//! Lakehouse cache-fill and object-store accessors for execution contexts.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use ::object_store::ObjectStoreExt;
#[cfg(any(test, feature = "test-hooks"))]
use bytes::Bytes;
use query_planning::error::{QueryError, QueryErrorCode, Result};

use super::{
    super::{
        CacheFillByteBudget, CacheFillBytePermit, DeploymentIoCapability, LakehouseCacheFillConfig,
        SourceStorageProfile,
        cache_events::{LakehouseCacheFillEventSink, LakehouseCacheFillEventSnapshot},
        cache_fill::CacheFillOperationPermit,
        object_store::{ObjectStoreCustomFetchResult, ObjectStoreFetchResult, S3ObjectStoreKey},
    },
    ExecutionContext, QueryServiceStreamPool,
};

impl ExecutionContext {
    /// Returns the service-level lakehouse cache-fill resource configuration.
    pub fn lakehouse_cache_fill_config(&self) -> &LakehouseCacheFillConfig {
        &self.lakehouse_cache_fill_config
    }

    /// Returns the service-level CUDA stream pool used by source-read workers.
    pub fn service_stream_pool(&self) -> QueryServiceStreamPool {
        QueryServiceStreamPool {
            pool: self.service_stream_pool.clone(),
            service_epoch: self.service_epoch.clone(),
            query_anchor: self.query_anchor.clone(),
        }
    }

    /// Timeout used by blocking lakehouse cache maintenance resource acquisition.
    pub fn lakehouse_cache_fill_acquire_timeout(&self) -> Duration {
        self.lakehouse_cache_fill_config.acquire_timeout
    }

    /// Returns the service-wide background lakehouse cache-fill byte budget.
    pub fn lakehouse_cache_fill_budget(&self) -> CacheFillByteBudget {
        self.lakehouse_cache_fill_budget.clone()
    }

    /// Maximum background lakehouse cache-fill operations admitted concurrently.
    pub fn lakehouse_cache_fill_max_concurrent_ops(&self) -> usize {
        self.lakehouse_cache_fill_concurrency.max_concurrent_ops()
    }

    /// Structured asynchronous lakehouse cache-fill completion/failure events.
    #[doc(hidden)]
    pub fn lakehouse_cache_fill_events(&self) -> LakehouseCacheFillEventSink {
        self.lakehouse_cache_fill_events.clone()
    }

    /// Returns credential-free background cache-fill events observed by this context so far.
    pub fn lakehouse_cache_fill_event_snapshots(&self) -> Vec<LakehouseCacheFillEventSnapshot> {
        self.lakehouse_cache_fill_events.events().iter().map(LakehouseCacheFillEventSnapshot::from).collect()
    }

    /// Drains credential-free background cache-fill events observed by this context so far.
    pub fn drain_lakehouse_cache_fill_event_snapshots(&self) -> Vec<LakehouseCacheFillEventSnapshot> {
        self.lakehouse_cache_fill_events.drain().iter().map(LakehouseCacheFillEventSnapshot::from).collect()
    }

    #[doc(hidden)]
    pub fn acquire_lakehouse_cache_fill_operation_with_timeout(
        &self,
        timeout: Duration,
        operation: &'static str,
    ) -> Result<Option<CacheFillOperationPermit>> {
        self.lakehouse_cache_fill_concurrency.acquire_with_timeout(&self.cancellation, timeout, operation)
    }

    #[doc(hidden)]
    pub fn acquire_lakehouse_cache_fill_budget_with_timeout(
        &self,
        bytes: u64,
        timeout: Duration,
        operation: &'static str,
    ) -> Result<Option<CacheFillBytePermit>> {
        self.lakehouse_cache_fill_budget.acquire_with_timeout(&self.cancellation, bytes, timeout, operation)
    }

    #[doc(hidden)]
    pub fn s3_object_store(&self, key: S3ObjectStoreKey) -> Result<Arc<dyn ::object_store::ObjectStore>> {
        self.source_object_stores.s3_store(key)
    }

    #[doc(hidden)]
    pub fn source_storage_profile(&self, id: &str) -> Result<&SourceStorageProfile> {
        self.source_object_stores.profile(id)
    }

    /// Resolves default-chain AWS credentials from the object-store provider.
    ///
    /// Remote KvikIO reads use this to pass resolved identity into lower layers
    /// that lack a default chain. Recent results are reused briefly; on a cache
    /// miss, resolution runs on the object-store control runtime while this call
    /// blocks for the result.
    #[doc(hidden)]
    pub fn resolve_s3_default_chain_credential(
        &self,
        key: S3ObjectStoreKey,
        operation: &'static str,
    ) -> Result<Arc<::object_store::aws::AwsCredential>> {
        if let Some(credential) = self.source_object_stores.recently_resolved_s3_credential(&key) {
            return Ok(credential);
        }
        let storage_profile_id = key.storage_profile_id.clone();
        let provider = self.source_object_stores.s3_credential_provider(key.clone())?;
        let (sender, receiver) = std::sync::mpsc::channel();
        self.object_store_control_runtime.spawn(async move {
            let credential = provider.get_credential().await;
            let _ = sender.send(credential);
        });
        let credential_unavailable = |query_error: QueryError| {
            let query_error = query_error
                .with_fact("resource", "remote_parquet_source")
                .with_fact("credential_source", "default_chain")
                .with_fact("reason", "default_chain_credentials_unavailable");
            match storage_profile_id.as_deref() {
                Some(profile_id) => query_error.with_fact("storage_profile_id", profile_id),
                None => query_error,
            }
        };
        let credential = receiver.recv().map_err(|_error| {
            credential_unavailable(QueryError::new(
                QueryErrorCode::Internal,
                operation,
                "source credential resolution worker stopped before returning a result",
            ))
        })?;
        let credential = credential.map_err(|error| {
            credential_unavailable(
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    operation,
                    "failed to resolve AWS default-chain credentials for native remote-direct read",
                )
                .with_source(error),
            )
        })?;
        self.source_object_stores.record_resolved_s3_credential(key, Arc::clone(&credential));
        Ok(credential)
    }

    #[doc(hidden)]
    pub fn spawn_cache_maintenance_fetch(
        &self,
        store: Arc<dyn ::object_store::ObjectStore>,
        location: ::object_store::path::Path,
        on_result: impl FnOnce(ObjectStoreFetchResult) + Send + 'static,
    ) {
        Self::spawn_cache_maintenance_fetch_on_runtime(
            self.object_store_control_runtime.as_ref(),
            store,
            location,
            on_result,
        )
    }

    /// Spawns caller-provided object-store control-plane work.
    ///
    /// The caller composes any combination of `get_range` / `get_ranges` calls
    /// inside the future. Elapsed time covers the entire future, including any
    /// rewrite work the caller performs inside it.
    #[doc(hidden)]
    pub fn spawn_object_store_control_task<F, Fut, T>(
        &self,
        future_factory: F,
        on_result: impl FnOnce(ObjectStoreCustomFetchResult<T>) + Send + 'static,
    ) -> tokio::task::JoinHandle<()>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.object_store_control_runtime.spawn(async move {
            let timer = Instant::now();
            let value = future_factory().await;
            on_result(ObjectStoreCustomFetchResult { value, elapsed: timer.elapsed() });
        })
    }

    /// Spawns caller-provided asynchronous lakehouse cache maintenance.
    #[doc(hidden)]
    pub fn spawn_lakehouse_cache_maintenance<F, Fut, T>(
        &self,
        future_factory: F,
        on_result: impl FnOnce(ObjectStoreCustomFetchResult<T>) + Send + 'static,
    ) -> tokio::task::JoinHandle<()>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: std::future::Future<Output = Result<T>> + Send + 'static,
        T: Send + 'static,
    {
        self.object_store_control_runtime.spawn(async move {
            let timer = Instant::now();
            let value = future_factory().await;
            on_result(ObjectStoreCustomFetchResult { value, elapsed: timer.elapsed() });
        })
    }

    /// Returns deployment-level IO capability diagnostics.
    pub fn deployment_io_capability(&self) -> &DeploymentIoCapability {
        &self.deployment_io_capability
    }

    #[cfg(any(test, feature = "test-hooks"))]
    #[cfg_attr(not(test), expect(dead_code, reason = "used by in-crate runtime source tests"))]
    pub(in crate::runtime) fn fetch_object_store_bytes_on_runtime(
        object_store_control_runtime: &tokio::runtime::Runtime,
        store: Arc<dyn ::object_store::ObjectStore>,
        location: ::object_store::path::Path,
    ) -> Result<Bytes> {
        let (sender, receiver) = std::sync::mpsc::channel();
        Self::spawn_cache_maintenance_fetch_on_runtime(object_store_control_runtime, store, location, move |result| {
            let _ = sender.send(result.bytes);
        });
        receiver.recv().map_err(|error| {
            QueryError::new(
                QueryErrorCode::Internal,
                "query_engine.lakehouse_cache.maintenance.receive",
                "object-store cache maintenance worker stopped before returning a result",
            )
            .with_source(error)
        })?
    }

    fn spawn_cache_maintenance_fetch_on_runtime(
        object_store_control_runtime: &tokio::runtime::Runtime,
        store: Arc<dyn ::object_store::ObjectStore>,
        location: ::object_store::path::Path,
        on_result: impl FnOnce(ObjectStoreFetchResult) + Send + 'static,
    ) {
        object_store_control_runtime.spawn(async move {
            let path_fact = query_planning::source::redacted_object_store_path_diagnostic(location.as_ref());
            let bytes = async move {
                let result = store.get(&location).await?;
                result.bytes().await
            }
            .await
            .map_err(|_error| {
                QueryError::new(
                    QueryErrorCode::CudfDependencyFailure,
                    "query_engine.lakehouse_cache.maintenance.object_store",
                    "failed to fetch lakehouse cache object through object_store",
                )
                .with_fact("path", path_fact)
            });
            on_result(ObjectStoreFetchResult { bytes });
        });
    }
}
