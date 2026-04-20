//! Durable cache for reusable GPU hash-join build artifacts.

use std::{collections::HashMap, sync::Mutex};

use crate::{
    error::{QueryError, Result},
    plan::HashJoinBuildArtifactId,
    runtime::resident::GpuHashJoinBuildHandle,
};

/// Durable cache for reusable GPU hash-join build artifacts.
///
/// Entries are keyed by plan-local [`HashJoinBuildArtifactId`] values. Reuse is
/// still validated against runtime, device placement, build keys, build schema,
/// and null-equality before an artifact is consumed.
#[derive(Default)]
pub struct GpuHashJoinBuildCache {
    cache: Mutex<HashMap<HashJoinBuildArtifactId, GpuHashJoinBuildHandle>>,
}

impl std::fmt::Debug for GpuHashJoinBuildCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let entry_count = self.cache.lock().ok().map(|cache| cache.len());
        f.debug_struct("GpuHashJoinBuildCache").field("entry_count", &entry_count).finish()
    }
}

impl GpuHashJoinBuildCache {
    /// Create an empty durable hash-join build cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of retained build artifacts.
    pub fn len(&self) -> Result<usize> {
        Ok(self.lock()?.len())
    }

    /// Whether the cache currently holds no build artifacts.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Sum of observed retained hash-table device bytes.
    pub fn retained_hash_table_bytes(&self) -> Result<u64> {
        Ok(self.lock()?.values().map(GpuHashJoinBuildHandle::device_allocated_size_bytes).sum())
    }

    pub(crate) fn get(&self, id: &HashJoinBuildArtifactId) -> Result<Option<GpuHashJoinBuildHandle>> {
        Ok(self.lock()?.get(id).cloned())
    }

    pub(crate) fn insert_if_absent(&self, id: HashJoinBuildArtifactId, handle: GpuHashJoinBuildHandle) -> Result<bool> {
        let mut cache = self.lock()?;
        if cache.contains_key(&id) {
            return Ok(false);
        }
        cache.insert(id, handle);
        Ok(true)
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, HashMap<HashJoinBuildArtifactId, GpuHashJoinBuildHandle>>> {
        self.cache.lock().map_err(|_| {
            QueryError::new(
                crate::error::QueryErrorCode::InvalidPlan,
                "nexus_query_engine.gpu_hash_join_build_cache",
                "native GPU hash-join build artifact cache lock was poisoned",
            )
        })
    }
}
