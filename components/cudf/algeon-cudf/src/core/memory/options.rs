//! Physical-base options and validation for the device memory manager.

use crate::core::error::{self, Result};

/// Physical base used by the process-local device memory manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMemoryManagerBase {
    /// Use CUDA's stream-ordered async memory pool.
    CudaAsync { initial_pool_size: Option<usize>, release_threshold: Option<usize> },
    /// Use RMM's pool resource around the raw CUDA resource.
    Pool { initial_pool_size: usize, maximum_pool_size: Option<usize> },
}

impl Default for DeviceMemoryManagerBase {
    fn default() -> Self {
        Self::CudaAsync { initial_pool_size: None, release_threshold: None }
    }
}

/// Options applied when a device memory manager is first created.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceMemoryManagerOptions {
    pub base: DeviceMemoryManagerBase,
    /// Immutable hard limit enforced by the process manager. `None` leaves the
    /// selected physical base unpartitioned.
    pub allocation_limit_bytes: Option<usize>,
}

impl DeviceMemoryManagerOptions {
    fn normalized(self) -> Self {
        let base = match self.base {
            DeviceMemoryManagerBase::CudaAsync { initial_pool_size, release_threshold } => {
                DeviceMemoryManagerBase::CudaAsync {
                    initial_pool_size: initial_pool_size.filter(|size| *size != 0),
                    release_threshold: release_threshold.filter(|threshold| *threshold != usize::MAX),
                }
            },
            DeviceMemoryManagerBase::Pool { initial_pool_size, maximum_pool_size } => DeviceMemoryManagerBase::Pool {
                initial_pool_size,
                maximum_pool_size: maximum_pool_size.filter(|size| *size != 0),
            },
        };
        Self { base, allocation_limit_bytes: self.allocation_limit_bytes }
    }

    /// Validate this configuration without consulting CUDA or process-global
    /// allocator state.
    pub fn validate(&self) -> Result<()> {
        if self.allocation_limit_bytes == Some(0) {
            return Err(invalid_memory_manager_options(
                "allocation_limit_bytes",
                "allocation limit must be greater than zero when configured",
            ));
        }
        let normalized = (*self).normalized();
        let Some(limit) = normalized.allocation_limit_bytes else {
            if let DeviceMemoryManagerBase::Pool { initial_pool_size, maximum_pool_size: Some(maximum) } =
                normalized.base
                && maximum < initial_pool_size
            {
                return Err(invalid_memory_manager_options(
                    "maximum_pool_size",
                    "maximum pool size must not be below the initial pool size",
                ));
            }
            return Ok(());
        };
        match normalized.base {
            DeviceMemoryManagerBase::CudaAsync { initial_pool_size: Some(initial), .. } if initial > limit => Err(
                invalid_memory_manager_options("initial_pool_size", "initial pool size exceeds the allocation limit"),
            ),
            DeviceMemoryManagerBase::Pool { initial_pool_size, .. } if initial_pool_size > limit => Err(
                invalid_memory_manager_options("initial_pool_size", "initial pool size exceeds the allocation limit"),
            ),
            DeviceMemoryManagerBase::Pool { initial_pool_size, maximum_pool_size: Some(maximum) }
                if maximum < initial_pool_size =>
            {
                Err(invalid_memory_manager_options(
                    "maximum_pool_size",
                    "maximum pool size must not be below the initial pool size",
                ))
            },
            DeviceMemoryManagerBase::Pool { maximum_pool_size: Some(maximum), .. } if maximum > limit => Err(
                invalid_memory_manager_options("maximum_pool_size", "maximum pool size exceeds the allocation limit"),
            ),
            _ => Ok(()),
        }
    }

    pub(super) fn to_sys(self) -> Result<cudf_sys::cudf_memory_manager_options_t> {
        self.validate()?;
        let mut out = cudf_sys::cudf_memory_manager_options_t {
            allocation_limit_bytes: self.allocation_limit_bytes.unwrap_or(0) as u64,
            ..Default::default()
        };
        match self.base {
            DeviceMemoryManagerBase::CudaAsync { initial_pool_size, release_threshold } => {
                out.base_kind = cudf_sys::CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC;
                out.initial_pool_size_bytes = initial_pool_size.unwrap_or(0) as u64;
                out.release_threshold_bytes = release_threshold.unwrap_or(usize::MAX) as u64;
            },
            DeviceMemoryManagerBase::Pool { initial_pool_size, maximum_pool_size } => {
                out.base_kind = cudf_sys::CUDF_MEMORY_MANAGER_BASE_POOL;
                out.initial_pool_size_bytes = initial_pool_size as u64;
                out.maximum_pool_size_bytes = maximum_pool_size.unwrap_or(0) as u64;
            },
        }
        Ok(out)
    }

    pub(super) fn from_sys(value: cudf_sys::cudf_memory_manager_options_t) -> Result<Self> {
        let initial_pool_size = memory_manager_option_bytes(value.initial_pool_size_bytes, "initial_pool_size_bytes")?;
        let base = match value.base_kind {
            cudf_sys::CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC => {
                let release_threshold = if value.release_threshold_bytes == u64::MAX {
                    None
                } else {
                    Some(memory_manager_option_bytes(value.release_threshold_bytes, "release_threshold_bytes")?)
                };
                DeviceMemoryManagerBase::CudaAsync {
                    initial_pool_size: (initial_pool_size != 0).then_some(initial_pool_size),
                    release_threshold,
                }
            },
            cudf_sys::CUDF_MEMORY_MANAGER_BASE_POOL => {
                let maximum_pool_size = if value.maximum_pool_size_bytes == 0 {
                    None
                } else {
                    Some(memory_manager_option_bytes(value.maximum_pool_size_bytes, "maximum_pool_size_bytes")?)
                };
                DeviceMemoryManagerBase::Pool { initial_pool_size, maximum_pool_size }
            },
            other => {
                return Err(error::Error::new_with(
                    error::ErrorCode::ForeignInvariantViolation,
                    "memory::device_memory_manager::options_for_device",
                    "FFI returned an unknown memory-manager base kind",
                )
                .with_detail("base_kind", other.to_string()));
            },
        };
        let allocation_limit_bytes = if value.allocation_limit_bytes == 0 {
            None
        } else {
            Some(memory_manager_option_bytes(value.allocation_limit_bytes, "allocation_limit_bytes")?)
        };
        let options = Self { base, allocation_limit_bytes };
        options.validate()?;
        Ok(options)
    }
}

fn invalid_memory_manager_options(field: &'static str, message: &'static str) -> error::Error {
    error::Error::new_with(error::ErrorCode::ValueOutOfRange, "memory::device_memory_manager::options", message)
        .with_detail("field", field)
        .with_detail("reason", "invalid_device_memory_manager_options")
}

fn memory_manager_option_bytes(value: u64, field: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        error::Error::new_with(
            error::ErrorCode::ValueOutOfRange,
            "memory::device_memory_manager::options_for_device",
            "installed device memory manager option does not fit usize",
        )
        .with_detail("field", field)
        .with_detail("value", value.to_string())
    })
}

/// Options for an opt-in query memory domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryMemoryDomainOptions {
    pub grant_bytes: usize,
    pub ceiling_bytes: usize,
}

impl QueryMemoryDomainOptions {
    /// Create a domain whose initial grant and immutable logical ceiling are equal.
    pub fn new(grant_bytes: usize) -> Self {
        Self { grant_bytes, ceiling_bytes: grant_bytes }
    }

    /// Set the immutable logical ceiling for grant resizing.
    pub fn with_ceiling_bytes(mut self, ceiling_bytes: usize) -> Self {
        self.ceiling_bytes = ceiling_bytes;
        self
    }
}
