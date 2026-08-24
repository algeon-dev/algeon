#[path = "gpu_backend_heterogeneous_active_limits.rs"]
mod gpu_backend_heterogeneous_active_limits;

#[cfg(feature = "qualification")]
#[path = "gpu_backend_fault_injection.rs"]
mod gpu_backend_fault_injection;
