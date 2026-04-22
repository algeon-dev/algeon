//! Runtime utilities for device-pinned execution resources and retained provenance.
//!
//! This module exposes the public runtime control surface:
//!
//! - [`Runtime`] owns one CUDA device binding plus default [`CudaStream`] and
//!   [`crate::core::memory::DeviceMemoryResource`] handles
//! - [`ExecutionOverrides`] borrows a stream and/or memory resource for one
//!   operation
//! - [`CudaStream`] and [`CudaEvent`] expose low-level CUDA ordering handles
//! - [`DeviceId`] is a validated CUDA device ordinal

pub mod device;
pub(crate) mod device_guard;
pub mod event;
pub mod execution_context;
pub(crate) mod execution_domain;
mod owner;
pub mod stream;

pub use device::DeviceId;
pub use event::CudaEvent;
pub use execution_context::ExecutionOverrides;
pub use owner::{Runtime, RuntimeBuilder};
pub use stream::CudaStream;
