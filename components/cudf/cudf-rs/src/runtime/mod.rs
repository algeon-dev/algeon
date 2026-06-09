//! Runtime control for per-operation execution authority and retained provenance.

pub mod device;
pub(crate) mod device_guard;
pub mod event;
pub mod execution_context;
pub(crate) mod execution_domain;
mod owner;
pub mod stream;

pub use device::DeviceId;
pub use event::CudaEvent;
#[doc(hidden)]
pub use execution_context::__DebugExecutionContext;
pub use execution_context::ExecutionContext;
pub use owner::{Runtime, RuntimeBuilder};
pub use stream::CudaStream;
