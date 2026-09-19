//! Common DataFusion-facing GPU function discovery and validation catalog.
//!
//! GPU SQL families provide descriptors and static validation through the
//! crate-private `GpuFunctionFamily` trait. This module owns the immutable
//! cross-family catalog plus the stable `gpu_*` metadata UDTFs; it deliberately
//! does not own execution or family-specific planning.

mod catalog;
mod descriptor;
mod family;
mod registration;
mod table_functions;
mod validation;

pub(crate) use catalog::GpuFunctionCatalog;
pub(crate) use descriptor::{GpuFunctionAvailability, GpuFunctionDescriptor};
pub(crate) use family::GpuFunctionFamily;
pub(crate) use registration::install_gpu_function_family;
pub(crate) use table_functions::gpu_metadata_table_functions;
pub use table_functions::{GPU_DESCRIBE_FUNCTION_FUNCTION, GPU_LIST_FUNCTIONS_FUNCTION, GPU_VALIDATE_CALL_FUNCTION};

#[cfg(feature = "server")]
#[doc(hidden)]
pub mod server_internal {
    pub use super::table_functions::is_builtin_gpu_metadata_table_source;
}

pub(crate) use validation::{GpuCallRequest, GpuCallValidation, parse_table_reference};
