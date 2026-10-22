//! Common DataFusion-facing GPU UDTF discovery and validation catalog.
//!
//! GPU SQL families provide descriptors and static validation through the
//! crate-private `GpuFunctionFamily` trait. This module owns the immutable
//! cross-family catalog plus the stable `gpu_*` metadata UDTFs; it deliberately
//! does not own execution or family-specific planning.

mod catalog;
pub(crate) mod dense_relation;
pub(crate) mod dense_relation_errors;
mod descriptor;
mod family;
mod input_ordering;
mod invocation;
mod named_args;
mod registration;
pub(crate) mod scalar_arguments;
mod table_functions;
mod unknown_function;
mod validation;

pub(crate) use catalog::GpuFunctionCatalog;
pub(crate) use descriptor::{GpuFunctionAvailability, GpuFunctionDescriptor, REQUIRED_FEATURE_DISABLED};
pub(crate) use family::GpuFunctionFamily;
pub(crate) use input_ordering::{
    fmt_input_orderings, input_ordering_expressions, input_orderings_with_expressions, resolve_input_ordering,
};
pub(crate) use invocation::GpuFunctionInvocation;
pub(crate) use named_args::{NamedArgumentDefinition, NamedArgumentKind, bind_named_arguments, unknown_name_message};
pub(crate) use registration::install_gpu_function_family;
pub(crate) use table_functions::gpu_metadata_table_functions;
#[cfg(feature = "server")]
pub(crate) use table_functions::is_builtin_gpu_metadata_table_source;
pub use table_functions::{GPU_DESCRIBE_FUNCTION_FUNCTION, GPU_LIST_FUNCTIONS_FUNCTION, GPU_VALIDATE_CALL_FUNCTION};
pub(crate) use validation::{GpuCallProblem, GpuCallProblems, GpuCallRequest, GpuCallValidation, argument_of};
