//! Narrow discovery and static-validation contract owned by GPU SQL families.

use datafusion::catalog::Session;

use super::{GpuCallRequest, GpuCallValidation, GpuFunctionDescriptor};

/// Family-owned adapter for the common GPU function catalog.
///
/// This intentionally does not abstract execution, lowering, or provider-owned
/// relation semantics. It only lets the shared DataFusion metadata surface
/// discover functions and delegate static validation to the owning family.
#[allow(clippy::double_must_use)]
#[async_trait::async_trait]
pub(crate) trait GpuFunctionFamily: Send + Sync {
    /// Stable unique identity for this installed family.
    fn family_id(&self) -> &'static str;

    /// Materialize this family's current session/build descriptors.
    fn descriptors(&self) -> Vec<GpuFunctionDescriptor>;

    /// Validate a parsed common request for a descriptor owned by this family.
    async fn validate(&self, session: &dyn Session, function_name: &str, request: &GpuCallRequest)
    -> GpuCallValidation;
}
