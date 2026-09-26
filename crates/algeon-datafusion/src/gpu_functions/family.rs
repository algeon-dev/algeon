//! Narrow discovery and static-validation contract owned by GPU SQL families.

use std::any::Any;

use datafusion::catalog::Session;

use super::{GpuCallRequest, GpuCallValidation, GpuFunctionDescriptor};

/// Family-owned adapter for the common GPU UDTF catalog.
///
/// This intentionally does not abstract execution, lowering, or provider-owned
/// relation semantics. It only lets the shared DataFusion metadata surface
/// discover functions and delegate static validation to the owning family.
#[async_trait::async_trait]
pub(crate) trait GpuFunctionFamily: Any + Send + Sync {
    /// Stable unique identity for this installed family.
    fn family_id(&self) -> &'static str;

    /// Materialize this family's current session/build descriptors.
    fn descriptors(&self) -> Vec<GpuFunctionDescriptor>;

    /// Whether installing `self` over the `installed` family with the same
    /// id is a no-op; any other reinstall is rejected as a conflict. Families
    /// with installation settings outside their descriptors compare them too.
    fn same_installation(&self, installed: &dyn GpuFunctionFamily) -> bool {
        self.descriptors() == installed.descriptors()
    }

    /// Validate a parsed common request for a descriptor owned by this family.
    async fn validate(&self, session: &dyn Session, function_name: &str, request: &GpuCallRequest)
    -> GpuCallValidation;
}
