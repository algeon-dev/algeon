//! Identity of one SQL call site of a GPU table function.

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_GPU_FUNCTION_INVOCATION_ID: AtomicU64 = AtomicU64::new(0);

/// One SQL invocation of a GPU table function, minted when the relation
/// planner builds the call's logical node.
///
/// DataFusion inlines a CTE or view at each reference by copying its logical
/// plan, so every copy of one call site carries the same identity and every
/// physical occurrence planned from those copies reads one evaluation per
/// query execution. Two call sites with identical text receive different
/// identities and evaluate independently. The identity also enters the
/// logical node's equality, which keeps DataFusion from treating two call
/// sites as one plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct GpuFunctionInvocation {
    family: &'static str,
    id: u64,
}

impl GpuFunctionInvocation {
    /// Mints an identity unique within this process for one call site of a
    /// function in `family`.
    pub(crate) fn mint(family: &'static str) -> Self {
        Self { family, id: NEXT_GPU_FUNCTION_INVOCATION_ID.fetch_add(1, Ordering::Relaxed) }
    }

    /// Name that binds the invocation's result in a consumer's GPU-resident
    /// input map and identifies its host occurrences in the attempt manifest.
    pub(crate) fn result_input_name(self) -> String {
        format!("__algeon_datafusion_{}_result_{}", self.family, self.id)
    }

    /// Name that binds the relation argument at `index` in the invocation's
    /// own native plan.
    pub(crate) fn relation_input_name(self, index: usize) -> String {
        format!("__algeon_datafusion_{}_input_{}_{index}", self.family, self.id)
    }
}
