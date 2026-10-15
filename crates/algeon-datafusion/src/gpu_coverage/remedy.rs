//! Stable actionable guidance for native candidate-selection outcomes.

use query_planning::plan::NativePlanCostReason;

use super::model::{QueryGpuRemedy, QueryGpuRemedyKind};

pub(super) const fn cost_remedy(reason: NativePlanCostReason) -> QueryGpuRemedy {
    match reason {
        NativePlanCostReason::NoColumnData => QueryGpuRemedy::new(
            "keep_cpu_without_column_data",
            QueryGpuRemedyKind::CpuPreferred,
            "The plan reads no column data, so there is no GPU work to accelerate; DataFusion produces its rows directly.",
        ),
        NativePlanCostReason::HostLiteralPassthrough => QueryGpuRemedy::new(
            "keep_cpu_for_host_literal_rows",
            QueryGpuRemedyKind::CpuPreferred,
            "The plan only returns rows it already holds in host memory; DataFusion returns them without a device round trip.",
        ),
    }
}

#[cfg(test)]
mod tests {
    use crate::capability::CapabilityReason;

    #[test]
    fn every_capability_reason_has_a_stable_nonempty_remedy() {
        for reason in CapabilityReason::ALL {
            let remedy = reason.facts().remedy;
            assert!(!remedy.code.is_empty(), "{reason:?}");
            assert!(!remedy.guidance.is_empty(), "{reason:?}");
        }
    }
}
