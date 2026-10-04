//! Stable actionable guidance for native candidate-selection outcomes.

use query_engine::plan::NativePlanCostReason;

use super::model::{QueryGpuRemedy, QueryGpuRemedyKind};

pub(super) const fn cost_remedy(reason: NativePlanCostReason) -> QueryGpuRemedy {
    match reason {
        NativePlanCostReason::SmallSelectiveTopK
        | NativePlanCostReason::SmallSelectiveResidualJoin
        | NativePlanCostReason::SmallSelectiveFullSort => QueryGpuRemedy::new(
            "keep_cpu_for_small_selective_work",
            QueryGpuRemedyKind::CpuPreferred,
            "The cost model expects CPU execution to be cheaper for this small selective shape; increase useful GPU work or keep the DataFusion path.",
        ),
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
    use super::*;
    use crate::capability::CapabilityReason;

    #[test]
    fn every_capability_reason_has_a_stable_nonempty_remedy() {
        for reason in CapabilityReason::ALL {
            let remedy = reason.facts().remedy;
            assert!(!remedy.code.is_empty(), "{reason:?}");
            assert!(!remedy.guidance.is_empty(), "{reason:?}");
        }
    }

    #[test]
    fn every_cost_reason_preserves_its_remedy_code_and_kind() {
        for reason in [
            NativePlanCostReason::SmallSelectiveTopK,
            NativePlanCostReason::SmallSelectiveResidualJoin,
            NativePlanCostReason::SmallSelectiveFullSort,
        ] {
            let remedy = cost_remedy(reason);
            assert_eq!(remedy.code, "keep_cpu_for_small_selective_work", "{reason:?}");
            assert_eq!(remedy.kind, QueryGpuRemedyKind::CpuPreferred, "{reason:?}");
        }
    }

    #[test]
    fn prescribed_remedies_remain_stable() {
        let local = CapabilityReason::NativeSourceNonLocalObjectStore.facts().remedy;
        assert_eq!(local.code, "stage_local_parquet");
        assert_eq!(local.kind, QueryGpuRemedyKind::UserAction);

        let unsupported = CapabilityReason::UnsupportedExpression.facts().remedy;
        assert_eq!(unsupported.code, "rewrite_expression");
        assert_eq!(unsupported.kind, QueryGpuRemedyKind::UserAction);

        let cost = cost_remedy(NativePlanCostReason::SmallSelectiveTopK);
        assert_eq!(cost.code, "keep_cpu_for_small_selective_work");
        assert_eq!(cost.kind, QueryGpuRemedyKind::CpuPreferred);
    }
}
