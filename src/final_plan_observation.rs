//! Single final-physical-plan observation walk shared by planning reports and
//! GPU coverage.
//!
//! `Rejected` is a terminal planning result, not an observable physical node;
//! this walk therefore only returns executable `DataFusion`, `Native`, or
//! `Mixed` dispositions.

use datafusion::physical_plan::ExecutionPlan;

#[cfg(feature = "cugraph")]
use crate::graph_sql::CugraphAlgoExec;
use crate::{
    native_exec::{NexusNativeComposedExec, NexusNativeExec, NexusNativeGpuProducerExec},
    planner::FinalPlanDisposition,
};

/// Final-plan facts derived from one physical-plan walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FinalPlanObservation {
    pub(crate) disposition: FinalPlanDisposition,
    pub(crate) native_fragment_count: usize,
    pub(crate) explicit_gpu_island_count: usize,
    pub(crate) datafusion_boundary_count: usize,
    pub(crate) host_materialization_count: usize,
}

impl FinalPlanObservation {
    pub(crate) const fn has_gpu_runtime(self) -> bool {
        self.native_fragment_count > 0 || self.explicit_gpu_island_count > 0
    }
}

/// Observe the exact executable physical plan once.
pub(crate) fn observe_final_plan(plan: &dyn ExecutionPlan) -> FinalPlanObservation {
    observe(plan)
}

fn observe(plan: &dyn ExecutionPlan) -> FinalPlanObservation {
    if let Some(input) = crate::native_exec::transparent_root_wrapper_input(plan) {
        return observe(input.as_ref());
    }
    if plan.downcast_ref::<NexusNativeExec>().is_some() {
        return FinalPlanObservation {
            disposition: FinalPlanDisposition::Native,
            native_fragment_count: 1,
            explicit_gpu_island_count: 0,
            datafusion_boundary_count: 0,
            host_materialization_count: 1,
        };
    }
    if plan.downcast_ref::<NexusNativeGpuProducerExec>().is_some() {
        return FinalPlanObservation {
            disposition: FinalPlanDisposition::Native,
            native_fragment_count: 1,
            explicit_gpu_island_count: 0,
            datafusion_boundary_count: 0,
            host_materialization_count: 0,
        };
    }
    if plan.downcast_ref::<NexusNativeComposedExec>().is_some() {
        let mut observation = FinalPlanObservation {
            disposition: FinalPlanDisposition::Native,
            native_fragment_count: 1,
            explicit_gpu_island_count: 0,
            datafusion_boundary_count: 0,
            host_materialization_count: 1,
        };
        for child in plan.children() {
            let child = observe(child.as_ref());
            if child.disposition != FinalPlanDisposition::Native {
                observation.disposition = FinalPlanDisposition::Mixed;
            }
            merge(&mut observation, child);
        }
        return observation;
    }
    #[cfg(feature = "cugraph")]
    if plan.downcast_ref::<CugraphAlgoExec>().is_some() {
        let mut observation = FinalPlanObservation {
            disposition: FinalPlanDisposition::Native,
            native_fragment_count: 0,
            explicit_gpu_island_count: 1,
            datafusion_boundary_count: 0,
            host_materialization_count: 1,
        };
        for child_plan in plan.children() {
            let child = observe(child_plan.as_ref());
            if !graph_child_stays_gpu_resident(child_plan.as_ref(), child) {
                observation.disposition = FinalPlanDisposition::Mixed;
                observation.datafusion_boundary_count += 1;
            }
            merge(&mut observation, child);
        }
        return observation;
    }

    let mut observation = FinalPlanObservation {
        disposition: FinalPlanDisposition::DataFusion,
        native_fragment_count: 0,
        explicit_gpu_island_count: 0,
        datafusion_boundary_count: 0,
        host_materialization_count: 0,
    };
    let mut directly_contains_gpu_runtime = false;
    for child in plan.children() {
        let child = observe(child.as_ref());
        directly_contains_gpu_runtime |= child.has_gpu_runtime();
        if child.disposition != FinalPlanDisposition::DataFusion {
            observation.disposition = FinalPlanDisposition::Mixed;
        }
        merge(&mut observation, child);
    }
    if directly_contains_gpu_runtime {
        observation.datafusion_boundary_count += 1;
        observation.disposition = FinalPlanDisposition::Mixed;
    }
    observation
}

#[cfg(feature = "cugraph")]
fn graph_child_stays_gpu_resident(plan: &dyn ExecutionPlan, observation: FinalPlanObservation) -> bool {
    if observation.disposition == FinalPlanDisposition::Native
        && (plan.downcast_ref::<NexusNativeExec>().is_some()
            || plan.downcast_ref::<NexusNativeComposedExec>().is_some())
    {
        return true;
    }
    if plan.name() != "CooperativeExec" {
        return false;
    }
    let children = plan.children();
    let [child] = children.as_slice() else {
        return false;
    };
    let child = observe(child.as_ref());
    child.disposition == FinalPlanDisposition::Native
        && (child.native_fragment_count > 0 && child.explicit_gpu_island_count == 0)
}

fn merge(target: &mut FinalPlanObservation, child: FinalPlanObservation) {
    target.native_fragment_count += child.native_fragment_count;
    target.explicit_gpu_island_count += child.explicit_gpu_island_count;
    target.datafusion_boundary_count += child.datafusion_boundary_count;
    target.host_materialization_count += child.host_materialization_count;
}
