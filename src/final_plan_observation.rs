//! Single final-physical-plan observation walk shared by planning reports and
//! GPU coverage.
//!
//! `Rejected` is a terminal planning result, not an observable physical node;
//! this walk therefore only returns executable `DataFusion`, `Native`, or
//! `Mixed` dispositions.

use datafusion::physical_plan::ExecutionPlan;

#[cfg(feature = "cugraph")]
use crate::cugraph_sql::CugraphAlgoExec;
use crate::{
    cuvs_sql::CuvsAlgoExec,
    native_exec::{
        NexusNativeComposedExec, NexusNativeComposedGpuProducerExec, NexusNativeExec, NexusNativeGpuProducerExec,
    },
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
    observe_with_output_boundary(plan, true)
}

/// Observe a node in the context in which its output is consumed.
///
/// Accelerator operators can execute through their ordinary DataFusion
/// `ExecutionPlan` method or through the internal GPU-producer contract. The
/// former owns a final device-to-host materialization; the latter must not be
/// counted as one merely because the same concrete type also supports a host
/// result. Keeping that context in this walk avoids reporting a false host
/// boundary for composed graph/vector plans.
fn observe_with_output_boundary(plan: &dyn ExecutionPlan, materializes_host: bool) -> FinalPlanObservation {
    if let Some(input) = crate::native_exec::transparent_root_wrapper_input(plan) {
        return observe_with_output_boundary(input.as_ref(), materializes_host);
    }
    if plan.name() == "CooperativeExec" {
        let children = plan.children();
        if let [child] = children.as_slice() {
            let wraps_gpu_producer = crate::native_exec::producer::gpu_resident_producer(plan).is_some();
            let wraps_graph_native_input = !materializes_host
                && (child.downcast_ref::<NexusNativeExec>().is_some()
                    || child.downcast_ref::<NexusNativeComposedExec>().is_some());
            if wraps_gpu_producer || wraps_graph_native_input {
                // DataFusion's one-level cooperative wrapper is
                // scheduling-only around an internal GPU producer or a
                // native cuGraph input. It does not change device ownership
                // or introduce an Arrow boundary.
                return observe_with_output_boundary(child.as_ref(), false);
            }
        }
    }
    if plan.downcast_ref::<NexusNativeExec>().is_some() {
        return FinalPlanObservation {
            disposition: FinalPlanDisposition::Native,
            native_fragment_count: 1,
            explicit_gpu_island_count: 0,
            datafusion_boundary_count: 0,
            host_materialization_count: usize::from(materializes_host),
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
            host_materialization_count: usize::from(materializes_host),
        };
        for child in plan.children() {
            let child = observe_as_gpu_producer(child.as_ref());
            if child.disposition != FinalPlanDisposition::Native {
                observation.disposition = FinalPlanDisposition::Mixed;
            }
            merge(&mut observation, child);
        }
        return observation;
    }
    if plan.downcast_ref::<NexusNativeComposedGpuProducerExec>().is_some() {
        let mut observation = FinalPlanObservation {
            disposition: FinalPlanDisposition::Native,
            native_fragment_count: 1,
            explicit_gpu_island_count: 0,
            datafusion_boundary_count: 0,
            host_materialization_count: 0,
        };
        for child in plan.children() {
            let child = observe_as_gpu_producer(child.as_ref());
            if child.disposition != FinalPlanDisposition::Native {
                observation.disposition = FinalPlanDisposition::Mixed;
            }
            merge(&mut observation, child);
        }
        return observation;
    }
    if plan.downcast_ref::<CuvsAlgoExec>().is_some() {
        // cuVS owns the native vector execution and its normal `ExecutionPlan`
        // path materializes only its final relational result.  Its children
        // remain on the device only when they satisfy the shared producer
        // contract; otherwise `CuvsAlgoExec` explicitly stages the Arrow
        // child before entering its native context.
        let mut observation = FinalPlanObservation {
            disposition: FinalPlanDisposition::Native,
            native_fragment_count: 1,
            explicit_gpu_island_count: 0,
            datafusion_boundary_count: 0,
            host_materialization_count: usize::from(materializes_host),
        };
        for child_plan in plan.children() {
            let child_is_gpu_producer =
                crate::native_exec::producer::gpu_resident_producer(child_plan.as_ref()).is_some();
            let child = if child_is_gpu_producer {
                observe_as_gpu_producer(child_plan.as_ref())
            } else {
                observe_with_output_boundary(child_plan.as_ref(), true)
            };
            if !child_is_gpu_producer || child.disposition != FinalPlanDisposition::Native {
                observation.disposition = FinalPlanDisposition::Mixed;
                observation.datafusion_boundary_count += 1;
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
            host_materialization_count: usize::from(materializes_host),
        };
        for child_plan in plan.children() {
            let native_input = crate::cugraph_sql::native_graph_input_execution_plan(child_plan.as_ref());
            let child_uses_native_gpu_input = native_input.is_some()
                || crate::native_exec::producer::gpu_resident_producer(child_plan.as_ref()).is_some();
            let child = if let Some(native_input) = native_input {
                observe_with_output_boundary(native_input, false)
            } else if child_uses_native_gpu_input {
                observe_with_output_boundary(child_plan.as_ref(), false)
            } else {
                observe_with_output_boundary(child_plan.as_ref(), true)
            };
            if !child_uses_native_gpu_input || child.disposition != FinalPlanDisposition::Native {
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
        let child = observe_with_output_boundary(child.as_ref(), true);
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

fn observe_as_gpu_producer(plan: &dyn ExecutionPlan) -> FinalPlanObservation {
    observe_with_output_boundary(plan, false)
}

fn merge(target: &mut FinalPlanObservation, child: FinalPlanObservation) {
    target.native_fragment_count += child.native_fragment_count;
    target.explicit_gpu_island_count += child.explicit_gpu_island_count;
    target.datafusion_boundary_count += child.datafusion_boundary_count;
    target.host_materialization_count += child.host_materialization_count;
}
