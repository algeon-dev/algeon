use datafusion::physical_plan::ExecutionPlan;

#[cfg(feature = "cugraph")]
use crate::cugraph_sql::CugraphAlgoExec;
use crate::{
    cuvs_sql::CuvsAlgoExec,
    native_exec::{
        AlgeonNativeComposedExec, AlgeonNativeComposedGpuProducerExec, AlgeonNativeExec, AlgeonNativeGpuProducerExec,
    },
    planner::FinalPlanDisposition,
};

/// Observable final execution-plan coverage derived from one physical-plan walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FinalPlanCoverage {
    pub(crate) disposition: FinalPlanDisposition,
    pub(crate) native_fragment_count: usize,
    pub(crate) explicit_gpu_island_count: usize,
    pub(crate) datafusion_boundary_count: usize,
    pub(crate) host_materialization_count: usize,
    pub(crate) has_gpu_runtime: bool,
}

impl FinalPlanCoverage {
    pub(crate) const fn new(
        disposition: FinalPlanDisposition,
        native_fragment_count: usize,
        explicit_gpu_island_count: usize,
        datafusion_boundary_count: usize,
        host_materialization_count: usize,
    ) -> Self {
        Self {
            disposition,
            native_fragment_count,
            explicit_gpu_island_count,
            datafusion_boundary_count,
            host_materialization_count,
            has_gpu_runtime: native_fragment_count > 0 || explicit_gpu_island_count > 0,
        }
    }

    pub const fn disposition(&self) -> FinalPlanDisposition {
        self.disposition
    }

    pub const fn native_fragment_count(&self) -> usize {
        self.native_fragment_count
    }

    pub const fn explicit_gpu_island_count(&self) -> usize {
        self.explicit_gpu_island_count
    }

    pub const fn datafusion_boundary_count(&self) -> usize {
        self.datafusion_boundary_count
    }

    pub const fn host_materialization_count(&self) -> usize {
        self.host_materialization_count
    }

    pub const fn has_gpu_runtime(&self) -> bool {
        self.has_gpu_runtime
    }
}

/// Observe the exact executable physical plan once.
pub fn observe_final_plan(plan: &dyn ExecutionPlan) -> FinalPlanCoverage {
    observe_with_output_boundary(plan, true)
}

fn observe_with_output_boundary(plan: &dyn ExecutionPlan, materializes_host: bool) -> FinalPlanCoverage {
    if let Some(input) = crate::native_exec::transparent_root_wrapper_input(plan) {
        return observe_with_output_boundary(input.as_ref(), materializes_host);
    }
    if plan.name() == "CooperativeExec" {
        let children = plan.children();
        if let [child] = children.as_slice() {
            let wraps_gpu_producer = crate::native_exec::gpu_resident_producer(plan).is_some();
            let wraps_graph_native_input = !materializes_host
                && (child.downcast_ref::<AlgeonNativeExec>().is_some()
                    || child.downcast_ref::<AlgeonNativeComposedExec>().is_some());
            if wraps_gpu_producer || wraps_graph_native_input {
                return observe_with_output_boundary(child.as_ref(), false);
            }
        }
    }
    if plan.downcast_ref::<AlgeonNativeExec>().is_some() {
        return FinalPlanCoverage::new(FinalPlanDisposition::Native, 1, 0, 0, usize::from(materializes_host));
    }
    if plan.downcast_ref::<AlgeonNativeGpuProducerExec>().is_some() {
        return FinalPlanCoverage::new(FinalPlanDisposition::Native, 1, 0, 0, 0);
    }
    if plan.downcast_ref::<AlgeonNativeComposedExec>().is_some() {
        let mut observation =
            FinalPlanCoverage::new(FinalPlanDisposition::Native, 1, 0, 0, usize::from(materializes_host));
        for child in plan.children() {
            let child = observe_as_gpu_producer(child.as_ref());
            if child.disposition != FinalPlanDisposition::Native {
                observation.disposition = FinalPlanDisposition::Mixed;
            }
            merge(&mut observation, child);
        }
        return observation;
    }
    if plan.downcast_ref::<AlgeonNativeComposedGpuProducerExec>().is_some() {
        let mut observation = FinalPlanCoverage::new(FinalPlanDisposition::Native, 1, 0, 0, 0);
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
        let mut observation =
            FinalPlanCoverage::new(FinalPlanDisposition::Native, 1, 0, 0, usize::from(materializes_host));
        for child_plan in plan.children() {
            let child_is_gpu_producer = crate::native_exec::gpu_resident_producer(child_plan.as_ref()).is_some();
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
        let mut observation =
            FinalPlanCoverage::new(FinalPlanDisposition::Native, 0, 1, 0, usize::from(materializes_host));
        for child_plan in plan.children() {
            let native_input = crate::cugraph_sql::native_graph_input_execution_plan(child_plan.as_ref());
            let child_uses_native_gpu_input =
                native_input.is_some() || crate::native_exec::gpu_resident_producer(child_plan.as_ref()).is_some();
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

    let mut observation = FinalPlanCoverage::new(FinalPlanDisposition::DataFusion, 0, 0, 0, 0);
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

fn observe_as_gpu_producer(plan: &dyn ExecutionPlan) -> FinalPlanCoverage {
    observe_with_output_boundary(plan, false)
}

fn merge(target: &mut FinalPlanCoverage, child: FinalPlanCoverage) {
    target.native_fragment_count += child.native_fragment_count;
    target.explicit_gpu_island_count += child.explicit_gpu_island_count;
    target.datafusion_boundary_count += child.datafusion_boundary_count;
    target.host_materialization_count += child.host_materialization_count;
    target.has_gpu_runtime |= child.has_gpu_runtime;
}
