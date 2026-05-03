use std::sync::Arc;

use datafusion::execution::SessionStateBuilder;
use datafusion_nexus::{
    gpu_coverage::{QueryGpuCudfNativeCoverage, cudf_native_coverage},
    native_exec::{
        NATIVE_CUDF_DATAFUSION_METRIC_NAMES, NativeCudfComposedExec, NativeCudfExec, NativeCudfExecutionConfig,
        NativeCudfGpuProducerExec, NativeCudfMetrics,
    },
    planner::{CudfNativePhysicalOptimizerRule, CudfOptimizerConfig, NexusNativeOptimizerConfig},
    session::{
        CudfSessionStateBuilderExt, cudf_native_physical_optimizer_rules, install_cudf_native_rule_with_session_config,
    },
    table_format::{CudfTableFormatGpuOnlyLeafGuardRule, CudfTableFormatScanExec},
};

fn main() {
    let builder = SessionStateBuilder::new_with_default_features();
    let _ = builder.with_cudf_native(NexusNativeOptimizerConfig::default());

    let builder = SessionStateBuilder::new_with_default_features();
    let _ = builder
        .with_cudf_native_rule(Arc::new(CudfNativePhysicalOptimizerRule::new(NexusNativeOptimizerConfig::default())));
}
