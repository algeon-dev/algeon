/// ClickBench adapter-specific report item builders.
pub mod clickbench;
/// TPC-DS adapter-specific report and parity item builders.
pub mod tpcds;
/// TPC-H adapter-specific report item builders.
pub mod tpch;

use datafusion_nexus::policy::GpuFallbackPolicy;

/// Parses the stable workload spelling of a GPU fallback policy.
///
/// Benchmark/report binaries deliberately keep their policy selection explicit:
/// their normal defaults remain `GpuPreferred`, while validation invocations can
/// request `gpu_only` without duplicating a workload runner.
pub fn parse_gpu_fallback_policy(value: &str) -> Result<GpuFallbackPolicy, String> {
    match value.trim().to_ascii_lowercase().replace(['_', '-'], "").as_str() {
        "gpuonly" => Ok(GpuFallbackPolicy::GpuOnly),
        "gpupreferred" => Ok(GpuFallbackPolicy::GpuPreferred),
        "cpuonly" => Ok(GpuFallbackPolicy::CpuOnly),
        _ => Err(format!("unsupported GPU fallback policy {value:?}; expected gpu_only, gpu_preferred, or cpu_only")),
    }
}

/// Returns the stable workload/report spelling of a GPU fallback policy.
///
/// Keep this paired with [`parse_gpu_fallback_policy`] so run-configuration
/// evidence is machine-readable and does not depend on Rust's debug format.
pub const fn gpu_fallback_policy_label(policy: GpuFallbackPolicy) -> &'static str {
    match policy {
        GpuFallbackPolicy::GpuOnly => "gpu_only",
        GpuFallbackPolicy::GpuPreferred => "gpu_preferred",
        GpuFallbackPolicy::CpuOnly => "cpu_only",
    }
}
