use datafusion_nexus::{
    capability::CapabilityReason,
    planner::NativeAdaptivePipelineCandidateDiagnostics,
};

fn main() {
    let _ = std::mem::size_of::<NativeAdaptivePipelineCandidateDiagnostics>();
    let _ = CapabilityReason::NativeAdaptivePipelineUnsupported;
}
