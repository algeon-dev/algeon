use nexus_query_engine::mode::NativePipelineExecutionMode;

fn main() {
    let _ = NativePipelineExecutionMode::StrictBounded;
    let _ = NativePipelineExecutionMode::Adaptive;
}
