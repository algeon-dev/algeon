use std::sync::Arc;

use datafusion_nexus::native_exec::NexusNativeExecutionConfig;

fn main() {
    let runtime = Arc::new(cudf::Runtime::new().unwrap());
    let _config = NexusNativeExecutionConfig::bounded_auto_sized().with_shared_runtime(runtime);
}
