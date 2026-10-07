//! Smoke tests for metadata-only Debug formatting on runtime resource types.

use cudf::{
    core::memory::DeviceMemoryResource,
    runtime::{CudaEvent, CudaStream},
};

#[test]
fn test_runtime_resource_debug_smoke_reports_device_and_handle_metadata() {
    let stream_output = format!("{:?}", CudaStream::new().unwrap());
    let event_output = format!("{:?}", CudaEvent::new().unwrap());
    let mr_output = format!("{:?}", DeviceMemoryResource::cuda().unwrap());

    assert!(stream_output.contains("CudaStream"), "debug output should identify CudaStream: {}", stream_output);
    assert!(stream_output.contains("device"), "debug output should contain stream device metadata: {}", stream_output);
    assert!(stream_output.contains("handle"), "debug output should contain stream handle metadata: {}", stream_output);

    assert!(event_output.contains("CudaEvent"), "debug output should identify CudaEvent: {}", event_output);
    assert!(event_output.contains("device"), "debug output should contain event device metadata: {}", event_output);
    assert!(event_output.contains("handle"), "debug output should contain event handle metadata: {}", event_output);

    assert!(
        mr_output.contains("DeviceMemoryResource"),
        "debug output should identify DeviceMemoryResource: {}",
        mr_output
    );
    assert!(mr_output.contains("device"), "debug output should contain memory-resource device metadata: {}", mr_output);
    assert!(mr_output.contains("handle"), "debug output should contain memory-resource handle metadata: {}", mr_output);
}
