//! Tests for metadata-only Debug formatting on runtime resource types.

use cudf::{
    core::memory::DeviceMemoryResource,
    runtime::{CudaEvent, CudaStream},
};

#[test]
fn test_cuda_stream_debug_reports_device_and_handle_identity() {
    let stream = CudaStream::new().unwrap();

    let output = format!("{:?}", stream);

    assert!(output.contains("CudaStream"), "debug output should contain the type name: {}", output);
    assert!(output.contains("device"), "debug output should contain device metadata: {}", output);
    assert!(output.contains("handle"), "debug output should contain handle metadata: {}", output);
}

#[test]
fn test_cuda_event_debug_reports_device_and_handle_identity() {
    let event = CudaEvent::new().unwrap();

    let output = format!("{:?}", event);

    assert!(output.contains("CudaEvent"), "debug output should contain the type name: {}", output);
    assert!(output.contains("device"), "debug output should contain device metadata: {}", output);
    assert!(output.contains("handle"), "debug output should contain handle metadata: {}", output);
}

#[test]
fn test_device_memory_resource_debug_reports_device_and_handle_identity() {
    let mr = DeviceMemoryResource::cuda().unwrap();

    let output = format!("{:?}", mr);

    assert!(output.contains("DeviceMemoryResource"), "debug output should contain the type name: {}", output);
    assert!(output.contains("device"), "debug output should contain device metadata: {}", output);
    assert!(output.contains("handle"), "debug output should contain handle metadata: {}", output);
}
