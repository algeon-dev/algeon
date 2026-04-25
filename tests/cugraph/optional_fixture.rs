//! Failure policy for selected Python-cuGraph acceptance tests.

/// Stops a selected acceptance test when its provisioned Python oracle is not
/// available. This is deliberately divergent rather than a pass-as-skip path.
pub fn handle_cugraph_python_oracle_unavailable(label: &str, message: impl AsRef<str>) -> ! {
    panic!("required {label} is unavailable: {}", message.as_ref().trim());
}
