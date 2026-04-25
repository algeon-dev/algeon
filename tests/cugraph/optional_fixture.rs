//! Failure policy for selected Python-cuGraph acceptance tests.

/// Stops a selected acceptance test when its provisioned Python oracle is not
/// available. This is deliberately divergent rather than a pass-as-skip path.
// The Cargo contract target does not invoke Python-oracle helpers; explicit
// tools runners and the BFS predicate fixture do.
#[cfg_attr(test, allow(dead_code))]
pub fn handle_cugraph_python_oracle_unavailable(label: &str, message: impl AsRef<str>) -> ! {
    panic!("required {label} is unavailable: {}", message.as_ref().trim());
}
