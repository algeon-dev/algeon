// Spec §2–§3: `cudf::Error` is a downstream-readable contract, not a
// downstream-authored error DSL. `Error::new_with` is crate-internal and
// must not be callable from integration tests or downstream crates.

fn main() {
    let _ = cudf::Error::new_with(cudf::ErrorCode::InvariantViolation, "downstream", "should not compile");
}
