//! Cargo test harness for the shared std-only RAPIDS build policy.

#[expect(dead_code, reason = "shared build support module included in test harness")]
#[path = "../scripts/build/rapids.rs"]
mod rapids;
