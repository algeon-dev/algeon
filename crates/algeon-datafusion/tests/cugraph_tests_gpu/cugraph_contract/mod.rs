//! Registry-driven cuGraph planning / capability / bounded
//! contract checks.
//!
//! These tests are CPU-only and run on every `cargo test --features
//! cugraph` invocation. They assert that every algorithm in the
//! cuGraph operation registry exposes the same schema/capability/
//! bounded surface, without per-algorithm boilerplate.

mod spec;
