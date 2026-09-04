//! CPU native planning contracts: fallback pressure through the public report API.
//!
//! Lowering white-box contracts live with the lowering owner as
//! `src/native/tests.rs` unit tests; this target only asserts
//! application-observable planning reports.

#[path = "support/analytical_olap_workload.rs"]
mod analytical_olap_workload;
#[path = "support/tpch_fallback_workload.rs"]
mod tpch_fallback_workload;
#[path = "support/tpch_session.rs"]
mod tpch_session;

#[path = "native_planning/fallback_pressure.rs"]
mod fallback_pressure;
