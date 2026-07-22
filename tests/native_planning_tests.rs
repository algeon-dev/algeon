//! CPU native planning contracts: lowering + fallback pressure.

#[path = "support/analytical_olap_workload.rs"]
mod analytical_olap_workload;
#[path = "support/tpch_fallback_workload.rs"]
mod tpch_fallback_workload;
#[path = "support/tpch_session.rs"]
mod tpch_session;

#[path = "native_planning/fallback_pressure.rs"]
mod fallback_pressure;
#[path = "native_planning/lowering.rs"]
mod lowering;
