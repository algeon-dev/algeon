//! Flight SQL integration tests with whole-device backend ownership.
//!
//! Queue, cancellation, and shutdown assertions use bounded deadlines, so the
//! target runs in the exclusive GPU lane rather than competing with unrelated
//! kernels from other test processes.

#[path = "server/server_auth_tests.rs"]
mod server_auth_tests;
#[path = "server/server_flight_sql_tests.rs"]
mod server_flight_sql_tests;
