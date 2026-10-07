//! Flight SQL integration tests with whole-device backend ownership.
//!
//! Queue, cancellation, and shutdown assertions use bounded deadlines, so the
//! target runs in the exclusive GPU lane rather than competing with unrelated
//! kernels from other test processes.

mod server_auth_tests;
mod server_flight_sql_tests;
