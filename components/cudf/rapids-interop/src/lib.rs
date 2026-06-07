//! Shared interop contract for RAPIDS Rust crates.
//!
//! This crate defines:
//!
//! - a borrowed GPU-view compatibility profile via [`BorrowedColumnView`] and
//!   [`BorrowedTableView`]
//! - a producer-rooted prepared export profile via [`PreparedColumnExport`] and
//!   [`PreparedTableExport`]
//! - activated read leases with explicit completion via
//!   [`ActiveColumnReadLease`] and [`ActiveTableReadLease`]
//! - an owned device transfer profile via [`OwnedColumnTransfer`] and
//!   [`OwnedTableTransfer`]
//! - explicit partitioned multi-device wrappers via
//!   [`PartitionedPreparedTableExport`] and
//!   [`PartitionedOwnedTableTransfer`]
//!
//! # Contract Summary
//!
//! - Borrowed compatibility and prepared export both describe existing buffers;
//!   neither transfers ownership.
//! - Borrowed compatibility covers low-level borrowed descriptors and their
//!   explicit execution binding facts.
//! - Prepared export plus activated lease covers read-start, completion, and
//!   backend-defined incomplete-drop behavior at the shared contract layer.
//! - Owned device transfer is a separate ownership-bearing contract with an
//!   explicit release contract.
//! - Partitioned multi-device export is explicit composition over multiple
//!   single-device table roots; it does not weaken the meaning of one
//!   [`PreparedTableExport`] or one [`OwnedTableTransfer`].
//! - Current device-export support covers fixed-width layouts, the binary-like
//!   string layout slice, `List`, `Struct`, and `Dictionary`.
//! - Current borrowed-compatibility support covers fixed-width layouts plus
//!   the current `STRING` binary-like slice and `DICTIONARY32` with supported
//!   recursive keys projections.
//! - Nested-layout support is still intentionally narrow in this slice:
//!   current coverage is the recursive semantic-column graph that can be
//!   expressed by the existing fixed-width, binary-like string, list, struct,
//!   and dictionary slices on one CUDA device.
//! - Consumer algorithms may still allocate or copy for outputs; this crate
//!   does not promise zero-copy output.
//! - `cudf-rs` producer adapters require explicit execution provenance and
//!   reject ambient/default-only objects at the interop boundary.
//! - User-facing types such as `cudf::Series` and `cudf::DataFrame` may
//!   provide convenience adapters, but they are not the shared contract.
//!
//! [`BorrowedColumnView`] and [`BorrowedTableView`] are low-level fact records
//! about producer-owned device memory. Consumers must not treat them as
//! ownership-bearing handles or cache them beyond the lifetime of the producer
//! borrow that created them.
//!
//! [`PreparedColumnExport`] and [`PreparedTableExport`] are producer-rooted
//! export contracts. Consumers activate access explicitly by providing
//! [`ReadStartAuthority`] plus [`CompletionAuthority`], then finish via
//! [`ActiveColumnReadLease::complete`] or [`ActiveTableReadLease::complete`],
//! which returns a shared [`LeaseCompletionResult`].

mod borrowed;
mod contract;
mod lease;
mod owned;
mod partitioned;
mod prepared;
mod traits;
mod types;
mod validation;

pub use borrowed::*;
pub use contract::*;
pub use lease::*;
pub use owned::*;
pub use partitioned::*;
pub use prepared::*;
pub use traits::*;
pub use types::*;
