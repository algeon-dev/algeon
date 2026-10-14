//! Shared interop contract for RAPIDS Rust crates.
//!
//! Published as an Algeon implementation dependency. Applications use
//! [`algeon-datafusion`](https://docs.rs/algeon-datafusion). See the
//! [API documentation boundary](https://algeon.dev/design#api-documentation).
//!
//! Producers and consumers import these types through the `rapids_interop`
//! library name; `cudf` re-exports them from `cudf::interop::contract`.
//!
//! A producer describes storage it already holds as a [`PreparedColumnExport`]
//! or [`PreparedTableExport`]. The export pairs a graph of [`SemanticColumn`]s
//! with the [`BufferBinding`]s behind it and a [`PreparedAccess`], and it never
//! owns or transfers the storage. A retained owner attached with
//! [`PreparedColumnExport::with_retained_owner`] or
//! [`PreparedTableExport::with_retained_owner`] keeps the storage alive while
//! the export, one of its clones, or a pending [`LeaseCompletion`] holds it.
//!
//! # Construction
//!
//! Raw addresses are descriptors. The unsafe `try_new` constructors require
//! live storage, valid contents and placement before granting safe read access.
//! Their graph checks return `Err` for structural mismatches:
//!
//! - Column ids and buffer ids are unique. The graph is acyclic and holds
//!   exactly the columns reachable from the root, through [`ColumnLayout`]
//!   children and dictionary keys.
//! - Each layout matches its [`TypeId`]. Value, offset, index, and validity
//!   buffers cover the column's logical range.
//! - With [`PreparedAccess::Host`], every binding is host memory. With
//!   [`PreparedAccess::Device`], every binding and the [`ProducerReadiness`]
//!   are on one CUDA device.
//! - A table's schema column ids are unique, names (when present) match the
//!   schema length, and every top-level column has `num_rows` rows.
//!
//! # Leases
//!
//! A consumer reads an export through a lease activated with a
//! [`ReadStartAuthority`] and a [`CompletionAuthority`]. Host exports accept
//! only the synchronous authorities, and their leases complete without backend
//! work. For a device export, a same-device completion stream must be on the
//! producer device, and the export's [`LeaseBackend`] starts, completes, or
//! drops each lease; a lease dropped without completing runs the backend's
//! drop fallback.
//! Active leases retain storage owners before returning from activation,
//! including when a caller forgets the lease. Consumers that expose borrowed
//! asynchronous reads must require an unsafe lifetime obligation; a destructor
//! cannot protect storage when it is forgotten.
//!
//! [`ActiveColumnReadLease::complete`] and [`ActiveTableReadLease::complete`]
//! return a [`LeaseCompletion`]. It keeps the export's retained owner alive
//! until the consumer's reads of producer storage have finished, and leaks the
//! owner if that wait fails while it is dropped. Failed incomplete cleanup
//! retains the owner and activation resources. Without an owner it aborts before
//! ending the producer borrow. Backend final-cleanup errors enter its runtime's
//! device-health quarantine. Explicit completion errors still run the fallback
//! until completion has been proved or a pending witness retains storage.

mod contract;
mod lease;
mod prepared;
mod types;
mod validation;

pub use contract::*;
pub use lease::*;
pub use prepared::*;
pub use types::*;
