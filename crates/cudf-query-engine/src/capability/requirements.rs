//! Re-export of [`reservation_requirement_scope`] so external callers can
//! resolve reservation scopes without depending on `capability::model`'s
//! internal organization.

#![deny(missing_docs)]

pub use super::model::reservation_requirement_scope;
