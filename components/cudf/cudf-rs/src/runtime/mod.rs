//! Runtime control for per-operation execution authority and retained provenance.

pub mod execution_context;
pub(crate) mod execution_domain;
mod owner;
pub mod stream;

pub use execution_context::*;
pub use owner::*;
pub use stream::*;
