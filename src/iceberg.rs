//! Compatibility facade for Iceberg integrations.
//!
//! Iceberg implementation code lives under [`crate::table_format::iceberg`].
//! This module preserves the historical `crate::iceberg` public path.

pub use crate::table_format::iceberg::*;
