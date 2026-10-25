//! CPU tests of Iceberg scan planning: plan shape, pruning, the storage
//! paths, catalog registration, native planning of a remote table, and every
//! rejection of a snapshot the Parquet scan would misread.

mod native_planning;
mod register;
mod rejections;
mod scan;
#[path = "../support/warehouse.rs"]
mod support;
