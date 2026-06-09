//! Column-name resolution and validation helpers.
//!
//! These are used by `DataFrame` and other high-level types to translate
//! user-supplied column names into positional indices and to validate
//! schema constraints.

use std::sync::Arc;

use crate::{
    core::{
        column::Column,
        error::{Error, ErrorKind, Result},
    },
    internal::{
        hash::{FastHashSet, FastIndexMap, fast_hash_set_with_capacity},
        temp_table::TempTableView,
    },
};

pub(crate) fn missing_column_error(operation: &'static str, name: impl Into<String>) -> Error {
    let name = name.into();
    Error::new(ErrorKind::NotFound, format!("column '{}' not found in schema", name))
        .with_operation(operation)
        .with_context("column_name", name)
}

/// Validate that `names` contains no duplicates.
///
/// # Errors
///
/// Returns an error with the first duplicate name found.
pub(crate) fn validate_no_duplicate_names(names: &[String], operation: &'static str) -> Result<()> {
    let mut seen = fast_hash_set_with_capacity(names.len());
    for name in names {
        if !seen.insert(name) {
            return Err(Error::new(ErrorKind::InvalidOperation, format!("duplicate column name '{}'", name))
                .with_operation(operation)
                .with_context("duplicate_name", name.clone()));
        }
    }
    Ok(())
}

/// Resolve a slice of column name strings to their positional indices in
/// an `IndexMap`.
///
/// # Errors
///
/// Returns an error if any name is not found.
pub(crate) fn resolve_column_names(
    columns: &FastIndexMap<String, Arc<Column>>,
    names: &[&str],
    operation: &'static str,
) -> Result<Vec<usize>> {
    names.iter().map(|&name| columns.get_index_of(name).ok_or_else(|| missing_column_error(operation, name))).collect()
}

/// Build a borrowed `TempTableView` containing only the columns at the given
/// positional indices from the `IndexMap`.
///
/// This is zero-copy with respect to device data: it allocates only temporary
/// FFI view handles over the selected columns.
pub(crate) fn build_borrowed_table_view_from_columns<'a>(
    columns: &'a FastIndexMap<String, Arc<Column>>,
    indices: &[usize],
) -> Result<TempTableView<'a>> {
    let views = indices
        .iter()
        .map(|&idx| {
            let (_, col) = columns.get_index(idx).expect("BUG: resolved index should always be valid");
            col.view()
        })
        .collect::<Result<Vec<_>>>()?;

    TempTableView::from_column_views(views)
}

/// Compute payload column indices: all positions NOT in `key_indices`,
/// preserving insertion order.
pub(crate) fn payload_indices(columns: &FastIndexMap<String, Arc<Column>>, key_indices: &[usize]) -> Vec<usize> {
    let key_set: FastHashSet<usize> = key_indices.iter().copied().collect();
    (0..columns.len()).filter(|idx| !key_set.contains(idx)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_no_duplicate_names_ok() {
        let names = vec!["a".into(), "b".into(), "c".into()];
        assert!(validate_no_duplicate_names(&names, "test").is_ok());
    }

    #[test]
    fn validate_no_duplicate_names_rejects_dup() {
        let names = vec!["a".into(), "b".into(), "a".into()];
        let err = validate_no_duplicate_names(&names, "test").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidOperation);
        assert!(err.context().iter().any(|(k, v)| *k == "duplicate_name" && v == "a"));
    }

    #[test]
    fn resolve_column_names_ok() {
        let mut map = crate::internal::hash::fast_index_map();
        map.insert("x".into(), Arc::new(Column::empty(crate::core::types::TypeId::Int32).unwrap()));
        map.insert("y".into(), Arc::new(Column::empty(crate::core::types::TypeId::Int64).unwrap()));
        map.insert("z".into(), Arc::new(Column::empty(crate::core::types::TypeId::Float64).unwrap()));

        let indices = resolve_column_names(&map, &["z", "x"], "test").unwrap();
        assert_eq!(indices, vec![2, 0]);
    }

    #[test]
    fn resolve_column_names_missing() {
        let mut map = crate::internal::hash::fast_index_map();
        map.insert("a".into(), Arc::new(Column::empty(crate::core::types::TypeId::Int32).unwrap()));

        let err = resolve_column_names(&map, &["missing"], "test").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
        assert!(err.context().iter().any(|(k, v)| *k == "column_name" && v == "missing"));
    }

    #[test]
    fn payload_indices_excludes_keys() {
        let mut map = crate::internal::hash::fast_index_map();
        map.insert("a".into(), Arc::new(Column::empty(crate::core::types::TypeId::Int32).unwrap()));
        map.insert("b".into(), Arc::new(Column::empty(crate::core::types::TypeId::Int32).unwrap()));
        map.insert("c".into(), Arc::new(Column::empty(crate::core::types::TypeId::Int32).unwrap()));
        map.insert("d".into(), Arc::new(Column::empty(crate::core::types::TypeId::Int32).unwrap()));

        let result = payload_indices(&map, &[1, 3]);
        assert_eq!(result, vec![0, 2]);
    }
}
