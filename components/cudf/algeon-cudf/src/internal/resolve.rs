//! Column-name resolution and validation helpers.
//!
//! These are used by `DataFrame` and other high-level types to translate
//! user-supplied column names into positional indices and to validate
//! schema constraints.

use std::sync::Arc;

use crate::{
    core::{
        column::Column,
        error::{self, Error, ErrorCode, FACT_COLUMN_INDEX, Result},
    },
    internal::{
        hash::{FastHashSet, FastIndexMap, fast_hash_set_with_capacity},
        temp_table::TempTableView,
    },
};

pub(crate) fn missing_column_error(operation: &'static str, name: impl Into<String>) -> Error {
    let name = name.into();
    Error::new_with(ErrorCode::MissingColumn, operation, "column not found in schema").with_column(name)
}

pub(crate) fn duplicate_column_name_error_with_contract(
    operation: &'static str,
    duplicate_name: impl Into<String>,
    column_name_count: usize,
    message: impl Into<String>,
    reason: &'static str,
    remedy: &'static str,
) -> Error {
    let duplicate_name = duplicate_name.into();
    Error::new_with(ErrorCode::DuplicateColumnName, operation, message)
        .with_detail("duplicate_name", duplicate_name)
        .with_detail("column_name_count", column_name_count.to_string())
        .with_detail("resource", "column_name")
        .with_detail("reason", reason)
        .with_detail("remedy", remedy)
}

pub(crate) fn duplicate_column_name_error(
    operation: &'static str,
    duplicate_name: impl Into<String>,
    column_name_count: usize,
    message: impl Into<String>,
) -> Error {
    duplicate_column_name_error_with_contract(
        operation,
        duplicate_name,
        column_name_count,
        message,
        "duplicate_column_name",
        "ensure column names are unique",
    )
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
            return Err(duplicate_column_name_error(
                operation,
                name.clone(),
                names.len(),
                format!("duplicate column name '{}'", name),
            ));
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
            let (_, col) = columns.get_index(idx).ok_or_else(|| {
                error::invariant_violation(
                    "internal::build_borrowed_table_view_from_columns",
                    "resolved column index should always remain valid",
                )
                .with_fact(FACT_COLUMN_INDEX, idx)
            })?;
            col.view()
        })
        .collect::<Result<Vec<_>>>()?;

    if views.is_empty() {
        let authoritative_domain =
            columns.values().next().map(|column| Arc::clone(column.execution_domain())).ok_or_else(|| {
                Error::new_with(
                    ErrorCode::EmptyConstructionRequiresRuntime,
                    "internal::build_borrowed_table_view_from_columns",
                    "building an empty borrowed table view requires an authoritative source column",
                )
            })?;
        return TempTableView::from_column_views_in_domain(
            views,
            authoritative_domain,
            "internal::build_borrowed_table_view_from_columns",
        );
    }

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
    use crate::{
        core::error::{ErrorKind, FactKeyName, FactValue},
        runtime::Runtime,
    };

    fn empty_column(type_id: crate::core::types::TypeId) -> Arc<Column> {
        Arc::new(Runtime::new().unwrap().empty_column(type_id).unwrap())
    }

    #[test]
    fn validate_no_duplicate_names_ok() {
        let names = vec!["a".into(), "b".into(), "c".into()];
        assert!(validate_no_duplicate_names(&names, "test").is_ok());
    }

    #[test]
    fn validate_no_duplicate_names_rejects_dup() {
        let names = vec!["a".into(), "b".into(), "a".into()];
        let err = validate_no_duplicate_names(&names, "test").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.details().iter().any(|(k, v)| *k == "duplicate_name" && v == "a"));
    }

    #[test]
    fn resolve_column_names_ok() {
        let mut map = crate::internal::hash::fast_index_map();
        map.insert("x".into(), empty_column(crate::core::types::TypeId::Int32));
        map.insert("y".into(), empty_column(crate::core::types::TypeId::Int64));
        map.insert("z".into(), empty_column(crate::core::types::TypeId::Float64));

        let indices = resolve_column_names(&map, &["z", "x"], "test").unwrap();
        assert_eq!(indices, vec![2, 0]);
    }

    #[test]
    fn resolve_column_names_missing() {
        let mut map = crate::internal::hash::fast_index_map();
        map.insert("a".into(), empty_column(crate::core::types::TypeId::Int32));

        let err = resolve_column_names(&map, &["missing"], "test").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::NotFound);
        assert_eq!(err.fact_value(FactKeyName::ColumnName), Some(&FactValue::Str("missing".into())));
    }

    #[test]
    fn payload_indices_excludes_keys() {
        let mut map = crate::internal::hash::fast_index_map();
        map.insert("a".into(), empty_column(crate::core::types::TypeId::Int32));
        map.insert("b".into(), empty_column(crate::core::types::TypeId::Int32));
        map.insert("c".into(), empty_column(crate::core::types::TypeId::Int32));
        map.insert("d".into(), empty_column(crate::core::types::TypeId::Int32));

        let result = payload_indices(&map, &[1, 3]);
        assert_eq!(result, vec![0, 2]);
    }
}
