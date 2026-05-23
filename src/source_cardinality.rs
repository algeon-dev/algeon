//! Source cardinality proofs from Arrow schema metadata.
//!
//! The only trusted unique-key input is the Arrow schema/field metadata that
//! travels with the scan being lowered. A unique-key predicate can bound a
//! reader to `max_rows = 1`, so evidence that does not flow with the scan
//! (such as a process-global path registry) risks silently truncating data.

use std::collections::{BTreeSet, HashMap};

use arrow::datatypes::Schema;
use nexus_query_engine::expr::{BinaryOp, Expr, LiteralValue};

/// Arrow schema/field metadata key carrying comma-separated source unique-key
/// column groups.
pub const SOURCE_UNIQUE_KEYS_METADATA_KEY: &str = "datafusion_nexus.source_unique_keys.v1";

pub fn schema_with_source_unique_keys(schema: &Schema, unique_keys: &[&[&str]]) -> Schema {
    let encoded = encode_source_unique_keys(unique_keys);
    if encoded.is_empty() {
        return schema.clone();
    }
    let mut metadata = schema.metadata().clone();
    metadata.insert(SOURCE_UNIQUE_KEYS_METADATA_KEY.to_string(), encoded);
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            let mut field = field.as_ref().clone();
            let field_unique_keys = unique_keys
                .iter()
                .filter(|key| key.iter().any(|column| column.trim() == field.name()))
                .copied()
                .collect::<Vec<_>>();
            let encoded = encode_source_unique_keys(&field_unique_keys);
            if !encoded.is_empty() {
                let mut metadata = field.metadata().clone();
                metadata.insert(SOURCE_UNIQUE_KEYS_METADATA_KEY.to_string(), encoded);
                field = field.with_metadata(metadata);
            }
            field
        })
        .collect::<Vec<_>>();
    Schema::new_with_metadata(fields, metadata)
}

pub(crate) fn reader_predicate_proves_source_at_most_one_row(schemas: &[&Schema], predicate: &Expr) -> bool {
    let mut unique_keys = BTreeSet::new();
    for schema in schemas {
        unique_keys.extend(source_unique_keys_from_schema(schema));
    }
    reader_predicate_proves_unique_keys_at_most_one_row(unique_keys.into_iter().collect(), predicate)
}

fn reader_predicate_proves_unique_keys_at_most_one_row(unique_keys: Vec<Vec<String>>, predicate: &Expr) -> bool {
    if unique_keys.is_empty() {
        return false;
    }
    let bound_columns = predicate_bound_columns(predicate);
    unique_keys.iter().any(|key| key.iter().all(|column| bound_columns.contains(column)))
}

fn encode_source_unique_keys(unique_keys: &[&[&str]]) -> String {
    unique_keys
        .iter()
        .filter_map(|key| {
            let columns =
                key.iter().map(|column| column.trim()).filter(|column| !column.is_empty()).collect::<Vec<_>>();
            (!columns.is_empty()).then(|| columns.join(","))
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn source_unique_keys_from_metadata(metadata: &HashMap<String, String>) -> Vec<Vec<String>> {
    metadata
        .get(SOURCE_UNIQUE_KEYS_METADATA_KEY)
        .into_iter()
        .flat_map(|value| value.split(';'))
        .filter_map(|key| {
            let columns = key
                .split(',')
                .map(str::trim)
                .filter(|column| !column.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            (!columns.is_empty()).then_some(columns)
        })
        .collect()
}

fn source_unique_keys_from_schema(schema: &Schema) -> Vec<Vec<String>> {
    let mut unique_keys = BTreeSet::new();
    unique_keys.extend(source_unique_keys_from_metadata(schema.metadata()));
    for field in schema.fields() {
        unique_keys.extend(source_unique_keys_from_metadata(field.metadata()));
    }
    unique_keys.into_iter().collect()
}

fn predicate_bound_columns(predicate: &Expr) -> BTreeSet<String> {
    let mut columns = BTreeSet::new();
    collect_predicate_bound_columns(predicate, &mut columns);
    columns
}

fn collect_predicate_bound_columns(predicate: &Expr, columns: &mut BTreeSet<String>) {
    match predicate {
        Expr::Binary { left, op: BinaryOp::And, right, .. } => {
            collect_predicate_bound_columns(left, columns);
            collect_predicate_bound_columns(right, columns);
        },
        Expr::Binary { left, op: BinaryOp::Eq, right, .. } => {
            if let Some(column) = equality_bound_column(left, right) {
                columns.insert(column.to_string());
            } else if let Some(column) = equality_bound_column(right, left) {
                columns.insert(column.to_string());
            }
        },
        Expr::Alias { expr, .. } => collect_predicate_bound_columns(expr, columns),
        _ => {},
    }
}

fn equality_bound_column<'a>(column_expr: &'a Expr, literal_expr: &Expr) -> Option<&'a str> {
    let column = match column_expr {
        Expr::Column(column) => column.name(),
        Expr::Alias { expr, .. } => return equality_bound_column(expr, literal_expr),
        _ => return None,
    };
    literal_is_non_null(literal_expr).then_some(column)
}

fn literal_is_non_null(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(value) => !matches!(value, LiteralValue::Null(_)),
        Expr::Alias { expr, .. } => literal_is_non_null(expr),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field};

    use super::*;

    #[test]
    fn source_unique_key_metadata_proves_equality_bound_key() {
        let schema =
            Schema::new(vec![Field::new("id", DataType::Int32, false), Field::new("v", DataType::Int32, true)]);
        let schema = schema_with_source_unique_keys(&schema, &[&["id"]]);
        let predicate = Expr::binary(Expr::column("id"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(7)));

        assert!(reader_predicate_proves_source_at_most_one_row(&[&schema], &predicate));
    }

    #[test]
    fn source_unique_key_metadata_requires_every_composite_key_column() {
        let schema = Schema::new(vec![
            Field::new("k1", DataType::Int32, false),
            Field::new("k2", DataType::Int32, false),
            Field::new("v", DataType::Int32, true),
        ]);
        let schema = schema_with_source_unique_keys(&schema, &[&["k1", "k2"]]);
        let partial = Expr::binary(Expr::column("k1"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(7)));
        let full = Expr::binary(
            partial.clone(),
            BinaryOp::And,
            Expr::binary(Expr::column("k2"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(9))),
        );

        assert!(!reader_predicate_proves_source_at_most_one_row(&[&schema], &partial));
        assert!(reader_predicate_proves_source_at_most_one_row(&[&schema], &full));
    }

    #[test]
    fn source_unique_key_metadata_ignores_null_equality() {
        let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
        let schema = schema_with_source_unique_keys(&schema, &[&["id"]]);
        let predicate =
            Expr::binary(Expr::column("id"), BinaryOp::Eq, Expr::literal(LiteralValue::Null(DataType::Int32)));

        assert!(!reader_predicate_proves_source_at_most_one_row(&[&schema], &predicate));
    }

    #[test]
    fn source_unique_key_metadata_rejects_cast_key_equality() {
        let schema = Schema::new(vec![Field::new("id", DataType::Int32, false)]);
        let schema = schema_with_source_unique_keys(&schema, &[&["id"]]);
        let predicate = Expr::binary(
            Expr::cast(Expr::column("id"), DataType::Float64),
            BinaryOp::Eq,
            Expr::literal(LiteralValue::Float64(7.0)),
        );

        assert!(!reader_predicate_proves_source_at_most_one_row(&[&schema], &predicate));
    }
}
