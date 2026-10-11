//! Source cardinality proofs from Arrow schema metadata.
//!
//! The only trusted unique-key input is the Arrow schema/field metadata that
//! travels with the scan being lowered. A unique-key predicate can bound a
//! reader to `max_rows = 1`. Evidence that does not flow with the scan, such as
//! a process-global path registry, risks truncating data.

use std::collections::{BTreeSet, HashMap};

use arrow::datatypes::Schema;
use query_planning::expr::{BinaryOp, Expr, LiteralValue};

/// Arrow schema/field metadata key carrying a JSON array of source unique-key
/// column-name arrays.
pub const SOURCE_UNIQUE_KEYS_METADATA_KEY: &str = "algeon_datafusion.source_unique_keys";

/// Returns a copy of `schema` with the declared source unique-key groups
/// recorded on the key columns' fields.
///
/// Unique-key evidence is source-scoped, so it is carried only on fields. A
/// schema-level entry would share one key across every source in a plan:
/// `build_join_schema` concatenates both sides' schema metadata and keeps the
/// last writer, so a join between two keyed sources replaces one
/// source's evidence with the other's. That also breaks physical optimizer
/// rules that declare `schema_check()`, which reject any schema metadata
/// change across the rule.
///
/// Column names are preserved, including whitespace and punctuation.
/// Empty key groups are omitted because they cannot prove a row bound.
pub fn schema_with_source_unique_keys(schema: &Schema, unique_keys: &[&[&str]]) -> Schema {
    if encode_source_unique_keys(unique_keys).is_empty() {
        return schema.clone();
    }
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            let mut field = field.as_ref().clone();
            let field_unique_keys = unique_keys
                .iter()
                .filter(|key| key.iter().any(|column| *column == field.name()))
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
    Schema::new_with_metadata(fields, schema.metadata().clone())
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
    let unique_keys = unique_keys.iter().filter(|key| !key.is_empty()).collect::<Vec<_>>();
    if unique_keys.is_empty() {
        return String::new();
    }
    serde_json::to_string(&unique_keys).expect("string-array JSON serialization cannot fail")
}

fn source_unique_keys_from_metadata(metadata: &HashMap<String, String>) -> Vec<Vec<String>> {
    let Some(value) = metadata.get(SOURCE_UNIQUE_KEYS_METADATA_KEY) else {
        return Vec::new();
    };
    let Ok(unique_keys) = serde_json::from_str::<Vec<Vec<String>>>(value) else {
        return Vec::new();
    };
    if unique_keys.iter().any(Vec::is_empty) {
        return Vec::new();
    }
    unique_keys
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
    fn source_unique_key_metadata_preserves_delimiters_and_whitespace_in_column_names() {
        for (name, other) in [("a;b", "a"), ("a,b", "a"), (" id ", "id")] {
            let schema = Schema::new(vec![
                Field::new(name, DataType::Int32, false),
                Field::new(other, DataType::Int32, false),
                Field::new("b", DataType::Int32, false),
            ]);
            let schema = schema_with_source_unique_keys(&schema, &[&[name]]);
            let unrelated = Expr::binary(
                Expr::binary(Expr::column(other), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(7))),
                BinaryOp::And,
                Expr::binary(Expr::column("b"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(9))),
            );
            let exact = Expr::binary(Expr::column(name), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(7)));
            assert!(!reader_predicate_proves_source_at_most_one_row(&[&schema], &unrelated), "{name}");
            assert!(reader_predicate_proves_source_at_most_one_row(&[&schema], &exact), "{name}");
        }
    }

    #[test]
    fn source_unique_key_metadata_rejects_malformed_values() {
        let predicate = Expr::binary(Expr::column("id"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(7)));
        for value in ["not json", r#"["id"]"#, "[[]]"] {
            let schema = Schema::new_with_metadata(
                vec![Field::new("id", DataType::Int32, false)],
                HashMap::from([(SOURCE_UNIQUE_KEYS_METADATA_KEY.to_string(), value.to_string())]),
            );

            assert!(!reader_predicate_proves_source_at_most_one_row(&[&schema], &predicate));
        }
    }

    #[test]
    fn source_unique_key_metadata_decodes_field_metadata() {
        let field = Field::new("id", DataType::Int32, false)
            .with_metadata(HashMap::from([(SOURCE_UNIQUE_KEYS_METADATA_KEY.to_string(), r#"[["id"]]"#.to_string())]));
        let schema = Schema::new(vec![field]);
        let predicate = Expr::binary(Expr::column("id"), BinaryOp::Eq, Expr::literal(LiteralValue::Int32(7)));

        assert!(reader_predicate_proves_source_at_most_one_row(&[&schema], &predicate));
    }

    #[test]
    fn source_unique_key_metadata_survives_join_without_replacing_another_source() {
        use datafusion::{common::JoinType, physical_plan::joins::utils::build_join_schema};

        let date_dim = schema_with_source_unique_keys(
            &Schema::new(vec![Field::new("d_date_sk", DataType::Int64, false)]),
            &[&["d_date_sk"]],
        );
        let store = schema_with_source_unique_keys(
            &Schema::new(vec![Field::new("s_store_sk", DataType::Int64, false)]),
            &[&["s_store_sk"]],
        );

        // `build_join_schema` concatenates schema metadata and keeps the last
        // writer, so a shared schema-level key would drop one side's evidence
        // and change the joined metadata with join orientation.
        for (left, right) in [(&date_dim, &store), (&store, &date_dim)] {
            let (joined, _) = build_join_schema(left, right, &JoinType::Inner);
            assert_eq!(joined.metadata().get(SOURCE_UNIQUE_KEYS_METADATA_KEY), None);
        }

        let (joined, _) = build_join_schema(&date_dim, &store, &JoinType::Inner);
        let date_key = Expr::binary(Expr::column("d_date_sk"), BinaryOp::Eq, Expr::literal(LiteralValue::Int64(7)));
        let store_key = Expr::binary(Expr::column("s_store_sk"), BinaryOp::Eq, Expr::literal(LiteralValue::Int64(9)));
        assert!(reader_predicate_proves_source_at_most_one_row(&[&joined], &date_key));
        assert!(reader_predicate_proves_source_at_most_one_row(&[&joined], &store_key));
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
