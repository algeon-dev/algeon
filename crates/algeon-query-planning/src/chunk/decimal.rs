use std::collections::{BTreeMap, BTreeSet};

use parquet::{
    basic::{ConvertedType, LogicalType, Type as ParquetPhysicalType},
    file::metadata::ParquetMetaData,
};

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    expr::Expr,
    source::{
        LocalParquetFile, LocalParquetSource, ParquetReadStrategy, ParquetReaderDecimalType,
        ParquetReaderDecimalTypeId, ParquetSourceLocation,
    },
};

#[doc(hidden)]
pub fn object_store_decimal_reader_predicate_requires_footer_contract(
    source: &LocalParquetSource,
    file: &LocalParquetFile,
) -> bool {
    matches!(file.resolved_source.read_strategy(), ParquetReadStrategy::RemoteDirect)
        && matches!(file.location, ParquetSourceLocation::ObjectStoreUri { .. })
        && source
            .reader_predicate
            .as_ref()
            .is_some_and(|predicate| source.reader_predicate_requires_source_contract(predicate))
}

#[doc(hidden)]
pub fn validate_object_store_reader_decimal_types(
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    reader_decimal_types: &[ParquetReaderDecimalType],
) -> Result<()> {
    if !object_store_decimal_reader_predicate_requires_footer_contract(source, file) {
        return Ok(());
    }
    let predicate = source.reader_predicate.as_ref().expect("contract check requires predicate");
    let missing = reader_predicate_decimal_columns(source, predicate)
        .into_iter()
        .filter(|column| !reader_decimal_types.iter().any(|reader_type| reader_type.column_name == *column))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }
    Err(reader_decimal_type_metadata_error(
        source,
        file,
        "reader_decimal_type_metadata_unavailable",
        missing.join(","),
        None,
    ))
}

#[doc(hidden)]
pub fn validate_supplied_reader_decimal_types_match_footer(
    source: &LocalParquetSource,
    file: &LocalParquetFile,
    supplied_reader_decimal_types: &[ParquetReaderDecimalType],
    footer_reader_decimal_types: &[ParquetReaderDecimalType],
) -> Result<()> {
    if supplied_reader_decimal_types.is_empty() {
        return Ok(());
    }
    let Some(predicate) = source.reader_predicate.as_ref() else {
        return Ok(());
    };
    let supplied_by_column = supplied_reader_decimal_types
        .iter()
        .map(|reader_type| (reader_type.column_name.as_str(), reader_type))
        .collect::<BTreeMap<_, _>>();
    let footer_by_column = footer_reader_decimal_types
        .iter()
        .map(|reader_type| (reader_type.column_name.as_str(), reader_type))
        .collect::<BTreeMap<_, _>>();
    let mismatched = reader_predicate_decimal_columns(source, predicate)
        .into_iter()
        .filter_map(|column| {
            let supplied = supplied_by_column.get(column.as_str())?;
            let footer = footer_by_column.get(column.as_str())?;
            ((supplied.type_id != footer.type_id) || (supplied.cudf_scale != footer.cudf_scale)).then_some(format!(
                "{}:{}:{}!=footer:{}:{}",
                column,
                supplied.type_id.as_str(),
                supplied.cudf_scale,
                footer.type_id.as_str(),
                footer.cudf_scale
            ))
        })
        .collect::<Vec<_>>();
    if mismatched.is_empty() {
        return Ok(());
    }
    Err(reader_decimal_type_metadata_error(
        source,
        file,
        "reader_decimal_type_metadata_mismatch",
        mismatched.join(","),
        Some("supplied_reader_decimal_types_do_not_match_footer"),
    ))
}

#[doc(hidden)]
pub fn reader_decimal_type_metadata_unavailable_error(
    source: &LocalParquetSource,
    file: &LocalParquetFile,
) -> QueryError {
    let missing = source
        .reader_predicate
        .as_ref()
        .map(|predicate| reader_predicate_decimal_columns(source, predicate).into_iter().collect::<Vec<_>>().join(","))
        .unwrap_or_default();
    reader_decimal_type_metadata_error(source, file, "reader_decimal_type_metadata_unavailable", missing, None)
}

fn reader_decimal_type_metadata_error(
    _source: &LocalParquetSource,
    file: &LocalParquetFile,
    reason: &'static str,
    columns: String,
    detail: Option<&'static str>,
) -> QueryError {
    let mut error = QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "query_engine.source_metadata.reader_decimal_type",
        "remote decimal reader predicates require Parquet footer reader type metadata",
    )
    .with_fact("reason", reason)
    .with_fact("path", file.location.diagnostic_location())
    .with_fact("stable_id", file.location.diagnostic_stable_id(file.stable_id.as_str()))
    .with_fact("read_strategy", file.resolved_source.read_strategy().as_str())
    .with_fact("remote_read_mode", "require_kvikio")
    .with_fact("missing_reader_decimal_columns", columns);
    if let Some(detail) = detail {
        error = error.with_fact("detail", detail);
    }
    error
}

fn reader_predicate_decimal_columns(source: &LocalParquetSource, expr: &Expr) -> BTreeSet<String> {
    let mut columns = BTreeSet::new();
    collect_reader_predicate_decimal_columns(source, expr, &mut columns);
    columns
}

fn collect_reader_predicate_decimal_columns(source: &LocalParquetSource, expr: &Expr, columns: &mut BTreeSet<String>) {
    match expr {
        Expr::Column(column) => {
            if source
                .schema
                .field_with_name(column.name())
                .is_ok_and(|field| matches!(field.data_type(), arrow::datatypes::DataType::Decimal128(_, _)))
            {
                columns.insert(column.name().to_string());
            }
        },
        Expr::Literal(_) => {},
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::Alias { expr: inner, .. }
        | Expr::Cast { expr: inner, .. }
        | Expr::TryCast { expr: inner, .. }
        | Expr::Like { expr: inner, .. }
        | Expr::Substring { expr: inner, .. }
        | Expr::Lower(inner)
        | Expr::Upper(inner)
        | Expr::Trim { expr: inner, .. }
        | Expr::LTrim { expr: inner, .. }
        | Expr::RTrim { expr: inner, .. }
        | Expr::StringContains { expr: inner, .. }
        | Expr::StartsWith { expr: inner, .. }
        | Expr::EndsWith { expr: inner, .. }
        | Expr::StringLength { expr: inner, .. }
        | Expr::RegexpReplace { expr: inner, .. }
        | Expr::Abs(inner)
        | Expr::Sqrt { expr: inner, .. }
        | Expr::Round { expr: inner, .. }
        | Expr::Extract { expr: inner, .. }
        | Expr::DateTrunc { expr: inner, .. } => collect_reader_predicate_decimal_columns(source, inner, columns),
        Expr::Binary { left, right, .. }
        | Expr::IsDistinctFrom { left, right }
        | Expr::IsNotDistinctFrom { left, right } => {
            collect_reader_predicate_decimal_columns(source, left, columns);
            collect_reader_predicate_decimal_columns(source, right, columns);
        },
        Expr::InList { expr, list, .. } => {
            collect_reader_predicate_decimal_columns(source, expr, columns);
            for item in list {
                collect_reader_predicate_decimal_columns(source, item, columns);
            }
        },
        Expr::Concat(exprs) => {
            for item in exprs {
                collect_reader_predicate_decimal_columns(source, item, columns);
            }
        },
        Expr::Case { branches, else_expr } => {
            for branch in branches {
                collect_reader_predicate_decimal_columns(source, &branch.when, columns);
                collect_reader_predicate_decimal_columns(source, &branch.then, columns);
            }
            collect_reader_predicate_decimal_columns(source, else_expr, columns);
        },
    }
}

#[doc(hidden)]
pub fn parquet_reader_decimal_types(
    metadata: &ParquetMetaData,
    source: &LocalParquetSource,
) -> Vec<ParquetReaderDecimalType> {
    source
        .read_columns
        .iter()
        .zip(&source.read_column_file_indices)
        .filter_map(|(column_name, file_index)| {
            let field = source.schema.field_with_name(column_name).ok()?;
            if !matches!(field.data_type(), arrow::datatypes::DataType::Decimal128(_, _)) {
                return None;
            }
            let descriptor = metadata.file_metadata().schema_descr().columns().get(*file_index)?;
            let (_precision, scale) = parquet_decimal_precision_scale(descriptor)?;
            let type_id = parquet_reader_decimal_type_id(descriptor)?;
            Some(ParquetReaderDecimalType::new(
                column_name.clone(),
                type_id,
                crate::datatypes::arrow_decimal_scale_to_cudf_scale(scale),
            ))
        })
        .collect()
}

fn parquet_decimal_precision_scale(descriptor: &parquet::schema::types::ColumnDescriptor) -> Option<(u8, i8)> {
    if let Some(LogicalType::Decimal(decimal)) = descriptor.logical_type_ref() {
        return Some((u8::try_from(decimal.precision).ok()?, i8::try_from(decimal.scale).ok()?));
    }
    (descriptor.converted_type() == ConvertedType::DECIMAL)
        .then(|| (descriptor.type_precision(), descriptor.type_scale()))
        .and_then(|(precision, scale)| Some((u8::try_from(precision).ok()?, i8::try_from(scale).ok()?)))
}

fn parquet_reader_decimal_type_id(
    descriptor: &parquet::schema::types::ColumnDescriptor,
) -> Option<ParquetReaderDecimalTypeId> {
    match descriptor.physical_type() {
        ParquetPhysicalType::INT32 => Some(ParquetReaderDecimalTypeId::Decimal32),
        ParquetPhysicalType::INT64 => Some(ParquetReaderDecimalTypeId::Decimal64),
        ParquetPhysicalType::FIXED_LEN_BYTE_ARRAY => match descriptor.type_length() {
            0..=4 => Some(ParquetReaderDecimalTypeId::Decimal32),
            5..=8 => Some(ParquetReaderDecimalTypeId::Decimal64),
            9..=16 => Some(ParquetReaderDecimalTypeId::Decimal128),
            _ => None,
        },
        ParquetPhysicalType::BYTE_ARRAY => {
            parquet_decimal_precision_scale(descriptor).and_then(|(precision, _)| match precision {
                1..=9 => Some(ParquetReaderDecimalTypeId::Decimal32),
                10..=18 => Some(ParquetReaderDecimalTypeId::Decimal64),
                19..=38 => Some(ParquetReaderDecimalTypeId::Decimal128),
                _ => None,
            })
        },
        _ => None,
    }
}
