use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

use super::*;

/// A source-schema column referenced by a local Parquet reader predicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceReaderPredicateColumn {
    pub(super) name: String,
    pub(super) source: SourceColumnRef,
}

impl SourceReaderPredicateColumn {
    /// Create a predicate column reference backed by one decoded read column.
    pub fn read_column(name: impl Into<String>, read_index: usize) -> Result<Self> {
        let name = checked_name(name.into(), OP_SOURCE_READER_PREDICATE, "reader predicate column")?;
        Ok(Self { name, source: SourceColumnRef::ReadColumn(read_index) })
    }

    /// Logical source-schema column name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Source column reference.
    pub fn source(&self) -> SourceColumnRef {
        self.source
    }
}

/// Comparison operators supported by source-aware reader predicate lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceReaderPredicateOp {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

impl SourceReaderPredicateOp {
    fn ast_operator(self) -> AstOperator {
        match self {
            Self::Equal => AstOperator::Equal,
            Self::NotEqual => AstOperator::NotEqual,
            Self::Less => AstOperator::Less,
            Self::LessEqual => AstOperator::LessEqual,
            Self::Greater => AstOperator::Greater,
            Self::GreaterEqual => AstOperator::GreaterEqual,
        }
    }
}

/// Literal value for a source-aware reader predicate.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SourceReaderPredicateLiteral {
    Boolean(bool),
    Int64(i64),
    UInt64(u64),
    Float64(f64),
    /// Decimal128 raw fixed-point value plus cudf scale.
    Decimal128 {
        raw: i128,
        scale: i32,
    },
    Date32(i32),
    String(String),
}

/// Source-aware local Parquet reader predicate.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SourceReaderPredicate {
    /// Binary comparison between a decoded read column and a literal.
    Comparison {
        column: SourceReaderPredicateColumn,
        op: SourceReaderPredicateOp,
        literal: SourceReaderPredicateLiteral,
    },
    /// Literal-list membership for one decoded read column.
    InList { column: SourceReaderPredicateColumn, literals: Vec<SourceReaderPredicateLiteral>, negated: bool },
    /// Null-aware boolean AND of two source reader predicates.
    And(Box<SourceReaderPredicate>, Box<SourceReaderPredicate>),
    /// Null-aware boolean OR of two source reader predicates.
    Or(Box<SourceReaderPredicate>, Box<SourceReaderPredicate>),
    /// Boolean NOT of one source reader predicate.
    Not(Box<SourceReaderPredicate>),
}

impl SourceReaderPredicate {
    /// Build a column/literal comparison.
    pub fn comparison(
        column: SourceReaderPredicateColumn,
        op: SourceReaderPredicateOp,
        literal: SourceReaderPredicateLiteral,
    ) -> Self {
        Self::Comparison { column, op, literal }
    }

    /// Build `column = literal`.
    pub fn equal(column: SourceReaderPredicateColumn, literal: SourceReaderPredicateLiteral) -> Self {
        Self::comparison(column, SourceReaderPredicateOp::Equal, literal)
    }

    /// Build `column >= literal`.
    pub fn greater_equal(column: SourceReaderPredicateColumn, literal: SourceReaderPredicateLiteral) -> Self {
        Self::comparison(column, SourceReaderPredicateOp::GreaterEqual, literal)
    }

    /// Build literal-list membership for one decoded read column.
    pub fn in_list(
        column: SourceReaderPredicateColumn,
        literals: Vec<SourceReaderPredicateLiteral>,
        negated: bool,
    ) -> Result<Self> {
        if literals.is_empty() {
            return Err(Error::new_with(
                ErrorCode::EmptyInput,
                OP_SOURCE_READER_PREDICATE,
                "source reader predicate IN list must not be empty",
            )
            .with_detail("reason", "empty_in_list"));
        }
        Ok(Self::InList { column, literals, negated })
    }

    /// Build a null-aware boolean AND.
    pub fn and(left: SourceReaderPredicate, right: SourceReaderPredicate) -> Self {
        Self::And(Box::new(left), Box::new(right))
    }

    /// Build a null-aware boolean OR.
    pub fn or(left: SourceReaderPredicate, right: SourceReaderPredicate) -> Self {
        Self::Or(Box::new(left), Box::new(right))
    }
}

/// Build a libcudf reader predicate in the same dtype domain used by the local
/// Parquet reader.
///
/// Decimal predicates are typed from the Parquet footer for the selected
/// source chunk. This keeps logical `Decimal128` source schemas compatible with
/// Parquet columns that libcudf evaluates as Decimal32, Decimal64, or
/// Decimal128.
pub fn build_local_parquet_reader_predicate(
    runtime: &Runtime,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    predicate: &SourceReaderPredicate,
) -> Result<AstExpression> {
    source.validate_chunk_partition_values(chunk, OP_SOURCE_READER_PREDICATE)?;
    let mut type_cache = SourceReaderPredicateTypeCache::default();
    lower_source_reader_predicate(runtime, source, chunk, predicate, &mut type_cache)
}

/// Build a libcudf reader predicate that is valid for every chunk in a
/// multi-file local Parquet batch.
///
/// Decimal predicates are batch-stable only when every selected chunk resolves
/// each predicate column to the same libcudf reader dtype. This protects mixed
/// Parquet decimal physical encodings from sharing a predicate scalar typed for
/// a different file.
pub fn build_local_parquet_reader_batch_predicate(
    runtime: &Runtime,
    source: &LocalParquetSource,
    chunks: &[SourceChunk],
    predicate: &SourceReaderPredicate,
) -> Result<AstExpression> {
    let Some(first) = chunks.first() else {
        return Err(Error::new_with(
            ErrorCode::EmptyInput,
            OP_SOURCE_READER_PREDICATE,
            "source reader predicate batch requires at least one source chunk",
        )
        .with_detail("reason", "empty_batch"));
    };
    source.validate_chunk_partition_values(first, OP_SOURCE_READER_PREDICATE)?;
    let mut type_cache = SourceReaderPredicateTypeCache::default();
    let first_types = source_reader_predicate_target_types(source, first, predicate, &mut type_cache)?;
    for chunk in chunks.iter().skip(1) {
        source.validate_chunk_partition_values(chunk, OP_SOURCE_READER_PREDICATE)?;
        let chunk_types = source_reader_predicate_target_types(source, chunk, predicate, &mut type_cache)?;
        if chunk_types != first_types {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                OP_SOURCE_READER_PREDICATE,
                "source reader predicate batch requires stable predicate column dtypes",
            )
            .with_detail("reason", "batch_reader_predicate_dtype_mismatch")
            .with_detail("first_chunk_index", first.chunk_index().to_string())
            .with_detail("first_file_index", first.file_index().to_string())
            .with_detail("chunk_index", chunk.chunk_index().to_string())
            .with_detail("file_index", chunk.file_index().to_string())
            .with_detail("first_types", format!("{first_types:?}"))
            .with_detail("chunk_types", format!("{chunk_types:?}")));
        }
    }
    lower_source_reader_predicate(runtime, source, first, predicate, &mut type_cache)
}

fn lower_source_reader_predicate(
    runtime: &Runtime,
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    predicate: &SourceReaderPredicate,
    type_cache: &mut SourceReaderPredicateTypeCache,
) -> Result<AstExpression> {
    match predicate {
        SourceReaderPredicate::Comparison { column, op, literal } => {
            let (read_index, target_type) = resolve_source_reader_column(source, chunk, column, type_cache)?;
            let left = source_reader_column_ast(read_index)?;
            let right = source_reader_literal_ast(runtime, literal, target_type)?;
            AstExpression::binary(op.ast_operator(), left, right).map_err(|error| {
                error
                    .reanchor(OP_SOURCE_READER_PREDICATE)
                    .with_detail("reason", "ast_binary_create_failed")
                    .with_detail("operator", format!("{op:?}"))
            })
        },
        SourceReaderPredicate::InList { column, literals, negated } => {
            if literals.is_empty() {
                return Err(Error::new_with(
                    ErrorCode::EmptyInput,
                    OP_SOURCE_READER_PREDICATE,
                    "source reader predicate IN list must not be empty",
                )
                .with_detail("reason", "empty_in_list"));
            }
            let (read_index, target_type) = resolve_source_reader_column(source, chunk, column, type_cache)?;
            let input = source_reader_column_ast(read_index)?;
            let values = literals
                .iter()
                .map(|literal| source_reader_literal_ast(runtime, literal, target_type))
                .collect::<Result<Vec<_>>>()?;
            if *negated {
                AstExpression::is_not_in_list(input, values)
            } else {
                AstExpression::is_in_list(input, values)
            }
            .map_err(|error| {
                error
                    .reanchor(OP_SOURCE_READER_PREDICATE)
                    .with_detail("reason", "ast_in_list_create_failed")
                    .with_detail("negated", negated.to_string())
                    .with_detail("value_count", literals.len().to_string())
            })
        },
        SourceReaderPredicate::And(left, right) => {
            let left = lower_source_reader_predicate(runtime, source, chunk, left, type_cache)?;
            let right = lower_source_reader_predicate(runtime, source, chunk, right, type_cache)?;
            AstExpression::binary(AstOperator::NullLogicalAnd, left, right)
                .map_err(|error| error.reanchor(OP_SOURCE_READER_PREDICATE).with_detail("reason", "ast_and_failed"))
        },
        SourceReaderPredicate::Or(left, right) => {
            let left = lower_source_reader_predicate(runtime, source, chunk, left, type_cache)?;
            let right = lower_source_reader_predicate(runtime, source, chunk, right, type_cache)?;
            AstExpression::binary(AstOperator::NullLogicalOr, left, right)
                .map_err(|error| error.reanchor(OP_SOURCE_READER_PREDICATE).with_detail("reason", "ast_or_failed"))
        },
        SourceReaderPredicate::Not(input) => {
            let input = lower_source_reader_predicate(runtime, source, chunk, input, type_cache)?;
            AstExpression::unary(AstOperator::Not, input)
                .map_err(|error| error.reanchor(OP_SOURCE_READER_PREDICATE).with_detail("reason", "ast_not_failed"))
        },
    }
}

fn source_reader_predicate_target_types(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    predicate: &SourceReaderPredicate,
    type_cache: &mut SourceReaderPredicateTypeCache,
) -> Result<Vec<(usize, DataType)>> {
    let mut types = Vec::new();
    collect_source_reader_predicate_target_types(source, chunk, predicate, &mut types, type_cache)?;
    Ok(types)
}

fn collect_source_reader_predicate_target_types(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    predicate: &SourceReaderPredicate,
    types: &mut Vec<(usize, DataType)>,
    type_cache: &mut SourceReaderPredicateTypeCache,
) -> Result<()> {
    match predicate {
        SourceReaderPredicate::Comparison { column, .. } | SourceReaderPredicate::InList { column, .. } => {
            types.push(resolve_source_reader_column(source, chunk, column, type_cache)?);
        },
        SourceReaderPredicate::And(left, right) | SourceReaderPredicate::Or(left, right) => {
            collect_source_reader_predicate_target_types(source, chunk, left, types, type_cache)?;
            collect_source_reader_predicate_target_types(source, chunk, right, types, type_cache)?;
        },
        SourceReaderPredicate::Not(input) => {
            collect_source_reader_predicate_target_types(source, chunk, input, types, type_cache)?;
        },
    }
    Ok(())
}

fn source_reader_column_ast(read_index: usize) -> Result<AstExpression> {
    let index = i32::try_from(read_index).map_err(|_| {
        Error::new_with(
            ErrorCode::IndexOutOfBounds,
            OP_SOURCE_READER_PREDICATE,
            "source reader predicate read column index exceeds i32 range",
        )
        .with_fact(FACT_COLUMN_INDEX, read_index)
        .with_detail("reason", "read_column_index_exceeds_i32")
    })?;
    AstExpression::left_column(index)
        .map_err(|error| error.reanchor(OP_SOURCE_READER_PREDICATE).with_detail("reason", "ast_column_failed"))
}

fn source_reader_literal_ast(
    runtime: &Runtime,
    literal: &SourceReaderPredicateLiteral,
    target_type: DataType,
) -> Result<AstExpression> {
    let scalar = source_reader_literal_scalar(runtime, literal, target_type)?;
    AstExpression::literal(scalar)
        .map_err(|error| error.reanchor(OP_SOURCE_READER_PREDICATE).with_detail("reason", "ast_literal_failed"))
}

fn source_reader_literal_scalar(
    runtime: &Runtime,
    literal: &SourceReaderPredicateLiteral,
    target_type: DataType,
) -> Result<crate::core::Scalar> {
    match target_type.id() {
        TypeId::Bool8 => match literal {
            SourceReaderPredicateLiteral::Boolean(value) => runtime.scalar(*value),
            _ => source_reader_literal_type_mismatch(literal, target_type),
        },
        TypeId::Int64 => match literal {
            SourceReaderPredicateLiteral::Int64(value) => runtime.scalar(*value),
            _ => source_reader_literal_type_mismatch(literal, target_type),
        },
        TypeId::UInt64 => match literal {
            SourceReaderPredicateLiteral::UInt64(value) => runtime.scalar(*value),
            _ => source_reader_literal_type_mismatch(literal, target_type),
        },
        TypeId::Float64 => match literal {
            SourceReaderPredicateLiteral::Float64(value) => runtime.scalar(*value),
            _ => source_reader_literal_type_mismatch(literal, target_type),
        },
        TypeId::String => match literal {
            SourceReaderPredicateLiteral::String(value) => runtime.string_scalar(value),
            _ => source_reader_literal_type_mismatch(literal, target_type),
        },
        TypeId::TimestampDays => match literal {
            SourceReaderPredicateLiteral::Date32(value) => {
                runtime.timestamp_raw_scalar(i64::from(*value), TimestampResolution::Days)
            },
            _ => source_reader_literal_type_mismatch(literal, target_type),
        },
        TypeId::Decimal32 | TypeId::Decimal64 | TypeId::Decimal128 => {
            let raw = source_reader_decimal_literal_raw(literal, target_type.scale())?;
            runtime.decimal_raw_scalar(raw, target_type)
        },
        other => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            OP_SOURCE_READER_PREDICATE,
            "source reader predicate literal target dtype is not supported",
        )
        .with_fact(FACT_TYPE_ID, other)
        .with_detail("reason", "unsupported_literal_target_dtype")),
    }
    .map_err(|error| {
        error
            .reanchor(OP_SOURCE_READER_PREDICATE)
            .with_detail_if_absent("target_type_id", target_type.id().to_string())
            .with_detail_if_absent("target_type_scale", target_type.scale().to_string())
    })
}

fn source_reader_literal_type_mismatch<T>(literal: &SourceReaderPredicateLiteral, target_type: DataType) -> Result<T> {
    Err(Error::new_with(
        ErrorCode::TypeMismatch,
        OP_SOURCE_READER_PREDICATE,
        "source reader predicate literal type does not match target column dtype",
    )
    .with_detail("reason", "literal_type_mismatch")
    .with_detail("literal", format!("{literal:?}"))
    .with_detail("target_type_id", target_type.id().to_string())
    .with_detail("target_type_scale", target_type.scale().to_string()))
}

fn source_reader_decimal_literal_raw(literal: &SourceReaderPredicateLiteral, target_scale: i32) -> Result<i128> {
    match literal {
        SourceReaderPredicateLiteral::Int64(value) => decimal_raw_for_integer_value(i128::from(*value), target_scale),
        SourceReaderPredicateLiteral::UInt64(value) => decimal_raw_for_integer_value(i128::from(*value), target_scale),
        SourceReaderPredicateLiteral::Decimal128 { raw, scale } => rescale_decimal_raw(*raw, *scale, target_scale),
        _ => source_reader_literal_type_mismatch(literal, DataType::fixed_point(TypeId::Decimal128, target_scale)?),
    }
}

fn resolve_source_reader_column(
    source: &LocalParquetSource,
    chunk: &SourceChunk,
    column: &SourceReaderPredicateColumn,
    type_cache: &mut SourceReaderPredicateTypeCache,
) -> Result<(usize, DataType)> {
    let read_index = match column.source {
        SourceColumnRef::ReadColumn(index) => index,
        _ => {
            return Err(Error::new_with(
                ErrorCode::UnsupportedOperation,
                OP_SOURCE_READER_PREDICATE,
                "source reader predicates can only reference decoded read columns",
            )
            .with_column(column.name.clone())
            .with_detail("reason", "predicate_column_not_read_column"));
        },
    };
    let read_column = source.read_columns.get(read_index).ok_or_else(|| {
        Error::new_with(
            ErrorCode::MissingColumn,
            OP_SOURCE_READER_PREDICATE,
            "source reader predicate column is missing from read projection",
        )
        .with_column(column.name.clone())
        .with_fact(FACT_COLUMN_INDEX, read_index)
        .with_detail("reason", "read_projection_column_missing")
    })?;
    let schema_column = source.schema.column(column.name.as_str()).ok_or_else(|| {
        schema_missing_error(OP_SOURCE_READER_PREDICATE, column.name.as_str())
            .with_detail("reason", "predicate_schema_column_missing")
    })?;
    let logical_type = schema_column.data_type();
    let target_type = if logical_type.id().is_decimal() {
        let reader_type = if let Some(reader_type) = chunk.reader_decimal_type(read_column) {
            reader_type
        } else if matches!(chunk.location(), ParquetSourceLocation::ObjectStoreUri(_)) {
            return Err(Error::new_with(
                ErrorCode::UnsupportedOperation,
                OP_SOURCE_READER_PREDICATE,
                "object-store decimal reader predicates require Parquet footer reader type metadata",
            )
            .with_column(column.name.clone())
            .with_detail("reason", "reader_decimal_type_metadata_unavailable")
            .with_detail("read_column", read_column.clone())
            .with_detail("read_index", read_index.to_string()));
        } else {
            type_cache.decimal_dtype(chunk.path(), read_column, read_index)?
        };
        if logical_type.scale() != reader_type.scale() {
            return Err(Error::new_with(
                ErrorCode::TypeMismatch,
                OP_SOURCE_READER_PREDICATE,
                "source reader predicate decimal scale does not match Parquet footer scale",
            )
            .with_column(column.name.clone())
            .with_detail("reason", "decimal_scale_mismatch")
            .with_detail("logical_scale", logical_type.scale().to_string())
            .with_detail("reader_scale", reader_type.scale().to_string())
            .with_detail("read_column", read_column.clone()));
        }
        reader_type
    } else {
        logical_type
    };
    Ok((read_index, target_type))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SourceReaderPredicateTypeCacheKey {
    path: PathBuf,
    read_column: String,
    read_index: usize,
}

#[derive(Debug, Default)]
struct SourceReaderPredicateTypeCache {
    decimal_types: HashMap<SourceReaderPredicateTypeCacheKey, DataType>,
}

impl SourceReaderPredicateTypeCache {
    fn decimal_dtype(&mut self, path: &Path, read_column: &str, read_index: usize) -> Result<DataType> {
        let key = SourceReaderPredicateTypeCacheKey {
            path: path.to_path_buf(),
            read_column: read_column.to_string(),
            read_index,
        };
        if let Some(dtype) = self.decimal_types.get(&key) {
            return Ok(*dtype);
        }
        let dtype = parquet_reader_decimal_dtype(path, read_column, read_index)?;
        self.decimal_types.insert(key, dtype);
        Ok(dtype)
    }
}

fn parquet_reader_decimal_dtype(path: &Path, read_column: &str, read_index: usize) -> Result<DataType> {
    let file = File::open(path).map_err(|source| {
        Error::new_with(
            ErrorCode::IoFailure,
            OP_SOURCE_READER_PREDICATE,
            "failed to open Parquet file for reader predicate footer typing",
        )
        .with_detail("reason", "parquet_footer_open_failed")
        .with_detail("path", path.display().to_string())
        .with_source(source)
    })?;
    let reader = SerializedFileReader::new(file).map_err(|source| {
        Error::new_with(
            ErrorCode::IoFailure,
            OP_SOURCE_READER_PREDICATE,
            "failed to read Parquet footer for reader predicate typing",
        )
        .with_detail("reason", "parquet_footer_read_failed")
        .with_detail("path", path.display().to_string())
        .with_source(source)
    })?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let descriptor = schema
        .columns()
        .get(read_index)
        .filter(|descriptor| parquet_descriptor_matches_column(descriptor, read_column))
        .or_else(|| {
            schema.columns().iter().find(|descriptor| parquet_descriptor_matches_column(descriptor, read_column))
        })
        .ok_or_else(|| {
            Error::new_with(
                ErrorCode::MissingColumn,
                OP_SOURCE_READER_PREDICATE,
                "source reader predicate column was not found in Parquet footer",
            )
            .with_column(read_column.to_string())
            .with_fact(FACT_COLUMN_INDEX, read_index)
            .with_detail("reason", "parquet_footer_column_missing")
            .with_detail("path", path.display().to_string())
        })?;
    let (precision, scale) = parquet_decimal_precision_scale(descriptor).ok_or_else(|| {
        Error::new_with(
            ErrorCode::TypeMismatch,
            OP_SOURCE_READER_PREDICATE,
            "source schema declares a decimal predicate column but Parquet footer column is not decimal",
        )
        .with_column(read_column.to_string())
        .with_fact(FACT_COLUMN_INDEX, read_index)
        .with_detail("reason", "parquet_footer_column_not_decimal")
        .with_detail("path", path.display().to_string())
        .with_detail("physical_type", format!("{:?}", descriptor.physical_type()))
    })?;
    let type_id = parquet_reader_decimal_type_id(descriptor, precision)?;
    DataType::fixed_point(type_id, -scale)
}

fn parquet_descriptor_matches_column(descriptor: &ColumnDescriptor, column_name: &str) -> bool {
    // `column_name` is a top-level schema column, so it may only match a leaf
    // whose full dotted path is exactly that name (depth 1). Matching on leaf
    // name alone would let a nested leaf like `s.id` shadow top-level `id`.
    let parts = descriptor.path().parts();
    parts.len() == 1 && parts[0] == column_name
}

fn parquet_decimal_precision_scale(descriptor: &ColumnDescriptor) -> Option<(i32, i32)> {
    if let Some(LogicalType::Decimal(decimal)) = descriptor.logical_type_ref() {
        return Some((decimal.precision, decimal.scale));
    }
    (descriptor.converted_type() == ConvertedType::DECIMAL)
        .then(|| (descriptor.type_precision(), descriptor.type_scale()))
}

fn parquet_reader_decimal_type_id(descriptor: &ColumnDescriptor, precision: i32) -> Result<TypeId> {
    match descriptor.physical_type() {
        ParquetPhysicalType::INT32 => Ok(TypeId::Decimal32),
        ParquetPhysicalType::INT64 => Ok(TypeId::Decimal64),
        ParquetPhysicalType::FIXED_LEN_BYTE_ARRAY => match descriptor.type_length() {
            0..=4 => Ok(TypeId::Decimal32),
            5..=8 => Ok(TypeId::Decimal64),
            9..=16 => Ok(TypeId::Decimal128),
            type_length => Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                OP_SOURCE_READER_PREDICATE,
                "Parquet decimal fixed byte width exceeds Decimal128 storage",
            )
            .with_detail("reason", "decimal_fixed_len_byte_width_unsupported")
            .with_detail("type_length", type_length.to_string())),
        },
        ParquetPhysicalType::BYTE_ARRAY => match precision {
            1..=9 => Ok(TypeId::Decimal32),
            10..=18 => Ok(TypeId::Decimal64),
            19..=38 => Ok(TypeId::Decimal128),
            precision => Err(Error::new_with(
                ErrorCode::UnsupportedDtype,
                OP_SOURCE_READER_PREDICATE,
                "Parquet decimal precision exceeds Decimal128 storage",
            )
            .with_detail("reason", "decimal_precision_unsupported")
            .with_detail("precision", precision.to_string())),
        },
        physical_type => Err(Error::new_with(
            ErrorCode::UnsupportedDtype,
            OP_SOURCE_READER_PREDICATE,
            "Parquet decimal physical type is not supported by reader predicates",
        )
        .with_detail("reason", "unsupported_decimal_physical_type")
        .with_detail("physical_type", format!("{physical_type:?}"))),
    }
}

fn decimal_raw_for_integer_value(value: i128, target_scale: i32) -> Result<i128> {
    if target_scale <= 0 {
        let multiplier = checked_pow10(target_scale.unsigned_abs())?;
        value.checked_mul(multiplier).ok_or_else(|| {
            Error::new_with(
                ErrorCode::ValueOutOfRange,
                OP_SOURCE_READER_PREDICATE,
                "integer source reader predicate literal raw value overflows target decimal scale",
            )
            .with_detail("reason", "integer_decimal_literal_overflow")
            .with_detail("target_scale", target_scale.to_string())
        })
    } else {
        let divisor = checked_pow10(target_scale as u32)?;
        if value % divisor != 0 {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                OP_SOURCE_READER_PREDICATE,
                "integer source reader predicate literal cannot be represented exactly at target decimal scale",
            )
            .with_detail("reason", "integer_decimal_literal_not_exact")
            .with_detail("target_scale", target_scale.to_string()));
        }
        Ok(value / divisor)
    }
}

fn rescale_decimal_raw(raw: i128, source_scale: i32, target_scale: i32) -> Result<i128> {
    let scale_delta = source_scale.checked_sub(target_scale).ok_or_else(|| {
        Error::new_with(
            ErrorCode::ValueOutOfRange,
            OP_SOURCE_READER_PREDICATE,
            "decimal source reader predicate literal scale alignment overflowed",
        )
        .with_detail("reason", "decimal_scale_alignment_overflow")
        .with_detail("source_scale", source_scale.to_string())
        .with_detail("target_scale", target_scale.to_string())
    })?;
    if scale_delta >= 0 {
        let multiplier = checked_pow10(scale_delta as u32)?;
        raw.checked_mul(multiplier).ok_or_else(|| {
            Error::new_with(
                ErrorCode::ValueOutOfRange,
                OP_SOURCE_READER_PREDICATE,
                "decimal source reader predicate literal raw value overflows target scale",
            )
            .with_detail("reason", "decimal_literal_rescale_overflow")
            .with_detail("source_scale", source_scale.to_string())
            .with_detail("target_scale", target_scale.to_string())
        })
    } else {
        let divisor = checked_pow10(scale_delta.unsigned_abs())?;
        if raw % divisor != 0 {
            return Err(Error::new_with(
                ErrorCode::ValueOutOfRange,
                OP_SOURCE_READER_PREDICATE,
                "decimal source reader predicate literal cannot be represented exactly at target scale",
            )
            .with_detail("reason", "decimal_literal_rescale_not_exact")
            .with_detail("source_scale", source_scale.to_string())
            .with_detail("target_scale", target_scale.to_string()));
        }
        Ok(raw / divisor)
    }
}

fn checked_pow10(exponent: u32) -> Result<i128> {
    let mut value = 1i128;
    for _ in 0..exponent {
        value = value.checked_mul(10).ok_or_else(|| {
            Error::new_with(
                ErrorCode::ValueOutOfRange,
                OP_SOURCE_READER_PREDICATE,
                "decimal source reader predicate scale factor overflows i128",
            )
            .with_detail("reason", "decimal_scale_factor_overflow")
            .with_detail("exponent", exponent.to_string())
        })?;
    }
    Ok(value)
}
