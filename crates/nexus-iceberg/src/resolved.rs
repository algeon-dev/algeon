use std::{fmt, sync::Arc};

use iceberg::{
    spec::{Literal, NestedField, PrimitiveLiteral, PrimitiveType, Type},
    table::Table,
};

use crate::{
    DefaultValue, Error, ErrorCode, ErrorContext, Operation, ResolvedSnapshot, ResolvedTableFacts, Result,
    S3FileIoConfigVerification, ScalarValue, ScanCandidate, ScanPlanningOptions, ScanRequest, Schema, SchemaField,
    SourceAccessPolicy, TableIdentifier, planner::ScanPlanner,
};

/// A frozen Iceberg table resolution.
///
/// The table handle, resolved snapshot/schema facts, and source-access policy
/// are captured together. Leaf scan planning and a later root CPU delegate
/// must consume this same value; the root derives native storage credentials
/// from the same policy rather than independently rediscovering a source.
#[derive(Clone)]
pub struct ResolvedTable {
    table: Table,
    facts: ResolvedTableFacts,
    source_access: SourceAccessPolicy,
    file_io_source_access_verification: S3FileIoConfigVerification,
}

impl ResolvedTable {
    /// Freezes a table's current snapshot/schema and its source-access policy.
    pub fn from_table(table: Table, source_access: SourceAccessPolicy) -> Result<Self> {
        let verification = source_access.verify_s3_file_io_properties(table.file_io().config().props());
        Self::from_table_with_file_io_verification(table, source_access, verification)
    }

    /// Freezes a table resolved through a leaf catalog that owns and enforces
    /// the source FileIO policy. The catalog adapter may carry separate
    /// metadata credentials in its raw FileIO properties, so raw-property
    /// comparison is not the effective source-access proof at this boundary.
    pub(crate) fn from_catalog_file_io(table: Table, source_access: SourceAccessPolicy) -> Result<Self> {
        Self::from_table_with_file_io_verification(table, source_access, S3FileIoConfigVerification::ExactPolicy)
    }

    fn from_table_with_file_io_verification(
        table: Table,
        source_access: SourceAccessPolicy,
        file_io_source_access_verification: S3FileIoConfigVerification,
    ) -> Result<Self> {
        let metadata = table.metadata();
        let current_snapshot = metadata.current_snapshot();
        let snapshot_id = current_snapshot.map(|snapshot| snapshot.snapshot_id());
        let sequence_number = current_snapshot.map(|snapshot| snapshot.sequence_number());
        let schema = match current_snapshot {
            Some(snapshot) => snapshot.schema(metadata).map_err(|source| {
                Error::from_iceberg_error(Operation::TableResolve, source)
                    .with_context(ErrorContext { snapshot_id, ..ErrorContext::default() })
            })?,
            None => table.current_schema_ref(),
        };
        let schema_facts = iceberg_schema_to_neutral(schema.as_ref())?;
        let table_identifier = TableIdentifier {
            namespace: table.identifier().namespace().as_ref().clone(),
            name: table.identifier().name().to_string(),
        };
        if table_identifier.namespace.is_empty() || table_identifier.name.trim().is_empty() {
            return Err(Error::new(
                ErrorCode::InvalidMetadata,
                Operation::TableResolve,
                "catalog returned a table with an invalid identifier",
            ));
        }

        let facts = ResolvedTableFacts {
            table: table_identifier,
            snapshot: ResolvedSnapshot { snapshot_id, sequence_number, schema_id: schema_facts.schema_id },
            schema: schema_facts,
        };
        Ok(Self { table, facts, source_access, file_io_source_access_verification })
    }

    /// Returns the frozen `iceberg-rust` table handle for CPU delegation or
    /// leaf manifest/task planning.
    pub const fn table(&self) -> &Table {
        &self.table
    }

    /// Returns credential-free snapshot/schema facts for candidate creation.
    pub const fn facts(&self) -> &ResolvedTableFacts {
        &self.facts
    }

    /// Returns the one source-access policy frozen with this table.
    pub const fn source_access_policy(&self) -> &SourceAccessPolicy {
        &self.source_access
    }

    /// Returns whether the frozen source-access policy is proven to match the
    /// table's configured FileIO S3 properties. This is intentionally only a
    /// safe verification state, never the properties themselves.
    pub const fn file_io_source_access_verification(&self) -> S3FileIoConfigVerification {
        self.file_io_source_access_verification
    }

    /// Plans a credential-free scan candidate against this table's frozen
    /// snapshot.
    pub async fn plan_scan(&self, request: ScanRequest, options: ScanPlanningOptions) -> Result<ScanCandidate> {
        ScanPlanner::new(self).plan(request, options).await
    }
}

impl fmt::Debug for ResolvedTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResolvedTable")
            .field("table", &"<redacted Iceberg table handle>")
            .field("facts", &self.facts)
            .field("source_access", &self.source_access)
            .field("file_io_source_access_verification", &self.file_io_source_access_verification)
            .finish()
    }
}

fn iceberg_schema_to_neutral(schema: &iceberg::spec::Schema) -> Result<Schema> {
    let arrow_schema = iceberg::arrow::schema_to_arrow_schema(schema).map_err(|error| {
        Error::new(ErrorCode::UnsupportedSchema, Operation::SchemaInterpretation, error.to_string())
            .with_context(ErrorContext { schema_id: Some(schema.schema_id()), ..ErrorContext::default() })
    })?;
    let fields = schema.as_struct().fields().iter().map(|field| neutral_field(field)).collect();
    Schema::new(schema.schema_id(), Arc::new(arrow_schema), fields)
}

fn neutral_field(field: &NestedField) -> SchemaField {
    let children = match field.field_type.as_ref() {
        Type::Struct(struct_type) => struct_type.fields().iter().map(|field| neutral_field(field)).collect(),
        Type::List(list_type) => vec![neutral_field(&list_type.element_field)],
        Type::Map(map_type) => vec![neutral_field(&map_type.key_field), neutral_field(&map_type.value_field)],
        Type::Primitive(_) | Type::Variant(_) => Vec::new(),
    };
    let mut neutral = SchemaField::new(field.id, field.name.clone(), field.required, field.field_type.to_string())
        .with_children(children);
    if let Some(default) = field.initial_default.as_ref() {
        neutral = neutral.with_initial_default(
            neutral_default(default, field.field_type.as_ref()).unwrap_or(DefaultValue::Unsupported),
        );
    }
    neutral
}

fn neutral_default(value: &Literal, field_type: &Type) -> Option<DefaultValue> {
    let primitive = value.as_primitive_literal()?;
    let Type::Primitive(field_type) = field_type else {
        return None;
    };
    let scalar = match (field_type, primitive) {
        (PrimitiveType::Boolean, PrimitiveLiteral::Boolean(value)) => ScalarValue::Boolean(value),
        (PrimitiveType::Int, PrimitiveLiteral::Int(value)) => ScalarValue::Int(value),
        (PrimitiveType::Long, PrimitiveLiteral::Long(value)) => ScalarValue::Long(value),
        (PrimitiveType::Float, PrimitiveLiteral::Float(value)) => {
            ScalarValue::Float { bits: value.into_inner().to_bits() }
        },
        (PrimitiveType::Double, PrimitiveLiteral::Double(value)) => {
            ScalarValue::Double { bits: value.into_inner().to_bits() }
        },
        (PrimitiveType::Decimal { scale, .. }, PrimitiveLiteral::Int128(unscaled)) => {
            ScalarValue::Decimal { unscaled, scale: i32::try_from(*scale).ok()? }
        },
        (PrimitiveType::Date, PrimitiveLiteral::Int(value)) => ScalarValue::Date(value),
        (PrimitiveType::Time, PrimitiveLiteral::Long(value)) => ScalarValue::Time(value),
        (PrimitiveType::Timestamp | PrimitiveType::Timestamptz, PrimitiveLiteral::Long(value)) => {
            ScalarValue::TimestampMicros(value)
        },
        (PrimitiveType::TimestampNs | PrimitiveType::TimestamptzNs, PrimitiveLiteral::Long(value)) => {
            ScalarValue::TimestampNanos(value)
        },
        (PrimitiveType::String, PrimitiveLiteral::String(value)) => ScalarValue::String(value),
        (PrimitiveType::Uuid, PrimitiveLiteral::UInt128(value)) => ScalarValue::Uuid(value.to_be_bytes()),
        (PrimitiveType::Fixed(_) | PrimitiveType::Binary, PrimitiveLiteral::Binary(value)) => {
            ScalarValue::Binary(value)
        },
        _ => return None,
    };
    Some(DefaultValue::Scalar(scalar))
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{Literal, NestedField, PrimitiveType, Type};

    use super::*;

    #[test]
    fn neutral_schema_retains_initial_defaults_for_old_files() {
        let field = NestedField::optional(17, "region", Type::Primitive(PrimitiveType::String))
            .with_initial_default(Literal::string("unknown"));

        let neutral = neutral_field(&field);

        assert_eq!(neutral.initial_default, Some(DefaultValue::Scalar(ScalarValue::String("unknown".to_string()))));
    }
}
