use std::{collections::HashSet, fmt};

use arrow::datatypes::SchemaRef;

use crate::{Error, ErrorCode, Operation, Result};

/// A catalog-independent Iceberg table identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TableIdentifier {
    pub namespace: Vec<String>,
    pub name: String,
}

impl TableIdentifier {
    pub fn new(namespace: Vec<String>, name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        if namespace.is_empty() || namespace.iter().any(|part| part.trim().is_empty()) || name.trim().is_empty() {
            return Err(invalid_candidate("table identifier must contain a non-empty namespace and name"));
        }
        Ok(Self { namespace, name })
    }

    pub fn display_name(&self) -> String {
        format!("{}.{}", self.namespace.join("."), self.name)
    }
}

/// Iceberg snapshot facts frozen with a resolved table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSnapshot {
    pub snapshot_id: Option<i64>,
    pub sequence_number: Option<i64>,
    pub schema_id: i32,
}

/// Canonical Arrow schema plus Iceberg field identities.
///
/// The Arrow schema is canonical for both a future CPU delegate and native
/// conversion. Iceberg field ids, roles, and defaults remain explicit because
/// Arrow alone cannot represent Iceberg schema evolution semantics.
#[derive(Clone)]
pub struct Schema {
    pub schema_id: i32,
    pub arrow_schema: SchemaRef,
    pub fields: Vec<SchemaField>,
}

impl Schema {
    pub fn new(schema_id: i32, arrow_schema: SchemaRef, fields: Vec<SchemaField>) -> Result<Self> {
        if arrow_schema.fields().len() != fields.len() {
            return Err(invalid_schema("canonical Arrow schema and Iceberg top-level fields have different lengths"));
        }
        if arrow_schema.fields().iter().zip(&fields).any(|(arrow, iceberg)| arrow.name() != &iceberg.name) {
            return Err(invalid_schema("canonical Arrow schema field names do not match Iceberg schema facts"));
        }
        let mut field_ids = HashSet::new();
        if fields.iter().any(|field| !field.collect_unique_ids(&mut field_ids)) {
            return Err(invalid_schema("Iceberg schema contains duplicate field ids"));
        }
        Ok(Self { schema_id, arrow_schema, fields })
    }

    pub fn contains_field_id(&self, field_id: i32) -> bool {
        self.fields.iter().any(|field| field.contains_field_id(field_id))
    }
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Schema")
            .field("schema_id", &self.schema_id)
            .field("arrow_schema", &self.arrow_schema)
            .field("fields", &self.fields)
            .finish()
    }
}

/// Iceberg schema field facts, including nested children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaField {
    pub field_id: i32,
    pub name: String,
    pub required: bool,
    pub iceberg_type: String,
    /// The initial default used for rows written before this field existed.
    /// Complex or otherwise unrepresentable defaults stay explicit so the
    /// root can decline only native execution while retaining CPU semantics.
    pub initial_default: Option<DefaultValue>,
    pub children: Vec<SchemaField>,
}

impl SchemaField {
    pub fn new(field_id: i32, name: impl Into<String>, required: bool, iceberg_type: impl Into<String>) -> Self {
        Self {
            field_id,
            name: name.into(),
            required,
            iceberg_type: iceberg_type.into(),
            initial_default: None,
            children: Vec::new(),
        }
    }

    pub fn with_initial_default(mut self, initial_default: DefaultValue) -> Self {
        self.initial_default = Some(initial_default);
        self
    }

    pub fn with_children(mut self, children: Vec<SchemaField>) -> Self {
        self.children = children;
        self
    }

    fn contains_field_id(&self, field_id: i32) -> bool {
        self.field_id == field_id || self.children.iter().any(|child| child.contains_field_id(field_id))
    }

    fn collect_unique_ids(&self, ids: &mut HashSet<i32>) -> bool {
        ids.insert(self.field_id) && self.children.iter().all(|child| child.collect_unique_ids(ids))
    }
}

/// Resolved, credential-free table facts copied into a scan candidate.
#[derive(Debug, Clone)]
pub struct ResolvedTableFacts {
    pub table: TableIdentifier,
    pub snapshot: ResolvedSnapshot,
    pub schema: Schema,
}

/// Request intent passed from a future adapter expression translator into the
/// leaf planner. It contains no DataFusion expression or execution-plan type.
#[derive(Debug, Clone, PartialEq)]
pub struct ScanRequest {
    pub projection: Projection,
    pub filter_bindings: Vec<ColumnBinding>,
    pub predicate: PredicateIntent,
    /// Iceberg-native predicate passed directly to the Iceberg task planner.
    ///
    /// The DataFusion adapter translates its own expression model into this value;
    /// the leaf never accepts a DataFusion expression. When it is present,
    /// the planner retains consumer-side residual intent because manifest
    /// pruning is not row-level predicate enforcement.
    #[cfg(feature = "catalog")]
    pub iceberg_predicate: Option<iceberg::expr::Predicate>,
    pub source_limit: SourceLimit,
    pub pruning: PruningIntent,
}

impl Default for ScanRequest {
    fn default() -> Self {
        Self {
            projection: Projection::AllColumns,
            filter_bindings: Vec::new(),
            predicate: PredicateIntent::none(),
            #[cfg(feature = "catalog")]
            iceberg_predicate: None,
            source_limit: SourceLimit::default(),
            pruning: PruningIntent::default(),
        }
    }
}

/// Projection intent and field bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Projection {
    AllColumns,
    Columns(Vec<ColumnBinding>),
}

/// Bound Iceberg column identity and how a consumer obtains its value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnBinding {
    pub field_id: i32,
    pub output_name: String,
    pub source: ColumnSource,
    pub role: ColumnRole,
    pub missing_default: Option<DefaultValue>,
}

/// Source of a projected or filter-only column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnSource {
    PhysicalFile,
    PartitionConstant,
    MetadataColumn,
    MissingWithDefault,
}

/// The reason a bound column is retained by the scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnRole {
    Projected,
    FilterOnly,
    ProjectedAndFilter,
}

impl ColumnRole {
    fn includes_filter(self) -> bool {
        matches!(self, Self::FilterOnly | Self::ProjectedAndFilter)
    }
}

/// Literal values used by partition constants and schema-evolution defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarValue {
    Null,
    Boolean(bool),
    Int(i32),
    Long(i64),
    /// IEEE-754 bits preserve NaN payloads and signed zero while remaining
    /// comparable in the neutral contract.
    Float {
        bits: u32,
    },
    /// IEEE-754 bits preserve NaN payloads and signed zero while remaining
    /// comparable in the neutral contract.
    Double {
        bits: u64,
    },
    Decimal {
        unscaled: i128,
        scale: i32,
    },
    String(String),
    Date(i32),
    Time(i64),
    TimestampMicros(i64),
    TimestampNanos(i64),
    Uuid([u8; 16]),
    Binary(Vec<u8>),
}

/// Schema-evolution default retained in a neutral scan request.
///
/// `Unsupported` represents an Iceberg literal that this neutral contract
/// deliberately does not materialize. It is still a valid CPU Iceberg table;
/// the root must reject only native admission for that candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefaultValue {
    Scalar(ScalarValue),
    Unsupported,
}

/// Whether the source enforced a predicate and what must remain for the
/// eventual consumer to evaluate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateIntent {
    pub enforcement: PredicateEnforcement,
    pub residual: Option<ResidualPredicate>,
}

impl PredicateIntent {
    pub const fn none() -> Self {
        Self { enforcement: PredicateEnforcement::NotPresent, residual: None }
    }
}

/// Predicate enforcement disposition known from Iceberg task planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredicateEnforcement {
    NotPresent,
    FullyEnforced,
    PartiallyEnforced,
    ConsumerMustEnforce,
}

/// Opaque canonical predicate identity for a residual that must be evaluated
/// by the eventual CPU/native consumer. The adapter owns expression decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidualPredicate {
    pub canonical: String,
}

/// Source-side limit intent. This is planning evidence, not native admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SourceLimit {
    pub row_limit: Option<u64>,
    pub file_limit: Option<usize>,
}

/// Pruning already applied by Iceberg planning. This preserves source semantics
/// without declaring any native capability or cache/read strategy.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PruningIntent {
    pub partition_pruning_applied: bool,
    pub manifest_pruning_applied: bool,
    pub file_pruning_applied: bool,
    pub remaining_partition_predicate: Option<String>,
}

/// A fully neutral, lossless Iceberg scan candidate.
///
/// It excludes `TableFormat*`, query-engine/native types, CPU/GPU source
/// strategy, cache policy, and credentials. The DataFusion adapter later converts it
/// to its own scan plan and applies those independent decisions.
#[derive(Debug, Clone)]
pub struct ScanCandidate {
    pub resolved_table: ResolvedTableFacts,
    pub request: ScanRequest,
    pub data_files: Vec<DataFile>,
    pub name_mapping: Option<NameMapping>,
}

impl ScanCandidate {
    pub fn try_new(
        resolved_table: ResolvedTableFacts,
        request: ScanRequest,
        data_files: Vec<DataFile>,
        name_mapping: Option<NameMapping>,
    ) -> Result<Self> {
        validate_resolved_table(&resolved_table)?;
        validate_request(&resolved_table.schema, &request)?;
        for file in &data_files {
            validate_data_file(&resolved_table.schema, file)?;
        }
        if let Some(mapping) = &name_mapping {
            mapping.validate()?;
        }
        Ok(Self { resolved_table, request, data_files, name_mapping })
    }

    /// Number of data files selected after Iceberg pruning.
    pub fn planned_data_file_count(&self) -> usize {
        self.data_files.len()
    }

    /// Number of associated delete files selected after Iceberg pruning.
    pub fn planned_delete_file_count(&self) -> usize {
        self.data_files.iter().map(|file| file.delete_files.len()).sum()
    }
}

/// A selected data-file range and all Iceberg semantic associations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataFile {
    pub location: String,
    pub format: FileFormat,
    pub file_size_bytes: u64,
    pub range: FileRange,
    pub record_count: Option<u64>,
    pub sequence: FileSequence,
    /// Safe marker for encrypted file metadata. Raw key metadata is never
    /// retained because it can contain plaintext decryption material.
    pub encryption_key_metadata_present: bool,
    pub partition_values: Vec<PartitionValue>,
    pub delete_files: Vec<DeleteFile>,
}

/// Data-file or delete-file format as defined by Iceberg.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Avro,
    Orc,
    Parquet,
    Puffin,
}

/// The byte range selected from a data file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileRange {
    pub start: u64,
    pub length: u64,
}

/// File-level sequence values necessary for delete semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FileSequence {
    pub data_sequence_number: Option<i64>,
    pub file_sequence_number: Option<i64>,
}

/// A delete file associated with a selected data file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeleteFile {
    pub location: String,
    pub format: FileFormat,
    pub content: DeleteContent,
    pub file_size_bytes: u64,
    pub sequence: FileSequence,
    /// Safe marker for encrypted file metadata. Raw key metadata is never
    /// retained because it can contain plaintext decryption material.
    pub encryption_key_metadata_present: bool,
    pub referenced_data_file: Option<String>,
    pub equality_field_ids: Vec<i32>,
    pub partition_spec_id: i32,
}

/// Delete content semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteContent {
    Position,
    Equality,
}

/// A partition value and the transform that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionValue {
    pub source_field_id: i32,
    pub partition_field_id: i32,
    pub transform: PartitionTransform,
    pub value: ScalarValue,
}

/// Iceberg partition transforms relevant to interpreting a partition value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionTransform {
    Identity,
    Bucket(u32),
    Truncate(u32),
    Year,
    Month,
    Day,
    Hour,
    Void,
    Unknown,
}

/// Iceberg's field-name to id fallback mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameMapping {
    pub fields: Vec<NameMappingField>,
}

/// One name-mapping field and its nested children.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameMappingField {
    pub field_id: Option<i32>,
    pub names: Vec<String>,
    pub fields: Vec<NameMappingField>,
}

impl NameMapping {
    fn validate(&self) -> Result<()> {
        if self.fields.iter().any(|field| !field.is_valid()) {
            return Err(invalid_candidate("Iceberg name mapping contains an empty name"));
        }
        Ok(())
    }
}

impl NameMappingField {
    fn is_valid(&self) -> bool {
        self.names.iter().all(|name| !name.trim().is_empty()) && self.fields.iter().all(Self::is_valid)
    }
}

fn validate_resolved_table(resolved: &ResolvedTableFacts) -> Result<()> {
    if resolved.snapshot.schema_id != resolved.schema.schema_id {
        return Err(invalid_schema("resolved snapshot schema id does not match canonical Iceberg schema"));
    }
    Ok(())
}

fn validate_request(schema: &Schema, request: &ScanRequest) -> Result<()> {
    let projection = match &request.projection {
        Projection::AllColumns => &[][..],
        Projection::Columns(bindings) => bindings,
    };
    validate_bindings(schema, projection, false)?;
    validate_bindings(schema, &request.filter_bindings, true)?;

    match (&request.predicate.enforcement, &request.predicate.residual) {
        (PredicateEnforcement::NotPresent | PredicateEnforcement::FullyEnforced, None)
        | (PredicateEnforcement::PartiallyEnforced | PredicateEnforcement::ConsumerMustEnforce, Some(_)) => {},
        _ => return Err(invalid_candidate("predicate enforcement and residual intent disagree")),
    }
    if request.source_limit.row_limit == Some(0) || request.source_limit.file_limit == Some(0) {
        return Err(invalid_candidate("source limit must be positive when present"));
    }
    Ok(())
}

fn validate_bindings(schema: &Schema, bindings: &[ColumnBinding], filter_bindings: bool) -> Result<()> {
    let mut ids = HashSet::new();
    for binding in bindings {
        if binding.output_name.trim().is_empty() || !schema.contains_field_id(binding.field_id) {
            return Err(invalid_schema("column binding references an absent schema field or has an empty output name"));
        }
        if !ids.insert(binding.field_id) {
            return Err(invalid_candidate("scan request repeats an Iceberg field binding"));
        }
        if filter_bindings && !binding.role.includes_filter() {
            return Err(invalid_candidate("filter bindings must carry a filter column role"));
        }
        if binding.source == ColumnSource::MissingWithDefault && binding.missing_default.is_none() {
            return Err(invalid_candidate("missing-default column binding must carry its Iceberg default"));
        }
    }
    Ok(())
}

fn validate_data_file(schema: &Schema, file: &DataFile) -> Result<()> {
    validate_location(&file.location)?;
    let Some(end) = file.range.start.checked_add(file.range.length) else {
        return Err(invalid_candidate("data-file range overflows"));
    };
    if end > file.file_size_bytes {
        return Err(invalid_candidate("data-file range exceeds file size"));
    }
    if file.partition_values.iter().any(|partition| !schema.contains_field_id(partition.source_field_id)) {
        return Err(invalid_schema("partition value references a source field absent from the frozen schema"));
    }
    for delete in &file.delete_files {
        validate_location(&delete.location)?;
        if delete.content == DeleteContent::Position
            && delete.referenced_data_file.as_deref().is_some_and(|referenced| referenced != file.location)
        {
            return Err(invalid_candidate("position delete must reference its selected data file"));
        }
        if delete.content == DeleteContent::Equality && delete.equality_field_ids.is_empty() {
            return Err(invalid_candidate("equality delete must carry equality field ids"));
        }
        if delete.equality_field_ids.iter().any(|field_id| !schema.contains_field_id(*field_id)) {
            return Err(invalid_schema("equality delete references a field absent from the frozen schema"));
        }
    }
    Ok(())
}

fn validate_location(location: &str) -> Result<()> {
    let lower = location.to_ascii_lowercase();
    let has_userinfo = lower.find("://").is_some_and(|scheme| {
        lower[scheme + 3..]
            .find(['/', '?', '#'])
            .is_none_or(|authority_end| lower[scheme + 3..scheme + 3 + authority_end].contains('@'))
    });
    if location.trim().is_empty()
        || has_userinfo
        || lower.contains("x-amz-signature=")
        || lower.contains("access_key=")
        || lower.contains("token=")
    {
        return Err(invalid_candidate("file locations must be canonical and credential-free"));
    }
    Ok(())
}

fn invalid_candidate(message: &'static str) -> Error {
    Error::new(ErrorCode::InvalidMetadata, Operation::ScanCandidateBuild, message)
}

fn invalid_schema(message: &'static str) -> Error {
    Error::new(ErrorCode::UnsupportedSchema, Operation::SchemaInterpretation, message)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    use super::*;

    fn resolved_table() -> ResolvedTableFacts {
        let schema = Schema::new(
            7,
            Arc::new(ArrowSchema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("region", DataType::Utf8, true),
            ])),
            vec![SchemaField::new(1, "id", true, "long"), SchemaField::new(2, "region", false, "string")],
        )
        .unwrap();
        ResolvedTableFacts {
            table: TableIdentifier::new(vec!["analytics".to_string()], "events").unwrap(),
            snapshot: ResolvedSnapshot { snapshot_id: Some(42), sequence_number: Some(11), schema_id: 7 },
            schema,
        }
    }

    #[test]
    fn candidate_preserves_projection_predicate_file_delete_partition_and_name_mapping_facts() {
        let request = ScanRequest {
            projection: Projection::Columns(vec![ColumnBinding {
                field_id: 1,
                output_name: "id".to_string(),
                source: ColumnSource::PhysicalFile,
                role: ColumnRole::Projected,
                missing_default: None,
            }]),
            filter_bindings: vec![ColumnBinding {
                field_id: 2,
                output_name: "region".to_string(),
                source: ColumnSource::PartitionConstant,
                role: ColumnRole::FilterOnly,
                missing_default: None,
            }],
            predicate: PredicateIntent {
                enforcement: PredicateEnforcement::PartiallyEnforced,
                residual: Some(ResidualPredicate { canonical: "field(1) > 10".to_string() }),
            },
            #[cfg(feature = "catalog")]
            iceberg_predicate: None,
            source_limit: SourceLimit { row_limit: Some(100), file_limit: Some(3) },
            pruning: PruningIntent {
                partition_pruning_applied: true,
                manifest_pruning_applied: true,
                file_pruning_applied: false,
                remaining_partition_predicate: Some("region = 'us'".to_string()),
            },
        };
        let data_location = "s3://warehouse/events/data-1.parquet".to_string();
        let candidate = ScanCandidate::try_new(
            resolved_table(),
            request,
            vec![DataFile {
                location: data_location.clone(),
                format: FileFormat::Parquet,
                file_size_bytes: 4096,
                range: FileRange { start: 128, length: 2048 },
                record_count: Some(50),
                sequence: FileSequence { data_sequence_number: Some(9), file_sequence_number: Some(11) },
                encryption_key_metadata_present: false,
                partition_values: vec![PartitionValue {
                    source_field_id: 2,
                    partition_field_id: 1002,
                    transform: PartitionTransform::Identity,
                    value: ScalarValue::String("us".to_string()),
                }],
                delete_files: vec![
                    DeleteFile {
                        location: "s3://warehouse/events/delete-pos.parquet".to_string(),
                        format: FileFormat::Parquet,
                        content: DeleteContent::Position,
                        file_size_bytes: 512,
                        sequence: FileSequence { data_sequence_number: Some(10), file_sequence_number: Some(12) },
                        encryption_key_metadata_present: false,
                        referenced_data_file: Some(data_location),
                        equality_field_ids: Vec::new(),
                        partition_spec_id: 3,
                    },
                    DeleteFile {
                        location: "s3://warehouse/events/delete-eq.parquet".to_string(),
                        format: FileFormat::Parquet,
                        content: DeleteContent::Equality,
                        file_size_bytes: 256,
                        sequence: FileSequence { data_sequence_number: Some(10), file_sequence_number: Some(13) },
                        encryption_key_metadata_present: false,
                        referenced_data_file: None,
                        equality_field_ids: vec![1],
                        partition_spec_id: 3,
                    },
                ],
            }],
            Some(NameMapping {
                fields: vec![NameMappingField {
                    field_id: Some(1),
                    names: vec!["id".to_string(), "event_id".to_string()],
                    fields: Vec::new(),
                }],
            }),
        )
        .unwrap();

        assert_eq!(candidate.resolved_table.snapshot.snapshot_id, Some(42));
        assert_eq!(candidate.resolved_table.schema.arrow_schema.field(0).name(), "id");
        assert_eq!(candidate.data_files[0].range, FileRange { start: 128, length: 2048 });
        assert_eq!(candidate.data_files[0].delete_files[1].equality_field_ids, vec![1]);
        assert_eq!(candidate.planned_delete_file_count(), 2);
        assert_eq!(candidate.name_mapping.as_ref().unwrap().fields[0].names[1], "event_id");
    }

    #[test]
    fn candidate_rejects_credential_bearing_locations_and_bad_delete_associations() {
        let data_file = DataFile {
            location: "s3://access:secret@warehouse/events/data.parquet".to_string(),
            format: FileFormat::Parquet,
            file_size_bytes: 10,
            range: FileRange { start: 0, length: 10 },
            record_count: None,
            sequence: FileSequence::default(),
            encryption_key_metadata_present: false,
            partition_values: Vec::new(),
            delete_files: Vec::new(),
        };
        let error =
            ScanCandidate::try_new(resolved_table(), ScanRequest::default(), vec![data_file], None).unwrap_err();
        assert_eq!(error.code(), ErrorCode::InvalidMetadata);
        assert_eq!(error.operation(), Operation::ScanCandidateBuild);
    }

    #[test]
    fn candidate_rejects_partition_and_equality_ids_absent_from_frozen_schema() {
        let valid_data_file = || DataFile {
            location: "file:///tmp/iceberg/data.parquet".to_string(),
            format: FileFormat::Parquet,
            file_size_bytes: 10,
            range: FileRange { start: 0, length: 10 },
            record_count: None,
            sequence: FileSequence::default(),
            encryption_key_metadata_present: false,
            partition_values: vec![PartitionValue {
                source_field_id: 2,
                partition_field_id: 1002,
                transform: PartitionTransform::Identity,
                value: ScalarValue::String("us".to_string()),
            }],
            delete_files: vec![DeleteFile {
                location: "file:///tmp/iceberg/delete.parquet".to_string(),
                format: FileFormat::Parquet,
                content: DeleteContent::Equality,
                file_size_bytes: 5,
                sequence: FileSequence::default(),
                encryption_key_metadata_present: false,
                referenced_data_file: None,
                equality_field_ids: vec![1],
                partition_spec_id: 0,
            }],
        };

        let mut unknown_partition_source = valid_data_file();
        unknown_partition_source.partition_values[0].source_field_id = 99;
        let partition_error =
            ScanCandidate::try_new(resolved_table(), ScanRequest::default(), vec![unknown_partition_source], None)
                .unwrap_err();
        assert_eq!(partition_error.code(), ErrorCode::UnsupportedSchema);
        assert_eq!(partition_error.operation(), Operation::SchemaInterpretation);

        let mut unknown_equality_id = valid_data_file();
        unknown_equality_id.delete_files[0].equality_field_ids = vec![99];
        let equality_error =
            ScanCandidate::try_new(resolved_table(), ScanRequest::default(), vec![unknown_equality_id], None)
                .unwrap_err();
        assert_eq!(equality_error.code(), ErrorCode::UnsupportedSchema);
        assert_eq!(equality_error.operation(), Operation::SchemaInterpretation);
    }
}
