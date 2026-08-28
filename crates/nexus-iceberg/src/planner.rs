use std::{
    collections::{HashMap, HashSet},
    future::Future,
    time::Duration,
};

use futures::StreamExt;
use iceberg::{
    scan::FileScanTask,
    spec::{
        DataContentType, DataFileFormat, Literal, NameMapping as VendorNameMapping, PrimitiveLiteral, PrimitiveType,
        TableProperties, Transform,
    },
};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::{
    ColumnBinding, DataFile, DeleteContent, DeleteFile, Error, ErrorCode, ErrorContext, FileFormat, FileRange,
    FileSequence, NameMapping, NameMappingField, Operation, PartitionTransform, PartitionValue, PredicateEnforcement,
    ResidualPredicate, ResolvedTable, Result, ScalarValue, ScanCandidate, ScanRequest,
};

/// Execution controls for one Iceberg metadata-planning operation.
///
/// A snapshot override is permitted only when it matches the snapshot frozen
/// by [`ResolvedTable`]. This prevents a catalog refresh or a caller
/// mistake from silently changing the snapshot behind the resolved table
/// facts.
#[derive(Debug, Clone)]
pub struct ScanPlanningOptions {
    /// Optional explicit assertion of the snapshot frozen in the resolved
    /// table. A differing value is rejected before any manifest reads.
    pub snapshot_id: Option<i64>,
    /// Whether Iceberg binds field names case-sensitively.
    pub case_sensitive: bool,
    /// Maximum concurrent Iceberg metadata operations. Must be non-zero.
    pub manifest_concurrency: usize,
    /// Whole-operation deadline, measured from the start of `plan`.
    pub timeout: Option<Duration>,
    /// Optional caller-controlled cancellation signal.
    pub cancellation: Option<CancellationToken>,
}

impl Default for ScanPlanningOptions {
    fn default() -> Self {
        Self { snapshot_id: None, case_sensitive: true, manifest_concurrency: 16, timeout: None, cancellation: None }
    }
}

/// Plans a frozen Iceberg table into a neutral [`ScanCandidate`].
///
/// This planner owns Iceberg table-scan construction, manifest interpretation,
/// task conversion, and timeout/cancellation handling. It deliberately does
/// not select a CPU/native execution strategy or expose source credentials.
/// [`ResolvedTable::plan_scan`] is its only entry point.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScanPlanner<'a> {
    resolved: &'a ResolvedTable,
}

impl<'a> ScanPlanner<'a> {
    pub(crate) const fn new(resolved: &'a ResolvedTable) -> Self {
        Self { resolved }
    }

    /// Plans Iceberg file tasks for the table's frozen snapshot.
    pub(crate) async fn plan(&self, mut request: ScanRequest, options: ScanPlanningOptions) -> Result<ScanCandidate> {
        let control = PlanningControl::new(&options, self.error_context());
        control.check(Operation::ScanPlanning)?;
        self.validate_options(&options)?;
        self.normalize_predicate_intent(&mut request)?;

        let snapshot = self.frozen_snapshot(&options)?;
        let name_mapping = self.extract_name_mapping()?;
        let scan_columns = self.bind_scan_columns(&request)?;

        let mut scan_builder = self
            .resolved
            .table()
            .scan()
            .with_case_sensitive(options.case_sensitive)
            .with_concurrency_limit(options.manifest_concurrency);
        if let Some(snapshot_id) = self.resolved.facts().snapshot.snapshot_id {
            scan_builder = scan_builder.snapshot_id(snapshot_id);
        }
        scan_builder = match scan_columns {
            None => scan_builder.select_all(),
            Some(columns) if columns.is_empty() => scan_builder.select_empty(),
            Some(columns) => scan_builder.select(columns),
        };
        if let Some(predicate) = request.iceberg_predicate.clone() {
            scan_builder = scan_builder.with_filter(predicate);
            request.pruning.partition_pruning_applied = true;
            request.pruning.manifest_pruning_applied = true;
            request.pruning.file_pruning_applied = true;
        }

        control.check(Operation::ScanPlanning)?;
        let scan = scan_builder.build().map_err(|source| self.source_error(Operation::ScanPlanning, source))?;
        control.check(Operation::ScanPlanning)?;

        // `FileScanTask` intentionally omits manifest sequence facts and the
        // referenced data file for position deletes. Interpret the frozen
        // manifest list alongside task planning so the neutral candidate keeps
        // those semantics without carrying encrypted key metadata.
        let semantics = match snapshot {
            Some(snapshot) => self.load_manifest_semantics(&snapshot, &control).await?,
            None => ManifestSemantics::default(),
        };

        let mut task_stream = control.wait(Operation::ManifestLoad, scan.plan_files()).await?;
        let mut data_files = Vec::new();
        loop {
            let next = control
                .wait(Operation::ManifestInterpretation, async {
                    match task_stream.next().await {
                        Some(task) => task.map(Some),
                        None => Ok(None),
                    }
                })
                .await?;
            let Some(task) = next else {
                break;
            };
            data_files.push(self.task_to_data_file(task, &semantics)?);
        }

        if let Some(file_limit) = request.source_limit.file_limit {
            data_files.truncate(file_limit);
        }

        ScanCandidate::try_new(self.resolved.facts().clone(), request, data_files, name_mapping)
            .map_err(|error| error.with_context(self.error_context()))
    }

    fn validate_options(&self, options: &ScanPlanningOptions) -> Result<()> {
        if options.manifest_concurrency == 0 {
            return Err(self.invalid_metadata(
                Operation::ScanPlanning,
                "Iceberg manifest planning concurrency must be greater than zero",
            ));
        }
        let frozen_snapshot_id = self.resolved.facts().snapshot.snapshot_id;
        if options.snapshot_id.is_some() && options.snapshot_id != frozen_snapshot_id {
            return Err(self.invalid_metadata(
                Operation::ScanPlanning,
                "requested snapshot does not match the frozen resolved-table snapshot",
            ));
        }
        Ok(())
    }

    fn frozen_snapshot(&self, options: &ScanPlanningOptions) -> Result<Option<iceberg::spec::SnapshotRef>> {
        let Some(snapshot_id) = self.resolved.facts().snapshot.snapshot_id else {
            if options.snapshot_id.is_some() {
                return Err(self.invalid_metadata(
                    Operation::ScanPlanning,
                    "an empty table cannot be planned with a snapshot override",
                ));
            }
            return Ok(None);
        };
        let metadata = self.resolved.table().metadata();
        let snapshot = metadata.snapshot_by_id(snapshot_id).cloned().ok_or_else(|| {
            self.invalid_metadata(Operation::ScanPlanning, "frozen Iceberg snapshot is absent from table metadata")
        })?;
        let schema =
            snapshot.schema(metadata).map_err(|source| self.source_error(Operation::SchemaInterpretation, source))?;
        if schema.schema_id() != self.resolved.facts().schema.schema_id {
            return Err(self.invalid_metadata(
                Operation::SchemaInterpretation,
                "frozen snapshot schema does not match resolved-table schema facts",
            ));
        }
        Ok(Some(snapshot))
    }

    fn normalize_predicate_intent(&self, request: &mut ScanRequest) -> Result<()> {
        let Some(predicate) = request.iceberg_predicate.as_ref() else {
            return Ok(());
        };
        match request.predicate.enforcement {
            PredicateEnforcement::NotPresent => {
                request.predicate.enforcement = PredicateEnforcement::ConsumerMustEnforce;
                request.predicate.residual = Some(ResidualPredicate { canonical: predicate.to_string() });
            },
            PredicateEnforcement::FullyEnforced => {
                return Err(self.invalid_metadata(
                    Operation::ScanPlanning,
                    "Iceberg manifest pruning cannot be declared as full row-level predicate enforcement",
                ));
            },
            PredicateEnforcement::PartiallyEnforced | PredicateEnforcement::ConsumerMustEnforce => {},
        }
        Ok(())
    }

    fn bind_scan_columns(&self, request: &ScanRequest) -> Result<Option<Vec<String>>> {
        let crate::Projection::Columns(bindings) = &request.projection else {
            return Ok(None);
        };
        let mut names = Vec::with_capacity(bindings.len() + request.filter_bindings.len());
        let mut field_ids = HashSet::with_capacity(bindings.len() + request.filter_bindings.len());
        for binding in bindings.iter().chain(&request.filter_bindings) {
            if field_ids.insert(binding.field_id) {
                names.push(self.bind_column_name(binding)?);
            }
        }
        Ok(Some(names))
    }

    fn bind_column_name(&self, binding: &ColumnBinding) -> Result<String> {
        self.resolved
            .facts()
            .schema
            .fields
            .iter()
            .find(|field| field.field_id == binding.field_id)
            .map(|field| field.name.clone())
            .ok_or_else(|| {
                let mut context = self.error_context();
                context.field_id = Some(binding.field_id);
                Error::new(
                    ErrorCode::UnsupportedSchema,
                    Operation::SchemaInterpretation,
                    "Iceberg task planning supports only top-level projection and filter fields",
                )
                .with_context(context)
            })
    }

    fn extract_name_mapping(&self) -> Result<Option<NameMapping>> {
        self.resolved
            .table()
            .metadata()
            .properties()
            .get(TableProperties::PROPERTY_DEFAULT_NAME_MAPPING)
            .map(|raw| {
                serde_json::from_str::<VendorNameMapping>(raw).map(|mapping| neutral_name_mapping(&mapping)).map_err(
                    |_source| {
                        self.invalid_metadata(
                            Operation::ManifestInterpretation,
                            "Iceberg schema name mapping property is invalid",
                        )
                    },
                )
            })
            .transpose()
    }

    async fn load_manifest_semantics(
        &self,
        snapshot: &iceberg::spec::SnapshotRef,
        control: &PlanningControl,
    ) -> Result<ManifestSemantics> {
        let manifest_list =
            control.wait(Operation::ManifestLoad, self.resolved.table().manifest_list_reader(snapshot).load()).await?;
        let mut semantics = ManifestSemantics::default();
        for manifest_file in manifest_list.entries() {
            let manifest = control
                .wait(Operation::ManifestLoad, self.resolved.table().manifest_reader().read(manifest_file))
                .await?;
            for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
                control.check(Operation::ManifestInterpretation)?;
                let sequence = FileSequence {
                    data_sequence_number: entry.sequence_number(),
                    file_sequence_number: entry.file_sequence_number,
                };
                match entry.content_type() {
                    DataContentType::Data => semantics.insert_data(
                        entry.file_path(),
                        ManifestDataFacts {
                            format: file_format(entry.file_format()),
                            file_size_bytes: entry.file_size_in_bytes(),
                            sequence,
                            encryption_key_metadata_present: entry.data_file().key_metadata().is_some(),
                        },
                        self,
                    )?,
                    DataContentType::PositionDeletes | DataContentType::EqualityDeletes => semantics.insert_delete(
                        entry.file_path(),
                        ManifestDeleteFacts {
                            format: file_format(entry.file_format()),
                            content: delete_content(entry.content_type()),
                            file_size_bytes: entry.file_size_in_bytes(),
                            sequence,
                            referenced_data_file: entry.data_file().referenced_data_file(),
                            equality_field_ids: entry.data_file().equality_ids().unwrap_or_default(),
                            partition_spec_id: manifest_file.partition_spec_id,
                            encryption_key_metadata_present: entry.data_file().key_metadata().is_some(),
                        },
                        self,
                    )?,
                }
            }
        }
        Ok(semantics)
    }

    fn task_to_data_file(&self, task: FileScanTask, semantics: &ManifestSemantics) -> Result<DataFile> {
        let manifest_data = semantics.data.get(task.data_file_path()).ok_or_else(|| {
            self.invalid_metadata(
                Operation::ManifestInterpretation,
                "Iceberg task data file is absent from the frozen manifest facts",
            )
        })?;
        let format = file_format(task.data_file_format());
        if format != manifest_data.format || task.file_size_in_bytes() != manifest_data.file_size_bytes {
            return Err(self.invalid_metadata(
                Operation::ManifestInterpretation,
                "Iceberg task data file disagrees with frozen manifest facts",
            ));
        }
        let partition_values = self.partition_values(&task)?;
        let delete_files = task
            .deletes()
            .iter()
            .map(|delete| self.task_to_delete_file(delete, semantics))
            .collect::<Result<Vec<_>>>()?;
        Ok(DataFile {
            location: task.data_file_path().to_owned(),
            format,
            file_size_bytes: task.file_size_in_bytes(),
            range: FileRange { start: task.start(), length: task.length() },
            record_count: task.record_count(),
            sequence: manifest_data.sequence,
            encryption_key_metadata_present: task.key_metadata().is_some()
                || manifest_data.encryption_key_metadata_present,
            partition_values,
            delete_files,
        })
    }

    fn task_to_delete_file(
        &self,
        task_delete: &iceberg::scan::FileScanTaskDeleteFile,
        semantics: &ManifestSemantics,
    ) -> Result<DeleteFile> {
        let manifest_delete = semantics.delete.get(&task_delete.file_path).ok_or_else(|| {
            self.invalid_metadata(
                Operation::ManifestInterpretation,
                "Iceberg task delete file is absent from the frozen manifest facts",
            )
        })?;
        if file_format(task_delete.file_format) != manifest_delete.format
            || delete_content(task_delete.file_type) != manifest_delete.content
            || task_delete.file_size_in_bytes != manifest_delete.file_size_bytes
            || task_delete.partition_spec_id != manifest_delete.partition_spec_id
            || task_delete.equality_ids.clone().unwrap_or_default() != manifest_delete.equality_field_ids
        {
            return Err(self.invalid_metadata(
                Operation::ManifestInterpretation,
                "Iceberg task delete file disagrees with frozen manifest facts",
            ));
        }
        Ok(DeleteFile {
            location: task_delete.file_path.clone(),
            format: manifest_delete.format,
            content: manifest_delete.content,
            file_size_bytes: manifest_delete.file_size_bytes,
            sequence: manifest_delete.sequence,
            encryption_key_metadata_present: task_delete.key_metadata.is_some()
                || manifest_delete.encryption_key_metadata_present,
            referenced_data_file: manifest_delete.referenced_data_file.clone(),
            equality_field_ids: manifest_delete.equality_field_ids.clone(),
            partition_spec_id: manifest_delete.partition_spec_id,
        })
    }

    fn partition_values(&self, task: &FileScanTask) -> Result<Vec<PartitionValue>> {
        let (Some(partition), Some(spec)) = (task.partition(), task.partition_spec()) else {
            if task.partition().is_some_and(|partition| !partition.fields().is_empty()) {
                return Err(self.invalid_metadata(
                    Operation::ManifestInterpretation,
                    "Iceberg task exposes partition values without a partition spec",
                ));
            }
            return Ok(Vec::new());
        };
        if partition.fields().len() != spec.fields().len() {
            return Err(self.invalid_metadata(
                Operation::ManifestInterpretation,
                "Iceberg task partition value count does not match its partition spec",
            ));
        }

        spec.fields()
            .iter()
            .zip(partition.fields())
            .map(|(partition_field, value)| {
                let source_field = task.schema().field_by_id(partition_field.source_id).ok_or_else(|| {
                    let mut context = self.error_context();
                    context.field_id = Some(partition_field.source_id);
                    Error::new(
                        ErrorCode::UnsupportedSchema,
                        Operation::SchemaInterpretation,
                        "Iceberg partition field references a source field absent from the snapshot schema",
                    )
                    .with_context(context)
                })?;
                let result_type = partition_field
                    .transform
                    .result_type(source_field.field_type.as_ref())
                    .map_err(|source| self.source_error(Operation::SchemaInterpretation, source))?;
                Ok(PartitionValue {
                    source_field_id: partition_field.source_id,
                    partition_field_id: partition_field.field_id,
                    transform: partition_transform(partition_field.transform),
                    value: partition_scalar(value.as_ref(), &result_type, self)?,
                })
            })
            .collect()
    }

    fn source_error(&self, operation: Operation, source: iceberg::Error) -> Error {
        Error::from_iceberg_error(operation, source).with_context(self.error_context())
    }

    fn invalid_metadata(&self, operation: Operation, message: &'static str) -> Error {
        Error::new(ErrorCode::InvalidMetadata, operation, message).with_context(self.error_context())
    }

    fn error_context(&self) -> ErrorContext {
        ErrorContext {
            table_identifier: Some(self.resolved.facts().table.display_name()),
            snapshot_id: self.resolved.facts().snapshot.snapshot_id,
            schema_id: Some(self.resolved.facts().schema.schema_id),
            ..ErrorContext::default()
        }
    }
}

#[derive(Debug, Default)]
struct ManifestSemantics {
    data: HashMap<String, ManifestDataFacts>,
    delete: HashMap<String, ManifestDeleteFacts>,
}

impl ManifestSemantics {
    fn insert_data(&mut self, location: &str, facts: ManifestDataFacts, planner: &ScanPlanner<'_>) -> Result<()> {
        if self.data.insert(location.to_string(), facts).is_some() {
            return Err(planner.invalid_metadata(
                Operation::ManifestInterpretation,
                "frozen Iceberg manifests contain duplicate live data-file locations",
            ));
        }
        Ok(())
    }

    fn insert_delete(&mut self, location: &str, facts: ManifestDeleteFacts, planner: &ScanPlanner<'_>) -> Result<()> {
        if self.delete.insert(location.to_string(), facts).is_some() {
            return Err(planner.invalid_metadata(
                Operation::ManifestInterpretation,
                "frozen Iceberg manifests contain duplicate live delete-file locations",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ManifestDataFacts {
    format: FileFormat,
    file_size_bytes: u64,
    sequence: FileSequence,
    encryption_key_metadata_present: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ManifestDeleteFacts {
    format: FileFormat,
    content: DeleteContent,
    file_size_bytes: u64,
    sequence: FileSequence,
    encryption_key_metadata_present: bool,
    referenced_data_file: Option<String>,
    equality_field_ids: Vec<i32>,
    partition_spec_id: i32,
}

struct PlanningControl {
    deadline: Option<Instant>,
    cancellation: Option<CancellationToken>,
    context: ErrorContext,
}

impl PlanningControl {
    fn new(options: &ScanPlanningOptions, context: ErrorContext) -> Self {
        Self {
            deadline: options.timeout.map(|timeout| Instant::now() + timeout),
            cancellation: options.cancellation.clone(),
            context,
        }
    }

    fn check(&self, operation: Operation) -> Result<()> {
        if self.cancellation.as_ref().is_some_and(CancellationToken::is_cancelled) {
            return Err(self.cancelled(operation));
        }
        if self.deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err(self.timed_out(operation));
        }
        Ok(())
    }

    async fn wait<T, F>(&self, operation: Operation, future: F) -> Result<T>
    where F: Future<Output = iceberg::Result<T>> {
        self.check(operation)?;
        let timeout = self.deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
        let result = match (timeout, self.cancellation.as_ref()) {
            (Some(timeout), Some(cancellation)) => {
                tokio::select! {
                    _ = cancellation.cancelled() => return Err(self.cancelled(operation)),
                    result = tokio::time::timeout(timeout, future) => result.map_err(|_elapsed| self.timed_out(operation))?,
                }
            },
            (Some(timeout), None) => {
                tokio::time::timeout(timeout, future).await.map_err(|_elapsed| self.timed_out(operation))?
            },
            (None, Some(cancellation)) => {
                tokio::select! {
                    _ = cancellation.cancelled() => return Err(self.cancelled(operation)),
                    result = future => result,
                }
            },
            (None, None) => future.await,
        };
        result.map_err(|source| Error::from_iceberg_error(operation, source).with_context(self.context.clone()))
    }

    fn cancelled(&self, operation: Operation) -> Error {
        Error::new(ErrorCode::Cancelled, operation, "Iceberg scan planning was cancelled")
            .with_context(self.context.clone())
    }

    fn timed_out(&self, operation: Operation) -> Error {
        Error::new(ErrorCode::Timeout, operation, "Iceberg scan planning exceeded its deadline")
            .with_context(self.context.clone())
    }
}

fn file_format(format: DataFileFormat) -> FileFormat {
    match format {
        DataFileFormat::Avro => FileFormat::Avro,
        DataFileFormat::Orc => FileFormat::Orc,
        DataFileFormat::Parquet => FileFormat::Parquet,
        DataFileFormat::Puffin => FileFormat::Puffin,
    }
}

fn delete_content(content: DataContentType) -> DeleteContent {
    match content {
        DataContentType::PositionDeletes => DeleteContent::Position,
        DataContentType::EqualityDeletes => DeleteContent::Equality,
        DataContentType::Data => unreachable!("data files never become delete-file facts"),
    }
}

fn partition_transform(transform: Transform) -> PartitionTransform {
    match transform {
        Transform::Identity => PartitionTransform::Identity,
        Transform::Bucket(buckets) => PartitionTransform::Bucket(buckets),
        Transform::Truncate(width) => PartitionTransform::Truncate(width),
        Transform::Year => PartitionTransform::Year,
        Transform::Month => PartitionTransform::Month,
        Transform::Day => PartitionTransform::Day,
        Transform::Hour => PartitionTransform::Hour,
        Transform::Void => PartitionTransform::Void,
        Transform::Unknown => PartitionTransform::Unknown,
    }
}

fn partition_scalar(
    value: Option<&Literal>,
    result_type: &iceberg::spec::Type,
    planner: &ScanPlanner<'_>,
) -> Result<ScalarValue> {
    let Some(value) = value else {
        return Ok(ScalarValue::Null);
    };
    let Some(primitive) = value.as_primitive_literal() else {
        return Err(planner.invalid_metadata(
            Operation::ManifestInterpretation,
            "Iceberg partition values must be primitive literals",
        ));
    };
    let iceberg::spec::Type::Primitive(result_type) = result_type else {
        return Err(planner.invalid_metadata(
            Operation::SchemaInterpretation,
            "Iceberg partition transform produced a non-primitive result type",
        ));
    };
    match (result_type, primitive) {
        (PrimitiveType::Boolean, PrimitiveLiteral::Boolean(value)) => Ok(ScalarValue::Boolean(value)),
        (PrimitiveType::Int, PrimitiveLiteral::Int(value)) => Ok(ScalarValue::Int(value)),
        (PrimitiveType::Long, PrimitiveLiteral::Long(value)) => Ok(ScalarValue::Long(value)),
        (PrimitiveType::Float, PrimitiveLiteral::Float(value)) => {
            Ok(ScalarValue::Float { bits: value.into_inner().to_bits() })
        },
        (PrimitiveType::Double, PrimitiveLiteral::Double(value)) => {
            Ok(ScalarValue::Double { bits: value.into_inner().to_bits() })
        },
        (PrimitiveType::Decimal { scale, .. }, PrimitiveLiteral::Int128(unscaled)) => Ok(ScalarValue::Decimal {
            unscaled,
            scale: i32::try_from(*scale).map_err(|_error| {
                planner.invalid_metadata(Operation::SchemaInterpretation, "Iceberg decimal scale exceeds i32")
            })?,
        }),
        (PrimitiveType::Date, PrimitiveLiteral::Int(value)) => Ok(ScalarValue::Date(value)),
        (PrimitiveType::Time, PrimitiveLiteral::Long(value)) => Ok(ScalarValue::Time(value)),
        (PrimitiveType::Timestamp | PrimitiveType::Timestamptz, PrimitiveLiteral::Long(value)) => {
            Ok(ScalarValue::TimestampMicros(value))
        },
        (PrimitiveType::TimestampNs | PrimitiveType::TimestamptzNs, PrimitiveLiteral::Long(value)) => {
            Ok(ScalarValue::TimestampNanos(value))
        },
        (PrimitiveType::String, PrimitiveLiteral::String(value)) => Ok(ScalarValue::String(value)),
        (PrimitiveType::Uuid, PrimitiveLiteral::UInt128(value)) => Ok(ScalarValue::Uuid(value.to_be_bytes())),
        (PrimitiveType::Fixed(_) | PrimitiveType::Binary, PrimitiveLiteral::Binary(value)) => {
            Ok(ScalarValue::Binary(value))
        },
        _ => Err(planner.invalid_metadata(
            Operation::ManifestInterpretation,
            "Iceberg partition literal does not match its transform result type",
        )),
    }
}

fn neutral_name_mapping(mapping: &VendorNameMapping) -> NameMapping {
    NameMapping { fields: mapping.fields().iter().map(neutral_name_mapping_field).collect() }
}

fn neutral_name_mapping_field(field: &iceberg::spec::MappedField) -> NameMappingField {
    NameMappingField {
        field_id: field.field_id(),
        names: field.names().to_vec(),
        fields: field.fields().iter().map(|field| neutral_name_mapping_field(field)).collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf, time::Duration};

    use iceberg::{
        TableIdent,
        expr::Reference,
        io::FileIO,
        spec::{
            DataContentType, DataFileBuilder, DataFileFormat, Literal, ManifestListWriter, ManifestWriterBuilder,
            Struct, TableMetadata, TableProperties,
        },
        table::Table,
    };

    use super::*;
    use crate::{
        ColumnRole, ColumnSource, ErrorStatus, PredicateEnforcement, Projection, SourceAccessPolicy, SourceLimit,
    };

    const METADATA_TEMPLATE: &str = include_str!("testdata/metadata_v2.json");

    struct ManifestFixture {
        _temp_dir: tempfile::TempDir,
        table_location: PathBuf,
        table: Table,
    }

    impl ManifestFixture {
        fn new() -> Self {
            let temp_dir = tempfile::TempDir::new().expect("create Iceberg fixture directory");
            let table_location = temp_dir.path().join("events");
            let metadata_dir = table_location.join("metadata");
            fs::create_dir_all(&metadata_dir).expect("create Iceberg metadata directory");

            let manifest_list_1_location = metadata_dir.join("manifest-list-1.avro");
            let manifest_list_2_location = metadata_dir.join("manifest-list-2.avro");
            let metadata_location = metadata_dir.join("v1.json");
            let metadata_json = METADATA_TEMPLATE
                .replace("{{ table_location }}", table_location.to_str().expect("UTF-8 table path"))
                .replace(
                    "{{ manifest_list_1_location }}",
                    manifest_list_1_location.to_str().expect("UTF-8 manifest path"),
                )
                .replace(
                    "{{ manifest_list_2_location }}",
                    manifest_list_2_location.to_str().expect("UTF-8 manifest path"),
                )
                .replace("{{ table_metadata_1_location }}", metadata_location.to_str().expect("UTF-8 metadata path"));
            let mut metadata_json =
                serde_json::from_str::<serde_json::Value>(&metadata_json).expect("parse fixture metadata JSON");
            metadata_json["properties"][TableProperties::PROPERTY_DEFAULT_NAME_MAPPING] =
                serde_json::Value::String(r#"[{"field-id":1,"names":["x","legacy_x"]}]"#.to_string());
            let metadata = serde_json::from_value::<TableMetadata>(metadata_json).expect("parse fixture metadata");
            let table = Table::builder()
                .metadata(metadata)
                .identifier(TableIdent::from_strs(["test", "events"]).expect("valid fixture table identifier"))
                .file_io(FileIO::new_with_fs())
                .metadata_location(metadata_location.to_str().expect("UTF-8 metadata path"))
                .runtime(iceberg::Runtime::current())
                .build()
                .expect("build fixture table");
            Self { _temp_dir: temp_dir, table_location, table }
        }

        async fn write_current_snapshot_manifests(&self) -> (String, String) {
            let snapshot = self.table.metadata().current_snapshot().expect("current fixture snapshot");
            let schema = snapshot.schema(self.table.metadata()).expect("fixture snapshot schema");
            let partition_spec = self.table.metadata().default_partition_spec();
            let data_location = self.table_location.join("data-0001.parquet").to_str().unwrap().to_string();
            let delete_location = self.table_location.join("delete-0001.parquet").to_str().unwrap().to_string();

            let data_manifest = {
                let output = self
                    .table
                    .file_io()
                    .new_output(self.table_location.join("metadata/data.avro").to_str().unwrap())
                    .expect("create data manifest output");
                let mut writer = ManifestWriterBuilder::new(
                    output,
                    Some(snapshot.snapshot_id()),
                    schema.clone(),
                    partition_spec.as_ref().clone(),
                )
                .build_v2_data();
                writer
                    .add_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::Data)
                            .file_path(data_location.clone())
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(4096)
                            .record_count(17)
                            .partition(Struct::from_iter([Some(Literal::long(100))]))
                            // Deliberately non-empty: the candidate must retain
                            // only the safe presence marker, never these bytes.
                            .key_metadata(Some(vec![0xA5, 0x5A]))
                            .build()
                            .expect("build data file"),
                        snapshot.sequence_number(),
                    )
                    .expect("add data manifest entry");
                writer.write_manifest_file().await.expect("write data manifest")
            };

            let delete_manifest = {
                let output = self
                    .table
                    .file_io()
                    .new_output(self.table_location.join("metadata/delete.avro").to_str().unwrap())
                    .expect("create delete manifest output");
                let mut writer = ManifestWriterBuilder::new(
                    output,
                    Some(snapshot.snapshot_id()),
                    schema,
                    partition_spec.as_ref().clone(),
                )
                .build_v2_deletes();
                writer
                    .add_file(
                        DataFileBuilder::default()
                            .partition_spec_id(0)
                            .content(DataContentType::PositionDeletes)
                            .file_path(delete_location.clone())
                            .file_format(DataFileFormat::Parquet)
                            .file_size_in_bytes(512)
                            .record_count(1)
                            .partition(Struct::from_iter([Some(Literal::long(100))]))
                            .referenced_data_file(Some(data_location.clone()))
                            .key_metadata(Some(vec![0x42]))
                            .build()
                            .expect("build delete file"),
                        snapshot.sequence_number(),
                    )
                    .expect("add delete manifest entry");
                writer.write_manifest_file().await.expect("write delete manifest")
            };

            let writer = self
                .table
                .file_io()
                .new_output(snapshot.manifest_list())
                .expect("create manifest-list output")
                .writer()
                .await
                .expect("open manifest-list output");
            let mut manifest_list = ManifestListWriter::v2(
                writer,
                snapshot.snapshot_id(),
                snapshot.parent_snapshot_id(),
                snapshot.sequence_number(),
            );
            manifest_list
                .add_manifests(vec![data_manifest, delete_manifest].into_iter())
                .expect("add manifests to list");
            manifest_list.close().await.expect("close manifest list");
            (data_location, delete_location)
        }

        fn resolved(&self) -> ResolvedTable {
            ResolvedTable::from_table(self.table.clone(), SourceAccessPolicy::s3_default_chain(None::<String>))
                .expect("resolve fixture table")
        }
    }

    #[tokio::test]
    async fn plans_real_manifest_tasks_with_projection_predicate_partition_delete_and_sequence_facts() {
        let fixture = ManifestFixture::new();
        let (data_location, delete_location) = fixture.write_current_snapshot_manifests().await;
        let resolved = fixture.resolved();
        let request = ScanRequest {
            projection: Projection::Columns(vec![ColumnBinding {
                field_id: 1,
                output_name: "customer_id".to_string(),
                source: ColumnSource::PartitionConstant,
                role: ColumnRole::Projected,
                missing_default: None,
            }]),
            filter_bindings: vec![ColumnBinding {
                field_id: 2,
                output_name: "y".to_string(),
                source: ColumnSource::PhysicalFile,
                role: ColumnRole::FilterOnly,
                missing_default: None,
            }],
            predicate: crate::PredicateIntent::none(),
            iceberg_predicate: Some(Reference::new("y").greater_than(iceberg::spec::Datum::long(0))),
            source_limit: SourceLimit { row_limit: Some(10), file_limit: Some(1) },
            pruning: crate::PruningIntent::default(),
        };

        let candidate = resolved.plan_scan(request, ScanPlanningOptions::default()).await.expect("plan fixture");

        assert_eq!(candidate.planned_data_file_count(), 1);
        assert_eq!(candidate.planned_delete_file_count(), 1);
        assert_eq!(candidate.resolved_table.snapshot.sequence_number, Some(1));
        assert_eq!(candidate.request.predicate.enforcement, PredicateEnforcement::ConsumerMustEnforce);
        assert!(candidate.request.predicate.residual.as_ref().unwrap().canonical.contains("y >"));
        assert!(candidate.request.pruning.partition_pruning_applied);
        assert!(candidate.request.pruning.manifest_pruning_applied);
        assert!(candidate.request.pruning.file_pruning_applied);

        let data_file = &candidate.data_files[0];
        assert_eq!(data_file.location, data_location);
        assert_eq!(data_file.sequence, FileSequence { data_sequence_number: Some(1), file_sequence_number: Some(1) });
        assert!(data_file.encryption_key_metadata_present);
        assert_eq!(data_file.partition_values.len(), 1);
        assert_eq!(data_file.partition_values[0].source_field_id, 1);
        assert_eq!(data_file.partition_values[0].partition_field_id, 1000);
        assert_eq!(data_file.partition_values[0].transform, PartitionTransform::Identity);
        assert_eq!(data_file.partition_values[0].value, ScalarValue::Long(100));

        let delete_file = &data_file.delete_files[0];
        assert_eq!(delete_file.location, delete_location);
        assert_eq!(delete_file.content, DeleteContent::Position);
        assert_eq!(delete_file.referenced_data_file.as_deref(), Some(data_location.as_str()));
        assert_eq!(delete_file.sequence, FileSequence { data_sequence_number: Some(1), file_sequence_number: Some(1) });
        assert!(delete_file.encryption_key_metadata_present);
        assert_eq!(
            candidate.name_mapping.as_ref().unwrap().fields[0].names,
            vec!["x".to_string(), "legacy_x".to_string()]
        );
    }

    #[tokio::test]
    async fn rejected_snapshot_timeout_and_cancellation_are_normalized_leaf_errors() {
        let fixture = ManifestFixture::new();
        let resolved = fixture.resolved();
        let frozen_snapshot_id = resolved.facts().snapshot.snapshot_id.expect("fixture snapshot");

        let snapshot_error = resolved
            .plan_scan(
                ScanRequest::default(),
                ScanPlanningOptions { snapshot_id: Some(frozen_snapshot_id + 1), ..Default::default() },
            )
            .await
            .expect_err("mismatched snapshot must fail");
        assert_eq!(snapshot_error.code(), ErrorCode::InvalidMetadata);
        assert_eq!(snapshot_error.operation(), Operation::ScanPlanning);

        let timeout_error = resolved
            .plan_scan(
                ScanRequest::default(),
                ScanPlanningOptions { timeout: Some(Duration::ZERO), ..Default::default() },
            )
            .await
            .expect_err("zero deadline must fail");
        assert_eq!(timeout_error.code(), ErrorCode::Timeout);
        assert_eq!(timeout_error.status(), ErrorStatus::Temporary);

        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let cancelled_error = resolved
            .plan_scan(
                ScanRequest::default(),
                ScanPlanningOptions { cancellation: Some(cancellation), ..Default::default() },
            )
            .await
            .expect_err("cancelled planning must fail");
        assert_eq!(cancelled_error.code(), ErrorCode::Cancelled);
        assert_eq!(cancelled_error.status(), ErrorStatus::Cancelled);
    }
}
