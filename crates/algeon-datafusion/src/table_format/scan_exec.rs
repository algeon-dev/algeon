#[cfg(feature = "iceberg")]
use std::error::Error as StdError;
use std::sync::Arc;

use arrow::{
    array::{Array, ArrayRef, make_array},
    datatypes::{DataType, Field, Schema, SchemaRef},
    error::ArrowError,
    record_batch::{RecordBatch, RecordBatchOptions},
};
use datafusion::{
    common::{DataFusionError, Result as DataFusionResult, Statistics, tree_node::TreeNodeRecursion},
    execution::{SendableRecordBatchStream, TaskContext},
    physical_expr::{EquivalenceProperties, PhysicalExpr},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, StatisticsArgs, StatisticsContext,
        metrics::MetricsSet, stream::RecordBatchStreamAdapter,
    },
};
use futures::{TryStreamExt, future::ready};
#[cfg(feature = "iceberg")]
use iceberg::{Error as IcebergError, ErrorKind as IcebergErrorKind};
#[cfg(feature = "iceberg")]
use object_store::Error as ObjectStoreError;
#[cfg(feature = "iceberg")]
use opendal::{Error as OpendalError, ErrorKind as OpendalErrorKind};

use crate::{
    capability::CapabilityReason,
    error::{DataFusionErrorExt, Error, ErrorCode},
    table_format::{TableFormat, TableFormatScanDiagnostics, TableFormatScanPlan},
};

const WITH_NEW_CHILDREN_OPERATION: &str = "algeon_datafusion.iceberg.scan_exec.with_new_children";
const EXECUTE_OPERATION: &str = "algeon_datafusion.iceberg.scan_exec.execute";
const CANONICAL_SCHEMA_OPERATION: &str = "algeon_datafusion.iceberg.scan_exec.canonical_schema";

/// Immutable native-inspection result attached to one executable Iceberg scan.
///
/// Native inspection is optional acceleration work. A capability miss or an
/// inspection failure never removes the CPU delegate from the physical plan.
#[derive(Debug, Clone)]
pub enum AlgeonIcebergNativeScanOutcome {
    /// Native scan facts are available for whole-candidate lowering.
    Available(TableFormatScanPlan),
    /// Inspection completed, but the resolved scan has a known native capability miss.
    NotSupported { scan_plan: TableFormatScanPlan, reason: CapabilityReason },
    /// Optional native inspection failed. The typed error remains available for
    /// diagnostics while the CPU delegate remains executable.
    InspectionFailed { resolved_scan: TableFormatScanPlan, error: Arc<Error> },
}

impl AlgeonIcebergNativeScanOutcome {
    /// Classifies successfully resolved scan facts without changing CPU executability.
    pub fn from_scan_plan(scan_plan: TableFormatScanPlan) -> Self {
        match scan_plan.first_not_supported_reason() {
            Some(reason) => Self::NotSupported { scan_plan, reason },
            None => Self::Available(scan_plan),
        }
    }

    /// Returns native scan facts when inspection produced them.
    pub fn scan_plan(&self) -> Option<&TableFormatScanPlan> {
        match self {
            Self::Available(scan_plan) | Self::NotSupported { scan_plan, .. } => Some(scan_plan),
            Self::InspectionFailed { .. } => None,
        }
    }

    /// Returns the immutable source facts resolved before optional native inspection.
    pub fn resolved_scan_plan(&self) -> &TableFormatScanPlan {
        match self {
            Self::Available(scan_plan) | Self::NotSupported { scan_plan, .. } => scan_plan,
            Self::InspectionFailed { resolved_scan, .. } => resolved_scan,
        }
    }

    /// Returns the typed optional-inspection failure, when inspection did not complete.
    pub fn inspection_error(&self) -> Option<&Error> {
        match self {
            Self::InspectionFailed { error, .. } => Some(error),
            Self::Available(_) | Self::NotSupported { .. } => None,
        }
    }

    /// Returns the stable candidate rejection reason, if native lowering must not start.
    pub fn rejection_reason(&self) -> Option<CapabilityReason> {
        match self {
            Self::Available(_) => None,
            Self::NotSupported { reason, .. } => Some(*reason),
            Self::InspectionFailed { .. } => Some(CapabilityReason::TableFormatPlanningFailed),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Available(_) => "available",
            Self::NotSupported { .. } => "not_supported",
            Self::InspectionFailed { .. } => "inspection_failed",
        }
    }
}

/// DataFusion-facing Iceberg scan with an executable CPU delegate and optional
/// immutable native scan facts.
///
/// The CPU delegate is a real child so DataFusion can inspect and rewrite its
/// physical properties. Native lowering replaces this entire wrapper only when
/// [`AlgeonIcebergNativeScanOutcome::Available`] is selected.
#[derive(Debug, Clone)]
pub struct AlgeonIcebergScanExec {
    cpu_delegate: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
    native_outcome: AlgeonIcebergNativeScanOutcome,
}

impl AlgeonIcebergScanExec {
    /// Creates a dual-mode scan with one canonical DataFusion-facing schema.
    pub fn try_new(
        cpu_delegate: Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
        native_outcome: AlgeonIcebergNativeScanOutcome,
    ) -> DataFusionResult<Self> {
        let delegate_schema = cpu_delegate.schema();
        validate_canonical_schema_compatibility(delegate_schema.as_ref(), schema.as_ref()).map_err(|mismatch| {
            canonical_schema_error(
                "Iceberg CPU delegate schema is incompatible with the canonical scan schema",
                delegate_schema.as_ref(),
                schema.as_ref(),
                mismatch,
                None,
            )
        })?;
        let delegate_equivalence_properties = cpu_delegate.properties().equivalence_properties().clone();
        let exact_data_types = delegate_schema
            .fields()
            .iter()
            .zip(schema.fields())
            .all(|(delegate, canonical)| delegate.data_type() == canonical.data_type());
        let equivalence_properties = if exact_data_types {
            delegate_equivalence_properties.with_new_schema(Arc::clone(&schema)).map_err(|error| {
                DataFusionError::from(
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        CANONICAL_SCHEMA_OPERATION,
                        "Iceberg CPU delegate properties are incompatible with the canonical scan schema",
                    )
                    .with_fact("cpu_delegate_field_count", delegate_schema.fields().len())
                    .with_fact("canonical_field_count", schema.fields().len())
                    .with_source(error),
                )
            })?
        } else {
            // DataFusion's property remapper requires exact nested DataType equality,
            // including child-field metadata. The compatibility proof above permits
            // only metadata differences, but that is not enough to preserve arbitrary
            // expression equivalences or orderings through the public remapping API.
            EquivalenceProperties::new(Arc::clone(&schema))
                .with_constraints(delegate_equivalence_properties.constraints().clone())
        };
        let properties =
            Arc::new(cpu_delegate.properties().as_ref().clone().with_eq_properties(equivalence_properties));
        Ok(Self { cpu_delegate, schema, properties, native_outcome })
    }

    pub fn cpu_delegate(&self) -> &Arc<dyn ExecutionPlan> {
        &self.cpu_delegate
    }

    pub fn native_outcome(&self) -> &AlgeonIcebergNativeScanOutcome {
        &self.native_outcome
    }

    pub fn scan_plan(&self) -> Option<&TableFormatScanPlan> {
        self.native_outcome.scan_plan()
    }

    pub fn resolved_scan_plan(&self) -> &TableFormatScanPlan {
        self.native_outcome.resolved_scan_plan()
    }

    pub fn table_format(&self) -> TableFormat {
        self.resolved_scan_plan().format.clone()
    }

    /// Returns credential-free facts from immutable source resolution.
    pub fn diagnostics(&self) -> TableFormatScanDiagnostics {
        self.resolved_scan_plan().diagnostics()
    }

    fn resolved_snapshot_id(&self) -> Option<&str> {
        self.resolved_scan_plan().snapshot_id.as_deref()
    }
}

impl DisplayAs for AlgeonIcebergScanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let diagnostics = self.diagnostics();
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                f,
                "AlgeonIcebergScanExec: native_outcome={}, native_inspection_error_code={}, snapshot_id={}, catalog_type={}, planned_data_file_count={}, planned_delete_file_count={}, admitted_native_source_file_count={}, native_source_read_capability={}, source_read_strategy={}, credential_mode={}, lakehouse_cache_status={}, lakehouse_cache_read_mode={}, lakehouse_cache_fill_mode={}, direct_device_read_disposition={}",
                self.native_outcome.as_str(),
                inspection_error_code(&self.native_outcome).unwrap_or_default(),
                diagnostics.snapshot_id.as_deref().unwrap_or_default(),
                diagnostics.catalog_type.as_deref().unwrap_or_default(),
                diagnostics.planned_data_file_count,
                diagnostics.planned_delete_file_count,
                diagnostics.admitted_native_source_file_count,
                diagnostics.native_source_read_capability,
                diagnostics.source_read_strategy,
                diagnostics.credential_mode,
                diagnostics.lakehouse_cache_status,
                diagnostics.lakehouse_cache_read_mode,
                diagnostics.lakehouse_cache_fill_mode,
                diagnostics.direct_device_read_disposition.code(),
            ),
            DisplayFormatType::TreeRender => {
                writeln!(f, "native_outcome={}", self.native_outcome.as_str())?;
                writeln!(
                    f,
                    "native_inspection_error_code={}",
                    inspection_error_code(&self.native_outcome).unwrap_or_default()
                )?;
                writeln!(f, "snapshot_id={}", diagnostics.snapshot_id.as_deref().unwrap_or_default())?;
                writeln!(f, "catalog_type={}", diagnostics.catalog_type.as_deref().unwrap_or_default())?;
                writeln!(f, "planned_data_file_count={}", diagnostics.planned_data_file_count)?;
                writeln!(f, "planned_delete_file_count={}", diagnostics.planned_delete_file_count)?;
                writeln!(f, "admitted_native_source_file_count={}", diagnostics.admitted_native_source_file_count)?;
                writeln!(f, "native_source_read_capability={}", diagnostics.native_source_read_capability)?;
                writeln!(f, "source_read_strategy={}", diagnostics.source_read_strategy)?;
                writeln!(f, "credential_mode={}", diagnostics.credential_mode)?;
                writeln!(f, "lakehouse_cache_status={}", diagnostics.lakehouse_cache_status)?;
                writeln!(f, "lakehouse_cache_read_mode={}", diagnostics.lakehouse_cache_read_mode)?;
                writeln!(f, "lakehouse_cache_fill_mode={}", diagnostics.lakehouse_cache_fill_mode)?;
                writeln!(f, "direct_device_read_disposition={}", diagnostics.direct_device_read_disposition.code())
            },
        }
    }
}

impl ExecutionPlan for AlgeonIcebergScanExec {
    fn name(&self) -> &str {
        "AlgeonIcebergScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
    ) -> DataFusionResult<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.cpu_delegate]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [cpu_delegate] = children.try_into().map_err(|children: Vec<_>| {
            DataFusionError::from(
                Error::new(
                    ErrorCode::InvalidChildCount,
                    WITH_NEW_CHILDREN_OPERATION,
                    "Iceberg scan wrapper requires exactly one executable CPU delegate",
                )
                .with_fact("actual_child_count", children.len())
                .with_fact("expected_child_count", 1),
            )
        })?;
        Ok(Arc::new(Self::try_new(cpu_delegate, Arc::clone(&self.schema), self.native_outcome.clone())?))
    }

    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        let stream = self
            .cpu_delegate
            .execute(partition, context)
            .map_err(|error| map_cpu_delegate_error(error, self.resolved_snapshot_id()))?;
        let schema = Arc::clone(&self.schema);
        let snapshot_id = self.resolved_snapshot_id().map(str::to_string);
        let stream_snapshot_id = snapshot_id.clone();
        let batch_schema = Arc::clone(&schema);
        let stream =
            stream.map_err(move |error| map_cpu_delegate_error(error, stream_snapshot_id.as_deref())).and_then(
                move |batch| ready(canonicalize_cpu_batch(batch, Arc::clone(&batch_schema), snapshot_id.as_deref())),
            );
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        self.cpu_delegate.metrics()
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        StatisticsContext::new().compute(self.cpu_delegate.as_ref(), &StatisticsArgs::new().with_partition(partition))
    }
}

fn inspection_error_code(outcome: &AlgeonIcebergNativeScanOutcome) -> Option<&'static str> {
    match outcome {
        AlgeonIcebergNativeScanOutcome::InspectionFailed { error, .. } => Some(error.code().as_str()),
        _ => None,
    }
}

fn canonicalize_cpu_batch(
    batch: RecordBatch,
    schema: SchemaRef,
    snapshot_id: Option<&str>,
) -> DataFusionResult<RecordBatch> {
    let (delegate_schema, columns, row_count) = batch.into_parts();
    validate_canonical_schema_compatibility(delegate_schema.as_ref(), schema.as_ref()).map_err(|mismatch| {
        canonical_schema_error(
            "Iceberg CPU delegate returned a batch incompatible with the canonical scan schema",
            delegate_schema.as_ref(),
            schema.as_ref(),
            mismatch,
            snapshot_id,
        )
    })?;
    let columns = columns
        .into_iter()
        .zip(schema.fields())
        .map(|(column, field)| canonicalize_cpu_array(column, field.data_type()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            let mut adapter = Error::new(
                ErrorCode::InvalidExecutionPlan,
                CANONICAL_SCHEMA_OPERATION,
                "Iceberg CPU delegate returned a batch incompatible with the canonical scan schema",
            )
            .with_fact("cpu_delegate_field_count", delegate_schema.fields().len())
            .with_fact("canonical_field_count", schema.fields().len());
            if let Some(snapshot_id) = snapshot_id {
                adapter = adapter.with_fact("snapshot_id", snapshot_id);
            }
            DataFusionError::from(adapter.with_source(error))
        })?;
    RecordBatch::try_new_with_options(
        Arc::clone(&schema),
        columns,
        &RecordBatchOptions::new().with_match_field_names(false).with_row_count(Some(row_count)),
    )
    .map_err(|error| {
        let mut adapter = Error::new(
            ErrorCode::InvalidExecutionPlan,
            CANONICAL_SCHEMA_OPERATION,
            "Iceberg CPU delegate returned a batch incompatible with the canonical scan schema",
        )
        .with_fact("cpu_delegate_field_count", delegate_schema.fields().len())
        .with_fact("canonical_field_count", schema.fields().len());
        if let Some(snapshot_id) = snapshot_id {
            adapter = adapter.with_fact("snapshot_id", snapshot_id);
        }
        DataFusionError::from(adapter.with_source(error))
    })
}

/// Rebuilds Arrow array data with the canonical field-bearing type while
/// retaining the delegate's validated buffers and values. Arrow places nested
/// field metadata in container data types, so assigning only a canonical
/// `RecordBatch` schema is insufficient for list, struct, map, union, and
/// other nested columns.
fn canonicalize_cpu_array(array: ArrayRef, canonical_data_type: &DataType) -> Result<ArrayRef, ArrowError> {
    if array.data_type() == canonical_data_type {
        return Ok(array);
    }

    let data = array.to_data();
    let child_types = canonical_array_child_data_types(canonical_data_type);
    if data.child_data().len() != child_types.len() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "canonical Arrow data type has {} child arrays but delegate array has {}",
            child_types.len(),
            data.child_data().len()
        )));
    }
    let child_data = data
        .child_data()
        .iter()
        .zip(child_types)
        .map(|(child, child_type)| {
            canonicalize_cpu_array(make_array(child.clone()), child_type).map(|child| child.to_data())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(make_array(data.into_builder().data_type(canonical_data_type.clone()).child_data(child_data).build()?))
}

fn canonical_array_child_data_types(data_type: &DataType) -> Vec<&DataType> {
    match data_type {
        DataType::List(field)
        | DataType::ListView(field)
        | DataType::LargeList(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => vec![field.data_type()],
        DataType::Struct(fields) => fields.iter().map(|field| field.data_type()).collect(),
        DataType::Union(fields, _) => fields.iter().map(|(_, field)| field.data_type()).collect(),
        DataType::Dictionary(_, value) => vec![value.as_ref()],
        DataType::RunEndEncoded(run_ends, values) => vec![run_ends.data_type(), values.data_type()],
        _ => Vec::new(),
    }
}

fn canonical_schema_error(
    message: &'static str,
    delegate_schema: &Schema,
    canonical_schema: &Schema,
    mismatch: String,
    snapshot_id: Option<&str>,
) -> DataFusionError {
    let mut error = Error::new(ErrorCode::InvalidExecutionPlan, CANONICAL_SCHEMA_OPERATION, message)
        .with_fact("cpu_delegate_field_count", delegate_schema.fields().len())
        .with_fact("canonical_field_count", canonical_schema.fields().len())
        .with_fact("schema_mismatch", mismatch);
    if let Some(snapshot_id) = snapshot_id {
        error = error.with_fact("snapshot_id", snapshot_id);
    }
    DataFusionError::from(error)
}

fn validate_canonical_schema_compatibility(delegate: &Schema, canonical: &Schema) -> Result<(), String> {
    if delegate.fields().len() != canonical.fields().len() {
        return Err(format!(
            "field count differs: delegate={} canonical={}",
            delegate.fields().len(),
            canonical.fields().len()
        ));
    }
    for (index, (delegate, canonical)) in delegate.fields().iter().zip(canonical.fields()).enumerate() {
        validate_canonical_field_compatibility(delegate, canonical, &format!("field[{index}]"))?;
    }
    Ok(())
}

fn validate_canonical_field_compatibility(delegate: &Field, canonical: &Field, path: &str) -> Result<(), String> {
    if delegate.name() != canonical.name() {
        return Err(format!("{path} name differs: delegate={} canonical={}", delegate.name(), canonical.name()));
    }
    if delegate.is_nullable() != canonical.is_nullable() {
        return Err(format!(
            "{path} nullability differs: delegate={} canonical={}",
            delegate.is_nullable(),
            canonical.is_nullable()
        ));
    }
    if delegate.dict_is_ordered() != canonical.dict_is_ordered() {
        return Err(format!(
            "{path} dictionary ordering differs: delegate={:?} canonical={:?}",
            delegate.dict_is_ordered(),
            canonical.dict_is_ordered()
        ));
    }
    validate_canonical_data_type_compatibility(delegate.data_type(), canonical.data_type(), path)
}

fn validate_canonical_data_type_compatibility(
    delegate: &DataType,
    canonical: &DataType,
    path: &str,
) -> Result<(), String> {
    match (delegate, canonical) {
        (DataType::List(delegate), DataType::List(canonical))
        | (DataType::ListView(delegate), DataType::ListView(canonical))
        | (DataType::LargeList(delegate), DataType::LargeList(canonical))
        | (DataType::LargeListView(delegate), DataType::LargeListView(canonical)) => {
            validate_canonical_field_compatibility(delegate, canonical, &format!("{path}.element"))
        },
        (DataType::FixedSizeList(delegate, delegate_size), DataType::FixedSizeList(canonical, canonical_size)) => {
            if delegate_size != canonical_size {
                return Err(format!(
                    "{path} fixed-size list length differs: delegate={delegate_size} canonical={canonical_size}"
                ));
            }
            validate_canonical_field_compatibility(delegate, canonical, &format!("{path}.element"))
        },
        (DataType::Struct(delegate), DataType::Struct(canonical)) => {
            if delegate.len() != canonical.len() {
                return Err(format!(
                    "{path} struct field count differs: delegate={} canonical={}",
                    delegate.len(),
                    canonical.len()
                ));
            }
            for (index, (delegate, canonical)) in delegate.iter().zip(canonical).enumerate() {
                validate_canonical_field_compatibility(delegate, canonical, &format!("{path}.field[{index}]"))?;
            }
            Ok(())
        },
        (DataType::Union(delegate, delegate_mode), DataType::Union(canonical, canonical_mode)) => {
            if delegate_mode != canonical_mode || delegate.len() != canonical.len() {
                return Err(format!("{path} union shape differs"));
            }
            for (index, ((delegate_id, delegate), (canonical_id, canonical))) in
                delegate.iter().zip(canonical.iter()).enumerate()
            {
                if delegate_id != canonical_id {
                    return Err(format!(
                        "{path}.field[{index}] union type id differs: delegate={delegate_id} canonical={canonical_id}"
                    ));
                }
                validate_canonical_field_compatibility(delegate, canonical, &format!("{path}.field[{index}]"))?;
            }
            Ok(())
        },
        (DataType::Dictionary(delegate_key, delegate_value), DataType::Dictionary(canonical_key, canonical_value)) => {
            validate_canonical_data_type_compatibility(delegate_key, canonical_key, &format!("{path}.key"))?;
            validate_canonical_data_type_compatibility(delegate_value, canonical_value, &format!("{path}.value"))
        },
        (DataType::Map(delegate, delegate_sorted), DataType::Map(canonical, canonical_sorted)) => {
            if delegate_sorted != canonical_sorted {
                return Err(format!(
                    "{path} map ordering differs: delegate={delegate_sorted} canonical={canonical_sorted}"
                ));
            }
            validate_canonical_field_compatibility(delegate, canonical, &format!("{path}.entries"))
        },
        (
            DataType::RunEndEncoded(delegate_run_ends, delegate_values),
            DataType::RunEndEncoded(canonical_run_ends, canonical_values),
        ) => {
            validate_canonical_field_compatibility(delegate_run_ends, canonical_run_ends, &format!("{path}.run_ends"))?;
            validate_canonical_field_compatibility(delegate_values, canonical_values, &format!("{path}.values"))
        },
        _ if delegate == canonical => Ok(()),
        _ => Err(format!("{path} data type differs: delegate={delegate} canonical={canonical}")),
    }
}

fn map_cpu_delegate_error(error: DataFusionError, snapshot_id: Option<&str>) -> DataFusionError {
    translate_iceberg_cpu_delegate_error(
        error,
        snapshot_id,
        EXECUTE_OPERATION,
        "Iceberg CPU scan failed while reading the resolved table snapshot",
    )
}

pub(crate) fn translate_iceberg_cpu_delegate_error(
    error: DataFusionError,
    snapshot_id: Option<&str>,
    operation: &'static str,
    message: &'static str,
) -> DataFusionError {
    if error.find_algeon_error().is_some() {
        return error;
    }

    #[cfg(feature = "iceberg")]
    if let Some(source) = find_iceberg_error(&error) {
        let mut adapter =
            Error::new(crate::table_format::iceberg::catalog::classify_iceberg_scan_error(source), operation, message)
                .with_fact("table_format", TableFormat::Iceberg.code())
                .with_fact("cpu_delegate", true);
        if let Some(snapshot_id) = snapshot_id {
            adapter = adapter.with_fact("snapshot_id", snapshot_id);
        }
        adapter = crate::table_format::iceberg::catalog::apply_iceberg_catalog_error_facts(adapter, source);
        return DataFusionError::from(adapter.with_source(error));
    }

    #[cfg(feature = "iceberg")]
    if let Some(source) = find_source_error::<OpendalError>(&error) {
        let mut adapter =
            Error::new(crate::table_format::iceberg::catalog::classify_opendal_scan_error(source), operation, message)
                .with_fact("table_format", TableFormat::Iceberg.code())
                .with_fact("cpu_delegate", true);
        if let Some(snapshot_id) = snapshot_id {
            adapter = adapter.with_fact("snapshot_id", snapshot_id);
        }
        adapter = crate::table_format::iceberg::catalog::apply_opendal_scan_error_facts(adapter, source);
        return DataFusionError::from(adapter.with_source(error));
    }

    #[cfg(feature = "iceberg")]
    if let Some(source) = find_source_error::<ObjectStoreError>(&error) {
        let mut adapter = Error::new(classify_object_store_error(source), operation, message)
            .with_fact("table_format", TableFormat::Iceberg.code())
            .with_fact("cpu_delegate", true)
            .with_fact("object_store_error_kind", object_store_error_kind(source));
        if let Some(snapshot_id) = snapshot_id {
            adapter = adapter.with_fact("snapshot_id", snapshot_id);
        }
        return DataFusionError::from(adapter.with_source(error));
    }

    if let Some(source) = find_io_error(&error) {
        let mut adapter = Error::new(classify_io_error(source), operation, message)
            .with_fact("table_format", TableFormat::Iceberg.code())
            .with_fact("cpu_delegate", true)
            .with_fact("io_error_kind", format!("{:?}", source.kind()));
        if let Some(snapshot_id) = snapshot_id {
            adapter = adapter.with_fact("snapshot_id", snapshot_id);
        }
        return DataFusionError::from(adapter.with_source(error));
    }

    let mut adapter = Error::new(classify_datafusion_delegate_error(&error), operation, message)
        .with_fact("table_format", TableFormat::Iceberg.code())
        .with_fact("cpu_delegate", true);
    if let Some(snapshot_id) = snapshot_id {
        adapter = adapter.with_fact("snapshot_id", snapshot_id);
    }
    DataFusionError::from(adapter.with_source(error))
}

fn classify_datafusion_delegate_error(error: &DataFusionError) -> ErrorCode {
    match error {
        DataFusionError::ResourcesExhausted(_) => ErrorCode::MemoryReservationFailed,
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => {
            classify_datafusion_delegate_error(source)
        },
        DataFusionError::Collection(errors) => {
            errors.first().map(classify_datafusion_delegate_error).unwrap_or(ErrorCode::SourceDependencyFailure)
        },
        DataFusionError::Shared(source) => classify_datafusion_delegate_error(source),
        DataFusionError::ExecutionJoin(source) if source.is_cancelled() => ErrorCode::Cancelled,
        _ => ErrorCode::SourceDependencyFailure,
    }
}

fn classify_io_error(source: &std::io::Error) -> ErrorCode {
    use std::io::ErrorKind;

    match source.kind() {
        ErrorKind::NotFound => ErrorCode::SourceObjectNotFound,
        ErrorKind::PermissionDenied => ErrorCode::SourceAccessDenied,
        ErrorKind::OutOfMemory => ErrorCode::MemoryReservationFailed,
        ErrorKind::ConnectionRefused
        | ErrorKind::ConnectionReset
        | ErrorKind::ConnectionAborted
        | ErrorKind::NotConnected
        | ErrorKind::BrokenPipe
        | ErrorKind::WouldBlock
        | ErrorKind::TimedOut
        | ErrorKind::Interrupted => ErrorCode::SourceTemporaryFailure,
        _ => ErrorCode::SourceDependencyFailure,
    }
}

fn find_io_error<'a>(mut error: &'a (dyn std::error::Error + 'static)) -> Option<&'a std::io::Error> {
    for _ in 0..32 {
        if let Some(error) = error.downcast_ref::<std::io::Error>() {
            return Some(error);
        }
        error = error.source()?;
    }
    None
}

#[cfg(feature = "iceberg")]
fn classify_object_store_error(source: &ObjectStoreError) -> ErrorCode {
    match source {
        ObjectStoreError::NotFound { .. } => ErrorCode::SourceObjectNotFound,
        ObjectStoreError::PermissionDenied { .. } => ErrorCode::SourceAccessDenied,
        ObjectStoreError::Unauthenticated { .. } => ErrorCode::SourceUnauthenticated,
        _ => ErrorCode::SourceDependencyFailure,
    }
}

#[cfg(feature = "iceberg")]
fn object_store_error_kind(source: &ObjectStoreError) -> &'static str {
    match source {
        ObjectStoreError::NotFound { .. } => "not_found",
        ObjectStoreError::PermissionDenied { .. } => "permission_denied",
        ObjectStoreError::Unauthenticated { .. } => "unauthenticated",
        ObjectStoreError::Generic { .. } => "generic",
        ObjectStoreError::InvalidPath { .. } => "invalid_path",
        ObjectStoreError::JoinError { .. } => "join_error",
        ObjectStoreError::NotSupported { .. } => "not_supported",
        ObjectStoreError::AlreadyExists { .. } => "already_exists",
        ObjectStoreError::Precondition { .. } => "precondition",
        ObjectStoreError::NotModified { .. } => "not_modified",
        ObjectStoreError::NotImplemented { .. } => "not_implemented",
        ObjectStoreError::UnknownConfigurationKey { .. } => "unknown_configuration_key",
        _ => "unknown",
    }
}

#[cfg(feature = "iceberg")]
fn find_iceberg_error<'a>(mut error: &'a (dyn StdError + 'static)) -> Option<&'a IcebergError> {
    for _ in 0..32 {
        if let Some(error) = error.downcast_ref::<IcebergError>() {
            return Some(error);
        }
        error = error.source()?;
    }
    None
}

#[cfg(feature = "iceberg")]
fn find_source_error<'a, E>(mut error: &'a (dyn StdError + 'static)) -> Option<&'a E>
where E: StdError + 'static {
    for _ in 0..32 {
        if let Some(error) = error.downcast_ref::<E>() {
            return Some(error);
        }
        error = error.source()?;
    }
    None
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow::{
        array::{Float32Array, ListArray},
        buffer::OffsetBuffer,
        compute::SortOptions,
        datatypes::{DataType, Field, Schema},
    };
    use datafusion::{
        physical_expr::{Partitioning, PhysicalSortExpr, expressions::Column},
        physical_plan::{
            collect,
            empty::EmptyExec,
            execution_plan::{Boundedness, EmissionType},
        },
    };
    use futures::{StreamExt, stream::iter};

    use super::*;
    use crate::error::ErrorStatus;

    #[derive(Debug, Clone, Copy)]
    enum DelegateBehavior {
        Success,
        ImmediateDataFusionError,
        StreamDataFusionError,
        ImmediateAlgeonError,
        StreamAlgeonError,
        ImmediateResourceError,
        #[cfg(feature = "iceberg")]
        ImmediateOpenDalUnauthenticated,
        #[cfg(feature = "iceberg")]
        ImmediateOpenDalPermissionDenied,
        #[cfg(feature = "iceberg")]
        StreamOpenDalNotFound,
        #[cfg(feature = "iceberg")]
        StreamOpenDalTemporary,
    }

    #[derive(Debug)]
    struct TestDelegateExec {
        inner: EmptyExec,
        behavior: DelegateBehavior,
        properties: Option<Arc<PlanProperties>>,
    }

    impl TestDelegateExec {
        fn new(behavior: DelegateBehavior) -> Self {
            Self { inner: EmptyExec::new(Arc::new(Schema::empty())), behavior, properties: None }
        }
    }

    impl DisplayAs for TestDelegateExec {
        fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.inner.fmt_as(t, f)
        }
    }

    impl ExecutionPlan for TestDelegateExec {
        fn name(&self) -> &str {
            "TestDelegateExec"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.properties.as_ref().unwrap_or_else(|| self.inner.properties())
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DataFusionResult<TreeNodeRecursion>,
        ) -> DataFusionResult<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            Vec::new()
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
            if children.is_empty() {
                Ok(self)
            } else {
                Err(DataFusionError::Internal("test delegate does not accept children".to_string()))
            }
        }

        fn execute(&self, partition: usize, context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
            match self.behavior {
                DelegateBehavior::Success => self.inner.execute(partition, context),
                DelegateBehavior::ImmediateDataFusionError => {
                    Err(DataFusionError::Execution("immediate delegate failure".to_string()))
                },
                DelegateBehavior::StreamDataFusionError => {
                    let schema = self.schema();
                    let stream = iter([Err(DataFusionError::Execution("stream delegate failure".to_string()))]);
                    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
                },
                DelegateBehavior::ImmediateAlgeonError => {
                    Err(Error::new(ErrorCode::Cancelled, "algeon_datafusion.test.delegate", "typed delegate failure")
                        .into())
                },
                DelegateBehavior::StreamAlgeonError => {
                    let schema = self.schema();
                    let error = Error::new(
                        ErrorCode::Cancelled,
                        "algeon_datafusion.test.delegate_stream",
                        "typed delegate stream failure",
                    );
                    let stream = iter([Err(error.into())]);
                    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
                },
                DelegateBehavior::ImmediateResourceError => {
                    Err(DataFusionError::ResourcesExhausted("delegate memory pool exhausted".to_string()))
                },
                #[cfg(feature = "iceberg")]
                DelegateBehavior::ImmediateOpenDalUnauthenticated => Err(iceberg_delegate_error(
                    OpendalError::new(
                        OpendalErrorKind::PermissionDenied,
                        "S3Error { code: \"InvalidAccessKeyId\", message: \"credential rejected\" }",
                    )
                    .with_context(
                        "response",
                        "Response { status: 403, headers: {\"x-amz-request-id\": \"req-auth-1\"} }",
                    ),
                )),
                #[cfg(feature = "iceberg")]
                DelegateBehavior::ImmediateOpenDalPermissionDenied => {
                    Err(iceberg_delegate_error(OpendalError::new(OpendalErrorKind::PermissionDenied, "access denied")))
                },
                #[cfg(feature = "iceberg")]
                DelegateBehavior::StreamOpenDalNotFound => {
                    let schema = self.schema();
                    let error = iceberg_delegate_error(OpendalError::new(OpendalErrorKind::NotFound, "object missing"));
                    let stream = iter([Err(error)]);
                    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
                },
                #[cfg(feature = "iceberg")]
                DelegateBehavior::StreamOpenDalTemporary => {
                    let schema = self.schema();
                    let error = iceberg_delegate_error(
                        OpendalError::new(OpendalErrorKind::Unexpected, "transient transport failure").set_temporary(),
                    );
                    let stream = iter([Err(error)]);
                    Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
                },
            }
        }
    }

    #[cfg(feature = "iceberg")]
    fn iceberg_delegate_error(source: OpendalError) -> DataFusionError {
        DataFusionError::External(Box::new(
            IcebergError::new(IcebergErrorKind::Unexpected, "Iceberg object read failed").with_source(source),
        ))
    }

    fn test_scan(behavior: DelegateBehavior) -> AlgeonIcebergScanExec {
        let delegate: Arc<dyn ExecutionPlan> = Arc::new(TestDelegateExec::new(behavior));
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg).with_snapshot_id("snapshot-42");
        let schema = delegate.schema();
        AlgeonIcebergScanExec::try_new(delegate, schema, AlgeonIcebergNativeScanOutcome::from_scan_plan(scan_plan))
            .unwrap()
    }

    #[test]
    fn immediate_delegate_error_is_translated_to_typed_source_failure() {
        let error =
            match test_scan(DelegateBehavior::ImmediateDataFusionError).execute(0, Arc::new(TaskContext::default())) {
                Err(error) => error,
                Ok(_) => panic!("delegate must fail immediately"),
            };
        let algeon = error.find_algeon_error().expect("wrapper must attach a typed Algeon error");

        assert_eq!(algeon.code(), ErrorCode::SourceDependencyFailure);
        assert_eq!(algeon.operation(), EXECUTE_OPERATION);
        assert_eq!(algeon.fact_value("table_format"), Some("iceberg"));
        assert_eq!(algeon.fact_value("cpu_delegate"), Some("true"));
        assert_eq!(algeon.fact_value("snapshot_id"), Some("snapshot-42"));
    }

    #[tokio::test]
    async fn stream_delegate_error_is_translated_to_typed_source_failure() {
        let mut stream = test_scan(DelegateBehavior::StreamDataFusionError)
            .execute(0, Arc::new(TaskContext::default()))
            .expect("delegate stream must be created");
        let error = stream.next().await.expect("stream must yield its failure").expect_err("stream item must fail");
        let algeon = error.find_algeon_error().expect("wrapper must attach a typed Algeon error");

        assert_eq!(algeon.code(), ErrorCode::SourceDependencyFailure);
        assert_eq!(algeon.operation(), EXECUTE_OPERATION);
        assert_eq!(algeon.fact_value("snapshot_id"), Some("snapshot-42"));
    }

    #[test]
    fn existing_algeon_delegate_error_is_preserved() {
        let error = match test_scan(DelegateBehavior::ImmediateAlgeonError).execute(0, Arc::new(TaskContext::default()))
        {
            Err(error) => error,
            Ok(_) => panic!("delegate must fail immediately"),
        };
        let algeon = error.find_algeon_error().expect("typed delegate error must remain discoverable");

        assert_eq!(algeon.code(), ErrorCode::Cancelled);
        assert_eq!(algeon.operation(), "algeon_datafusion.test.delegate");
        assert_eq!(algeon.fact_value("cpu_delegate"), None);
    }

    #[tokio::test]
    async fn existing_algeon_delegate_stream_error_is_preserved() {
        let mut stream = test_scan(DelegateBehavior::StreamAlgeonError)
            .execute(0, Arc::new(TaskContext::default()))
            .expect("delegate stream must be created");
        let error = stream.next().await.expect("stream must yield its failure").expect_err("stream item must fail");
        let algeon = error.find_algeon_error().expect("typed delegate error must remain discoverable");

        assert_eq!(algeon.code(), ErrorCode::Cancelled);
        assert_eq!(algeon.operation(), "algeon_datafusion.test.delegate_stream");
        assert_eq!(algeon.fact_value("cpu_delegate"), None);
    }

    #[test]
    fn immediate_resource_error_preserves_temporary_resource_classification() {
        let error =
            match test_scan(DelegateBehavior::ImmediateResourceError).execute(0, Arc::new(TaskContext::default())) {
                Err(error) => error,
                Ok(_) => panic!("delegate must fail immediately"),
            };
        let algeon = error.find_algeon_error().expect("wrapper must attach a typed Algeon error");

        assert_eq!(algeon.code(), ErrorCode::MemoryReservationFailed);
        assert_eq!(algeon.status(), ErrorStatus::Temporary);
    }

    #[cfg(feature = "iceberg")]
    #[test]
    fn immediate_iceberg_opendal_auth_error_preserves_unauthenticated_classification() {
        let error = match test_scan(DelegateBehavior::ImmediateOpenDalUnauthenticated)
            .execute(0, Arc::new(TaskContext::default()))
        {
            Err(error) => error,
            Ok(_) => panic!("delegate must fail immediately"),
        };
        let algeon = error.find_algeon_error().expect("wrapper must attach a typed Algeon error");

        assert_eq!(algeon.code(), ErrorCode::SourceUnauthenticated);
        assert_eq!(algeon.status(), ErrorStatus::Permanent);
        assert_eq!(algeon.fact_value("opendal_error_kind"), Some("PermissionDenied"));
        assert_eq!(algeon.fact_value("aws_error_code"), Some("InvalidAccessKeyId"));
        assert_eq!(algeon.fact_value("aws_request_id"), Some("req-auth-1"));
    }

    #[cfg(feature = "iceberg")]
    #[test]
    fn immediate_opendal_permission_error_preserves_access_denied_classification() {
        let error = match test_scan(DelegateBehavior::ImmediateOpenDalPermissionDenied)
            .execute(0, Arc::new(TaskContext::default()))
        {
            Err(error) => error,
            Ok(_) => panic!("delegate must fail immediately"),
        };
        let algeon = error.find_algeon_error().expect("wrapper must attach a typed Algeon error");

        assert_eq!(algeon.code(), ErrorCode::SourceAccessDenied);
        assert_eq!(algeon.status(), ErrorStatus::Permanent);
        assert_eq!(algeon.fact_value("opendal_error_kind"), Some("PermissionDenied"));
    }

    #[cfg(feature = "iceberg")]
    #[tokio::test]
    async fn stream_opendal_not_found_error_preserves_not_found_classification() {
        let mut stream = test_scan(DelegateBehavior::StreamOpenDalNotFound)
            .execute(0, Arc::new(TaskContext::default()))
            .expect("delegate stream must be created");
        let error = stream.next().await.expect("stream must yield its failure").expect_err("stream item must fail");
        let algeon = error.find_algeon_error().expect("wrapper must attach a typed Algeon error");

        assert_eq!(algeon.code(), ErrorCode::SourceObjectNotFound);
        assert_eq!(algeon.status(), ErrorStatus::Permanent);
        assert_eq!(algeon.fact_value("opendal_error_kind"), Some("NotFound"));
    }

    #[cfg(feature = "iceberg")]
    #[tokio::test]
    async fn stream_opendal_temporary_error_preserves_retryability() {
        let mut stream = test_scan(DelegateBehavior::StreamOpenDalTemporary)
            .execute(0, Arc::new(TaskContext::default()))
            .expect("delegate stream must be created");
        let error = stream.next().await.expect("stream must yield its failure").expect_err("stream item must fail");
        let algeon = error.find_algeon_error().expect("wrapper must attach a typed Algeon error");

        assert_eq!(algeon.code(), ErrorCode::SourceTemporaryFailure);
        assert_eq!(algeon.status(), ErrorStatus::Temporary);
        assert_eq!(algeon.fact_value("opendal_temporary"), Some("true"));
    }

    #[tokio::test]
    async fn successful_cpu_delegate_executes_through_wrapper() {
        let scan: Arc<dyn ExecutionPlan> = Arc::new(test_scan(DelegateBehavior::Success));
        let batches =
            collect(scan, Arc::new(TaskContext::default())).await.expect("successful delegate must remain executable");

        assert!(batches.is_empty());
    }

    #[test]
    fn wrapper_properties_use_canonical_schema_without_changing_delegate_execution_properties() {
        let delegate_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false)
                .with_metadata(HashMap::from([("PARQUET:field_id".to_string(), "1".to_string())])),
        ]));
        let canonical_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let delegate: Arc<dyn ExecutionPlan> = Arc::new(EmptyExec::new(delegate_schema));
        let scan_plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&canonical_schema))
            .with_native_source_schema(Arc::clone(&canonical_schema));
        let scan = AlgeonIcebergScanExec::try_new(
            Arc::clone(&delegate),
            Arc::clone(&canonical_schema),
            AlgeonIcebergNativeScanOutcome::from_scan_plan(scan_plan),
        )
        .unwrap();

        assert_eq!(scan.schema().as_ref(), canonical_schema.as_ref());
        assert_eq!(scan.properties().equivalence_properties().schema().as_ref(), canonical_schema.as_ref());
        assert_eq!(
            scan.properties().output_partitioning().partition_count(),
            delegate.properties().output_partitioning().partition_count(),
        );
        assert_eq!(scan.properties().boundedness, delegate.properties().boundedness);
        assert_eq!(scan.properties().emission_type, delegate.properties().emission_type);
        assert_eq!(scan.properties().evaluation_type, delegate.properties().evaluation_type);
        assert_eq!(scan.properties().scheduling_type, delegate.properties().scheduling_type);
    }

    #[test]
    fn canonical_batch_adapter_preserves_zero_column_row_count() {
        let delegate_schema = Arc::new(Schema::empty());
        let batch = RecordBatch::try_new_with_options(
            delegate_schema,
            Vec::new(),
            &RecordBatchOptions::new().with_row_count(Some(7)),
        )
        .unwrap();
        let canonical_schema = Arc::new(Schema::empty());

        let batch = canonicalize_cpu_batch(batch, Arc::clone(&canonical_schema), Some("snapshot-42")).unwrap();

        assert_eq!(batch.num_rows(), 7);
        assert_eq!(batch.num_columns(), 0);
        assert_eq!(batch.schema().as_ref(), canonical_schema.as_ref());
    }

    #[test]
    fn canonical_batch_adapter_retypes_nested_field_metadata() {
        let delegate_element = Arc::new(
            Field::new("element", DataType::Float32, false)
                .with_metadata(HashMap::from([("PARQUET:field_id".to_string(), "2".to_string())])),
        );
        let delegate_schema = Arc::new(Schema::new(vec![
            Field::new("embedding", DataType::List(Arc::clone(&delegate_element)), false)
                .with_metadata(HashMap::from([("PARQUET:field_id".to_string(), "1".to_string())])),
        ]));
        let canonical_element = Arc::new(Field::new("element", DataType::Float32, false));
        let canonical_schema =
            Arc::new(Schema::new(vec![Field::new("embedding", DataType::List(Arc::clone(&canonical_element)), false)]));
        let batch = RecordBatch::try_new(
            delegate_schema,
            vec![Arc::new(ListArray::new(
                delegate_element,
                OffsetBuffer::new(vec![0_i32, 2].into()),
                Arc::new(Float32Array::from(vec![1.0_f32, 2.0])),
                None,
            ))],
        )
        .unwrap();

        let batch = canonicalize_cpu_batch(batch, Arc::clone(&canonical_schema), Some("snapshot-42")).unwrap();

        assert_eq!(batch.schema().as_ref(), canonical_schema.as_ref());
        assert_eq!(batch.column(0).data_type(), canonical_schema.field(0).data_type());
    }

    #[test]
    fn canonical_schema_compatibility_ignores_only_recursive_metadata() {
        let delegate_child = Arc::new(
            Field::new("element", DataType::Utf8, false)
                .with_metadata(HashMap::from([("PARQUET:field_id".to_string(), "2".to_string())])),
        );
        let canonical_child = Arc::new(Field::new("element", DataType::Utf8, false));
        let delegate = Schema::new(vec![
            Field::new("names", DataType::List(delegate_child), false)
                .with_metadata(HashMap::from([("PARQUET:field_id".to_string(), "1".to_string())])),
        ]);
        let canonical = Schema::new(vec![Field::new("names", DataType::List(canonical_child), false)]);

        validate_canonical_schema_compatibility(&delegate, &canonical).unwrap();

        let renamed = Schema::new(vec![Field::new(
            "names",
            DataType::List(Arc::new(Field::new("item", DataType::Utf8, false))),
            false,
        )]);
        assert!(validate_canonical_schema_compatibility(&delegate, &renamed).unwrap_err().contains("name differs"));

        let retyped = Schema::new(vec![Field::new(
            "names",
            DataType::List(Arc::new(Field::new("element", DataType::Int64, false))),
            false,
        )]);
        assert!(
            validate_canonical_schema_compatibility(&delegate, &retyped).unwrap_err().contains("data type differs")
        );
    }

    #[test]
    fn nested_metadata_fallback_drops_unprovable_ordering() {
        let delegate_schema = Arc::new(Schema::new(vec![Field::new(
            "names",
            DataType::List(Arc::new(
                Field::new("element", DataType::Utf8, false)
                    .with_metadata(HashMap::from([("PARQUET:field_id".to_string(), "2".to_string())])),
            )),
            false,
        )]));
        let canonical_schema = Arc::new(Schema::new(vec![Field::new(
            "names",
            DataType::List(Arc::new(Field::new("element", DataType::Utf8, false))),
            false,
        )]));
        let ordering = PhysicalSortExpr::new(Arc::new(Column::new("names", 0)), SortOptions::default());
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new_with_orderings(Arc::clone(&delegate_schema), [[ordering]]),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        let delegate: Arc<dyn ExecutionPlan> = Arc::new(TestDelegateExec {
            inner: EmptyExec::new(delegate_schema),
            behavior: DelegateBehavior::Success,
            properties: Some(properties),
        });
        assert!(delegate.properties().output_ordering().is_some());

        let scan = AlgeonIcebergScanExec::try_new(
            delegate,
            canonical_schema,
            AlgeonIcebergNativeScanOutcome::from_scan_plan(TableFormatScanPlan::new(TableFormat::Iceberg)),
        )
        .unwrap();

        assert!(scan.properties().output_ordering().is_none());
    }
}
