//! Iceberg metadata planning adapter for table-format scan facts.
//!
//! This module consumes `iceberg-rust` planning output and preserves table
//! format facts at the adapter boundary. It does not execute Arrow readers.

use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
    sync::Arc,
    time::Duration,
};

use arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef, UnionFields};
use async_trait::async_trait;
use bytes::BytesMut;
use datafusion::{
    catalog::Session,
    common::{DataFusionError, ScalarValue, project_schema},
    datasource::{TableProvider, TableType},
    logical_expr::{Expr, Operator, TableProviderFilterPushDown},
    physical_plan::ExecutionPlan,
};
use futures::{StreamExt, future::BoxFuture};
use iceberg::{
    expr::Reference,
    io::{FileIO, FileRead},
    metadata_columns::{
        RESERVED_FIELD_ID_DELETE_FILE_POS, RESERVED_FIELD_ID_POS, get_metadata_field_id, is_metadata_column_name,
    },
    scan::{FileScanTask, FileScanTaskDeleteFile},
    spec::{
        DEFAULT_SCHEMA_NAME_MAPPING, DataContentType, DataFile, DataFileFormat, Datum, Literal, Manifest,
        ManifestContentType, ManifestEntry, ManifestFile, ManifestList, NameMapping as IcebergNameMapping, NestedField,
        PartitionSpec, PrimitiveLiteral, Schema as IcebergSchema, SchemaRef as IcebergSchemaRef, SnapshotRef, Struct,
        Transform, Type,
    },
    table::Table,
};
use iceberg_datafusion::IcebergStaticTableProvider;
use nexus_query_engine::datatypes::is_supported_arrow_type;
use parquet::file::{
    FOOTER_SIZE,
    metadata::{FooterTail, ParquetMetaData, ParquetMetaDataReader},
};

use crate::{
    error::{Error, ErrorCode, Result},
    table_format::{
        scan::{
            TableFormat, TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile,
            TableFormatDataFileFormat, TableFormatDeleteFile, TableFormatDeleteFileContent, TableFormatLiteral,
            TableFormatParquetReadStrategyConfig, TableFormatParquetSourceMetadata, TableFormatPartitionTransform,
            TableFormatPartitionValue, TableFormatPlanner, TableFormatPlanningDeadline, TableFormatPredicate,
            TableFormatPredicateOp, TableFormatReadCapability, TableFormatScalarValue, TableFormatScanPlan,
            TableFormatScanRequest, TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode,
            credential_free_endpoint_identity, resolve_table_format_parquet_source_with_metadata,
        },
        scan_exec::{NexusIcebergNativeScanOutcome, NexusIcebergScanExec, translate_iceberg_cpu_delegate_error},
    },
};

const PLAN_SCAN_OPERATION: &str = "datafusion_nexus.table_format.iceberg.plan_scan";
const PLAN_SCAN_TIMEOUT_OPERATION: &str = "datafusion_nexus.table_format.iceberg.plan_scan_timeout";
const BUILD_CPU_DELEGATE_OPERATION: &str = "datafusion_nexus.table_format.iceberg.build_cpu_delegate";
const TASK_CONVERSION_OPERATION: &str = "datafusion_nexus.table_format.iceberg.convert_file_scan_tasks";
// Manifest planning is remote-IO bound; this keeps large snapshots moving without unbounded fanout.
const REMOTE_ICEBERG_METADATA_PLANNING_CONCURRENCY: usize = 128;
// Footer probes fetch larger Parquet tails, so they use a tighter transient-buffer gate.
const REMOTE_FOOTER_FIELD_ID_PROOF_CONCURRENCY: usize = 32;
mod footer;
mod predicate;
mod schema;
mod tasks;

use footer::*;
use predicate::*;
pub use schema::iceberg_schema_to_datafusion_arrow_schema;
use schema::*;
pub use tasks::iceberg_file_scan_tasks_to_table_format_scan_plan;
use tasks::*;

type SharedTableFormatPlanner = Arc<dyn TableFormatPlanner + Send + Sync>;

/// DataFusion table provider that exposes one Iceberg scan through both an
/// executable upstream CPU delegate and optional native scan facts.
///
/// The provider treats DataFusion filters as inexact pushdown: DataFusion keeps
/// a parent filter for correctness, while native lowering can still see the
/// filter context and the scan can retain DataFusion's source-level limit
/// bound.
#[derive(Clone)]
pub struct NexusIcebergTableProvider {
    schema: SchemaRef,
    cpu_source: NexusIcebergCpuSource,
    resolved_snapshot_id: Option<i64>,
    planner: SharedTableFormatPlanner,
    catalog_type: Option<String>,
    scan_planning_timeout: Option<Duration>,
}

#[derive(Clone)]
enum NexusIcebergCpuSource {
    ResolvedTable(Table),
    InjectedProvider(Arc<dyn TableProvider>),
}

impl NexusIcebergTableProvider {
    /// Builds a provider from an `iceberg-rust` table handle.
    ///
    /// The table schema is converted eagerly; file planning remains lazy and
    /// happens in `TableProvider::scan`.
    pub fn try_new(table: Table) -> Result<Self> {
        Self::try_new_with_optional_parquet_read_strategy_config(table, None)
    }

    /// Builds a provider from an `iceberg-rust` table handle with an explicit
    /// Parquet read-strategy config.
    pub fn try_new_with_parquet_read_strategy_config(
        table: Table,
        config: TableFormatParquetReadStrategyConfig,
    ) -> Result<Self> {
        Self::try_new_with_optional_parquet_read_strategy_config(table, Some(config))
    }

    fn try_new_with_optional_parquet_read_strategy_config(
        table: Table,
        config: Option<TableFormatParquetReadStrategyConfig>,
    ) -> Result<Self> {
        let resolved_snapshot_id = table.metadata().current_snapshot_id();
        let iceberg_schema = match resolved_snapshot_id {
            Some(snapshot_id) => table
                .metadata()
                .snapshot_by_id(snapshot_id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        PLAN_SCAN_OPERATION,
                        "resolved Iceberg snapshot is missing from table metadata",
                    )
                    .with_fact("table_format", TableFormat::Iceberg.code())
                    .with_fact("snapshot_id", snapshot_id)
                })?
                .schema(table.metadata())
                .map_err(|error| iceberg_planning_error("failed to resolve Iceberg snapshot schema", error))?,
            None => table.current_schema_ref().clone(),
        };
        let schema = iceberg_schema_to_datafusion_arrow_schema(iceberg_schema.as_ref())?;
        let planner = match config {
            Some(config) => IcebergTableFormatPlanner::new(table.clone()).with_parquet_read_strategy_config(config),
            None => IcebergTableFormatPlanner::new(table.clone()),
        };
        Ok(Self {
            schema,
            cpu_source: NexusIcebergCpuSource::ResolvedTable(table),
            resolved_snapshot_id,
            planner: Arc::new(planner),
            catalog_type: None,
            scan_planning_timeout: None,
        })
    }

    /// Builds a provider from an executable CPU provider and a native metadata planner.
    pub fn from_cpu_provider_and_native_planner(
        cpu_provider: Arc<dyn TableProvider>,
        resolved_snapshot_id: Option<i64>,
        planner: Arc<dyn TableFormatPlanner + Send + Sync>,
    ) -> Self {
        let schema = Arc::new(strip_arrow_schema_metadata(cpu_provider.schema().as_ref().clone()));
        Self {
            schema,
            cpu_source: NexusIcebergCpuSource::InjectedProvider(cpu_provider),
            resolved_snapshot_id,
            planner,
            catalog_type: None,
            scan_planning_timeout: None,
        }
    }

    /// Records a stable catalog label in scan requests and diagnostics.
    pub fn with_catalog_type(mut self, catalog_type: impl Into<String>) -> Self {
        self.catalog_type = Some(catalog_type.into());
        self
    }

    /// Applies a wall-clock budget to metadata scan planning.
    ///
    /// `None` preserves unbounded library behavior. The Flight SQL server uses
    /// this to keep one pathological Iceberg table from monopolizing the
    /// statement-admission slot.
    pub fn with_scan_planning_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.scan_planning_timeout = timeout;
        self
    }
}

impl fmt::Debug for NexusIcebergTableProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NexusIcebergTableProvider")
            .field("schema", &self.schema)
            .field("resolved_snapshot_id", &self.resolved_snapshot_id)
            .field("catalog_type", &self.catalog_type)
            .field("scan_planning_timeout", &self.scan_planning_timeout)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for NexusIcebergTableProvider {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let snapshot_id = self.resolved_snapshot_id.map(|snapshot_id| snapshot_id.to_string());
        let cpu_provider: Arc<dyn TableProvider> = match &self.cpu_source {
            NexusIcebergCpuSource::ResolvedTable(table) => {
                let provider = match self.resolved_snapshot_id {
                    Some(snapshot_id) => {
                        IcebergStaticTableProvider::try_new_from_table_snapshot(table.clone(), snapshot_id).await
                    },
                    None => IcebergStaticTableProvider::try_new_from_table(table.clone()).await,
                }
                .map_err(|error| {
                    translate_iceberg_cpu_delegate_error(
                        DataFusionError::External(Box::new(error)),
                        snapshot_id.as_deref(),
                        BUILD_CPU_DELEGATE_OPERATION,
                        "failed to build the executable Iceberg CPU scan provider",
                    )
                })?;
                Arc::new(provider)
            },
            NexusIcebergCpuSource::InjectedProvider(provider) => Arc::clone(provider),
        };
        let cpu_delegate = cpu_provider.scan(state, projection, filters, limit).await.map_err(|error| {
            translate_iceberg_cpu_delegate_error(
                error,
                snapshot_id.as_deref(),
                BUILD_CPU_DELEGATE_OPERATION,
                "failed to build the executable Iceberg CPU scan delegate",
            )
        })?;
        let canonical_schema = project_schema(&self.schema, projection)?;
        let projected_column_names = projection
            .map(|indices| indices.iter().map(|index| self.schema.field(*index).name().clone()).collect::<Vec<_>>());
        let predicate = datafusion_filters_to_table_format_predicate(filters);
        let mut request = TableFormatScanRequest::default()
            .with_case_sensitive(true)
            .with_scan_limit(limit)
            .with_predicate(predicate);
        request.projected_column_names = projected_column_names;
        request.snapshot_id = snapshot_id;
        if let Some(catalog_type) = self.catalog_type.as_ref() {
            request.catalog_type = Some(catalog_type.clone());
        }
        let resolved_scan =
            resolve_scan_with_optional_timeout(Arc::clone(&self.planner), request.clone(), self.scan_planning_timeout)
                .await
                .map_err(DataFusionError::from)?;
        let resolved_rejection = resolved_scan.first_rejection_reason();
        let native_outcome = match inspect_native_scan_with_optional_timeout(
            Arc::clone(&self.planner),
            request,
            resolved_scan.clone(),
            self.scan_planning_timeout,
        )
        .await
        {
            // Native inspection may trim a predicate-free limited scan. That
            // can remove a file-local rejection, so classify the inspected
            // facts rather than preserving the pre-inspection rejection.
            Ok(scan_plan) => NexusIcebergNativeScanOutcome::from_scan_plan(scan_plan),
            // Inspection is optional acceleration work. A known resolved
            // capability miss remains more precise than the inspection
            // failure; only otherwise-admissible source facts report the
            // inspection failure as native evidence.
            Err(_error) if resolved_rejection.is_some() => NexusIcebergNativeScanOutcome::from_scan_plan(resolved_scan),
            Err(error) => NexusIcebergNativeScanOutcome::InspectionFailed { resolved_scan, error: Arc::new(error) },
        };
        Ok(Arc::new(NexusIcebergScanExec::try_new(cpu_delegate, canonical_schema, native_outcome)?))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }
}

async fn resolve_scan_with_optional_timeout(
    planner: SharedTableFormatPlanner,
    request: TableFormatScanRequest,
    timeout: Option<Duration>,
) -> Result<TableFormatScanPlan> {
    let Some(timeout) = timeout else {
        return planner.resolve_scan(request).await;
    };
    let deadline = TableFormatPlanningDeadline::from_timeout(timeout);
    let timeout_ms = deadline.timeout_ms();
    let catalog_type = request.catalog_type.clone();
    let snapshot_id = request.snapshot_id.clone();
    let planning_failure_facts = planner.planning_failure_facts();

    let mut planning_task = tokio::spawn(async move { planner.resolve_scan_with_deadline(request, deadline).await });
    tokio::select! {
        biased;

        result = &mut planning_task => match result {
            Ok(result) => result,
            Err(error) => Err(iceberg_planning_task_error(error)),
        },
        _ = tokio::time::sleep(timeout) => {
            planning_task.abort();
            Err(iceberg_scan_planning_timeout_error(
                timeout_ms,
                catalog_type,
                snapshot_id,
                planning_failure_facts,
                "source_resolution_provider_timeout",
            ))
        },
    }
}

async fn inspect_native_scan_with_optional_timeout(
    planner: SharedTableFormatPlanner,
    request: TableFormatScanRequest,
    resolved_scan: TableFormatScanPlan,
    timeout: Option<Duration>,
) -> Result<TableFormatScanPlan> {
    let Some(timeout) = timeout else {
        return planner.inspect_native_scan(request, resolved_scan).await;
    };
    let deadline = TableFormatPlanningDeadline::from_timeout(timeout);
    let timeout_ms = deadline.timeout_ms();
    let catalog_type = request.catalog_type.clone();
    let snapshot_id = request.snapshot_id.clone();
    let planning_failure_facts = planner.planning_failure_facts();

    let mut planning_task =
        tokio::spawn(async move { planner.inspect_native_scan_with_deadline(request, resolved_scan, deadline).await });
    tokio::select! {
        biased;

        result = &mut planning_task => match result {
            Ok(result) => result,
            Err(error) => Err(iceberg_planning_task_error(error)),
        },
        _ = tokio::time::sleep(timeout) => {
            planning_task.abort();
            Err(iceberg_scan_planning_timeout_error(
                timeout_ms,
                catalog_type,
                snapshot_id,
                planning_failure_facts,
                "native_inspection_provider_timeout",
            ))
        },
    }
}

fn iceberg_planning_task_error(error: tokio::task::JoinError) -> Error {
    let (code, join_status) =
        if error.is_cancelled() { (ErrorCode::Cancelled, "cancelled") } else { (ErrorCode::Internal, "panic") };
    Error::new(code, PLAN_SCAN_TIMEOUT_OPERATION, "Iceberg table-format scan planning task failed")
        .with_fact("table_format", TableFormat::Iceberg.code())
        .with_fact("planning_task_join_status", join_status)
        .with_source(error)
}

fn iceberg_scan_planning_timeout_error(
    timeout_ms: u128,
    catalog_type: Option<String>,
    snapshot_id: Option<String>,
    planning_failure_facts: Vec<(&'static str, String)>,
    stage: &'static str,
) -> Error {
    let mut error = Error::new(
        ErrorCode::PlanningTimeout,
        PLAN_SCAN_TIMEOUT_OPERATION,
        "Iceberg table-format scan planning exceeded the configured time budget",
    )
    .with_fact("table_format", TableFormat::Iceberg.code())
    .with_fact("planning_timeout", true)
    .with_fact("planning_timeout_ms", timeout_ms)
    .with_fact("planning_timeout_stage", stage)
    .with_fact("timeout_retryable", false)
    .with_fact("timeout_budget_increase_required", true);
    if let Some(catalog_type) = catalog_type {
        error = error.with_fact("catalog_type", catalog_type);
    }
    if let Some(snapshot_id) = snapshot_id {
        error = error.with_fact("snapshot_id", snapshot_id);
    }
    for (key, value) in planning_failure_facts {
        if error.fact_value(key).is_none() {
            error = error.with_fact(key, value);
        }
    }
    error
}

#[derive(Debug, Clone)]
struct IcebergScanPlanningTimeoutContext {
    deadline: TableFormatPlanningDeadline,
    catalog_type: Option<String>,
    snapshot_id: Option<String>,
    planning_failure_facts: Vec<(&'static str, String)>,
}

impl IcebergScanPlanningTimeoutContext {
    fn new(
        deadline: TableFormatPlanningDeadline,
        request: &TableFormatScanRequest,
        planning_failure_facts: Vec<(&'static str, String)>,
    ) -> Self {
        Self {
            deadline,
            catalog_type: request.catalog_type.clone(),
            snapshot_id: request.snapshot_id.clone(),
            planning_failure_facts,
        }
    }

    fn check(&self, stage: &'static str) -> Result<()> {
        if self.deadline.is_elapsed() { Err(self.error(stage)) } else { Ok(()) }
    }

    fn remaining(&self, stage: &'static str) -> Result<Duration> {
        self.deadline.remaining().ok_or_else(|| self.error(stage))
    }

    fn error(&self, stage: &'static str) -> Error {
        iceberg_scan_planning_timeout_error(
            self.deadline.timeout_ms(),
            self.catalog_type.clone(),
            self.snapshot_id.clone(),
            self.planning_failure_facts.clone(),
            stage,
        )
    }
}

/// Planner that resolves an `iceberg-rust` table scan into adapter-owned scan
/// facts for later native admission.
///
/// Snapshot ids are parsed as Iceberg `i64` ids. `None` projection selects all
/// columns; an empty projection requests `select_empty`. Row-selection support
/// is recorded as an unsupported feature until native scan facts can represent
/// page-index semantics.
#[derive(Debug, Clone)]
pub struct IcebergTableFormatPlanner {
    table: Table,
    catalog_type: Option<String>,
    parquet_read_strategy_config: TableFormatParquetReadStrategyConfig,
}

impl IcebergTableFormatPlanner {
    pub fn new(table: Table) -> Self {
        Self {
            table,
            catalog_type: None,
            parquet_read_strategy_config: TableFormatParquetReadStrategyConfig::from_env(),
        }
    }

    /// Records a default catalog label when the scan request does not provide one.
    pub fn with_catalog_type(mut self, catalog_type: impl Into<String>) -> Self {
        self.catalog_type = Some(catalog_type.into());
        self
    }

    /// Sets the table-format Parquet read-strategy config used while resolving
    /// Iceberg data-file scan facts.
    pub fn with_parquet_read_strategy_config(mut self, config: TableFormatParquetReadStrategyConfig) -> Self {
        self.parquet_read_strategy_config = config;
        self
    }

    /// Resolves Iceberg scan metadata into adapter-owned immutable scan facts.
    ///
    /// The result is metadata-only: data files, delete files, partition values,
    /// projected columns, and unsupported-feature facts. It does not read data
    /// files and must not expose credentials in diagnostics.
    pub async fn resolve_table_format_scan(&self, request: TableFormatScanRequest) -> Result<TableFormatScanPlan> {
        self.resolve_table_format_scan_with_timeout_context(request, None).await
    }

    async fn resolve_table_format_scan_with_timeout_context(
        &self,
        request: TableFormatScanRequest,
        timeout_context: Option<IcebergScanPlanningTimeoutContext>,
    ) -> Result<TableFormatScanPlan> {
        if let Some(context) = timeout_context.as_ref() {
            context.check("start")?;
        }
        let row_selection_enabled = request.row_selection_enabled;
        let mut scan_builder = self
            .table
            .scan()
            .with_case_sensitive(request.case_sensitive)
            .with_concurrency_limit(REMOTE_ICEBERG_METADATA_PLANNING_CONCURRENCY)
            .with_row_group_filtering_enabled(request.row_group_filtering_enabled)
            .with_row_selection_enabled(request.row_selection_enabled);

        let parsed_snapshot_id = request
            .snapshot_id
            .as_deref()
            .map(|snapshot_id| {
                snapshot_id.parse::<i64>().map_err(|error| {
                    Error::new(
                        ErrorCode::InvalidExecutionPlan,
                        PLAN_SCAN_OPERATION,
                        "invalid Iceberg snapshot id in table-format scan request",
                    )
                    .with_fact("table_format", TableFormat::Iceberg.code())
                    .with_fact("snapshot_id", snapshot_id)
                    .with_source(error)
                })
            })
            .transpose()?;
        if let Some(snapshot_id) = parsed_snapshot_id {
            scan_builder = scan_builder.snapshot_id(snapshot_id);
        }

        scan_builder = match request.projected_column_names.as_ref() {
            Some(column_names) if column_names.is_empty() => scan_builder.select_empty(),
            Some(column_names) => scan_builder.select(column_names),
            None => scan_builder.select_all(),
        };
        let iceberg_predicate = request.predicate.as_ref().and_then(table_format_predicate_to_iceberg_predicate);
        let parent_filter_enforces_iceberg_predicate = iceberg_predicate.is_some();
        if let Some(predicate) = iceberg_predicate {
            scan_builder = scan_builder.with_filter(predicate);
        }

        if let Some(context) = timeout_context.as_ref() {
            context.check("before_build")?;
        }
        let scan = scan_builder
            .build()
            .map_err(|error| iceberg_planning_error("failed to build Iceberg table scan", error))?;
        if let Some(context) = timeout_context.as_ref() {
            context.check("after_build")?;
        }
        let catalog_type = request.catalog_type.clone().or_else(|| self.catalog_type.clone());
        let snapshot_id = request
            .snapshot_id
            .clone()
            .or_else(|| self.table.metadata().current_snapshot_id().map(|id| id.to_string()));
        let mut plan = if let Some(plan) = self
            .try_plan_predicate_free_limited_scan(
                &request,
                scan.snapshot().cloned(),
                snapshot_id.clone(),
                catalog_type.clone(),
                timeout_context.as_ref(),
            )
            .await?
        {
            plan
        } else {
            let mut task_stream = if let Some(context) = timeout_context.as_ref() {
                context.check("before_plan_files")?;
                tokio::time::timeout(context.remaining("plan_files")?, scan.plan_files())
                    .await
                    .map_err(|_elapsed| context.error("plan_files"))?
                    .map_err(|error| iceberg_planning_error("failed to plan Iceberg data files", error))?
            } else {
                scan.plan_files()
                    .await
                    .map_err(|error| iceberg_planning_error("failed to plan Iceberg data files", error))?
            };

            let mut tasks = Vec::new();
            loop {
                let task = if let Some(context) = timeout_context.as_ref() {
                    context.check("before_next_file_scan_task")?;
                    tokio::time::timeout(context.remaining("next_file_scan_task")?, task_stream.next())
                        .await
                        .map_err(|_elapsed| context.error("next_file_scan_task"))?
                } else {
                    task_stream.next().await
                };
                let Some(task) = task else {
                    break;
                };
                if let Some(context) = timeout_context.as_ref() {
                    context.check("after_next_file_scan_task")?;
                }
                tasks.push(
                    task.map_err(|error| iceberg_planning_error("failed to resolve Iceberg file scan task", error))?,
                );
            }

            iceberg_file_scan_tasks_to_table_format_scan_plan_with_filter_enforcement(
                snapshot_id,
                catalog_type,
                tasks,
                parent_filter_enforces_iceberg_predicate,
                timeout_context.as_ref(),
            )?
        }
        .with_scan_limit(request.scan_limit);
        if row_selection_enabled {
            plan.unsupported_features.push(TableFormatUnsupportedFeature::new(
                TableFormatUnsupportedFeatureCode::IcebergRowSelectionUnsupported,
                "Iceberg row selection/page-index semantics are not represented by Milestone B native scan facts",
            ));
        }
        Ok(plan)
    }

    /// Enriches resolved scan facts with acceleration-only source and footer evidence.
    pub async fn inspect_table_format_scan(
        &self,
        request: TableFormatScanRequest,
        resolved_scan: TableFormatScanPlan,
    ) -> Result<TableFormatScanPlan> {
        self.inspect_table_format_scan_with_timeout_context(request, resolved_scan, None).await
    }

    async fn inspect_table_format_scan_with_timeout_context(
        &self,
        request: TableFormatScanRequest,
        mut plan: TableFormatScanPlan,
        timeout_context: Option<IcebergScanPlanningTimeoutContext>,
    ) -> Result<TableFormatScanPlan> {
        if let Some(context) = timeout_context.as_ref() {
            context.check("native_inspection_start")?;
        }
        attach_native_parquet_source_strategies(&mut plan, &self.parquet_read_strategy_config)?;
        let request_has_predicate = request.predicate.is_some();
        attach_remote_delete_file_row_counts_for_scan_limit(
            &mut plan,
            self.table.file_io(),
            request_has_predicate,
            timeout_context.as_ref(),
        )
        .await?;
        trim_predicate_free_scan_limit_data_files(&mut plan, request_has_predicate);
        attach_remote_footer_schemas_for_field_id_proof(&mut plan, self.table.file_io(), timeout_context.as_ref())
            .await?;
        Ok(plan)
    }

    async fn try_plan_predicate_free_limited_scan(
        &self,
        request: &TableFormatScanRequest,
        snapshot: Option<SnapshotRef>,
        snapshot_id: Option<String>,
        catalog_type: Option<String>,
        timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
    ) -> Result<Option<TableFormatScanPlan>> {
        if request.predicate.is_some() {
            return Ok(None);
        }
        let Some(scan_limit) = request.scan_limit else {
            return Ok(None);
        };
        let Some(snapshot) = snapshot else {
            return Ok(None);
        };
        if scan_limit == 0 {
            return Ok(Some(TableFormatScanPlan::new(TableFormat::Iceberg)));
        }
        if let Some(context) = timeout_context {
            context.check("before_predicate_free_limit_manifest_list")?;
        }

        let snapshot_schema = snapshot
            .schema(self.table.metadata())
            .map_err(|error| iceberg_planning_error("failed to resolve Iceberg snapshot schema", error))?;
        let field_ids =
            Arc::new(iceberg_project_field_ids(snapshot_schema.as_ref(), request.projected_column_names.as_deref())?);
        let name_mapping = default_schema_name_mapping(&self.table)?;
        let manifest_list = load_iceberg_manifest_list(&self.table, &snapshot, timeout_context).await?;
        if !manifest_list.entries().iter().any(|manifest| manifest.content == ManifestContentType::Deletes) {
            return Ok(None);
        }

        let delete_index = load_limited_delete_index(&manifest_list, self.table.file_io(), timeout_context).await?;
        let logical_schema = iceberg_schema_to_datafusion_arrow_schema(snapshot_schema.as_ref())?;
        let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_logical_schema(Arc::clone(&logical_schema))
            .with_native_source_schema(logical_schema);
        if let Some(snapshot_id) = snapshot_id {
            plan = plan.with_snapshot_id(snapshot_id);
        }
        if let Some(catalog_type) = catalog_type {
            plan = plan.with_catalog_type(catalog_type);
        }

        let mut data_manifests = manifest_list
            .entries()
            .iter()
            .filter(|manifest| manifest.content == ManifestContentType::Data)
            .filter(|manifest| manifest.has_added_files() || manifest.has_existing_files())
            .cloned()
            .collect::<Vec<_>>();
        data_manifests.sort_by_key(|manifest| Reverse(manifest_live_rows(manifest)));

        let limit = u64::try_from(scan_limit).unwrap_or(u64::MAX);
        let mut task_index = 0usize;
        for manifest_file in data_manifests {
            if let Some(context) = timeout_context {
                context.check("before_predicate_free_limit_data_manifest")?;
            }
            let data_partition_spec_id = manifest_file.partition_spec_id;
            let manifest = load_iceberg_manifest(
                self.table.file_io(),
                &manifest_file,
                timeout_context,
                "predicate_free_limit_data_manifest",
            )
            .await?;
            let mut entries = manifest
                .entries()
                .iter()
                .filter(|entry| {
                    entry.is_alive() && entry.content_type() == DataContentType::Data && entry.record_count() > 0
                })
                .cloned()
                .collect::<Vec<_>>();
            entries.sort_by_key(|entry| Reverse(entry.record_count()));

            for entry in entries {
                if let Some(context) = timeout_context {
                    context.check("before_predicate_free_limit_data_file")?;
                }
                let partition_spec = self.table.metadata().partition_spec_by_id(data_partition_spec_id).cloned();
                let limited_deletes = delete_index.deletes_for_data_file(
                    entry.data_file(),
                    data_partition_spec_id,
                    entry.sequence_number(),
                );
                let deletes =
                    limited_deletes.iter().map(LimitedDeleteFile::to_file_scan_task_delete_file).collect::<Vec<_>>();
                let task = file_scan_task_from_manifest_entry(
                    entry.as_ref(),
                    Arc::clone(&snapshot_schema),
                    Arc::clone(&field_ids),
                    partition_spec,
                    name_mapping.clone(),
                    deletes,
                    request.case_sensitive,
                );
                append_task_facts(&mut plan, task_index, task, false)?;
                let data_file_index = plan.data_files.len() - 1;
                attach_manifest_delete_metadata(&mut plan, data_file_index, &limited_deletes);
                task_index += 1;

                if scan_plan_live_row_lower_bound(&plan).is_some_and(|rows| rows >= limit) {
                    return Ok(Some(plan));
                }
            }
        }

        Ok(None)
    }
}

impl TableFormatPlanner for IcebergTableFormatPlanner {
    fn resolve_scan(&self, request: TableFormatScanRequest) -> BoxFuture<'_, Result<TableFormatScanPlan>> {
        Box::pin(async move { self.resolve_table_format_scan(request).await })
    }

    fn resolve_scan_with_deadline(
        &self,
        request: TableFormatScanRequest,
        deadline: TableFormatPlanningDeadline,
    ) -> BoxFuture<'_, Result<TableFormatScanPlan>> {
        Box::pin(async move {
            let timeout_context =
                IcebergScanPlanningTimeoutContext::new(deadline, &request, self.planning_failure_facts());
            self.resolve_table_format_scan_with_timeout_context(request, Some(timeout_context)).await
        })
    }

    fn inspect_native_scan(
        &self,
        request: TableFormatScanRequest,
        resolved_scan: TableFormatScanPlan,
    ) -> BoxFuture<'_, Result<TableFormatScanPlan>> {
        Box::pin(async move { self.inspect_table_format_scan(request, resolved_scan).await })
    }

    fn inspect_native_scan_with_deadline(
        &self,
        request: TableFormatScanRequest,
        resolved_scan: TableFormatScanPlan,
        deadline: TableFormatPlanningDeadline,
    ) -> BoxFuture<'_, Result<TableFormatScanPlan>> {
        Box::pin(async move {
            let timeout_context =
                IcebergScanPlanningTimeoutContext::new(deadline, &request, self.planning_failure_facts());
            self.inspect_table_format_scan_with_timeout_context(request, resolved_scan, Some(timeout_context)).await
        })
    }

    fn planning_failure_facts(&self) -> Vec<(&'static str, String)> {
        iceberg_metadata_planning_failure_facts(&self.table)
    }
}

fn iceberg_metadata_planning_failure_facts(table: &Table) -> Vec<(&'static str, String)> {
    let metadata = table.metadata();
    let snapshots = metadata.snapshots().collect::<Vec<_>>();
    let current_snapshot = metadata.current_snapshot();
    vec![
        ("iceberg_snapshot_count", snapshots.len().to_string()),
        ("iceberg_snapshot_log_count", metadata.history().len().to_string()),
        ("iceberg_metadata_log_count", metadata.metadata_log().len().to_string()),
        (
            "iceberg_snapshot_manifest_list_pointer_count",
            snapshots.iter().filter(|snapshot| !snapshot.manifest_list().trim().is_empty()).count().to_string(),
        ),
        ("iceberg_current_snapshot_id", metadata.current_snapshot_id().map(|id| id.to_string()).unwrap_or_default()),
        (
            "iceberg_current_snapshot_has_manifest_list",
            current_snapshot.is_some_and(|snapshot| !snapshot.manifest_list().trim().is_empty()).to_string(),
        ),
    ]
}

fn default_schema_name_mapping(table: &Table) -> Result<Option<Arc<IcebergNameMapping>>> {
    table
        .metadata()
        .properties()
        .get(DEFAULT_SCHEMA_NAME_MAPPING)
        .map(|raw| {
            serde_json::from_str::<IcebergNameMapping>(raw).map(Arc::new).map_err(|error| {
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    PLAN_SCAN_OPERATION,
                    "failed to parse Iceberg schema.name-mapping.default table property",
                )
                .with_fact("table_format", TableFormat::Iceberg.code())
                .with_fact("property", DEFAULT_SCHEMA_NAME_MAPPING)
                .with_source(error)
            })
        })
        .transpose()
}
fn iceberg_planning_error(message: &'static str, source: iceberg::Error) -> Error {
    let error = Error::new(
        crate::table_format::iceberg::catalog::classify_iceberg_scan_error(&source),
        PLAN_SCAN_OPERATION,
        message,
    )
    .with_fact("table_format", TableFormat::Iceberg.code());
    crate::table_format::iceberg::catalog::apply_iceberg_catalog_error_facts(error, &source).with_source(source)
}

#[cfg(test)]
mod tests;
