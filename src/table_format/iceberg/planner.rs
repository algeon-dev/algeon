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
        DataContentType, DataFile, DataFileFormat, Datum, Literal, Manifest, ManifestContentType, ManifestEntry,
        ManifestFile, ManifestList, NestedField, PartitionSpec, PrimitiveLiteral, Schema as IcebergSchema,
        SchemaRef as IcebergSchemaRef, SnapshotRef, Struct, Transform, Type,
    },
    table::Table,
};
use nexus_query_engine::is_supported_arrow_type;
use parquet::file::{
    FOOTER_SIZE,
    metadata::{FooterTail, ParquetMetaData, ParquetMetaDataReader},
};

use crate::{
    Error, ErrorCode, Result,
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
        scan_exec::CudfTableFormatScanExec,
    },
};

const PLAN_SCAN_OPERATION: &str = "datafusion_nexus.table_format.iceberg.plan_scan";
const PLAN_SCAN_TIMEOUT_OPERATION: &str = "datafusion_nexus.table_format.iceberg.plan_scan_timeout";
const TASK_CONVERSION_OPERATION: &str = "datafusion_nexus.table_format.iceberg.convert_file_scan_tasks";
// Manifest planning is remote-IO bound; this keeps large snapshots moving without unbounded fanout.
const REMOTE_ICEBERG_METADATA_PLANNING_CONCURRENCY: usize = 128;
// Footer probes fetch larger Parquet tails, so they use a tighter transient-buffer gate.
const REMOTE_FOOTER_FIELD_ID_PROOF_CONCURRENCY: usize = 32;
type SharedTableFormatPlanner = Arc<dyn TableFormatPlanner + Send + Sync>;

/// DataFusion table provider that plans Iceberg metadata into adapter-owned
/// scan facts instead of delegating to the upstream Arrow reader path.
///
/// The provider treats DataFusion filters as inexact pushdown: DataFusion keeps
/// a parent filter for correctness, while native lowering can still see the
/// filter context and the scan can retain DataFusion's source-level limit
/// bound.
#[derive(Clone)]
pub struct CudfIcebergTableProvider {
    schema: SchemaRef,
    planner: SharedTableFormatPlanner,
    catalog_type: Option<String>,
    scan_planning_timeout: Option<Duration>,
}

impl CudfIcebergTableProvider {
    /// Builds a provider from an `iceberg-rust` table handle.
    ///
    /// The table schema is converted eagerly; file planning remains lazy and
    /// happens in `TableProvider::scan`.
    pub fn try_new(table: Table) -> Result<Self> {
        let schema = iceberg_schema_to_datafusion_arrow_schema(table.current_schema_ref().as_ref())?;
        Ok(Self::from_table_format_planner(schema, Arc::new(IcebergTableFormatPlanner::new(table))))
    }

    /// Builds a provider from an `iceberg-rust` table handle with an explicit
    /// Parquet read-strategy config.
    pub fn try_new_with_parquet_read_strategy_config(
        table: Table,
        config: TableFormatParquetReadStrategyConfig,
    ) -> Result<Self> {
        let schema = iceberg_schema_to_datafusion_arrow_schema(table.current_schema_ref().as_ref())?;
        Ok(Self::from_table_format_planner(
            schema,
            Arc::new(IcebergTableFormatPlanner::new(table).with_parquet_read_strategy_config(config)),
        ))
    }

    /// Builds a provider around an already supplied table-format planner.
    ///
    /// This keeps tests and custom catalog integrations credential-free at the
    /// DataFusion provider boundary.
    pub fn from_table_format_planner(schema: SchemaRef, planner: Arc<dyn TableFormatPlanner + Send + Sync>) -> Self {
        Self { schema, planner, catalog_type: None, scan_planning_timeout: None }
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

impl fmt::Debug for CudfIcebergTableProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudfIcebergTableProvider")
            .field("schema", &self.schema)
            .field("catalog_type", &self.catalog_type)
            .field("scan_planning_timeout", &self.scan_planning_timeout)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for CudfIcebergTableProvider {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        let projected_schema = project_schema(&self.schema, projection)?;
        let projected_column_names = projection
            .map(|indices| indices.iter().map(|index| self.schema.field(*index).name().clone()).collect::<Vec<_>>());
        let predicate = datafusion_filters_to_table_format_predicate(filters);
        let mut request = TableFormatScanRequest::default()
            .with_case_sensitive(true)
            .with_scan_limit(limit)
            .with_predicate(predicate);
        request.projected_column_names = projected_column_names;
        if let Some(catalog_type) = self.catalog_type.as_ref() {
            request.catalog_type = Some(catalog_type.clone());
        }
        let scan_plan = plan_scan_with_optional_timeout(Arc::clone(&self.planner), request, self.scan_planning_timeout)
            .await
            .map_err(DataFusionError::from)?;
        Ok(Arc::new(CudfTableFormatScanExec::new(scan_plan, projected_schema)))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }
}

async fn plan_scan_with_optional_timeout(
    planner: SharedTableFormatPlanner,
    request: TableFormatScanRequest,
    timeout: Option<Duration>,
) -> Result<TableFormatScanPlan> {
    let Some(timeout) = timeout else {
        return planner.plan_scan(request).await;
    };
    let deadline = TableFormatPlanningDeadline::from_timeout(timeout);
    let timeout_ms = deadline.timeout_ms();
    let catalog_type = request.catalog_type.clone();
    let snapshot_id = request.snapshot_id.clone();
    let planning_failure_facts = planner.planning_failure_facts();

    let mut planning_task = tokio::spawn(async move { planner.plan_scan_with_deadline(request, deadline).await });
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
                "provider_timeout",
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

    /// Resolves Iceberg scan metadata into adapter-owned scan facts.
    ///
    /// The result is metadata-only: data files, delete files, partition values,
    /// projected columns, and unsupported-feature facts. It does not read data
    /// files and must not expose credentials in diagnostics.
    pub async fn plan_table_format_scan(&self, request: TableFormatScanRequest) -> Result<TableFormatScanPlan> {
        self.plan_table_format_scan_with_timeout_context(request, None).await
    }

    async fn plan_table_format_scan_with_timeout_context(
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
        let request_has_predicate = request.predicate.is_some();
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

            iceberg_file_scan_tasks_to_table_format_scan_plan_with_config_and_filter_enforcement(
                snapshot_id,
                catalog_type,
                tasks,
                &self.parquet_read_strategy_config,
                parent_filter_enforces_iceberg_predicate,
                timeout_context.as_ref(),
            )?
        }
        .with_scan_limit(request.scan_limit);
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
        if row_selection_enabled {
            plan.unsupported_features.push(TableFormatUnsupportedFeature::new(
                TableFormatUnsupportedFeatureCode::IcebergRowSelectionUnsupported,
                "Iceberg row selection/page-index semantics are not represented by Milestone B native scan facts",
            ));
        }
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
                    deletes,
                    request.case_sensitive,
                );
                append_task_facts(&mut plan, task_index, task, &self.parquet_read_strategy_config, false)?;
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
    fn plan_scan(&self, request: TableFormatScanRequest) -> BoxFuture<'_, Result<TableFormatScanPlan>> {
        Box::pin(async move { self.plan_table_format_scan(request).await })
    }

    fn plan_scan_with_deadline(
        &self,
        request: TableFormatScanRequest,
        deadline: TableFormatPlanningDeadline,
    ) -> BoxFuture<'_, Result<TableFormatScanPlan>> {
        Box::pin(async move {
            let timeout_context =
                IcebergScanPlanningTimeoutContext::new(deadline, &request, self.planning_failure_facts());
            self.plan_table_format_scan_with_timeout_context(request, Some(timeout_context)).await
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

fn datafusion_filters_to_table_format_predicate(filters: &[Expr]) -> Option<TableFormatPredicate> {
    filters
        .iter()
        .filter_map(datafusion_filter_to_table_format_predicate)
        .reduce(|left, right| TableFormatPredicate::And(Box::new(left), Box::new(right)))
}

#[cfg(test)]
fn datafusion_filters_to_iceberg_predicate(filters: &[Expr]) -> Option<iceberg::expr::Predicate> {
    datafusion_filters_to_table_format_predicate(filters).as_ref().and_then(table_format_predicate_to_iceberg_predicate)
}

fn datafusion_filter_to_table_format_predicate(expr: &Expr) -> Option<TableFormatPredicate> {
    datafusion_filter_to_table_format_predicate_with_partial_and(expr, true)
}

fn datafusion_filter_to_table_format_predicate_exact(expr: &Expr) -> Option<TableFormatPredicate> {
    datafusion_filter_to_table_format_predicate_with_partial_and(expr, false)
}

fn datafusion_filter_to_table_format_predicate_with_partial_and(
    expr: &Expr,
    allow_partial_and: bool,
) -> Option<TableFormatPredicate> {
    match expr {
        Expr::BinaryExpr(binary) => match binary.op {
            Operator::And => match (
                datafusion_filter_to_table_format_predicate_with_partial_and(&binary.left, allow_partial_and),
                datafusion_filter_to_table_format_predicate_with_partial_and(&binary.right, allow_partial_and),
            ) {
                (Some(left), Some(right)) => Some(TableFormatPredicate::And(Box::new(left), Box::new(right))),
                (Some(left), None) if allow_partial_and => Some(left),
                (None, Some(right)) if allow_partial_and => Some(right),
                (None, None) => None,
                (Some(_), None) | (None, Some(_)) => None,
            },
            Operator::Or => Some(TableFormatPredicate::Or(
                Box::new(datafusion_filter_to_table_format_predicate_with_partial_and(
                    &binary.left,
                    allow_partial_and,
                )?),
                Box::new(datafusion_filter_to_table_format_predicate_with_partial_and(
                    &binary.right,
                    allow_partial_and,
                )?),
            )),
            Operator::Eq | Operator::NotEq | Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq => {
                datafusion_comparison_to_table_format_predicate(&binary.left, binary.op, &binary.right).or_else(|| {
                    datafusion_comparison_to_table_format_predicate(
                        &binary.right,
                        reverse_operator(binary.op)?,
                        &binary.left,
                    )
                })
            },
            _ => None,
        },
        Expr::Not(expr) => datafusion_filter_to_table_format_predicate_exact(expr)
            .map(|predicate| TableFormatPredicate::Not(Box::new(predicate))),
        Expr::IsNull(expr) => datafusion_column_reference(expr).map(TableFormatPredicate::IsNull),
        Expr::IsNotNull(expr) => datafusion_column_reference(expr).map(TableFormatPredicate::IsNotNull),
        Expr::Column(column) => Some(TableFormatPredicate::BooleanColumn(column.name().to_string())),
        _ => None,
    }
}

fn datafusion_comparison_to_table_format_predicate(
    left: &Expr,
    operator: Operator,
    right: &Expr,
) -> Option<TableFormatPredicate> {
    let column = datafusion_column_reference(left)?;
    let literal = datafusion_literal_to_table_format_literal(right)?;
    let op = match operator {
        Operator::Eq => TableFormatPredicateOp::Eq,
        Operator::NotEq => TableFormatPredicateOp::NotEq,
        Operator::Lt => TableFormatPredicateOp::Lt,
        Operator::LtEq => TableFormatPredicateOp::LtEq,
        Operator::Gt => TableFormatPredicateOp::Gt,
        Operator::GtEq => TableFormatPredicateOp::GtEq,
        _ => return None,
    };
    Some(TableFormatPredicate::Comparison { column, op, literal })
}

fn table_format_predicate_to_iceberg_predicate(predicate: &TableFormatPredicate) -> Option<iceberg::expr::Predicate> {
    match predicate {
        TableFormatPredicate::And(left, right) => Some(
            table_format_predicate_to_iceberg_predicate(left)?.and(table_format_predicate_to_iceberg_predicate(right)?),
        ),
        TableFormatPredicate::Or(left, right) => Some(
            table_format_predicate_to_iceberg_predicate(left)?.or(table_format_predicate_to_iceberg_predicate(right)?),
        ),
        TableFormatPredicate::Not(predicate) => {
            table_format_predicate_to_iceberg_predicate(predicate).map(std::ops::Not::not)
        },
        TableFormatPredicate::BooleanColumn(column) => Some(Reference::new(column).equal_to(Datum::bool(true))),
        TableFormatPredicate::IsNull(column) => Some(Reference::new(column).is_null()),
        TableFormatPredicate::IsNotNull(column) => Some(Reference::new(column).is_not_null()),
        TableFormatPredicate::Comparison { column, op, literal } => {
            table_format_comparison_to_iceberg_predicate(column, *op, literal)
        },
    }
}

fn table_format_comparison_to_iceberg_predicate(
    column: &str,
    operator: TableFormatPredicateOp,
    literal: &TableFormatLiteral,
) -> Option<iceberg::expr::Predicate> {
    let reference = Reference::new(column);
    let datum = table_format_literal_to_iceberg_datum(literal);
    match operator {
        TableFormatPredicateOp::Eq => Some(reference.equal_to(datum)),
        TableFormatPredicateOp::NotEq => Some(reference.not_equal_to(datum)),
        TableFormatPredicateOp::Lt => Some(reference.less_than(datum)),
        TableFormatPredicateOp::LtEq => Some(reference.less_than_or_equal_to(datum)),
        TableFormatPredicateOp::Gt => Some(reference.greater_than(datum)),
        TableFormatPredicateOp::GtEq => Some(reference.greater_than_or_equal_to(datum)),
    }
}

fn datafusion_column_reference(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(column) => Some(column.name().to_string()),
        _ => None,
    }
}

fn datafusion_literal_to_table_format_literal(expr: &Expr) -> Option<TableFormatLiteral> {
    match expr {
        Expr::Literal(value, _) => scalar_value_to_table_format_literal(value),
        _ => None,
    }
}

fn scalar_value_to_table_format_literal(value: &ScalarValue) -> Option<TableFormatLiteral> {
    match value {
        ScalarValue::Boolean(Some(value)) => Some(TableFormatLiteral::Boolean(*value)),
        ScalarValue::Int8(Some(value)) => Some(TableFormatLiteral::Int32(i32::from(*value))),
        ScalarValue::Int16(Some(value)) => Some(TableFormatLiteral::Int32(i32::from(*value))),
        ScalarValue::Int32(Some(value)) => Some(TableFormatLiteral::Int32(*value)),
        ScalarValue::Int64(Some(value)) => Some(TableFormatLiteral::Int64(*value)),
        ScalarValue::Float32(Some(value)) => Some(TableFormatLiteral::Float32(*value)),
        ScalarValue::Float64(Some(value)) => Some(TableFormatLiteral::Float64(*value)),
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => {
            Some(TableFormatLiteral::Utf8(value.clone()))
        },
        ScalarValue::Date32(Some(value)) => Some(TableFormatLiteral::Date32(*value)),
        _ => None,
    }
}

fn table_format_literal_to_iceberg_datum(literal: &TableFormatLiteral) -> Datum {
    match literal {
        TableFormatLiteral::Boolean(value) => Datum::bool(*value),
        TableFormatLiteral::Int32(value) => Datum::int(*value),
        TableFormatLiteral::Int64(value) => Datum::long(*value),
        TableFormatLiteral::Float32(value) => Datum::float(*value),
        TableFormatLiteral::Float64(value) => Datum::double(*value),
        TableFormatLiteral::Utf8(value) => Datum::string(value),
        TableFormatLiteral::Date32(value) => Datum::date(*value),
    }
}

fn reverse_operator(operator: Operator) -> Option<Operator> {
    match operator {
        Operator::Eq => Some(Operator::Eq),
        Operator::NotEq => Some(Operator::NotEq),
        Operator::Lt => Some(Operator::Gt),
        Operator::LtEq => Some(Operator::GtEq),
        Operator::Gt => Some(Operator::Lt),
        Operator::GtEq => Some(Operator::LtEq),
        _ => None,
    }
}

/// Converts already planned Iceberg file tasks into adapter-owned scan facts.
///
/// Keeping this conversion public gives tests and later provider code a stable
/// way to validate native admission facts without invoking catalog credentials
/// or executing Arrow readers.
pub fn iceberg_file_scan_tasks_to_table_format_scan_plan(
    snapshot_id: Option<String>,
    catalog_type: Option<String>,
    tasks: impl IntoIterator<Item = FileScanTask>,
) -> Result<TableFormatScanPlan> {
    iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(
        snapshot_id,
        catalog_type,
        tasks,
        &TableFormatParquetReadStrategyConfig::from_env(),
    )
}

fn iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(
    snapshot_id: Option<String>,
    catalog_type: Option<String>,
    tasks: impl IntoIterator<Item = FileScanTask>,
    parquet_read_strategy_config: &TableFormatParquetReadStrategyConfig,
) -> Result<TableFormatScanPlan> {
    iceberg_file_scan_tasks_to_table_format_scan_plan_with_config_and_filter_enforcement(
        snapshot_id,
        catalog_type,
        tasks,
        parquet_read_strategy_config,
        false,
        None,
    )
}

fn iceberg_file_scan_tasks_to_table_format_scan_plan_with_config_and_filter_enforcement(
    snapshot_id: Option<String>,
    catalog_type: Option<String>,
    tasks: impl IntoIterator<Item = FileScanTask>,
    parquet_read_strategy_config: &TableFormatParquetReadStrategyConfig,
    parent_filter_enforces_task_predicate: bool,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
) -> Result<TableFormatScanPlan> {
    if let Some(context) = timeout_context {
        context.check("before_convert_file_scan_tasks")?;
    }
    let mut tasks = tasks.into_iter().collect::<Vec<_>>();
    let logical_schema =
        tasks.first().map(|task| iceberg_schema_to_datafusion_arrow_schema(task.schema())).transpose()?;
    let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg);

    if let Some(snapshot_id) = snapshot_id {
        plan = plan.with_snapshot_id(snapshot_id);
    }
    if let Some(catalog_type) = catalog_type {
        plan = plan.with_catalog_type(catalog_type);
    }
    if let Some(schema) = logical_schema {
        plan = plan.with_logical_schema(Arc::clone(&schema)).with_native_source_schema(schema);
    }
    if let Some(mapping) = count_star_probe_column_mapping_for_tasks(&tasks)? {
        plan.pure_filter_columns.push(mapping);
    }

    for (task_index, task) in tasks.drain(..).enumerate() {
        if let Some(context) = timeout_context {
            context.check("convert_file_scan_task")?;
        }
        append_task_facts(
            &mut plan,
            task_index,
            task,
            parquet_read_strategy_config,
            parent_filter_enforces_task_predicate,
        )?;
    }

    Ok(plan)
}

fn iceberg_project_field_ids(schema: &IcebergSchema, column_names: Option<&[String]>) -> Result<Vec<i32>> {
    let column_names = column_names
        .map(|column_names| column_names.to_vec())
        .unwrap_or_else(|| schema.as_struct().fields().iter().map(|field| field.name.clone()).collect());
    let mut field_ids = Vec::with_capacity(column_names.len());
    for column_name in column_names {
        if is_metadata_column_name(&column_name) {
            field_ids.push(get_metadata_field_id(&column_name).map_err(|error| {
                iceberg_planning_error("failed to bind Iceberg metadata column in table-format scan", error)
            })?);
            continue;
        }
        let field_id = schema.field_id_by_name(&column_name).ok_or_else(|| {
            Error::new(
                ErrorCode::InvalidExecutionPlan,
                PLAN_SCAN_OPERATION,
                "Iceberg scan projection references a column that is absent from the snapshot schema",
            )
            .with_fact("table_format", TableFormat::Iceberg.code())
            .with_fact("column", column_name.clone())
        })?;
        if schema.as_struct().field_by_id(field_id).is_none() {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                PLAN_SCAN_OPERATION,
                "Iceberg scan projection references a nested field that native table-format scans cannot bind",
            )
            .with_fact("table_format", TableFormat::Iceberg.code())
            .with_fact("column", column_name)
            .with_fact("field_id", field_id));
        }
        field_ids.push(field_id);
    }
    Ok(field_ids)
}

async fn load_iceberg_manifest_list(
    table: &Table,
    snapshot: &SnapshotRef,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
) -> Result<ManifestList> {
    let reader = table.manifest_list_reader(snapshot);
    let load = reader.load();
    if let Some(context) = timeout_context {
        tokio::time::timeout(context.remaining("predicate_free_limit_manifest_list")?, load)
            .await
            .map_err(|_elapsed| context.error("predicate_free_limit_manifest_list"))?
            .map_err(|error| iceberg_planning_error("failed to load Iceberg manifest list", error))
    } else {
        load.await.map_err(|error| iceberg_planning_error("failed to load Iceberg manifest list", error))
    }
}

async fn load_iceberg_manifest(
    file_io: &FileIO,
    manifest_file: &ManifestFile,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
    stage: &'static str,
) -> Result<Manifest> {
    let load = manifest_file.load_manifest(file_io);
    if let Some(context) = timeout_context {
        tokio::time::timeout(context.remaining(stage)?, load)
            .await
            .map_err(|_elapsed| context.error(stage))?
            .map_err(|error| iceberg_planning_error("failed to load Iceberg manifest", error))
    } else {
        load.await.map_err(|error| iceberg_planning_error("failed to load Iceberg manifest", error))
    }
}

async fn load_limited_delete_index(
    manifest_list: &ManifestList,
    file_io: &FileIO,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
) -> Result<LimitedDeleteIndex> {
    if let Some(context) = timeout_context {
        context.check("before_predicate_free_limit_delete_manifests")?;
    }
    let requests = manifest_list
        .entries()
        .iter()
        .filter(|manifest| manifest.content == ManifestContentType::Deletes)
        .filter(|manifest| manifest.has_added_files() || manifest.has_existing_files())
        .cloned()
        .collect::<Vec<_>>();
    let fetches = requests.into_iter().map(|manifest_file| {
        let file_io = file_io.clone();
        async move {
            let partition_spec_id = manifest_file.partition_spec_id;
            load_iceberg_manifest(&file_io, &manifest_file, None, "predicate_free_limit_delete_manifest")
                .await
                .map(|manifest| (partition_spec_id, manifest))
        }
    });
    let mut manifests = futures::stream::iter(fetches).buffer_unordered(REMOTE_ICEBERG_METADATA_PLANNING_CONCURRENCY);
    let mut index = LimitedDeleteIndex::default();
    loop {
        let result = if let Some(context) = timeout_context {
            tokio::time::timeout(context.remaining("predicate_free_limit_delete_manifest")?, manifests.next())
                .await
                .map_err(|_elapsed| context.error("predicate_free_limit_delete_manifest"))?
        } else {
            manifests.next().await
        };
        let Some(result) = result else {
            break;
        };
        let (partition_spec_id, manifest) = result?;
        index.extend_manifest(partition_spec_id, &manifest);
    }
    Ok(index)
}

#[derive(Default)]
struct LimitedDeleteIndex {
    files: Vec<LimitedDeleteFile>,
}

impl LimitedDeleteIndex {
    fn extend_manifest(&mut self, partition_spec_id: i32, manifest: &Manifest) {
        self.files.extend(
            manifest
                .entries()
                .iter()
                .filter(|entry| entry.is_alive() && entry.content_type() != DataContentType::Data)
                .map(|entry| LimitedDeleteFile::from_manifest_entry(partition_spec_id, entry.as_ref())),
        );
    }

    fn deletes_for_data_file(
        &self,
        data_file: &DataFile,
        data_partition_spec_id: i32,
        data_sequence_number: Option<i64>,
    ) -> Vec<LimitedDeleteFile> {
        self.files
            .iter()
            .filter(|delete_file| {
                delete_file.applies_to_data_file(data_file, data_partition_spec_id, data_sequence_number)
            })
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone)]
struct LimitedDeleteFile {
    file_path: String,
    file_size_in_bytes: u64,
    file_type: DataContentType,
    partition_spec_id: i32,
    equality_ids: Option<Vec<i32>>,
    sequence_number: Option<i64>,
    record_count: u64,
    partition: Struct,
    referenced_data_file: Option<String>,
    position_lower_bound: Option<u64>,
    position_upper_bound: Option<u64>,
}

impl LimitedDeleteFile {
    fn from_manifest_entry(partition_spec_id: i32, entry: &ManifestEntry) -> Self {
        let data_file = entry.data_file();
        Self {
            file_path: entry.file_path().to_string(),
            file_size_in_bytes: entry.file_size_in_bytes(),
            file_type: entry.content_type(),
            partition_spec_id,
            equality_ids: data_file.equality_ids(),
            sequence_number: entry.sequence_number(),
            record_count: entry.record_count(),
            partition: data_file.partition().clone(),
            referenced_data_file: data_file.referenced_data_file(),
            position_lower_bound: iceberg_position_delete_bound(data_file.lower_bounds()),
            position_upper_bound: iceberg_position_delete_bound(data_file.upper_bounds()),
        }
    }

    fn applies_to_data_file(
        &self,
        data_file: &DataFile,
        data_partition_spec_id: i32,
        data_sequence_number: Option<i64>,
    ) -> bool {
        match self.file_type {
            DataContentType::PositionDeletes => {
                self.partition_spec_id == data_partition_spec_id
                    && self.partition == *data_file.partition()
                    && position_delete_sequence_applies(self.sequence_number, data_sequence_number)
                    && self.referenced_data_file.as_deref().is_none_or(|referenced| referenced == data_file.file_path())
            },
            DataContentType::EqualityDeletes => {
                equality_delete_sequence_applies(self.sequence_number, data_sequence_number)
                    && (self.partition.fields().is_empty()
                        || (self.partition_spec_id == data_partition_spec_id
                            && self.partition == *data_file.partition()))
            },
            DataContentType::Data => false,
        }
    }

    fn to_file_scan_task_delete_file(&self) -> FileScanTaskDeleteFile {
        FileScanTaskDeleteFile::builder()
            .with_file_path(self.file_path.clone())
            .with_file_size_in_bytes(self.file_size_in_bytes)
            .with_file_type(self.file_type)
            .with_partition_spec_id(self.partition_spec_id)
            .with_equality_ids(self.equality_ids.clone())
            .build()
    }
}

fn iceberg_position_delete_bound(bounds: &HashMap<i32, Datum>) -> Option<u64> {
    [RESERVED_FIELD_ID_DELETE_FILE_POS, RESERVED_FIELD_ID_POS]
        .into_iter()
        .find_map(|field_id| bounds.get(&field_id).and_then(iceberg_position_bound_value))
}

fn iceberg_position_bound_value(value: &Datum) -> Option<u64> {
    match value.literal() {
        PrimitiveLiteral::Long(value) => u64::try_from(*value).ok(),
        PrimitiveLiteral::Int(value) => u64::try_from(*value).ok(),
        _ => None,
    }
}

fn attach_manifest_delete_metadata(
    plan: &mut TableFormatScanPlan,
    data_file_index: usize,
    delete_files: &[LimitedDeleteFile],
) {
    let Some(data_file) = plan.data_files.get_mut(data_file_index) else {
        return;
    };
    let data_file_id = data_file.id.clone();
    for limited_delete in delete_files {
        for delete_file in data_file
            .position_delete_files
            .iter_mut()
            .filter(|delete_file| delete_file.location == limited_delete.file_path)
        {
            delete_file.row_count = Some(limited_delete.record_count);
            delete_file.referenced_data_file = limited_delete.referenced_data_file.clone();
            delete_file.position_lower_bound = limited_delete.position_lower_bound;
            delete_file.position_upper_bound = limited_delete.position_upper_bound;
        }
        for delete_file in plan.delete_files.iter_mut().filter(|delete_file| {
            delete_file.data_file_id == data_file_id && delete_file.location == limited_delete.file_path
        }) {
            delete_file.row_count = Some(limited_delete.record_count);
            delete_file.referenced_data_file = limited_delete.referenced_data_file.clone();
            delete_file.position_lower_bound = limited_delete.position_lower_bound;
            delete_file.position_upper_bound = limited_delete.position_upper_bound;
        }
    }
}

fn position_delete_sequence_applies(delete_sequence_number: Option<i64>, data_sequence_number: Option<i64>) -> bool {
    data_sequence_number
        .map(|data_sequence_number| delete_sequence_number >= Some(data_sequence_number))
        .unwrap_or(true)
}

fn equality_delete_sequence_applies(delete_sequence_number: Option<i64>, data_sequence_number: Option<i64>) -> bool {
    data_sequence_number.map(|data_sequence_number| delete_sequence_number > Some(data_sequence_number)).unwrap_or(true)
}

fn file_scan_task_from_manifest_entry(
    entry: &ManifestEntry,
    schema: IcebergSchemaRef,
    field_ids: Arc<Vec<i32>>,
    partition_spec: Option<Arc<PartitionSpec>>,
    deletes: Vec<FileScanTaskDeleteFile>,
    case_sensitive: bool,
) -> FileScanTask {
    FileScanTask::builder()
        .with_file_size_in_bytes(entry.file_size_in_bytes())
        .with_start(0)
        .with_length(entry.file_size_in_bytes())
        .with_record_count(Some(entry.record_count()))
        .with_data_file_path(entry.file_path().to_string())
        .with_data_file_format(entry.file_format())
        .with_schema(schema)
        .with_project_field_ids(field_ids.to_vec())
        .with_predicate(None)
        .with_deletes(deletes)
        .with_partition(Some(entry.data_file().partition().clone()))
        .with_partition_spec(partition_spec)
        .with_name_mapping(None)
        .with_case_sensitive(case_sensitive)
        .build()
}

fn manifest_live_rows(manifest: &ManifestFile) -> u64 {
    manifest.added_rows_count.unwrap_or(0).saturating_add(manifest.existing_rows_count.unwrap_or(0))
}

fn scan_plan_live_row_lower_bound(plan: &TableFormatScanPlan) -> Option<u64> {
    plan.data_files.iter().try_fold(0u64, |rows, file| Some(rows.saturating_add(data_file_live_row_lower_bound(file)?)))
}

fn trim_predicate_free_scan_limit_data_files(plan: &mut TableFormatScanPlan, request_has_predicate: bool) {
    if request_has_predicate {
        return;
    }
    let Some(limit) = plan.scan_limit else {
        return;
    };
    let limit = u64::try_from(limit).unwrap_or(u64::MAX);
    if limit == 0 {
        plan.data_files.clear();
        return;
    }
    let Some(total_live_lower_bound) = plan
        .data_files
        .iter()
        .try_fold(0u64, |rows, file| Some(rows.saturating_add(data_file_live_row_lower_bound(file)?)))
    else {
        return;
    };
    if total_live_lower_bound < limit {
        return;
    }

    plan.data_files.sort_by(|left, right| {
        let left_rows = data_file_live_row_lower_bound(left).unwrap_or(0);
        let right_rows = data_file_live_row_lower_bound(right).unwrap_or(0);
        right_rows.cmp(&left_rows)
    });

    let mut retained = 0usize;
    let mut rows = 0u64;
    for file in &plan.data_files {
        let live_lower_bound = data_file_live_row_lower_bound(file).unwrap_or(0);
        retained += 1;
        rows = rows.saturating_add(live_lower_bound);
        if rows >= limit {
            break;
        }
    }
    if rows >= limit && retained < plan.data_files.len() {
        let original_data_file_ids = plan.data_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
        let original_delete_file_ids = plan.delete_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
        plan.data_files.truncate(retained);
        let retained_data_file_ids = plan.data_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
        plan.delete_files.retain(|delete_file| retained_data_file_ids.contains(&delete_file.data_file_id));
        let retained_delete_file_ids = plan.delete_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
        prune_dropped_file_local_unsupported_features(
            plan,
            &original_data_file_ids,
            &original_delete_file_ids,
            &retained_data_file_ids,
            &retained_delete_file_ids,
        );
    }
}

fn data_file_live_row_lower_bound(file: &TableFormatDataFile) -> Option<u64> {
    let data_rows = file.row_count?;
    let delete_rows = file
        .position_delete_files
        .iter()
        .try_fold(0u64, |rows, delete_file| Some(rows.saturating_add(delete_file.row_count?)))?;
    Some(data_rows.saturating_sub(delete_rows))
}

async fn attach_remote_delete_file_row_counts_for_scan_limit(
    plan: &mut TableFormatScanPlan,
    file_io: &FileIO,
    request_has_predicate: bool,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
) -> Result<()> {
    if request_has_predicate || plan.scan_limit.is_none() {
        return Ok(());
    }
    if let Some(context) = timeout_context {
        context.check("before_remote_delete_footer_row_count")?;
    }
    let limit = plan.scan_limit.and_then(|limit| u64::try_from(limit).ok()).unwrap_or(u64::MAX);
    let mut candidate_indices = plan
        .data_files
        .iter()
        .enumerate()
        .filter_map(|(index, file)| file.row_count.map(|rows| (index, rows)))
        .collect::<Vec<_>>();
    candidate_indices.sort_by(|(_left_index, left_rows), (_right_index, right_rows)| right_rows.cmp(left_rows));

    let mut retained_indices = Vec::new();
    let mut retained_live_rows = 0u64;
    for (data_file_index, _rows) in candidate_indices {
        attach_remote_delete_file_row_counts_for_data_file(plan, file_io, data_file_index, timeout_context).await?;
        let Some(live_rows) = plan.data_files.get(data_file_index).and_then(data_file_live_row_lower_bound) else {
            continue;
        };
        if live_rows == 0 {
            continue;
        }
        retained_indices.push(data_file_index);
        retained_live_rows = retained_live_rows.saturating_add(live_rows);
        if retained_live_rows >= limit {
            let original_data_file_ids = plan.data_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
            let original_delete_file_ids = plan.delete_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
            let retained_data_files =
                retained_indices.iter().filter_map(|index| plan.data_files.get(*index).cloned()).collect::<Vec<_>>();
            let retained_data_file_ids = retained_data_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
            plan.data_files = retained_data_files;
            plan.delete_files.retain(|delete_file| retained_data_file_ids.contains(&delete_file.data_file_id));
            let retained_delete_file_ids = plan.delete_files.iter().map(|file| file.id.clone()).collect::<HashSet<_>>();
            prune_dropped_file_local_unsupported_features(
                plan,
                &original_data_file_ids,
                &original_delete_file_ids,
                &retained_data_file_ids,
                &retained_delete_file_ids,
            );
            return Ok(());
        }
    }

    Ok(())
}

fn prune_dropped_file_local_unsupported_features(
    plan: &mut TableFormatScanPlan,
    original_data_file_ids: &HashSet<String>,
    original_delete_file_ids: &HashSet<String>,
    retained_data_file_ids: &HashSet<String>,
    retained_delete_file_ids: &HashSet<String>,
) {
    plan.unsupported_features.retain(|feature| {
        let Some(file_id) = feature.file_id.as_deref() else {
            return true;
        };
        let known_file_local = original_data_file_ids.contains(file_id) || original_delete_file_ids.contains(file_id);
        !known_file_local || retained_data_file_ids.contains(file_id) || retained_delete_file_ids.contains(file_id)
    });
}

async fn attach_remote_delete_file_row_counts_for_data_file(
    plan: &mut TableFormatScanPlan,
    file_io: &FileIO,
    data_file_index: usize,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
) -> Result<()> {
    let Some(data_file) = plan.data_files.get(data_file_index) else {
        return Ok(());
    };
    let requests = data_file
        .position_delete_files
        .iter()
        .enumerate()
        .filter_map(|(delete_index, delete_file)| {
            if delete_file.row_count.is_some()
                || delete_file.format != TableFormatDataFileFormat::Parquet
                || !delete_file.read_capability.is_native_parquet_capable()
            {
                return None;
            }
            Some((delete_index, delete_file.id.clone(), delete_file.location.clone(), delete_file.size_bytes))
        })
        .collect::<Vec<_>>();
    let fetches = requests.into_iter().map(|(delete_index, delete_id, location, size_bytes)| {
        let file_io = file_io.clone();
        async move {
            (delete_index, delete_id, remote_iceberg_parquet_footer_row_count(&file_io, &location, size_bytes).await)
        }
    });
    let mut row_counts = futures::stream::iter(fetches).buffer_unordered(REMOTE_FOOTER_FIELD_ID_PROOF_CONCURRENCY);

    loop {
        let result = if let Some(context) = timeout_context {
            tokio::time::timeout(context.remaining("remote_delete_footer_row_count")?, row_counts.next())
                .await
                .map_err(|_elapsed| context.error("remote_delete_footer_row_count"))?
        } else {
            row_counts.next().await
        };
        let Some((delete_index, delete_id, row_count)) = result else {
            break;
        };
        if let Some(row_count) = row_count {
            if let Some(delete_file) = plan
                .data_files
                .get_mut(data_file_index)
                .and_then(|data_file| data_file.position_delete_files.get_mut(delete_index))
            {
                delete_file.row_count = Some(row_count);
            }
            if let Some(delete_file) = plan.delete_files.iter_mut().find(|delete_file| delete_file.id == delete_id) {
                delete_file.row_count = Some(row_count);
            }
        }
    }

    Ok(())
}

async fn attach_remote_footer_schemas_for_field_id_proof(
    plan: &mut TableFormatScanPlan,
    file_io: &FileIO,
    timeout_context: Option<&IcebergScanPlanningTimeoutContext>,
) -> Result<()> {
    if !scan_plan_needs_field_id_footer_schema(plan) {
        return Ok(());
    }

    if let Some(context) = timeout_context {
        context.check("before_remote_footer_schema")?;
    }

    let requests = plan
        .data_files
        .iter()
        .enumerate()
        .filter(|(_index, data_file)| {
            data_file.footer_schema.is_none()
                && !data_file.read_capability.is_native_local_parquet_capable()
                && data_file.is_native_parquet_representable()
        })
        .map(|(index, data_file)| (index, data_file.location.clone(), data_file.size_bytes))
        .collect::<Vec<_>>();
    let fetches = requests.into_iter().map(|(index, location, size_bytes)| {
        let file_io = file_io.clone();
        async move { (index, remote_iceberg_parquet_footer(&file_io, &location, size_bytes).await) }
    });
    let mut footers = futures::stream::iter(fetches).buffer_unordered(REMOTE_FOOTER_FIELD_ID_PROOF_CONCURRENCY);

    loop {
        let result = if let Some(context) = timeout_context {
            tokio::time::timeout(context.remaining("remote_footer_schema")?, footers.next())
                .await
                .map_err(|_elapsed| context.error("remote_footer_schema"))?
        } else {
            footers.next().await
        };
        let Some((index, footer)) = result else {
            break;
        };
        if let Some(footer) = footer
            && let Some(data_file) = plan.data_files.get_mut(index)
        {
            data_file.footer_schema = Some(footer.schema);
            data_file.row_group_null_counts = footer.row_group_null_counts;
        }
    }

    Ok(())
}

fn scan_plan_needs_field_id_footer_schema(plan: &TableFormatScanPlan) -> bool {
    plan.format == TableFormat::Iceberg
        && plan.projected_columns.iter().chain(plan.pure_filter_columns.iter()).any(|mapping| {
            mapping.source == TableFormatColumnSource::DirectTopLevelFileColumn && mapping.field_id.is_some()
        })
}

async fn remote_iceberg_parquet_footer_row_count(
    file_io: &FileIO,
    location: &str,
    known_file_size: Option<u64>,
) -> Option<u64> {
    remote_iceberg_parquet_footer(file_io, location, known_file_size).await.and_then(|footer| footer.row_count)
}

struct RemoteIcebergParquetFooter {
    schema: SchemaRef,
    row_count: Option<u64>,
    row_group_null_counts: BTreeMap<usize, Vec<Option<u64>>>,
}

async fn remote_iceberg_parquet_footer(
    file_io: &FileIO,
    location: &str,
    known_file_size: Option<u64>,
) -> Option<RemoteIcebergParquetFooter> {
    let input = file_io.new_input(location).ok()?;
    let file_size = match known_file_size {
        Some(file_size) => file_size,
        None => input.metadata().await.ok()?.size,
    };
    if file_size < FOOTER_SIZE as u64 {
        return None;
    }

    let reader = input.reader().await.ok()?;
    let prefetch = (64 * 1024).min(file_size);
    let tail_start = file_size - prefetch;
    let tail = reader.read(tail_start..file_size).await.ok()?;
    if tail.len() < FOOTER_SIZE {
        return None;
    }
    let footer_tail_bytes: [u8; FOOTER_SIZE] = tail[tail.len() - FOOTER_SIZE..].try_into().ok()?;
    let footer_tail = FooterTail::try_from(footer_tail_bytes).ok()?;
    if footer_tail.is_encrypted_footer() {
        return None;
    }

    let total_footer_len = footer_tail.metadata_length() + FOOTER_SIZE;
    if total_footer_len as u64 > file_size {
        return None;
    }
    let metadata_bytes = if total_footer_len as u64 <= prefetch {
        let start = tail.len() - total_footer_len;
        let end = tail.len() - FOOTER_SIZE;
        tail.slice(start..end)
    } else {
        let metadata_start = file_size - total_footer_len as u64;
        let metadata_end = file_size - FOOTER_SIZE as u64;
        reader.read(metadata_start..metadata_end).await.ok()?
    };

    let mut footer = BytesMut::with_capacity(total_footer_len);
    footer.extend_from_slice(&metadata_bytes);
    footer.extend_from_slice(&footer_tail_bytes);
    let footer = footer.freeze();

    let mut metadata_reader = ParquetMetaDataReader::new();
    metadata_reader.try_parse_sized(&footer, file_size).ok()?;
    let metadata = metadata_reader.finish().ok()?;
    let file_metadata = metadata.file_metadata();
    let schema =
        parquet::arrow::parquet_to_arrow_schema(file_metadata.schema_descr(), file_metadata.key_value_metadata())
            .ok()
            .map(|schema| Arc::new(schema) as SchemaRef)?;
    let row_count = u64::try_from(file_metadata.num_rows()).ok();
    let row_group_null_counts = parquet_footer_row_group_null_counts(&metadata);
    Some(RemoteIcebergParquetFooter { schema, row_count, row_group_null_counts })
}

fn parquet_footer_row_group_null_counts(metadata: &ParquetMetaData) -> BTreeMap<usize, Vec<Option<u64>>> {
    let schema = metadata.file_metadata().schema_descr();
    let root_field_count = schema.root_schema().get_fields().len();
    (0..root_field_count)
        .filter_map(|root_index| {
            let leaf_index = single_leaf_primitive_root_column_index(schema, root_index)?;
            let counts = metadata
                .row_groups()
                .iter()
                .map(|row_group| {
                    row_group
                        .columns()
                        .get(leaf_index)
                        .and_then(|column| column.statistics())
                        .and_then(|statistics| statistics.null_count_opt())
                })
                .collect::<Vec<_>>();
            Some((root_index, counts))
        })
        .collect()
}

fn single_leaf_primitive_root_column_index(
    schema: &parquet::schema::types::SchemaDescriptor,
    root_index: usize,
) -> Option<usize> {
    if !schema.root_schema().get_fields().get(root_index)?.is_primitive() {
        return None;
    }
    let mut leaf_indices =
        (0..schema.num_columns()).filter(|leaf_index| schema.get_column_root_idx(*leaf_index) == root_index);
    let leaf_index = leaf_indices.next()?;
    leaf_indices.next().is_none().then_some(leaf_index)
}

fn count_star_probe_column_mapping_for_tasks(tasks: &[FileScanTask]) -> Result<Option<TableFormatColumnMapping>> {
    if tasks.is_empty()
        || !tasks.iter().all(|task| task.project_field_ids.is_empty())
        || !tasks.iter().any(|task| task.record_count.is_none())
    {
        return Ok(None);
    }

    let schema = tasks[0].schema();
    for field in schema.as_struct().fields() {
        if !count_star_probe_field_is_direct_file_column_for_missing_count_tasks(tasks, field.id)? {
            continue;
        }
        let data_type = iceberg_field_to_arrow_data_type(field)?;
        if !is_supported_arrow_type(&data_type) {
            continue;
        }
        return Ok(Some(TableFormatColumnMapping::new(
            Some(field.id),
            field.name.clone(),
            Some(field.name.clone()),
            data_type,
            !field.required,
            TableFormatColumnSource::DirectTopLevelFileColumn,
            TableFormatColumnRole::PureFilter,
        )));
    }

    Ok(None)
}

fn count_star_probe_field_is_direct_file_column_for_missing_count_tasks(
    tasks: &[FileScanTask],
    field_id: i32,
) -> Result<bool> {
    for task in tasks.iter().filter(|task| task.record_count.is_none()) {
        let Some(field) = task.schema().as_struct().field_by_id(field_id) else {
            return Ok(false);
        };
        if task.partition_spec.as_deref().is_some_and(|spec| identity_partition_source_field(spec, field_id)) {
            return Ok(false);
        }
        let data_type = iceberg_field_to_arrow_data_type(field)?;
        if !is_supported_arrow_type(&data_type) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn append_task_facts(
    plan: &mut TableFormatScanPlan,
    task_index: usize,
    task: FileScanTask,
    parquet_read_strategy_config: &TableFormatParquetReadStrategyConfig,
    parent_filter_enforces_task_predicate: bool,
) -> Result<()> {
    let file_id = if task.data_file_path.is_empty() {
        format!("iceberg-data-file-{task_index}")
    } else {
        format!(
            "iceberg-data-file-{task_index}-h{}",
            fnv1a64_hex(credential_free_endpoint_identity(&task.data_file_path).as_bytes())
        )
    };
    let file_format = table_format_data_file_format(task.data_file_format);
    let read_capability = read_capability_for_location(&task.data_file_path);
    let mut source_metadata =
        TableFormatParquetSourceMetadata::from_size_and_row_count(Some(task.file_size_in_bytes), task.record_count);
    if let Some(content_identity) = iceberg_task_content_identity(plan.snapshot_id.as_deref(), &task) {
        source_metadata = source_metadata.with_content_identity(content_identity);
    }
    let resolved_parquet_source =
        if task.data_file_format == DataFileFormat::Parquet && read_capability.is_native_parquet_capable() {
            Some(resolve_table_format_parquet_source_with_metadata(
                &task.data_file_path,
                &read_capability,
                parquet_read_strategy_config,
                &source_metadata,
            )?)
        } else {
            None
        };
    let mut data_file = TableFormatDataFile::new(
        file_id.clone(),
        task.data_file_path.clone(),
        file_format.clone(),
        read_capability.clone(),
    )
    .with_size_bytes(task.file_size_in_bytes);
    if let Some(resolved_parquet_source) = resolved_parquet_source {
        data_file = data_file.with_resolved_parquet_source(resolved_parquet_source);
    }
    if let Some(record_count) = task.record_count {
        data_file = data_file.with_row_count(record_count);
    }
    append_partition_values(plan, &mut data_file, &file_id, &task)?;
    if task.data_file_format != DataFileFormat::Parquet {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergNonParquetDataFile,
            file_id.clone(),
            format!("Iceberg data file format {} is not native Parquet", task.data_file_format),
        ));
    }

    if !read_capability.is_native_parquet_capable() {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergObjectStoreReadUnsupported,
            file_id.clone(),
            format!("Iceberg data file location requires {} read capability", read_capability.detail()),
        ));
    }

    if task.predicate.is_some() && !parent_filter_enforces_task_predicate {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergResidualPredicateUnsupported,
            file_id.clone(),
            "Iceberg reader predicate pushdown is not represented by Milestone B native scan facts",
        ));
    }

    if task.name_mapping.is_some() {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergNameMappingUnsupported,
            file_id.clone(),
            "Iceberg name mapping requires explicit native field-id mapping support",
        ));
    }

    let partition_spec = task.partition_spec.as_deref();
    for field_id in task.project_field_ids.iter().copied() {
        append_projected_column_mapping(plan, task.schema(), partition_spec, field_id)?;
    }

    for (delete_index, delete_file) in task.deletes.iter().enumerate() {
        append_delete_file_facts(
            plan,
            &mut data_file,
            &file_id,
            delete_index,
            delete_file,
            parquet_read_strategy_config,
        )?;
    }

    plan.data_files.push(data_file);

    Ok(())
}

fn append_partition_values(
    plan: &mut TableFormatScanPlan,
    data_file: &mut TableFormatDataFile,
    file_id: &str,
    task: &FileScanTask,
) -> Result<()> {
    let Some(partition) = task.partition.as_ref().filter(|partition| !partition.fields().is_empty()) else {
        return Ok(());
    };
    let Some(partition_spec) = task.partition_spec.as_ref() else {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergPartitionTransformUnsupported,
            file_id,
            "Iceberg partition values are present but partition spec facts are not exposed by this scan task",
        ));
        return Ok(());
    };

    let mut values = Vec::new();
    for (partition_index, partition_field) in partition_spec.fields().iter().enumerate() {
        let Some(partition_value) = partition.fields().get(partition_index) else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                TASK_CONVERSION_OPERATION,
                "Iceberg partition value count does not match the partition spec",
            )
            .with_fact("table_format", TableFormat::Iceberg.code())
            .with_fact("data_file_id", file_id)
            .with_fact("partition_field", partition_field.name.as_str())
            .with_fact("partition_index", partition_index));
        };
        let Some(source_field) = task.schema().field_by_id(partition_field.source_id) else {
            return Err(Error::new(
                ErrorCode::InvalidExecutionPlan,
                TASK_CONVERSION_OPERATION,
                "Iceberg partition source field is absent from the scan schema",
            )
            .with_fact("table_format", TableFormat::Iceberg.code())
            .with_fact("data_file_id", file_id)
            .with_fact("source_field_id", partition_field.source_id)
            .with_fact("partition_field", partition_field.name.as_str()));
        };
        let transform = table_format_partition_transform(&partition_field.transform);
        let transform_code = transform.code().to_string();
        let result_type = partition_field.transform.result_type(source_field.field_type.as_ref()).map_err(|error| {
            Error::new(
                ErrorCode::UnsupportedType,
                TASK_CONVERSION_OPERATION,
                "failed to resolve Iceberg partition transform result type",
            )
            .with_fact("table_format", TableFormat::Iceberg.code())
            .with_fact("data_file_id", file_id)
            .with_fact("source_field_id", partition_field.source_id)
            .with_fact("partition_transform", transform_code)
            .with_source(error)
        })?;
        let data_type = iceberg_type_to_arrow_data_type(&result_type)?;
        let Some(value) = table_format_partition_scalar_value(partition_value.as_ref(), &data_type) else {
            continue;
        };
        let column_name = if partition_field.transform == Transform::Identity {
            source_field.name.clone()
        } else {
            partition_field.name.clone()
        };
        values.push(TableFormatPartitionValue::new(column_name, data_type, !source_field.required, transform, value));
    }
    data_file.partition_values = values;
    Ok(())
}

fn append_projected_column_mapping(
    plan: &mut TableFormatScanPlan,
    schema: &IcebergSchema,
    partition_spec: Option<&PartitionSpec>,
    field_id: i32,
) -> Result<()> {
    let Some(field) = schema.field_by_id(field_id) else {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergFieldIdMappingUnsupported,
            field_id.to_string(),
            format!("Iceberg projected field id {field_id} is absent from the scan schema"),
        ));
        return Ok(());
    };

    let data_type = iceberg_field_to_arrow_data_type(field)?;
    let source = if schema.as_struct().field_by_id(field_id).is_some() {
        if partition_spec.is_some_and(|spec| identity_partition_source_field(spec, field_id)) {
            TableFormatColumnSource::IdentityPartitionValue
        } else {
            TableFormatColumnSource::DirectTopLevelFileColumn
        }
    } else {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergNestedProjectionUnsupported,
            field_id.to_string(),
            format!(
                "Iceberg projected field id {field_id} is nested and cannot lower to a top-level native source column"
            ),
        ));
        TableFormatColumnSource::UnsupportedNestedProjection
    };

    let mut mapping = TableFormatColumnMapping::new(
        Some(field_id),
        field.name.clone(),
        Some(field.name.clone()),
        data_type,
        !field.required,
        source,
        TableFormatColumnRole::Projected,
    );
    if let Some(default_value) = field
        .initial_default
        .as_ref()
        .and_then(|default| table_format_partition_scalar_value(Some(default), &mapping.data_type))
    {
        mapping = mapping.with_missing_default_value(default_value);
    }
    plan.projected_columns.push(mapping);

    Ok(())
}

fn identity_partition_source_field(partition_spec: &PartitionSpec, field_id: i32) -> bool {
    partition_spec.fields().iter().any(|partition_field| {
        partition_field.source_id == field_id && partition_field.transform == Transform::Identity
    })
}

fn table_format_partition_transform(transform: &Transform) -> TableFormatPartitionTransform {
    match transform {
        Transform::Identity => TableFormatPartitionTransform::Identity,
        Transform::Bucket(buckets) => TableFormatPartitionTransform::Bucket { buckets: *buckets },
        Transform::Truncate(width) => TableFormatPartitionTransform::Truncate { width: *width },
        Transform::Year => TableFormatPartitionTransform::Year,
        Transform::Month => TableFormatPartitionTransform::Month,
        Transform::Day => TableFormatPartitionTransform::Day,
        Transform::Hour => TableFormatPartitionTransform::Hour,
        Transform::Void => TableFormatPartitionTransform::Unknown("void".to_string()),
        Transform::Unknown => TableFormatPartitionTransform::Unknown("unknown".to_string()),
    }
}

fn table_format_partition_scalar_value(
    value: Option<&Literal>,
    data_type: &DataType,
) -> Option<TableFormatScalarValue> {
    let Some(value) = value else {
        return Some(TableFormatScalarValue::Null);
    };
    let primitive = value.as_primitive_literal()?;
    match primitive {
        PrimitiveLiteral::Boolean(value) => Some(TableFormatScalarValue::Boolean(value)),
        PrimitiveLiteral::Int(value) if matches!(data_type, DataType::Date32) => {
            Some(TableFormatScalarValue::Date32(value))
        },
        PrimitiveLiteral::Int(value) => Some(TableFormatScalarValue::Int32(value)),
        PrimitiveLiteral::Long(value) => Some(TableFormatScalarValue::Int64(value)),
        PrimitiveLiteral::Int128(value) if matches!(data_type, DataType::Decimal128(_, _)) => {
            Some(TableFormatScalarValue::Int128(value))
        },
        PrimitiveLiteral::UInt128(value) if matches!(data_type, DataType::Decimal128(_, _)) => {
            i128::try_from(value).ok().map(TableFormatScalarValue::Int128)
        },
        PrimitiveLiteral::Float(value) => Some(TableFormatScalarValue::Float64(f64::from(value.into_inner()))),
        PrimitiveLiteral::Double(value) => Some(TableFormatScalarValue::Float64(value.into_inner())),
        PrimitiveLiteral::String(value) => Some(TableFormatScalarValue::Utf8(value)),
        PrimitiveLiteral::Binary(_)
        | PrimitiveLiteral::Int128(_)
        | PrimitiveLiteral::UInt128(_)
        | PrimitiveLiteral::AboveMax
        | PrimitiveLiteral::BelowMin => None,
    }
}

fn append_delete_file_facts(
    plan: &mut TableFormatScanPlan,
    data_file: &mut TableFormatDataFile,
    data_file_id: &str,
    delete_index: usize,
    delete_file: &FileScanTaskDeleteFile,
    parquet_read_strategy_config: &TableFormatParquetReadStrategyConfig,
) -> Result<()> {
    let delete_id = if delete_file.file_path.is_empty() {
        format!("{data_file_id}#delete-{delete_index}")
    } else {
        format!(
            "{data_file_id}#delete-{delete_index}-h{}",
            fnv1a64_hex(credential_free_endpoint_identity(&delete_file.file_path).as_bytes())
        )
    };
    let format = table_format_data_file_format_from_path(&delete_file.file_path);
    let content = table_format_delete_content(delete_file.file_type);
    let read_capability = read_capability_for_location(&delete_file.file_path);
    let resolved_parquet_source = if format == TableFormatDataFileFormat::Parquet
        && content == TableFormatDeleteFileContent::PositionDeletes
        && read_capability.is_native_parquet_capable()
    {
        let metadata =
            TableFormatParquetSourceMetadata::from_size_and_row_count(Some(delete_file.file_size_in_bytes), None);
        Some(resolve_table_format_parquet_source_with_metadata(
            &delete_file.file_path,
            &read_capability,
            parquet_read_strategy_config,
            &metadata,
        )?)
    } else {
        None
    };
    let mut table_delete = TableFormatDeleteFile::for_data_file(
        delete_id.clone(),
        data_file_id.to_string(),
        delete_file.file_path.clone(),
        format.clone(),
        content.clone(),
    )
    .with_size_bytes(delete_file.file_size_in_bytes)
    .with_read_capability(read_capability.clone());
    if let Some(resolved_parquet_source) = resolved_parquet_source {
        table_delete = table_delete.with_resolved_parquet_source(resolved_parquet_source);
    }

    let supported = table_delete.is_supported_position_delete();
    plan.delete_files.push(table_delete.clone());
    if supported {
        data_file.push_position_delete_file(table_delete);
    } else {
        let reason = if content != TableFormatDeleteFileContent::PositionDeletes {
            format!("Iceberg {} delete files are not supported by native position-delete execution", content.code())
        } else if format != TableFormatDataFileFormat::Parquet {
            format!("Iceberg position delete file format {} is not native Parquet", format.code())
        } else {
            format!("Iceberg position delete file location requires {} read capability", read_capability.detail())
        };
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergDeleteFilesUnsupported,
            delete_id,
            reason,
        ));
    }
    Ok(())
}

fn iceberg_task_content_identity(snapshot_id: Option<&str>, task: &FileScanTask) -> Option<String> {
    let snapshot_id = snapshot_id?.trim();
    if snapshot_id.is_empty() {
        return None;
    }
    let path_hash = fnv1a64_hex(credential_free_endpoint_identity(&task.data_file_path).as_bytes());
    let rows = task.record_count.map_or_else(|| "unknown".to_string(), |rows| rows.to_string());
    Some(format!("iceberg_snapshot_{snapshot_id}_file_{path_hash}_size_{}_rows_{rows}", task.file_size_in_bytes))
}

fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Converts an Iceberg schema into the Arrow schema exposed to DataFusion.
///
/// `iceberg-rust` includes Parquet/Iceberg field-id metadata on Arrow fields.
/// DataFusion logical projections and aliases often drop that metadata, and
/// its physical/logical schema check treats the metadata difference as a hard
/// mismatch. Keep Iceberg field ids in [`TableFormatColumnMapping::field_id`]
/// and expose metadata-free Arrow schemas at DataFusion boundaries.
pub fn iceberg_schema_to_datafusion_arrow_schema(schema: &IcebergSchema) -> Result<SchemaRef> {
    let arrow_schema = iceberg::arrow::schema_to_arrow_schema(schema).map_err(|error| {
        Error::new(
            ErrorCode::UnsupportedType,
            TASK_CONVERSION_OPERATION,
            "failed to convert Iceberg schema to Arrow schema",
        )
        .with_fact("table_format", TableFormat::Iceberg.code())
        .with_source(error)
    })?;
    Ok(Arc::new(strip_arrow_schema_metadata(arrow_schema)))
}

fn strip_arrow_schema_metadata(schema: Schema) -> Schema {
    Schema::new(schema.fields().iter().map(strip_arrow_field_metadata).collect::<Vec<_>>())
}

fn strip_arrow_field_metadata(field: &FieldRef) -> Field {
    let stripped_type = strip_arrow_data_type_metadata(field.data_type());
    #[allow(deprecated)]
    let mut stripped = if let Some(dict_id) = field.dict_id() {
        #[allow(deprecated)]
        Field::new_dict(field.name(), stripped_type, field.is_nullable(), dict_id, false)
    } else {
        Field::new(field.name(), stripped_type, field.is_nullable())
    };
    if let Some(dict_is_ordered) = field.dict_is_ordered() {
        stripped = stripped.with_dict_is_ordered(dict_is_ordered);
    }
    stripped
}

fn strip_arrow_data_type_metadata(data_type: &DataType) -> DataType {
    match data_type {
        DataType::List(field) => DataType::List(Arc::new(strip_arrow_field_metadata(field))),
        DataType::ListView(field) => DataType::ListView(Arc::new(strip_arrow_field_metadata(field))),
        DataType::FixedSizeList(field, size) => {
            DataType::FixedSizeList(Arc::new(strip_arrow_field_metadata(field)), *size)
        },
        DataType::LargeList(field) => DataType::LargeList(Arc::new(strip_arrow_field_metadata(field))),
        DataType::LargeListView(field) => DataType::LargeListView(Arc::new(strip_arrow_field_metadata(field))),
        DataType::Struct(fields) => {
            DataType::Struct(Fields::from(fields.iter().map(strip_arrow_field_metadata).collect::<Vec<_>>()))
        },
        DataType::Union(fields, mode) => {
            let stripped_fields = fields.iter().map(|(_, field)| strip_arrow_field_metadata(field)).collect::<Vec<_>>();
            let type_ids = fields.iter().map(|(type_id, _)| type_id).collect::<Vec<_>>();
            DataType::Union(
                UnionFields::try_new(type_ids, stripped_fields)
                    .expect("stripping Arrow metadata preserves valid union type ids"),
                *mode,
            )
        },
        DataType::Dictionary(key_type, value_type) => DataType::Dictionary(
            Box::new(strip_arrow_data_type_metadata(key_type)),
            Box::new(strip_arrow_data_type_metadata(value_type)),
        ),
        DataType::Map(field, sorted) => DataType::Map(Arc::new(strip_arrow_field_metadata(field)), *sorted),
        DataType::RunEndEncoded(run_ends, values) => DataType::RunEndEncoded(
            Arc::new(strip_arrow_field_metadata(run_ends)),
            Arc::new(strip_arrow_field_metadata(values)),
        ),
        other => other.clone(),
    }
}

fn iceberg_field_to_arrow_data_type(field: &NestedField) -> Result<DataType> {
    iceberg_type_to_arrow_data_type(field.field_type.as_ref())
}

fn iceberg_type_to_arrow_data_type(field_type: &Type) -> Result<DataType> {
    let data_type = iceberg::arrow::type_to_arrow_type(field_type).map_err(|error| {
        Error::new(
            ErrorCode::UnsupportedType,
            TASK_CONVERSION_OPERATION,
            "failed to convert Iceberg field type to Arrow data type",
        )
        .with_fact("table_format", TableFormat::Iceberg.code())
        .with_source(error)
    })?;
    Ok(strip_arrow_data_type_metadata(&data_type))
}

fn table_format_data_file_format(format: DataFileFormat) -> TableFormatDataFileFormat {
    match format {
        DataFileFormat::Parquet => TableFormatDataFileFormat::Parquet,
        DataFileFormat::Orc => TableFormatDataFileFormat::Orc,
        DataFileFormat::Avro => TableFormatDataFileFormat::Avro,
        DataFileFormat::Puffin => TableFormatDataFileFormat::Unknown("puffin".to_string()),
    }
}

fn table_format_data_file_format_from_path(path: &str) -> TableFormatDataFileFormat {
    let extension = path.rsplit_once('.').map(|(_, extension)| extension.to_ascii_lowercase());
    match extension.as_deref() {
        Some("parquet") => TableFormatDataFileFormat::Parquet,
        Some("orc") => TableFormatDataFileFormat::Orc,
        Some("avro") => TableFormatDataFileFormat::Avro,
        Some(format) => TableFormatDataFileFormat::Unknown(format.to_string()),
        None => TableFormatDataFileFormat::Unknown("unknown".to_string()),
    }
}

fn table_format_delete_content(content: DataContentType) -> TableFormatDeleteFileContent {
    match content {
        DataContentType::PositionDeletes => TableFormatDeleteFileContent::PositionDeletes,
        DataContentType::EqualityDeletes => TableFormatDeleteFileContent::EqualityDeletes,
        DataContentType::Data => TableFormatDeleteFileContent::Unknown("data".to_string()),
    }
}

fn read_capability_for_location(location: &str) -> TableFormatReadCapability {
    let Some((scheme, _)) = location.split_once("://") else {
        return TableFormatReadCapability::local_file_system();
    };

    if scheme.eq_ignore_ascii_case("file") {
        TableFormatReadCapability::local_file_system()
    } else if scheme.is_empty() {
        TableFormatReadCapability::unsupported("empty_uri_scheme")
    } else {
        TableFormatReadCapability::object_store(scheme.to_ascii_lowercase())
    }
}

fn iceberg_planning_error(message: &'static str, source: iceberg::Error) -> Error {
    Error::new(ErrorCode::InvalidExecutionPlan, PLAN_SCAN_OPERATION, message)
        .with_fact("table_format", TableFormat::Iceberg.code())
        .with_source(source)
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use datafusion::logical_expr::{Cast, col, lit};
    use iceberg::{
        scan::FileScanTask,
        spec::{
            DataContentType, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema as IcebergSchema,
            Struct, Type,
        },
    };
    use nexus_query_engine::source::{CredentialMode, ResolvedParquetSource};

    use super::*;
    use crate::{
        FallbackReason,
        table_format::scan::{TableFormatLakehouseCachePolicy, TableFormatParquetReadStrategyPreference},
    };

    fn iceberg_schema() -> Arc<IcebergSchema> {
        Arc::new(
            IcebergSchema::builder()
                .with_fields([Arc::new(NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)))])
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn iceberg_task_conversion_uses_injected_parquet_read_strategy_config() {
        let config = TableFormatParquetReadStrategyConfig {
            preference: TableFormatParquetReadStrategyPreference::LocalCache,
            lakehouse_cache_root: Some(PathBuf::from("/tmp/cudf-iceberg-test-cache")),
            lakehouse_cache_policy: TableFormatLakehouseCachePolicy::FillBeforeRead,
            storage_endpoint: Some("http://rustfs.local:9000/minio".to_string()),
            storage_profile_id: Some("rest-s3".to_string()),
            ..Default::default()
        };
        let task = FileScanTask {
            file_size_in_bytes: 4096,
            start: 0,
            length: 4096,
            record_count: Some(17),
            data_file_path: "s3://bucket/table/data.parquet".to_string(),
            data_file_format: DataFileFormat::Parquet,
            schema: iceberg_schema(),
            project_field_ids: vec![1],
            predicate: None,
            deletes: vec![],
            partition: Some(Struct::empty()),
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
        };

        let plan = iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(
            Some("123".to_string()),
            Some("rest".to_string()),
            [task],
            &config,
        )
        .unwrap();
        let resolved = plan.data_files[0].resolved_parquet_source.as_ref().unwrap();

        assert_eq!(resolved.storage_profile_id.as_deref(), Some("rest-s3"));
        assert_eq!(resolved.storage_domain.as_deref(), Some("bucket@http://rustfs.local:9000/minio"));
        assert!(resolved.content_identity.as_deref().is_some_and(|identity| {
            identity.starts_with("iceberg_snapshot_123_file_") && identity.ends_with("_size_4096_rows_17")
        }));
        assert_eq!(resolved.cache_status.as_str(), "fill_before_read");
    }

    #[test]
    fn iceberg_snapshot_content_identity_is_complete_object_scoped() {
        let first = FileScanTask {
            file_size_in_bytes: 4096,
            start: 0,
            length: 2048,
            record_count: None,
            data_file_path: "s3://bucket/table/data.parquet".to_string(),
            data_file_format: DataFileFormat::Parquet,
            schema: iceberg_schema(),
            project_field_ids: vec![1],
            predicate: None,
            deletes: vec![],
            partition: Some(Struct::empty()),
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
        };
        let second = FileScanTask { start: 2048, length: 2048, record_count: None, ..first.clone() };

        assert_eq!(
            iceberg_task_content_identity(Some("123"), &first),
            iceberg_task_content_identity(Some("123"), &second)
        );
    }

    #[test]
    fn iceberg_snapshot_content_identity_hashes_object_path() {
        let task = FileScanTask {
            file_size_in_bytes: 4096,
            start: 0,
            length: 4096,
            record_count: Some(17),
            data_file_path: "s3://secret-bucket/table/prefix/data.parquet?token=do-not-leak".to_string(),
            data_file_format: DataFileFormat::Parquet,
            schema: iceberg_schema(),
            project_field_ids: vec![1],
            predicate: None,
            deletes: vec![],
            partition: Some(Struct::empty()),
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
        };

        let identity = iceberg_task_content_identity(Some("123"), &task).unwrap();

        assert!(identity.starts_with("iceberg_snapshot_123_file_"));
        assert!(identity.ends_with("_size_4096_rows_17"));
        assert!(!identity.contains("s3://"));
        assert!(!identity.contains("secret-bucket"));
        assert!(!identity.contains("do-not-leak"));
    }

    #[test]
    fn iceberg_data_file_id_hashes_credential_free_object_path() {
        let task = FileScanTask {
            file_size_in_bytes: 4096,
            start: 0,
            length: 4096,
            record_count: Some(17),
            data_file_path: "s3://secret-bucket/table/prefix/data.parquet?X-Amz-Signature=do-not-leak".to_string(),
            data_file_format: DataFileFormat::Parquet,
            schema: iceberg_schema(),
            project_field_ids: vec![1],
            predicate: None,
            deletes: vec![],
            partition: Some(Struct::empty()),
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
        };

        let plan =
            iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default())
                .unwrap();
        let file_id = &plan.data_files[0].id;

        assert!(file_id.starts_with("iceberg-data-file-0-h"));
        assert!(!file_id.contains("s3://"));
        assert!(!file_id.contains("secret-bucket"));
        assert!(!file_id.contains("X-Amz-Signature"));
        assert!(!file_id.contains("do-not-leak"));
    }

    #[test]
    fn iceberg_position_delete_bounds_use_reserved_manifest_fields() {
        assert_eq!(
            iceberg_position_delete_bound(&std::collections::HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_POS,
                Datum::long(7),
            )])),
            Some(7)
        );
        assert_eq!(
            iceberg_position_delete_bound(&std::collections::HashMap::from([(RESERVED_FIELD_ID_POS, Datum::long(9))])),
            Some(9)
        );
        assert_eq!(
            iceberg_position_delete_bound(&std::collections::HashMap::from([(
                RESERVED_FIELD_ID_DELETE_FILE_POS,
                Datum::long(-1),
            )])),
            None
        );
    }

    #[test]
    fn predicate_free_scan_limit_trims_data_files_before_footer_proof() {
        let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_scan_limit(Some(11))
            .with_data_file(
                TableFormatDataFile::new(
                    "file-0",
                    "s3://bucket/table/data-0.parquet",
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::object_store("s3"),
                )
                .with_row_count(5),
            )
            .with_data_file(
                TableFormatDataFile::new(
                    "file-1",
                    "s3://bucket/table/data-1.parquet",
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::object_store("s3"),
                )
                .with_row_count(7),
            )
            .with_data_file(
                TableFormatDataFile::new(
                    "file-2",
                    "s3://bucket/table/data-2.parquet",
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::object_store("s3"),
                )
                .with_row_count(100),
            );

        trim_predicate_free_scan_limit_data_files(&mut plan, false);

        assert_eq!(plan.data_files.iter().map(|file| file.id.as_str()).collect::<Vec<_>>(), vec!["file-2"]);
    }

    #[test]
    fn predicate_free_scan_limit_prunes_dropped_file_local_unsupported_features() {
        let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_scan_limit(Some(3))
            .with_data_file(
                TableFormatDataFile::new(
                    "dropped-file",
                    "s3://bucket/table/dropped.orc",
                    TableFormatDataFileFormat::Unknown("orc".to_string()),
                    TableFormatReadCapability::object_store("s3"),
                )
                .with_row_count(2),
            )
            .with_data_file(
                TableFormatDataFile::new(
                    "retained-file",
                    "s3://bucket/table/retained.parquet",
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::object_store("s3"),
                )
                .with_row_count(10),
            )
            .with_unsupported_feature(TableFormatUnsupportedFeature::for_file(
                TableFormatUnsupportedFeatureCode::IcebergNonParquetDataFile,
                "dropped-file",
                "dropped file should not keep blocking the trimmed plan",
            ));

        trim_predicate_free_scan_limit_data_files(&mut plan, false);

        assert_eq!(plan.data_files.iter().map(|file| file.id.as_str()).collect::<Vec<_>>(), vec!["retained-file"]);
        assert!(plan.unsupported_features.is_empty());
    }

    #[test]
    fn predicate_free_scan_limit_keeps_all_files_when_delete_semantics_are_present() {
        let delete_file = FileScanTaskDeleteFile {
            file_path: "s3://bucket/table/delete.parquet".to_string(),
            file_size_in_bytes: 512,
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
        };
        let task = FileScanTask {
            file_size_in_bytes: 4096,
            start: 0,
            length: 4096,
            record_count: Some(17),
            data_file_path: "s3://bucket/table/data.parquet".to_string(),
            data_file_format: DataFileFormat::Parquet,
            schema: iceberg_schema(),
            project_field_ids: vec![1],
            predicate: None,
            deletes: vec![delete_file],
            partition: Some(Struct::empty()),
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
        };
        let mut plan =
            iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default())
                .unwrap()
                .with_scan_limit(Some(1));

        trim_predicate_free_scan_limit_data_files(&mut plan, false);

        assert_eq!(plan.data_files.len(), 1);
        assert_eq!(plan.delete_files.len(), 1);
    }

    #[test]
    fn predicate_free_scan_limit_trims_delete_files_when_live_lower_bound_covers_limit() {
        let delete_0 = TableFormatDeleteFile::for_data_file(
            "delete-0",
            "file-0",
            "s3://bucket/table/delete-0.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatDeleteFileContent::PositionDeletes,
        )
        .with_row_count(9);
        let delete_1 = TableFormatDeleteFile::for_data_file(
            "delete-1",
            "file-1",
            "s3://bucket/table/delete-1.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatDeleteFileContent::PositionDeletes,
        )
        .with_row_count(5);
        let delete_2 = TableFormatDeleteFile::for_data_file(
            "delete-2",
            "file-2",
            "s3://bucket/table/delete-2.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatDeleteFileContent::PositionDeletes,
        )
        .with_row_count(1);
        let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_scan_limit(Some(2))
            .with_data_file(
                TableFormatDataFile::new(
                    "file-0",
                    "s3://bucket/table/data-0.parquet",
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::object_store("s3"),
                )
                .with_row_count(10)
                .with_position_delete_files(vec![delete_0.clone()]),
            )
            .with_data_file(
                TableFormatDataFile::new(
                    "file-1",
                    "s3://bucket/table/data-1.parquet",
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::object_store("s3"),
                )
                .with_row_count(10)
                .with_position_delete_files(vec![delete_1.clone()]),
            )
            .with_data_file(
                TableFormatDataFile::new(
                    "file-2",
                    "s3://bucket/table/data-2.parquet",
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::object_store("s3"),
                )
                .with_row_count(10)
                .with_position_delete_files(vec![delete_2.clone()]),
            );
        plan.delete_files = vec![delete_0, delete_1, delete_2];

        trim_predicate_free_scan_limit_data_files(&mut plan, false);

        assert_eq!(plan.data_files.iter().map(|file| file.id.as_str()).collect::<Vec<_>>(), vec!["file-2"]);
        assert_eq!(plan.delete_files.iter().map(|file| file.id.as_str()).collect::<Vec<_>>(), vec!["delete-2"]);
    }

    #[test]
    fn limited_delete_matching_preserves_iceberg_sequence_partition_and_reference_semantics() {
        let data_file = test_data_file("s3://bucket/table/data.parquet", 7);
        let same_sequence_position_delete =
            test_limited_delete(DataContentType::PositionDeletes, 7, Some(10), None, None);
        let older_position_delete = test_limited_delete(DataContentType::PositionDeletes, 7, Some(9), None, None);
        let wrong_spec_position_delete = test_limited_delete(DataContentType::PositionDeletes, 8, Some(10), None, None);
        let referenced_position_delete = test_limited_delete(
            DataContentType::PositionDeletes,
            7,
            Some(10),
            None,
            Some("s3://bucket/table/data.parquet"),
        );
        let other_referenced_position_delete = test_limited_delete(
            DataContentType::PositionDeletes,
            7,
            Some(10),
            None,
            Some("s3://bucket/table/other.parquet"),
        );
        let same_sequence_global_equality_delete =
            test_limited_delete(DataContentType::EqualityDeletes, 99, Some(10), Some(vec![1]), None);
        let newer_global_equality_delete =
            test_limited_delete(DataContentType::EqualityDeletes, 99, Some(11), Some(vec![1]), None);

        assert!(same_sequence_position_delete.applies_to_data_file(&data_file, 7, Some(10)));
        assert!(!older_position_delete.applies_to_data_file(&data_file, 7, Some(10)));
        assert!(!wrong_spec_position_delete.applies_to_data_file(&data_file, 7, Some(10)));
        assert!(referenced_position_delete.applies_to_data_file(&data_file, 7, Some(10)));
        assert!(!other_referenced_position_delete.applies_to_data_file(&data_file, 7, Some(10)));
        assert!(!same_sequence_global_equality_delete.applies_to_data_file(&data_file, 7, Some(10)));
        assert!(newer_global_equality_delete.applies_to_data_file(&data_file, 7, Some(10)));
    }

    #[test]
    fn iceberg_delete_file_id_hashes_credential_free_object_path() {
        let delete_file = FileScanTaskDeleteFile {
            file_path: "s3://secret-bucket/table/delete.parquet?X-Amz-Signature=do-not-leak".to_string(),
            file_size_in_bytes: 512,
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
        };
        let task = FileScanTask {
            file_size_in_bytes: 4096,
            start: 0,
            length: 4096,
            record_count: Some(17),
            data_file_path: "s3://secret-bucket/table/prefix/data.parquet?X-Amz-Signature=data-token".to_string(),
            data_file_format: DataFileFormat::Parquet,
            schema: iceberg_schema(),
            project_field_ids: vec![1],
            predicate: None,
            deletes: vec![delete_file],
            partition: Some(Struct::empty()),
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
        };

        let plan =
            iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default())
                .unwrap();
        let delete_id = &plan.delete_files[0].id;

        assert!(delete_id.contains("#delete-0-h"));
        assert!(!delete_id.contains("s3://"));
        assert!(!delete_id.contains("secret-bucket"));
        assert!(!delete_id.contains("X-Amz-Signature"));
        assert!(!delete_id.contains("do-not-leak"));
        assert!(!delete_id.contains("data-token"));
    }

    #[test]
    fn iceberg_task_delete_attachment_preserves_data_row_groups() {
        let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg);
        let mut data_file = TableFormatDataFile::new(
            "data-0",
            "s3://bucket/table/data.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_row_groups(vec![2, 3]);
        let delete_file = FileScanTaskDeleteFile {
            file_path: "s3://bucket/table/delete.parquet".to_string(),
            file_size_in_bytes: 512,
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
        };

        append_delete_file_facts(&mut plan, &mut data_file, "data-0", 0, &delete_file, &Default::default()).unwrap();

        assert_eq!(data_file.row_groups, Some(vec![2, 3]));
        assert_eq!(data_file.position_delete_files.len(), 1);
        assert_eq!(plan.delete_files.len(), 1);
    }

    fn test_data_file(path: &str, partition_spec_id: i32) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .partition(Struct::empty())
            .partition_spec_id(partition_spec_id)
            .record_count(10)
            .file_size_in_bytes(1024)
            .build()
            .unwrap()
    }

    fn test_limited_delete(
        file_type: DataContentType,
        partition_spec_id: i32,
        sequence_number: Option<i64>,
        equality_ids: Option<Vec<i32>>,
        referenced_data_file: Option<&str>,
    ) -> LimitedDeleteFile {
        LimitedDeleteFile {
            file_path: "s3://bucket/table/delete.parquet".to_string(),
            file_size_in_bytes: 512,
            file_type,
            partition_spec_id,
            equality_ids,
            sequence_number,
            record_count: 1,
            partition: Struct::empty(),
            referenced_data_file: referenced_data_file.map(str::to_string),
            position_lower_bound: None,
            position_upper_bound: None,
        }
    }

    #[test]
    fn attach_manifest_delete_metadata_copies_row_count_and_position_bounds_to_associated_delete_files() {
        let table_delete = TableFormatDeleteFile::for_data_file(
            "delete-0",
            "data-0",
            "s3://bucket/table/delete.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatDeleteFileContent::PositionDeletes,
        )
        .with_read_capability(TableFormatReadCapability::object_store("s3"))
        .with_resolved_parquet_source(ResolvedParquetSource::remote_direct(
            "s3://bucket/table/delete.parquet",
            CredentialMode::SharedProcessEnv,
        ));
        let data_file = TableFormatDataFile::new(
            "data-0",
            "s3://bucket/table/data.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatReadCapability::object_store("s3"),
        )
        .with_position_delete_files(vec![table_delete.clone()]);
        let mut plan =
            TableFormatScanPlan::new(TableFormat::Iceberg).with_data_file(data_file).with_delete_file(table_delete);
        let mut limited_delete = test_limited_delete(
            DataContentType::PositionDeletes,
            0,
            Some(10),
            None,
            Some("s3://bucket/table/data.parquet"),
        );
        limited_delete.record_count = 7;
        limited_delete.position_lower_bound = Some(11);
        limited_delete.position_upper_bound = Some(29);

        attach_manifest_delete_metadata(&mut plan, 0, &[limited_delete]);

        let associated_delete = &plan.data_files[0].position_delete_files[0];
        assert_eq!(associated_delete.row_count, Some(7));
        assert_eq!(associated_delete.referenced_data_file.as_deref(), Some("s3://bucket/table/data.parquet"));
        assert_eq!(associated_delete.position_lower_bound, Some(11));
        assert_eq!(associated_delete.position_upper_bound, Some(29));
        assert_eq!(plan.delete_files[0].row_count, Some(7));
        assert_eq!(plan.delete_files[0].referenced_data_file.as_deref(), Some("s3://bucket/table/data.parquet"));
        assert_eq!(plan.delete_files[0].position_lower_bound, Some(11));
        assert_eq!(plan.delete_files[0].position_upper_bound, Some(29));
    }

    #[test]
    fn iceberg_position_delete_files_are_associated_for_native_execution() {
        let delete_file = FileScanTaskDeleteFile {
            file_path: "s3://bucket/table/delete.parquet".to_string(),
            file_size_in_bytes: 512,
            file_type: DataContentType::PositionDeletes,
            partition_spec_id: 0,
            equality_ids: None,
        };
        let task = FileScanTask {
            file_size_in_bytes: 4096,
            start: 0,
            length: 4096,
            record_count: Some(17),
            data_file_path: "s3://bucket/table/data.parquet".to_string(),
            data_file_format: DataFileFormat::Parquet,
            schema: iceberg_schema(),
            project_field_ids: vec![1],
            predicate: None,
            deletes: vec![delete_file],
            partition: Some(Struct::empty()),
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
        };

        let plan =
            iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default())
                .unwrap();

        assert!(plan.unsupported_features.is_empty());
        assert_eq!(plan.delete_files.len(), 1);
        assert_eq!(plan.data_files[0].position_delete_files.len(), 1);
        assert_eq!(plan.delete_files[0].data_file_id, plan.data_files[0].id);
        assert_eq!(plan.delete_files[0].id, plan.data_files[0].position_delete_files[0].id);
        assert!(plan.delete_files[0].is_supported_position_delete());
        assert!(plan.is_native_admissible());
    }

    #[test]
    fn iceberg_equality_delete_files_remain_structured_rejections() {
        let delete_file = FileScanTaskDeleteFile {
            file_path: "s3://bucket/table/delete.parquet".to_string(),
            file_size_in_bytes: 512,
            file_type: DataContentType::EqualityDeletes,
            partition_spec_id: 0,
            equality_ids: Some(vec![1]),
        };
        let task = FileScanTask {
            file_size_in_bytes: 4096,
            start: 0,
            length: 4096,
            record_count: Some(17),
            data_file_path: "s3://bucket/table/data.parquet".to_string(),
            data_file_format: DataFileFormat::Parquet,
            schema: iceberg_schema(),
            project_field_ids: vec![1],
            predicate: None,
            deletes: vec![delete_file],
            partition: Some(Struct::empty()),
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
        };

        let plan =
            iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default())
                .unwrap();

        assert_eq!(plan.data_files[0].position_delete_files.len(), 0);
        assert_eq!(plan.unsupported_fallback_reasons(), vec![FallbackReason::IcebergDeleteFilesUnsupported]);
        assert_eq!(plan.first_rejection_reason(), Some(FallbackReason::IcebergDeleteFilesUnsupported));
        assert!(!plan.is_native_admissible());
    }

    #[test]
    fn iceberg_task_conversion_uses_weak_identity_without_snapshot() {
        let task = FileScanTask {
            file_size_in_bytes: 4096,
            start: 0,
            length: 4096,
            record_count: Some(17),
            data_file_path: "s3://bucket/table/data.parquet".to_string(),
            data_file_format: DataFileFormat::Parquet,
            schema: iceberg_schema(),
            project_field_ids: vec![1],
            predicate: None,
            deletes: vec![],
            partition: Some(Struct::empty()),
            partition_spec: None,
            name_mapping: None,
            case_sensitive: true,
        };

        let plan =
            iceberg_file_scan_tasks_to_table_format_scan_plan_with_config(None, None, [task], &Default::default())
                .unwrap();
        let resolved = plan.data_files[0].resolved_parquet_source.as_ref().unwrap();

        assert_eq!(resolved.content_identity.as_deref(), Some("weak_size_4096_rows_17"));
    }

    #[test]
    fn datafusion_filters_convert_to_iceberg_metadata_predicate() {
        let filters = vec![
            datafusion::logical_expr::binary_expr(
                col("id"),
                Operator::GtEq,
                lit(datafusion::common::ScalarValue::Int64(Some(10))),
            ),
            datafusion::logical_expr::binary_expr(
                lit(datafusion::common::ScalarValue::Utf8(Some("BUILDING".to_string()))),
                Operator::Eq,
                col("status"),
            ),
        ];

        let predicate = datafusion_filters_to_iceberg_predicate(&filters).unwrap();
        let rendered = predicate.to_string();

        assert!(rendered.contains("id >= 10"));
        assert!(rendered.contains("status = \"BUILDING\""));
    }

    #[test]
    fn unsupported_datafusion_filter_does_not_block_supported_filter_pushdown() {
        let filters = vec![datafusion::logical_expr::binary_expr(
            datafusion::logical_expr::binary_expr(
                col("id"),
                Operator::Lt,
                lit(datafusion::common::ScalarValue::Int64(Some(99))),
            ),
            Operator::And,
            lit(datafusion::common::ScalarValue::Boolean(Some(true))),
        )];

        let predicate = datafusion_filters_to_iceberg_predicate(&filters).unwrap();

        assert_eq!(predicate.to_string(), "id < 99");
    }

    #[test]
    fn cast_filters_are_not_pushed_into_iceberg_metadata_predicates() {
        let filters = vec![datafusion::logical_expr::binary_expr(
            Expr::Cast(Cast::new(Box::new(col("float_col")), DataType::Int32)),
            Operator::Eq,
            lit(datafusion::common::ScalarValue::Int32(Some(1))),
        )];

        assert!(datafusion_filters_to_iceberg_predicate(&filters).is_none());
    }

    #[test]
    fn not_of_partially_supported_and_is_not_pushed_into_iceberg_metadata_predicates() {
        let supported = datafusion::logical_expr::binary_expr(
            col("id"),
            Operator::Lt,
            lit(datafusion::common::ScalarValue::Int64(Some(99))),
        );
        let unsupported = datafusion::logical_expr::binary_expr(
            Expr::Cast(Cast::new(Box::new(col("float_col")), DataType::Int32)),
            Operator::Eq,
            lit(datafusion::common::ScalarValue::Int32(Some(1))),
        );
        let filters =
            vec![Expr::Not(Box::new(datafusion::logical_expr::binary_expr(supported, Operator::And, unsupported)))];

        assert!(datafusion_filters_to_iceberg_predicate(&filters).is_none());
    }
}
