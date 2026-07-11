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
            DataContentType, DataFileBuilder, DataFileFormat, ManifestStatus, MappedField, NameMapping, NestedField,
            PrimitiveType, Schema as IcebergSchema, Struct, Type,
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
    fn predicate_free_limit_manifest_entry_task_carries_table_name_mapping() {
        let data_file = DataFileBuilder::default()
            .partition_spec_id(0)
            .content(DataContentType::Data)
            .file_path("s3://bucket/table/data.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(4096)
            .record_count(17)
            .partition(Struct::empty())
            .key_metadata(None)
            .build()
            .unwrap();
        let entry = ManifestEntry {
            status: ManifestStatus::Added,
            snapshot_id: Some(7),
            sequence_number: Some(11),
            file_sequence_number: Some(13),
            data_file,
        };
        let name_mapping = Arc::new(NameMapping::new(vec![MappedField::new(
            Some(1),
            vec!["id".to_string(), "old_id".to_string()],
            vec![],
        )]));

        let task = file_scan_task_from_manifest_entry(
            &entry,
            iceberg_schema(),
            Arc::new(vec![1]),
            None,
            Some(Arc::clone(&name_mapping)),
            Vec::new(),
            true,
        );

        let task_mapping = task.name_mapping.expect("predicate-free LIMIT task should carry table name mapping");
        assert_eq!(task_mapping.fields(), name_mapping.fields());
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
    fn predicate_free_scan_limit_does_not_trim_equality_delete_files_by_metadata_lower_bound() {
        let delete_0 = TableFormatDeleteFile::for_data_file(
            "delete-0",
            "file-0",
            "s3://bucket/table/delete-0.parquet",
            TableFormatDataFileFormat::Parquet,
            TableFormatDeleteFileContent::EqualityDeletes,
        )
        .with_row_count(10)
        .with_equality_ids([1]);
        let mut plan = TableFormatScanPlan::new(TableFormat::Iceberg)
            .with_scan_limit(Some(1))
            .with_data_file(
                TableFormatDataFile::new(
                    "file-0",
                    "s3://bucket/table/data-0.parquet",
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::object_store("s3"),
                )
                .with_row_count(10)
                .with_equality_delete_files(vec![delete_0.clone()]),
            )
            .with_data_file(
                TableFormatDataFile::new(
                    "file-1",
                    "s3://bucket/table/data-1.parquet",
                    TableFormatDataFileFormat::Parquet,
                    TableFormatReadCapability::object_store("s3"),
                )
                .with_row_count(1),
            )
            .with_delete_file(delete_0);

        assert_eq!(scan_plan_live_row_lower_bound(&plan), None);

        trim_predicate_free_scan_limit_data_files(&mut plan, false);

        assert_eq!(plan.data_files.iter().map(|file| file.id.as_str()).collect::<Vec<_>>(), vec!["file-0", "file-1"]);
        assert_eq!(plan.delete_files.iter().map(|file| file.id.as_str()).collect::<Vec<_>>(), vec!["delete-0"]);
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
    fn iceberg_equality_delete_files_without_footer_field_id_proof_are_structured_rejections() {
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
        assert_eq!(plan.data_files[0].equality_delete_files.len(), 0);
        assert_eq!(plan.delete_files.len(), 1);
        assert_eq!(plan.delete_files[0].data_file_id, plan.data_files[0].id);
        assert_eq!(plan.delete_files[0].equality_ids.as_deref(), Some([1].as_slice()));
        assert!(plan.delete_files[0].is_supported_equality_delete());
        assert_eq!(plan.unsupported_fallback_reasons(), vec![FallbackReason::IcebergDeleteFilesUnsupported]);
        assert_eq!(plan.first_rejection_reason(), Some(FallbackReason::IcebergDeleteFilesUnsupported));
        assert!(!plan.is_native_admissible());
    }

    #[test]
    fn iceberg_equality_delete_files_without_key_ids_remain_structured_rejections() {
        let delete_file = FileScanTaskDeleteFile {
            file_path: "s3://bucket/table/delete.parquet".to_string(),
            file_size_in_bytes: 512,
            file_type: DataContentType::EqualityDeletes,
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

        assert_eq!(plan.data_files[0].equality_delete_files.len(), 0);
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
