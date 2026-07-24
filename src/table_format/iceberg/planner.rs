//! Iceberg metadata planning adapter for table-format scan facts.
//!
//! This module consumes `iceberg-rust` planning output and preserves table
//! format facts at the adapter boundary. It does not execute Arrow readers.

use std::{
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
use datafusion_nexus_iceberg::{
    IcebergColumnBinding, IcebergColumnRole, IcebergColumnSource, IcebergPredicateIntent, IcebergProjection,
    IcebergScanPlanningOptions, IcebergScanRequest, IcebergSourceLimit, ResolvedIcebergTable, S3CredentialMode,
    S3CredentialRef, S3FileIoConfigVerification, SourceAccessPolicy,
};
use futures::{StreamExt, future::BoxFuture};
use iceberg::{
    expr::Reference,
    io::{FileIO, FileRead},
    scan::{FileScanTask, FileScanTaskDeleteFile},
    spec::{
        DataContentType, DataFileFormat, Datum, Literal, NameMapping as IcebergNameMapping, NestedField, PartitionSpec,
        PrimitiveLiteral, Schema as IcebergSchema, Transform, Type,
    },
    table::Table,
};
use iceberg_datafusion::IcebergStaticTableProvider;
use nexus_query_engine::{
    datatypes::is_supported_arrow_type,
    runtime::{S3SourceStorageProfile, SourceStorageProfile},
    source::CredentialMode,
};
use parquet::file::{
    FOOTER_SIZE,
    metadata::{FooterTail, ParquetMetaData, ParquetMetaDataReader},
};

use super::adapter::root_error_from_iceberg;
use crate::{
    error::{Error, ErrorCode, Result},
    table_format::{
        scan::{
            DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID, TableFormat, TableFormatColumnMapping,
            TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile, TableFormatDataFileFormat,
            TableFormatDeleteFile, TableFormatDeleteFileContent, TableFormatLiteral,
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
    ResolvedTable(Box<ResolvedIcebergTable>),
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
        let resolved =
            ResolvedIcebergTable::from_table(table, root_environment_source_access_policy()).map_err(|error| {
                root_error_from_iceberg(
                    error,
                    PLAN_SCAN_OPERATION,
                    "failed to freeze the Iceberg table snapshot for root adaptation",
                )
            })?;
        Self::try_new_resolved_with_optional_parquet_read_strategy_config(resolved, config)
    }

    /// Builds a provider from the frozen leaf table resolution. The same
    /// resolution supplies the CPU delegate and all native candidate planning;
    /// neither path may rediscover a newer snapshot or source-access policy.
    pub fn try_new_resolved(resolved: ResolvedIcebergTable) -> Result<Self> {
        Self::try_new_resolved_with_optional_parquet_read_strategy_config(resolved, None)
    }

    /// Builds a provider from a frozen leaf table resolution with root-owned
    /// native source-strategy configuration.
    pub fn try_new_resolved_with_parquet_read_strategy_config(
        resolved: ResolvedIcebergTable,
        config: TableFormatParquetReadStrategyConfig,
    ) -> Result<Self> {
        Self::try_new_resolved_with_optional_parquet_read_strategy_config(resolved, Some(config))
    }

    fn try_new_resolved_with_optional_parquet_read_strategy_config(
        resolved: ResolvedIcebergTable,
        config: Option<TableFormatParquetReadStrategyConfig>,
    ) -> Result<Self> {
        let resolved_snapshot_id = resolved.facts().snapshot.snapshot_id;
        let schema = Arc::new(strip_arrow_schema_metadata(resolved.facts().schema.arrow_schema.as_ref().clone()));
        let planner = match config {
            Some(config) => IcebergTableFormatPlanner::new(resolved.clone()).with_parquet_read_strategy_config(config),
            None => IcebergTableFormatPlanner::new(resolved.clone()),
        };
        Ok(Self {
            schema,
            cpu_source: NexusIcebergCpuSource::ResolvedTable(Box::new(resolved)),
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

    /// Returns the source-storage profile derived from the same frozen leaf
    /// policy as this provider's CPU delegate and native candidate planner.
    ///
    /// An injected CPU provider has no leaf source policy and therefore does
    /// not expose a profile here.
    pub fn native_source_storage_profile(&self) -> Option<SourceStorageProfile> {
        match &self.cpu_source {
            NexusIcebergCpuSource::ResolvedTable(resolved) => {
                Some(native_source_storage_profile_from_resolved(resolved))
            },
            NexusIcebergCpuSource::InjectedProvider(_) => None,
        }
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
            NexusIcebergCpuSource::ResolvedTable(resolved) => {
                let provider = match self.resolved_snapshot_id {
                    Some(snapshot_id) => {
                        IcebergStaticTableProvider::try_new_from_table_snapshot(resolved.table().clone(), snapshot_id)
                            .await
                    },
                    None => IcebergStaticTableProvider::try_new_from_table(resolved.table().clone()).await,
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
        let resolved_not_supported = resolved_scan.first_not_supported_reason();
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
            Err(_error) if resolved_not_supported.is_some() => {
                NexusIcebergNativeScanOutcome::from_scan_plan(resolved_scan)
            },
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
    resolved: ResolvedIcebergTable,
    catalog_type: Option<String>,
    parquet_read_strategy_config: TableFormatParquetReadStrategyConfig,
}

impl IcebergTableFormatPlanner {
    /// Freezes a raw Iceberg table before creating the root inspection planner.
    ///
    /// Prefer [`Self::new`] when a catalog integration already resolved the
    /// table: that lets its CPU delegate and native planner share one exact
    /// snapshot and source-access policy.
    pub fn try_new(table: Table) -> Result<Self> {
        let resolved =
            ResolvedIcebergTable::from_table(table, root_environment_source_access_policy()).map_err(|error| {
                root_error_from_iceberg(
                    error,
                    PLAN_SCAN_OPERATION,
                    "failed to freeze the Iceberg table snapshot for root inspection",
                )
            })?;
        Ok(Self::new(resolved))
    }

    /// Builds a root native-inspection planner over one frozen leaf table resolution.
    ///
    /// Manifest/task planning is delegated to `ResolvedIcebergTable`; this
    /// root planner owns only source strategy and footer inspection.
    pub fn new(resolved: ResolvedIcebergTable) -> Self {
        let parquet_read_strategy_config = parquet_read_strategy_config_from_resolved_source_access(
            &resolved,
            TableFormatParquetReadStrategyConfig::from_env(),
        );
        Self { resolved, catalog_type: None, parquet_read_strategy_config }
    }

    /// Records a default catalog label when the scan request does not provide one.
    pub fn with_catalog_type(mut self, catalog_type: impl Into<String>) -> Self {
        self.catalog_type = Some(catalog_type.into());
        self
    }

    /// Sets the table-format Parquet read-strategy config used while resolving
    /// Iceberg data-file scan facts. Leaf-owned source-access facts always
    /// override environment-derived storage identity so the CPU delegate and
    /// native plan use the one policy frozen in `ResolvedIcebergTable`.
    pub fn with_parquet_read_strategy_config(mut self, config: TableFormatParquetReadStrategyConfig) -> Self {
        self.parquet_read_strategy_config =
            parquet_read_strategy_config_from_resolved_source_access(&self.resolved, config);
        self
    }

    /// Builds the exact native source-storage profile for this frozen table.
    ///
    /// Callers installing a resolved Iceberg table into an execution context
    /// must register this profile under its stable id. It is constructed from
    /// the leaf's frozen policy rather than from a later environment lookup.
    pub fn native_source_storage_profile(&self) -> SourceStorageProfile {
        native_source_storage_profile_from_resolved(&self.resolved)
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
        let leaf_request = iceberg_scan_request_from_table_format_request(&self.resolved, &request)?;
        let snapshot_id = request
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
        let options = IcebergScanPlanningOptions {
            snapshot_id,
            case_sensitive: request.case_sensitive,
            manifest_concurrency: REMOTE_ICEBERG_METADATA_PLANNING_CONCURRENCY,
            timeout: timeout_context.as_ref().map(|context| context.remaining("leaf_plan_scan")).transpose()?,
            cancellation: None,
        };
        let candidate = self.resolved.plan_scan(leaf_request, options).await.map_err(|error| {
            root_error_from_iceberg(
                error,
                PLAN_SCAN_OPERATION,
                "failed to plan the frozen Iceberg snapshot for root adaptation",
            )
        })?;
        if let Some(context) = timeout_context.as_ref() {
            context.check("after_leaf_plan_scan")?;
        }
        let catalog_type = request.catalog_type.as_deref().or(self.catalog_type.as_deref());
        let mut plan = super::adapter::iceberg_scan_candidate_to_table_format_scan_plan(candidate, catalog_type)?
            .with_scan_limit(request.scan_limit);
        append_unverified_remote_s3_source_access_rejection(
            &mut plan,
            self.resolved.file_io_source_access_verification(),
        );
        if request.row_selection_enabled {
            plan.unsupported_features.push(TableFormatUnsupportedFeature::new(
                TableFormatUnsupportedFeatureCode::IcebergRowSelectionUnsupported,
                "Iceberg row selection/page-index semantics are not represented by native scan facts",
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
            self.resolved.table().file_io(),
            request_has_predicate,
            timeout_context.as_ref(),
        )
        .await?;
        trim_predicate_free_scan_limit_data_files(&mut plan, request_has_predicate);
        attach_remote_footer_schemas_for_field_id_proof(
            &mut plan,
            self.resolved.table().file_io(),
            timeout_context.as_ref(),
        )
        .await?;
        Ok(plan)
    }
}

/// Remote S3 native reads need proof that their native profile is derived from
/// the same FileIO configuration used by the executable CPU delegate. Local
/// FileIO intentionally has no S3 properties and therefore stays eligible.
fn append_unverified_remote_s3_source_access_rejection(
    plan: &mut TableFormatScanPlan,
    file_io_source_access_verification: S3FileIoConfigVerification,
) {
    let has_remote_s3_source = plan
        .data_files
        .iter()
        .map(|file| file.location.as_str())
        .chain(plan.delete_files.iter().map(|file| file.location.as_str()))
        .any(is_s3_location);
    if has_remote_s3_source && file_io_source_access_verification != S3FileIoConfigVerification::ExactPolicy {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::new(
            TableFormatUnsupportedFeatureCode::IcebergFileIoSourceAccessUnverified,
            "Iceberg remote S3 FileIO configuration is not proven to match the frozen source-access policy",
        ));
    }
}

fn is_s3_location(location: &str) -> bool {
    location.split_once(':').is_some_and(|(scheme, _)| scheme.eq_ignore_ascii_case("s3"))
}

/// Translates root table-format request intent into the DataFusion-free leaf
/// request contract. The root owns DataFusion predicate parsing; the leaf owns
/// binding those names to the frozen Iceberg snapshot and planning its tasks.
fn iceberg_scan_request_from_table_format_request(
    resolved: &ResolvedIcebergTable,
    request: &TableFormatScanRequest,
) -> Result<IcebergScanRequest> {
    let projection = match request.projected_column_names.as_deref() {
        None => IcebergProjection::AllColumns,
        Some(columns) => IcebergProjection::Columns(
            columns
                .iter()
                .map(|column| {
                    iceberg_top_level_binding(resolved, column, request.case_sensitive, IcebergColumnRole::Projected)
                })
                .collect::<Result<Vec<_>>>()?,
        ),
    };
    let filter_bindings = request
        .predicate
        .as_ref()
        .map(table_format_predicate_columns)
        .unwrap_or_default()
        .into_iter()
        .map(|column| {
            iceberg_top_level_binding(resolved, &column, request.case_sensitive, IcebergColumnRole::FilterOnly)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(IcebergScanRequest {
        projection,
        filter_bindings,
        predicate: IcebergPredicateIntent::none(),
        iceberg_predicate: request.predicate.as_ref().and_then(table_format_predicate_to_iceberg_predicate),
        source_limit: IcebergSourceLimit { row_limit: request.scan_limit.map(|limit| limit as u64), file_limit: None },
        pruning: Default::default(),
    })
}

fn iceberg_top_level_binding(
    resolved: &ResolvedIcebergTable,
    name: &str,
    case_sensitive: bool,
    role: IcebergColumnRole,
) -> Result<IcebergColumnBinding> {
    let matches_name = |candidate: &str| {
        if case_sensitive { candidate == name } else { candidate.eq_ignore_ascii_case(name) }
    };
    let matches = resolved.facts().schema.fields.iter().filter(|field| matches_name(&field.name)).collect::<Vec<_>>();
    let [field] = matches.as_slice() else {
        let message = if matches.is_empty() {
            "Iceberg scan request references a top-level field absent from the frozen schema"
        } else {
            "Iceberg scan request is ambiguous under case-insensitive field binding"
        };
        return Err(Error::new(ErrorCode::InvalidExecutionPlan, PLAN_SCAN_OPERATION, message)
            .with_fact("table_format", TableFormat::Iceberg.code())
            .with_fact("column", name));
    };
    Ok(IcebergColumnBinding {
        field_id: field.field_id,
        output_name: name.to_string(),
        source: IcebergColumnSource::PhysicalFile,
        role,
        missing_default: field.initial_default.clone(),
    })
}

fn table_format_predicate_columns(predicate: &TableFormatPredicate) -> Vec<String> {
    fn append(predicate: &TableFormatPredicate, columns: &mut Vec<String>) {
        match predicate {
            TableFormatPredicate::And(left, right) | TableFormatPredicate::Or(left, right) => {
                append(left, columns);
                append(right, columns);
            },
            TableFormatPredicate::Not(predicate) => append(predicate, columns),
            TableFormatPredicate::BooleanColumn(column)
            | TableFormatPredicate::IsNull(column)
            | TableFormatPredicate::IsNotNull(column)
            | TableFormatPredicate::Comparison { column, .. } => {
                if !columns.contains(column) {
                    columns.push(column.clone());
                }
            },
        }
    }

    let mut columns = Vec::new();
    append(predicate, &mut columns);
    columns
}

/// Projects a frozen leaf source-access policy onto root-native inspection
/// configuration. Cache policy remains root-owned, but endpoint and
/// credential-mode facts may not be rediscovered from a later environment.
fn parquet_read_strategy_config_from_resolved_source_access(
    resolved: &ResolvedIcebergTable,
    mut config: TableFormatParquetReadStrategyConfig,
) -> TableFormatParquetReadStrategyConfig {
    let source_access = resolved.source_access_policy().s3_access();
    config.storage_endpoint = source_access.endpoint.map(credential_free_endpoint_identity);
    let requires_scoped_profile = source_access.endpoint.is_some()
        || source_access.region.is_some()
        || source_access.path_style_access.is_some()
        || !matches!(source_access.credential.mode(), S3CredentialMode::DefaultChain);
    if requires_scoped_profile {
        config.storage_profile_id = Some(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID.to_string());
        config.credential_mode = CredentialMode::ScopedResolved;
        config.remote_direct_profile_unsupported_reason =
            native_source_storage_profile_from_resolved(resolved).remote_direct_profile_unsupported_reason();
    } else {
        config.storage_profile_id = None;
        config.credential_mode = CredentialMode::SharedProcessEnv;
    }
    config
}

/// Converts leaf-owned source access into the root native-engine profile once.
/// The returned value deliberately stays out of scan candidates and reports:
/// it can hold static credentials, while the candidate must remain
/// credential-free.
fn native_source_storage_profile_from_resolved(resolved: &ResolvedIcebergTable) -> SourceStorageProfile {
    source_access_policy_to_native_source_storage_profile(resolved.source_access_policy())
}

/// Converts one frozen leaf source-access policy into the native-engine
/// profile that must be registered before a statement executes. This is the
/// sole root crossing for static credential material; callers must not put the
/// returned profile in diagnostics or scan candidates.
pub fn source_access_policy_to_native_source_storage_profile(
    source_access_policy: &SourceAccessPolicy,
) -> SourceStorageProfile {
    let source_access = source_access_policy.s3_access();
    let mut s3 = S3SourceStorageProfile::default_chain();
    if let Some(region) = source_access.region {
        s3 = s3.with_region(region.to_string());
    }
    if let Some(endpoint) = source_access.endpoint {
        s3 = s3.with_endpoint(endpoint.to_string());
    }
    if let Some(path_style_access) = source_access.path_style_access {
        s3 = s3.with_path_style_access(path_style_access);
    }
    s3 = match source_access.credential {
        S3CredentialRef::DefaultChain => s3,
        S3CredentialRef::Static { access_key_id, secret_access_key, session_token } => {
            s3.with_static_credentials(access_key_id, secret_access_key, session_token.map(str::to_string))
        },
        S3CredentialRef::Unsigned => s3.with_unsigned_requests(),
    };
    SourceStorageProfile::s3(DEFAULT_TABLE_FORMAT_S3_SOURCE_STORAGE_PROFILE_ID, s3)
}

/// Freezes the legacy root-constructor source policy from the process exactly
/// once. New catalog callers should pass `ResolvedIcebergTable` directly;
/// this bridge keeps raw-table constructors from letting leaf planning and
/// native inspection independently reinterpret the environment.
fn root_environment_source_access_policy() -> SourceAccessPolicy {
    let read = |name| super::non_empty(std::env::var(name).ok());
    let region = read(super::NEXUS_ICEBERG_S3_REGION_ENV)
        .or_else(|| read(super::AWS_REGION_ENV))
        .or_else(|| read(super::AWS_DEFAULT_REGION_ENV));
    let endpoint = read(super::NEXUS_ICEBERG_S3_ENDPOINT_ENV)
        .or_else(|| read(crate::table_format::AWS_ENDPOINT_URL_S3_ENV))
        .or_else(|| read(crate::table_format::AWS_ENDPOINT_URL_ENV));
    let path_style_access = read(super::NEXUS_ICEBERG_S3_PATH_STYLE_ENV).and_then(|value| value.parse::<bool>().ok());
    let explicit_credentials = (
        read(super::NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV),
        read(super::NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV),
        read(super::NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV),
    );
    let aws_credentials = (
        read(super::AWS_ACCESS_KEY_ID_ENV),
        read(super::AWS_SECRET_ACCESS_KEY_ENV),
        read(super::AWS_SESSION_TOKEN_ENV).or_else(|| read(super::AWS_SECURITY_TOKEN_ENV)),
    );

    let source_access = match endpoint {
        Some(endpoint) => SourceAccessPolicy::s3_custom_endpoint(endpoint, region, path_style_access),
        None => SourceAccessPolicy::s3_default_chain(region),
    };
    let mut s3 = source_access.as_s3().clone();
    s3 = if let (Some(access_key_id), Some(secret_access_key), session_token) = explicit_credentials {
        s3.with_static_credentials(access_key_id, secret_access_key, session_token)
    } else if s3.endpoint().is_some()
        && super::s3_credential_source_default_chain_opt_in(|name| std::env::var(name).ok())
    {
        s3.with_default_chain()
    } else if s3.endpoint().is_none()
        && let (Some(access_key_id), Some(secret_access_key), session_token) = aws_credentials
    {
        s3.with_static_credentials(access_key_id, secret_access_key, session_token)
    } else {
        s3
    };
    SourceAccessPolicy::S3(s3)
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
        iceberg_metadata_planning_failure_facts(self.resolved.table())
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

#[cfg(test)]
mod tests;
