//! Iceberg metadata planning adapter for table-format scan facts.
//!
//! This module consumes `iceberg-rust` planning output and preserves table
//! format facts at the adapter boundary. It does not execute Arrow readers.

use std::{any::Any, fmt, sync::Arc};

use arrow::datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef, UnionFields};
use async_trait::async_trait;
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
    scan::{FileScanTask, FileScanTaskDeleteFile},
    spec::{DataContentType, DataFileFormat, Datum, NestedField, Schema as IcebergSchema, Type},
    table::Table,
};

use crate::{
    Error, ErrorCode, Result,
    table_format::{
        scan::{
            TableFormat, TableFormatColumnMapping, TableFormatColumnRole, TableFormatColumnSource, TableFormatDataFile,
            TableFormatDataFileFormat, TableFormatDeleteFile, TableFormatDeleteFileContent, TableFormatLiteral,
            TableFormatParquetReadStrategyConfig, TableFormatParquetSourceMetadata, TableFormatPlanner,
            TableFormatPredicate, TableFormatPredicateOp, TableFormatReadCapability, TableFormatScanPlan,
            TableFormatScanRequest, TableFormatUnsupportedFeature, TableFormatUnsupportedFeatureCode,
            credential_free_endpoint_identity, resolve_table_format_parquet_source_with_metadata,
        },
        scan_exec::CudfTableFormatScanExec,
    },
};

const PLAN_SCAN_OPERATION: &str = "datafusion_nexus.table_format.iceberg.plan_scan";
const TASK_CONVERSION_OPERATION: &str = "datafusion_nexus.table_format.iceberg.convert_file_scan_tasks";
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
        Self { schema, planner, catalog_type: None }
    }

    /// Records a stable catalog label in scan requests and diagnostics.
    pub fn with_catalog_type(mut self, catalog_type: impl Into<String>) -> Self {
        self.catalog_type = Some(catalog_type.into());
        self
    }
}

impl fmt::Debug for CudfIcebergTableProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudfIcebergTableProvider")
            .field("schema", &self.schema)
            .field("catalog_type", &self.catalog_type)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for CudfIcebergTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

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
        let scan_plan = self.planner.plan_scan(request).await.map_err(DataFusionError::from)?;
        Ok(Arc::new(CudfTableFormatScanExec::new(scan_plan, projected_schema)))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
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
        let row_selection_enabled = request.row_selection_enabled;
        let mut scan_builder = self
            .table
            .scan()
            .with_case_sensitive(request.case_sensitive)
            .with_row_group_filtering_enabled(request.row_group_filtering_enabled)
            .with_row_selection_enabled(request.row_selection_enabled);

        if let Some(snapshot_id) = request.snapshot_id.as_deref() {
            let parsed_snapshot_id = snapshot_id.parse::<i64>().map_err(|error| {
                Error::new(
                    ErrorCode::InvalidExecutionPlan,
                    PLAN_SCAN_OPERATION,
                    "invalid Iceberg snapshot id in table-format scan request",
                )
                .with_fact("table_format", TableFormat::Iceberg.code())
                .with_fact("snapshot_id", snapshot_id)
                .with_source(error)
            })?;
            scan_builder = scan_builder.snapshot_id(parsed_snapshot_id);
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

        let scan = scan_builder
            .build()
            .map_err(|error| iceberg_planning_error("failed to build Iceberg table scan", error))?;
        let mut task_stream = scan
            .plan_files()
            .await
            .map_err(|error| iceberg_planning_error("failed to plan Iceberg data files", error))?;

        let mut tasks = Vec::new();
        while let Some(task) = task_stream.next().await {
            tasks
                .push(task.map_err(|error| iceberg_planning_error("failed to resolve Iceberg file scan task", error))?);
        }

        let catalog_type = request.catalog_type.or_else(|| self.catalog_type.clone());
        let snapshot_id =
            request.snapshot_id.or_else(|| self.table.metadata().current_snapshot_id().map(|id| id.to_string()));
        let mut plan = iceberg_file_scan_tasks_to_table_format_scan_plan_with_config_and_filter_enforcement(
            snapshot_id,
            catalog_type,
            tasks,
            &self.parquet_read_strategy_config,
            parent_filter_enforces_iceberg_predicate,
        )?
        .with_scan_limit(request.scan_limit);
        if row_selection_enabled {
            plan.unsupported_features.push(TableFormatUnsupportedFeature::new(
                TableFormatUnsupportedFeatureCode::IcebergRowSelectionUnsupported,
                "Iceberg row selection/page-index semantics are not represented by Milestone B native scan facts",
            ));
        }
        Ok(plan)
    }
}

impl TableFormatPlanner for IcebergTableFormatPlanner {
    fn plan_scan<'a>(&'a self, request: TableFormatScanRequest) -> BoxFuture<'a, Result<TableFormatScanPlan>> {
        Box::pin(async move { self.plan_table_format_scan(request).await })
    }
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
    )
}

fn iceberg_file_scan_tasks_to_table_format_scan_plan_with_config_and_filter_enforcement(
    snapshot_id: Option<String>,
    catalog_type: Option<String>,
    tasks: impl IntoIterator<Item = FileScanTask>,
    parquet_read_strategy_config: &TableFormatParquetReadStrategyConfig,
    parent_filter_enforces_task_predicate: bool,
) -> Result<TableFormatScanPlan> {
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

    for (task_index, task) in tasks.drain(..).enumerate() {
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
    plan.data_files.push(data_file);

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

    if task.partition.as_ref().is_some_and(|partition| !partition.fields().is_empty()) {
        plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
            TableFormatUnsupportedFeatureCode::IcebergPartitionTransformUnsupported,
            file_id.clone(),
            "Iceberg partition values are present but partition transform facts are not exposed by this scan task",
        ));
    }

    for field_id in task.project_field_ids.iter().copied() {
        append_projected_column_mapping(plan, task.schema(), field_id)?;
    }

    for (delete_index, delete_file) in task.deletes.iter().enumerate() {
        append_delete_file_facts(plan, &file_id, delete_index, delete_file);
    }

    Ok(())
}

fn append_projected_column_mapping(
    plan: &mut TableFormatScanPlan,
    schema: &IcebergSchema,
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
        TableFormatColumnSource::DirectTopLevelFileColumn
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

    plan.projected_columns.push(TableFormatColumnMapping::new(
        Some(field_id),
        field.name.clone(),
        Some(field.name.clone()),
        data_type,
        !field.required,
        source,
        TableFormatColumnRole::Projected,
    ));

    Ok(())
}

fn append_delete_file_facts(
    plan: &mut TableFormatScanPlan,
    data_file_id: &str,
    delete_index: usize,
    delete_file: &FileScanTaskDeleteFile,
) {
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
    let table_delete =
        TableFormatDeleteFile::new(delete_id.clone(), delete_file.file_path.clone(), format, content.clone())
            .with_size_bytes(delete_file.file_size_in_bytes);
    plan.delete_files.push(table_delete);
    plan.unsupported_features.push(TableFormatUnsupportedFeature::for_file(
        TableFormatUnsupportedFeatureCode::IcebergDeleteFilesUnsupported,
        delete_id,
        format!("Iceberg {} delete files require Milestone 2 row-level semantics", content.code()),
    ));
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
    iceberg::arrow::type_to_arrow_type(field_type).map_err(|error| {
        Error::new(
            ErrorCode::UnsupportedType,
            TASK_CONVERSION_OPERATION,
            "failed to convert Iceberg field type to Arrow data type",
        )
        .with_fact("table_format", TableFormat::Iceberg.code())
        .with_source(error)
    })
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
        spec::{DataContentType, DataFileFormat, NestedField, PrimitiveType, Schema as IcebergSchema, Struct, Type},
    };

    use super::*;
    use crate::table_format::scan::{TableFormatLakehouseCachePolicy, TableFormatParquetReadStrategyPreference};

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
