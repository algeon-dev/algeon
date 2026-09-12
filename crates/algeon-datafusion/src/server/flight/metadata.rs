use std::sync::Arc;

use super::*;

const FLIGHT_SQL_TABLE_TYPE_TABLE: &str = "TABLE";
/// Flight SQL standard `table_type` string for a view.
const FLIGHT_SQL_TABLE_TYPE_VIEW: &str = "VIEW";

/// The set advertised by `CommandGetTableTypes`. Stored sorted lexicographically
/// as required by the Flight SQL protobuf contract for `GetTableTypes`.
const FLIGHT_SQL_TABLE_TYPES: &[&str] = &[FLIGHT_SQL_TABLE_TYPE_TABLE, FLIGHT_SQL_TABLE_TYPE_VIEW];

/// Schema for `CommandGetTableTypes` responses. Pinned to `table_type: utf8 not
/// null` to match the Flight SQL protobuf contract. This duplicates the schema
/// defined inside `arrow_flight::sql::metadata::table_types`, which is
/// crate-private in arrow-rs 58.3 (and on main). Once arrow-rs re-exports
/// `GetTableTypesBuilder`, switch both this helper and `do_get_table_types` to
/// use it directly.
pub(super) fn flight_sql_table_types_schema() -> Arc<Schema> {
    use arrow::datatypes::{DataType, Field};
    static SCHEMA: std::sync::OnceLock<Arc<Schema>> = std::sync::OnceLock::new();
    SCHEMA.get_or_init(|| Arc::new(Schema::new(vec![Field::new("table_type", DataType::Utf8, false)]))).clone()
}

/// Schema for `CommandGetPrimaryKeys` responses. Pinned to the columns
/// specified by the Flight SQL protobuf documentation
/// (`catalog_name`, `db_schema_name`, `table_name`, `column_name`,
/// `key_name`, `key_sequence`). DataFusion does not model primary-key
/// constraints, so the response is always an empty batch; but it must
/// still carry this exact schema or arrow-flight clients refuse it.
pub(super) fn flight_sql_primary_keys_schema() -> Arc<Schema> {
    static SCHEMA: std::sync::OnceLock<Arc<Schema>> = std::sync::OnceLock::new();
    SCHEMA
        .get_or_init(|| {
            Arc::new(Schema::new(vec![
                Field::new("catalog_name", DataType::Utf8, true),
                Field::new("db_schema_name", DataType::Utf8, true),
                Field::new("table_name", DataType::Utf8, false),
                Field::new("column_name", DataType::Utf8, false),
                Field::new("key_name", DataType::Utf8, true),
                Field::new("key_sequence", DataType::Int32, false),
            ]))
        })
        .clone()
}

/// Schema for `CommandGetImportedKeys` / `CommandGetExportedKeys` /
/// `CommandGetCrossReference` responses. The three RPCs share this 13-column
/// foreign-key descriptor schema per the Flight SQL protobuf spec. DataFusion
/// does not model foreign-key constraints, so the body is always empty; but
/// the schema must still match exactly or arrow-flight clients refuse it.
pub(super) fn flight_sql_foreign_keys_schema() -> Arc<Schema> {
    static SCHEMA: std::sync::OnceLock<Arc<Schema>> = std::sync::OnceLock::new();
    SCHEMA
        .get_or_init(|| {
            Arc::new(Schema::new(vec![
                Field::new("pk_catalog_name", DataType::Utf8, true),
                Field::new("pk_db_schema_name", DataType::Utf8, true),
                Field::new("pk_table_name", DataType::Utf8, false),
                Field::new("pk_column_name", DataType::Utf8, false),
                Field::new("fk_catalog_name", DataType::Utf8, true),
                Field::new("fk_db_schema_name", DataType::Utf8, true),
                Field::new("fk_table_name", DataType::Utf8, false),
                Field::new("fk_column_name", DataType::Utf8, false),
                Field::new("key_sequence", DataType::Int32, false),
                Field::new("fk_key_name", DataType::Utf8, true),
                Field::new("pk_key_name", DataType::Utf8, true),
                Field::new("update_rule", DataType::UInt8, false),
                Field::new("delete_rule", DataType::UInt8, false),
            ]))
        })
        .clone()
}

/// Empty-row `RecordBatch` matching [`flight_sql_foreign_keys_schema`].
pub(super) fn empty_foreign_keys_batch() -> Result<arrow::record_batch::RecordBatch, arrow::error::ArrowError> {
    let schema = flight_sql_foreign_keys_schema();
    let columns: Vec<arrow::array::ArrayRef> = vec![
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<&str>::new())),
        Arc::new(StringArray::from(Vec::<&str>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<&str>::new())),
        Arc::new(StringArray::from(Vec::<&str>::new())),
        Arc::new(Int32Array::from(Vec::<i32>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(UInt8Array::from(Vec::<u8>::new())),
        Arc::new(UInt8Array::from(Vec::<u8>::new())),
    ];
    arrow::record_batch::RecordBatch::try_new(schema, columns)
}

/// Empty-row `RecordBatch` matching [`flight_sql_primary_keys_schema`].
pub(super) fn empty_primary_keys_batch() -> Result<arrow::record_batch::RecordBatch, arrow::error::ArrowError> {
    let schema = flight_sql_primary_keys_schema();
    let columns: Vec<arrow::array::ArrayRef> = vec![
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(StringArray::from(Vec::<&str>::new())),
        Arc::new(StringArray::from(Vec::<&str>::new())),
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
        Arc::new(Int32Array::from(Vec::<i32>::new())),
    ];
    arrow::record_batch::RecordBatch::try_new(schema, columns)
}

/// Build the single-batch response for `CommandGetTableTypes`.
///
/// `FLIGHT_SQL_TABLE_TYPES` is the source of truth; keep it sorted at authoring
/// time instead of re-sorting here.
pub(super) fn build_table_types_batch() -> Result<arrow::record_batch::RecordBatch, arrow::error::ArrowError> {
    debug_assert!(FLIGHT_SQL_TABLE_TYPES.windows(2).all(|w| w[0] < w[1]), "FLIGHT_SQL_TABLE_TYPES must be sorted");
    let column = arrow::array::StringArray::from_iter_values(FLIGHT_SQL_TABLE_TYPES.iter().copied());
    arrow::record_batch::RecordBatch::try_new(flight_sql_table_types_schema(), vec![Arc::new(column)])
}

pub(super) struct GetTablesPrefilters {
    catalog: Option<String>,
    db_schema_filter_pattern: Option<String>,
    table_name_filter_pattern: Option<String>,
    table_types: Vec<String>,
}

impl From<&CommandGetTables> for GetTablesPrefilters {
    fn from(query: &CommandGetTables) -> Self {
        Self {
            catalog: query.catalog.clone(),
            db_schema_filter_pattern: query.db_schema_filter_pattern.clone(),
            table_name_filter_pattern: query.table_name_filter_pattern.clone(),
            table_types: query.table_types.clone(),
        }
    }
}

impl GetTablesPrefilters {
    pub(super) fn catalog_matches(&self, catalog_name: &str) -> bool {
        self.catalog.as_deref().is_none_or(|catalog| catalog == catalog_name)
    }

    pub(super) fn schema_matches(&self, schema_name: &str) -> bool {
        metadata_like_filter_matches(self.db_schema_filter_pattern.as_deref(), schema_name)
    }

    pub(super) fn table_name_matches(&self, table_name: &str) -> bool {
        metadata_like_filter_matches(self.table_name_filter_pattern.as_deref(), table_name)
    }

    pub(super) fn table_type_matches(&self, table_type: &str) -> bool {
        self.table_types.is_empty() || self.table_types.iter().any(|filter| filter == table_type)
    }

    pub(super) fn has_table_type_filter(&self) -> bool {
        !self.table_types.is_empty()
    }
}

pub(super) fn metadata_like_filter_matches(pattern: Option<&str>, value: &str) -> bool {
    let Some(pattern) = pattern else {
        return true;
    };
    let values = StringArray::from(vec![value]);
    let patterns = StringArray::from(vec![pattern]);
    arrow::compute::kernels::comparison::like(&values, &patterns)
        .map(|matches| matches.is_valid(0) && matches.value(0))
        .unwrap_or(true)
}

/// Map DataFusion's `TableType` to the Flight SQL wire string. `Temporary` has
/// no dedicated Flight SQL value, so it surfaces as `"TABLE"` (matching how
/// `information_schema.tables` reports temp tables).
pub(super) fn table_type_to_flight_sql(table_type: TableType) -> &'static str {
    match table_type {
        TableType::View => FLIGHT_SQL_TABLE_TYPE_VIEW,
        TableType::Base | TableType::Temporary => FLIGHT_SQL_TABLE_TYPE_TABLE,
    }
}

/// Build a `FlightInfo` for a Flight SQL metadata command whose `do_get` echoes
/// the original command back as the ticket payload (matching how
/// `get_flight_info_sql_info` is wired).
pub(super) fn metadata_flight_info<C>(
    command: C,
    descriptor: FlightDescriptor,
    schema: &Schema,
) -> Result<FlightInfo, Status>
where
    C: ProstMessageExt,
{
    let ticket = Ticket { ticket: command.as_any().encode_to_vec().into() };
    let endpoint = FlightEndpoint::new().with_ticket(ticket);
    let info = FlightInfo::new()
        .try_with_schema(schema)
        .map_err(|error| {
            arrow_error_to_status_for_operation(
                error,
                "algeon_datafusion.server.flight.metadata_flight_info.schema",
                "failed to build Flight SQL metadata FlightInfo schema",
            )
        })?
        .with_descriptor(descriptor)
        .with_endpoint(endpoint)
        .with_ordered(false);
    Ok(info)
}

/// Wrap a single pre-built `RecordBatch` as a Flight `DoGetStream`. Used by the
/// metadata RPCs where the entire response fits in one batch produced by the
/// arrow-flight metadata builders.
pub(super) fn metadata_record_batch_stream(
    schema: Arc<Schema>,
    batch: arrow::record_batch::RecordBatch,
    operation: &'static str,
    message: &'static str,
) -> BoxStream<'static, Result<arrow_flight::FlightData, Status>> {
    let stream = encode_flight_data_stream(schema, futures::stream::once(async move { Ok(batch) }))
        .map_err(move |error| arrow_flight_error_to_status_for_operation(error, operation, message));
    Box::pin(stream)
}

pub(super) fn sql_info_data() -> Result<arrow_flight::sql::metadata::SqlInfoData, Status> {
    let mut builder = SqlInfoDataBuilder::new();
    builder.append(SqlInfo::FlightSqlServerName, "algeon-datafusion");
    builder.append(SqlInfo::FlightSqlServerVersion, env!("CARGO_PKG_VERSION"));
    builder.append(SqlInfo::FlightSqlServerArrowVersion, "1.3");
    builder.append(SqlInfo::FlightSqlServerReadOnly, false);
    builder.append(SqlInfo::FlightSqlServerSql, true);
    builder.append(SqlInfo::FlightSqlServerTransaction, SqlSupportedTransaction::None as i32);
    builder.append(SqlInfo::FlightSqlServerCancel, true);
    builder.append(SqlInfo::FlightSqlServerBulkIngestion, false);
    builder.append(SqlInfo::FlightSqlServerIngestTransactionsSupported, false);

    builder.append(SqlInfo::SqlDdlCatalog, false);
    builder.append(SqlInfo::SqlDdlSchema, true);
    builder.append(SqlInfo::SqlDdlTable, true);
    builder.append(SqlInfo::SqlIdentifierCase, SqlSupportedCaseSensitivity::SqlCaseSensitivityLowercase as i32);
    builder.append(SqlInfo::SqlIdentifierQuoteChar, r#"""#);
    builder.append(SqlInfo::SqlQuotedIdentifierCase, SqlSupportedCaseSensitivity::SqlCaseSensitivityUnknown as i32);
    builder.append(SqlInfo::SqlAllTablesAreSelectable, true);
    builder.append(SqlInfo::SqlNullOrdering, SqlNullOrdering::SqlNullsSortedHigh as i32);
    builder.append(SqlInfo::SqlSearchStringEscape, "\\");
    builder.append(SqlInfo::SqlExtraNameCharacters, "_");
    builder.append(SqlInfo::SqlSupportsColumnAliasing, true);
    builder.append(SqlInfo::SqlNullPlusNullIsNull, true);
    builder.append(SqlInfo::SqlSupportsTableCorrelationNames, true);
    builder.append(SqlInfo::SqlSupportsDifferentTableCorrelationNames, true);
    builder.append(SqlInfo::SqlSupportsExpressionsInOrderBy, true);
    builder.append(SqlInfo::SqlSupportsOrderByUnrelated, true);
    builder.append(
        SqlInfo::SqlSupportedGroupBy,
        sql_info_bitmask([
            SqlSupportedGroupBy::SqlGroupByUnrelated as i32,
            SqlSupportedGroupBy::SqlGroupByBeyondSelect as i32,
        ]),
    );
    builder.append(SqlInfo::SqlSupportsLikeEscapeClause, true);
    builder.append(SqlInfo::SqlSupportsNonNullableColumns, true);
    builder.append(
        SqlInfo::SqlSupportedGrammar,
        sql_info_bitmask([SupportedSqlGrammar::SqlMinimumGrammar as i32, SupportedSqlGrammar::SqlCoreGrammar as i32]),
    );
    builder.append(SqlInfo::SqlAnsi92SupportedLevel, sql_info_bitmask([SQL_ANSI92_ENTRY_SQL_BIT_INDEX]));
    builder.append(SqlInfo::SqlSupportsIntegrityEnhancementFacility, false);
    builder.append(SqlInfo::SqlOuterJoinsSupportLevel, SqlOuterJoinsSupportLevel::SqlFullOuterJoins as i32);
    builder.append(SqlInfo::SqlSchemaTerm, "schema");
    builder.append(SqlInfo::SqlProcedureTerm, "");
    builder.append(SqlInfo::SqlCatalogTerm, "catalog");
    builder.append(SqlInfo::SqlCatalogAtStart, true);
    builder.append(SqlInfo::SqlSchemasSupportedActions, 0);
    builder.append(SqlInfo::SqlCatalogsSupportedActions, 0);
    builder.append(SqlInfo::SqlSupportedPositionedCommands, 0);
    builder.append(SqlInfo::SqlSelectForUpdateSupported, false);
    builder.append(SqlInfo::SqlStoredProceduresSupported, false);
    builder.append(SqlInfo::SqlSupportedSubqueries, 0);
    builder.append(SqlInfo::SqlCorrelatedSubqueriesSupported, false);
    builder.append(
        SqlInfo::SqlSupportedUnions,
        sql_info_bitmask([SqlSupportedUnions::SqlUnion as i32, SqlSupportedUnions::SqlUnionAll as i32]),
    );
    builder.append(SqlInfo::SqlDefaultTransactionIsolation, SqlTransactionIsolationLevel::SqlTransactionNone as i32);
    builder.append(SqlInfo::SqlTransactionsSupported, false);
    builder.append(SqlInfo::SqlSupportedTransactionsIsolationLevels, 0);
    builder.append(SqlInfo::SqlDataDefinitionCausesTransactionCommit, false);
    builder.append(SqlInfo::SqlDataDefinitionsInTransactionsIgnored, false);
    builder.append(
        SqlInfo::SqlSupportedResultSetTypes,
        sql_info_bitmask([SqlSupportedResultSetType::SqlResultSetTypeForwardOnly as i32]),
    );
    builder.append(SqlInfo::SqlSupportedConcurrenciesForResultSetUnspecified, 0);
    builder.append(
        SqlInfo::SqlSupportedConcurrenciesForResultSetForwardOnly,
        sql_info_bitmask([SqlSupportedResultSetConcurrency::SqlResultSetConcurrencyReadOnly as i32]),
    );
    builder.append(SqlInfo::SqlSupportedConcurrenciesForResultSetScrollSensitive, 0);
    builder.append(SqlInfo::SqlSupportedConcurrenciesForResultSetScrollInsensitive, 0);
    builder.append(SqlInfo::SqlBatchUpdatesSupported, false);
    builder.append(SqlInfo::SqlSavepointsSupported, false);
    builder.append(SqlInfo::SqlNamedParametersSupported, false);
    builder.append(SqlInfo::SqlLocatorsUpdateCopy, false);
    builder.append(SqlInfo::SqlStoredFunctionsUsingCallSyntaxSupported, false);
    builder.build().map_err(|error| {
        arrow_flight_error_to_status_for_operation(
            error,
            "algeon_datafusion.server.flight.sql_info_data.build",
            "failed to build Flight SQL SqlInfo metadata",
        )
    })
}

pub(super) fn sql_info_bitmask(values: impl IntoIterator<Item = i32>) -> i32 {
    values.into_iter().fold(0, |mask, value| {
        let shift = u32::try_from(value).expect("Flight SQL SqlInfo bit index must be non-negative");
        mask | 1_i32.checked_shl(shift).expect("Flight SQL SqlInfo bit index must fit in i32")
    })
}

#[cfg(test)]
mod get_tables_prefilter_tests {
    use std::{
        collections::{BTreeMap, BTreeSet},
        sync::Mutex,
        time::Duration,
    };

    use arrow::datatypes::SchemaRef;
    use datafusion::{
        catalog::{CatalogProvider, SchemaProvider},
        datasource::{MemTable, TableProvider},
    };

    use super::*;

    #[derive(Debug)]
    struct CountingCatalog {
        schema: Arc<CountingSchemaProvider>,
    }

    impl CatalogProvider for CountingCatalog {
        fn schema_names(&self) -> Vec<String> {
            vec!["public".to_string()]
        }

        fn schema(&self, name: &str) -> Option<Arc<dyn SchemaProvider>> {
            (name == "public").then(|| Arc::clone(&self.schema) as Arc<dyn SchemaProvider>)
        }
    }

    #[derive(Debug)]
    struct CountingSchemaProvider {
        tables: BTreeMap<String, (TableType, SchemaRef)>,
        table_type_error_names: BTreeSet<String>,
        table_calls: Mutex<Vec<String>>,
        table_type_calls: Mutex<Vec<String>>,
    }

    impl CountingSchemaProvider {
        fn new(tables: impl IntoIterator<Item = (&'static str, TableType)>) -> Arc<Self> {
            Arc::new(Self {
                tables: tables
                    .into_iter()
                    .map(|(name, table_type)| (name.to_string(), (table_type, table_schema(name))))
                    .collect(),
                table_type_error_names: BTreeSet::new(),
                table_calls: Mutex::default(),
                table_type_calls: Mutex::default(),
            })
        }

        fn with_table_type_errors(self: Arc<Self>, names: impl IntoIterator<Item = &'static str>) -> Arc<Self> {
            let mut this = Arc::into_inner(self).expect("builder has the only Arc");
            this.table_type_error_names = names.into_iter().map(str::to_string).collect();
            Arc::new(this)
        }

        fn table_calls(&self) -> Vec<String> {
            self.table_calls.lock().unwrap().clone()
        }

        fn table_type_calls(&self) -> Vec<String> {
            self.table_type_calls.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl SchemaProvider for CountingSchemaProvider {
        fn table_names(&self) -> Vec<String> {
            self.tables.keys().cloned().collect()
        }

        async fn table(&self, name: &str) -> datafusion::common::Result<Option<Arc<dyn TableProvider>>> {
            self.table_calls.lock().unwrap().push(name.to_string());
            let Some((_, schema)) = self.tables.get(name) else {
                return Ok(None);
            };
            let provider = MemTable::try_new(Arc::clone(schema), vec![vec![]])?;
            Ok(Some(Arc::new(provider)))
        }

        async fn table_type(&self, name: &str) -> datafusion::common::Result<Option<TableType>> {
            self.table_type_calls.lock().unwrap().push(name.to_string());
            if self.table_type_error_names.contains(name) {
                return Err(datafusion::common::DataFusionError::Execution(format!("table_type failed for {name}")));
            }
            Ok(self.tables.get(name).map(|(table_type, _)| *table_type))
        }

        fn table_exist(&self, name: &str) -> bool {
            self.tables.contains_key(name)
        }
    }

    fn table_schema(name: &str) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(format!("{name}_id"), DataType::Int32, false)]))
    }

    fn service_with_schema(schema: Arc<CountingSchemaProvider>) -> DatafusionAlgeonFlightSqlService {
        let ctx = Arc::new(SessionContext::new());
        ctx.register_catalog("counting", Arc::new(CountingCatalog { schema }));
        DatafusionAlgeonFlightSqlService::new(
            ctx,
            AlgeonNativeOptimizerConfig::default(),
            Arc::new(QueryAdmission::new_for_tests()),
            Duration::from_secs(30),
        )
    }

    fn get_tables_query(
        table_name_filter_pattern: Option<&str>,
        table_types: Vec<&str>,
        include_schema: bool,
    ) -> CommandGetTables {
        CommandGetTables {
            catalog: Some("counting".to_string()),
            db_schema_filter_pattern: Some("public".to_string()),
            table_name_filter_pattern: table_name_filter_pattern.map(str::to_string),
            table_types: table_types.into_iter().map(str::to_string).collect(),
            include_schema,
        }
    }

    fn table_names_from_batch(batch: &arrow::record_batch::RecordBatch) -> Vec<String> {
        let index = batch.schema().index_of("table_name").unwrap();
        let array = batch.column(index).as_any().downcast_ref::<StringArray>().unwrap();
        (0..array.len()).map(|row| array.value(row).to_string()).collect()
    }

    #[tokio::test]
    async fn include_schema_prefilters_table_name_before_table_load() {
        let schema = CountingSchemaProvider::new([("other", TableType::Base), ("target", TableType::Base)]);
        let service = service_with_schema(Arc::clone(&schema));

        let (_, batch) =
            service.build_get_tables_response(get_tables_query(Some("target"), vec![], true)).await.unwrap();

        assert_eq!(table_names_from_batch(&batch), vec!["target"]);
        assert!(schema.table_type_calls().is_empty());
        assert_eq!(schema.table_calls(), vec!["target"]);
    }

    #[tokio::test]
    async fn include_schema_missing_table_filter_loads_no_tables() {
        let schema = CountingSchemaProvider::new([("other", TableType::Base), ("target", TableType::Base)]);
        let service = service_with_schema(Arc::clone(&schema));

        let (_, batch) =
            service.build_get_tables_response(get_tables_query(Some("missing"), vec![], true)).await.unwrap();

        assert_eq!(batch.num_rows(), 0);
        assert!(schema.table_type_calls().is_empty());
        assert!(schema.table_calls().is_empty());
    }

    #[tokio::test]
    async fn include_schema_applies_table_type_filter_before_table_load() {
        let schema = CountingSchemaProvider::new([("base_a", TableType::Base), ("base_b", TableType::Base)]);
        let service = service_with_schema(Arc::clone(&schema));

        let (_, batch) = service.build_get_tables_response(get_tables_query(None, vec!["VIEW"], true)).await.unwrap();

        assert_eq!(batch.num_rows(), 0);
        assert_eq!(schema.table_type_calls(), vec!["base_a", "base_b"]);
        assert!(schema.table_calls().is_empty());
    }

    #[tokio::test]
    async fn include_schema_skips_table_type_errors_when_prefiltering_type() {
        let schema = CountingSchemaProvider::new([("broken", TableType::Base), ("view", TableType::View)])
            .with_table_type_errors(["broken"]);
        let service = service_with_schema(Arc::clone(&schema));

        let (_, batch) = service.build_get_tables_response(get_tables_query(None, vec!["VIEW"], true)).await.unwrap();

        assert_eq!(table_names_from_batch(&batch), vec!["view"]);
        assert_eq!(schema.table_type_calls(), vec!["broken", "view"]);
        assert_eq!(schema.table_calls(), vec!["view"]);
    }

    #[tokio::test]
    async fn escaped_like_prefilter_matches_arrow_semantics() {
        let schema = CountingSchemaProvider::new([("fooXbar", TableType::Base), ("foo_bar", TableType::Base)]);
        let service = service_with_schema(Arc::clone(&schema));

        let (_, batch) =
            service.build_get_tables_response(get_tables_query(Some(r"foo\_%"), vec![], true)).await.unwrap();

        assert_eq!(table_names_from_batch(&batch), vec!["foo_bar"]);
        assert_eq!(schema.table_calls(), vec!["foo_bar"]);
    }

    #[test]
    pub(super) fn metadata_like_filter_matches_flight_sql_wildcards() {
        assert!(metadata_like_filter_matches(Some("target%"), "target_table"));
        assert!(metadata_like_filter_matches(Some("target_"), "target1"));
        assert!(!metadata_like_filter_matches(Some("target_"), "target12"));
        assert!(!metadata_like_filter_matches(Some("target%"), "other"));
        assert!(metadata_like_filter_matches(Some(r"foo\_%"), "foo_bar"));
        assert!(!metadata_like_filter_matches(Some(r"foo\_%"), "fooXbar"));
    }
}
