#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_sql_info_advertises_core_server_capabilities() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let requested = vec![
        SqlInfo::FlightSqlServerName,
        SqlInfo::FlightSqlServerSql,
        SqlInfo::FlightSqlServerReadOnly,
        SqlInfo::FlightSqlServerTransaction,
        SqlInfo::FlightSqlServerCancel,
        SqlInfo::SqlIdentifierQuoteChar,
        SqlInfo::SqlNullOrdering,
        SqlInfo::SqlSupportedGroupBy,
        SqlInfo::SqlOuterJoinsSupportLevel,
        SqlInfo::SqlSchemaTerm,
        SqlInfo::SqlCatalogTerm,
        SqlInfo::SqlCatalogAtStart,
        SqlInfo::SqlSupportedUnions,
        SqlInfo::SqlTransactionsSupported,
        SqlInfo::SqlSupportedResultSetTypes,
        SqlInfo::SqlSupportedConcurrenciesForResultSetForwardOnly,
    ];
    let flight_info = client.get_sql_info(requested).await.unwrap();
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();

    server.abort();

    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 16);
    assert_eq!(sql_info_string(batch, SqlInfo::FlightSqlServerName), "datafusion-nexus");
    assert!(sql_info_bool(batch, SqlInfo::FlightSqlServerSql));
    assert!(!sql_info_bool(batch, SqlInfo::FlightSqlServerReadOnly));
    assert_eq!(sql_info_i32(batch, SqlInfo::FlightSqlServerTransaction), SqlSupportedTransaction::None as i32);
    assert!(sql_info_bool(batch, SqlInfo::FlightSqlServerCancel));
    assert_eq!(sql_info_string(batch, SqlInfo::SqlIdentifierQuoteChar), r#"""#);
    assert_eq!(sql_info_i32(batch, SqlInfo::SqlNullOrdering), SqlNullOrdering::SqlNullsSortedHigh as i32);
    assert_eq!(
        sql_info_i32(batch, SqlInfo::SqlSupportedGroupBy),
        sql_info_bitmask([
            SqlSupportedGroupBy::SqlGroupByUnrelated as i32,
            SqlSupportedGroupBy::SqlGroupByBeyondSelect as i32,
        ])
    );
    assert_eq!(
        sql_info_i32(batch, SqlInfo::SqlOuterJoinsSupportLevel),
        SqlOuterJoinsSupportLevel::SqlFullOuterJoins as i32
    );
    assert_eq!(sql_info_string(batch, SqlInfo::SqlSchemaTerm), "schema");
    assert_eq!(sql_info_string(batch, SqlInfo::SqlCatalogTerm), "catalog");
    assert!(sql_info_bool(batch, SqlInfo::SqlCatalogAtStart));
    assert_eq!(
        sql_info_i32(batch, SqlInfo::SqlSupportedUnions),
        sql_info_bitmask([SqlSupportedUnions::SqlUnion as i32, SqlSupportedUnions::SqlUnionAll as i32,])
    );
    assert!(!sql_info_bool(batch, SqlInfo::SqlTransactionsSupported));
    assert_eq!(
        sql_info_i32(batch, SqlInfo::SqlSupportedResultSetTypes),
        sql_info_bitmask([SqlSupportedResultSetType::SqlResultSetTypeForwardOnly as i32])
    );
    assert_eq!(
        sql_info_i32(batch, SqlInfo::SqlSupportedConcurrenciesForResultSetForwardOnly),
        sql_info_bitmask([SqlSupportedResultSetConcurrency::SqlResultSetConcurrencyReadOnly as i32])
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_catalogs_lists_default_catalog() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.get_catalogs().await.unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    let names = collect_string_column(&batches, "catalog_name");
    assert!(names.contains(&"datafusion".to_string()), "expected default catalog, got {names:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_db_schemas_lists_default_schemas() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client
        .get_db_schemas(CommandGetDbSchemas { catalog: Some("datafusion".into()), db_schema_filter_pattern: None })
        .await
        .unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    // `information_schema` is exposed by `SessionState`, not by the catalog
    // provider, so it does not surface here. The default `public` schema does.
    let schemas = collect_string_column(&batches, "db_schema_name");
    assert!(schemas.contains(&"public".to_string()), "missing public: {schemas:?}");
    let catalogs = collect_string_column(&batches, "catalog_name");
    assert!(catalogs.iter().all(|c| c == "datafusion"), "filter leaked: {catalogs:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_tables_distinguishes_views_from_base_tables() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    // Seed both a real base table and a view so we can assert that the
    // metadata layer reports each with the correct Flight SQL table_type.
    let dir = tempfile::Builder::new().prefix("datafusion-nexus-flight-sql-metadata-").tempdir().unwrap();
    let parquet_path = dir.path().join("metadata_probe.parquet");
    write_update_edges_parquet(&parquet_path);
    let source_ddl = format!(
        "CREATE EXTERNAL TABLE metadata_probe_base STORED AS PARQUET LOCATION '{}'",
        parquet_path.to_string_lossy()
    );
    let _ = execute_batches(&mut client, &source_ddl).await;
    let _ =
        execute_batches(&mut client, "CREATE VIEW metadata_probe_view AS SELECT src AS id FROM metadata_probe_base")
            .await;

    let flight_info = client
        .get_tables(CommandGetTables {
            catalog: Some("datafusion".into()),
            db_schema_filter_pattern: Some("public".into()),
            table_name_filter_pattern: Some("metadata_probe%".into()),
            table_types: vec![],
            include_schema: false,
        })
        .await
        .unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    let pairs = collect_name_and_type(&batches);
    assert!(
        pairs.contains(&("metadata_probe_base".to_string(), "TABLE".to_string())),
        "missing seeded base table or wrong type: {pairs:?}"
    );
    assert!(
        pairs.contains(&("metadata_probe_view".to_string(), "VIEW".to_string())),
        "missing seeded view or wrong type: {pairs:?}"
    );
    let catalogs = collect_string_column(&batches, "catalog_name");
    assert!(catalogs.iter().all(|c| c == "datafusion"), "catalog filter leaked: {catalogs:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_tables_respects_table_types_filter() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let _ = execute_batches(&mut client, "CREATE TABLE table_types_filter_base AS SELECT 1 AS x").await;
    let _ =
        execute_batches(&mut client, "CREATE VIEW table_types_filter_view AS SELECT x FROM table_types_filter_base")
            .await;

    let flight_info = client
        .get_tables(CommandGetTables {
            catalog: Some("datafusion".into()),
            db_schema_filter_pattern: Some("public".into()),
            table_name_filter_pattern: Some("table_types_filter%".into()),
            table_types: vec!["VIEW".into()],
            include_schema: false,
        })
        .await
        .unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    let pairs = collect_name_and_type(&batches);
    assert_eq!(pairs, vec![("table_types_filter_view".to_string(), "VIEW".to_string())]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_tables_with_include_schema_emits_arrow_schema() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let _ = execute_batches(
        &mut client,
        "CREATE TABLE include_schema_probe AS \
         SELECT 1::INT AS id, 'a' AS label",
    )
    .await;

    let flight_info = client
        .get_tables(CommandGetTables {
            catalog: Some("datafusion".into()),
            db_schema_filter_pattern: Some("public".into()),
            table_name_filter_pattern: Some("include_schema_probe".into()),
            table_types: vec![],
            include_schema: true,
        })
        .await
        .unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    let pairs = collect_name_and_type(&batches);
    assert_eq!(pairs, vec![("include_schema_probe".to_string(), "TABLE".to_string())]);

    let schemas: Vec<arrow::datatypes::Schema> = batches
        .iter()
        .filter_map(|batch| {
            let index = batch.schema().index_of("table_schema").ok()?;
            let bytes = batch
                .column(index)
                .as_any()
                .downcast_ref::<arrow::array::BinaryArray>()
                .expect("table_schema column must be Binary");
            (0..bytes.len()).map(|i| arrow::ipc::convert::try_schema_from_ipc_buffer(bytes.value(i)).unwrap()).next()
        })
        .collect();
    assert_eq!(schemas.len(), 1, "expected exactly one table_schema row");
    let field_names: Vec<String> = schemas[0].fields().iter().map(|f| f.name().clone()).collect();
    assert_eq!(field_names, vec!["id".to_string(), "label".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_table_types_returns_table_and_view() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.get_table_types().await.unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    let types = collect_string_column(&batches, "table_type");
    assert_eq!(types, vec!["TABLE".to_string(), "VIEW".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_primary_keys_returns_empty_with_schema() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let cmd = CommandGetPrimaryKeys { catalog: None, db_schema: None, table: "any_table".to_string() };
    let flight_info = client.get_primary_keys(cmd).await.unwrap();
    let (schema, batches) = collect_metadata_schema_and_batches(&mut client, &flight_info).await;

    server.abort();

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0, "primary keys must be empty for DataFusion catalog");
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        field_names,
        vec!["catalog_name", "db_schema_name", "table_name", "column_name", "key_name", "key_sequence"]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_foreign_keys_rpcs_return_empty_with_shared_schema() {
    let (listener, addr) = free_local_addr();
    let mut config = test_server_config();
    declare_test_gpu_ownership(&mut config);
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve_with_listener(config, listener).await });
    let mut client = connect_with_retry(addr).await;

    let imported_info = client
        .get_imported_keys(CommandGetImportedKeys { catalog: None, db_schema: None, table: "t".to_string() })
        .await
        .unwrap();
    let exported_info = client
        .get_exported_keys(CommandGetExportedKeys { catalog: None, db_schema: None, table: "t".to_string() })
        .await
        .unwrap();
    let xref_info = client
        .get_cross_reference(CommandGetCrossReference {
            pk_catalog: None,
            pk_db_schema: None,
            pk_table: "p".to_string(),
            fk_catalog: None,
            fk_db_schema: None,
            fk_table: "f".to_string(),
        })
        .await
        .unwrap();

    let (imported_schema, imported) = collect_metadata_schema_and_batches(&mut client, &imported_info).await;
    let (exported_schema, exported) = collect_metadata_schema_and_batches(&mut client, &exported_info).await;
    let (xref_schema, xref) = collect_metadata_schema_and_batches(&mut client, &xref_info).await;

    server.abort();

    let expected_names = vec![
        "pk_catalog_name",
        "pk_db_schema_name",
        "pk_table_name",
        "pk_column_name",
        "fk_catalog_name",
        "fk_db_schema_name",
        "fk_table_name",
        "fk_column_name",
        "key_sequence",
        "fk_key_name",
        "pk_key_name",
        "update_rule",
        "delete_rule",
    ];
    for (label, schema, batches) in [
        ("imported", &imported_schema, &imported),
        ("exported", &exported_schema, &exported),
        ("cross_reference", &xref_schema, &xref),
    ] {
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 0, "{label} keys must be empty");
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, expected_names, "{label} keys schema mismatch");
        let update_rule = schema.field_with_name("update_rule").unwrap();
        assert_eq!(update_rule.data_type(), &arrow::datatypes::DataType::UInt8, "{label} update_rule must be UInt8");
    }
    // Reference the UInt8Array import to keep it warning-free even when batches are empty.
    let _ = UInt8Array::from(Vec::<u8>::new());
}
