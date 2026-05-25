#![cfg(feature = "server")]

use std::{net::TcpListener, time::Duration};

use arrow::array::{Array, BooleanArray, Int32Array, Int64Array, StringArray, UInt32Array, UInt8Array, UnionArray};
use arrow_flight::{
    error::FlightError,
    sql::{
        CommandGetCrossReference, CommandGetDbSchemas, CommandGetExportedKeys, CommandGetImportedKeys,
        CommandGetPrimaryKeys, CommandGetTables, SqlInfo, SqlNullOrdering, SqlOuterJoinsSupportLevel,
        SqlSupportedGroupBy, SqlSupportedResultSetConcurrency, SqlSupportedResultSetType, SqlSupportedTransaction,
        SqlSupportedUnions, client::FlightSqlServiceClient,
    },
};
use datafusion_cudf::server::{
    ServerConfig,
    error::{ERROR_CODE_METADATA_KEY, ERROR_KIND_METADATA_KEY},
    serve,
};
use futures::TryStreamExt;
use tonic::{Code, transport::Endpoint};

const QUERY_ID_METADATA_KEY: &str = "x-datafusion-cudf-query-id";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_client_can_run_select_one() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.execute("select 1 as one".to_string(), None).await.unwrap();
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();

    server.abort();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(batches[0].num_columns(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(1));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_view_statement_executes_once_and_persists_across_statements() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let batches = execute_batches(
        &mut client,
        "CREATE VIEW target_edges AS \
         SELECT * FROM (VALUES (0::BIGINT, 1::BIGINT), (1::BIGINT, 2::BIGINT)) AS t(src, dst)",
    )
    .await;
    assert_eq!(batches.iter().map(|batch| batch.num_rows()).sum::<usize>(), 0);

    let batches = execute_batches(&mut client, "SELECT COUNT(*) AS edge_count FROM target_edges").await;

    server.abort();

    assert_eq!(batches.len(), 1);
    assert_eq!(int_value(batches[0].column(0)), Some(2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn statement_planning_errors_include_query_id_metadata() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.execute("selekt 1".to_string(), None).await.unwrap_err();

    server.abort();

    let FlightError::Tonic(status) = error else {
        panic!("expected tonic status, got {error:?}");
    };
    assert_eq!(status.code(), Code::InvalidArgument);
    let query_id = status.metadata().get(QUERY_ID_METADATA_KEY).expect("query_id metadata must be present");
    let query_id = query_id.to_str().expect("query_id metadata must be ASCII");
    assert!(!query_id.is_empty());
    assert_error_metadata(&status, "sql_parse", "invalid_input");
    assert!(status.message().contains("code=sql_parse"));
    assert!(status.message().contains("kind=invalid_input"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_table_errors_are_not_found_with_structured_code() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.execute("select * from does_not_exist".to_string(), None).await.unwrap_err();

    server.abort();

    let FlightError::Tonic(status) = error else {
        panic!("expected tonic status, got {error:?}");
    };
    assert_eq!(status.code(), Code::NotFound);
    let query_id = status.metadata().get(QUERY_ID_METADATA_KEY).expect("query_id metadata must be present");
    let query_id = query_id.to_str().expect("query_id metadata must be ASCII");
    assert!(!query_id.is_empty());
    assert_error_metadata(&status, "object_not_found", "not_found");
    assert!(status.message().contains("code=object_not_found"));
    assert!(status.message().contains("kind=not_found"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unconsumed_statement_ticket_holds_admission_permit() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    config.max_in_flight = 1;
    config.statement_ticket_ttl = Duration::from_secs(30);

    let server = tokio::spawn(async move { serve(config).await });
    let mut first_client = connect_with_retry(addr).await;
    let mut second_client = connect_with_retry(addr).await;

    let first_info = first_client.execute("select 1 as one".to_string(), None).await.unwrap();
    let first_ticket = first_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();

    let mut second = tokio::spawn(async move { second_client.execute("select 2 as two".to_string(), None).await });
    assert!(
        tokio::time::timeout(Duration::from_millis(150), &mut second).await.is_err(),
        "second GetFlightInfo must wait while the first ticket holds the only admission permit"
    );

    let batches = first_client.do_get(first_ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(int_value(batches[0].column(0)), Some(1));

    let second_info = tokio::time::timeout(Duration::from_secs(5), second).await.unwrap().unwrap().unwrap();
    let second_ticket = second_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = first_client.do_get(second_ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();

    server.abort();

    assert_eq!(int_value(batches[0].column(0)), Some(2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abandoned_statement_ticket_expires_and_releases_admission_permit() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;
    config.max_in_flight = 1;
    config.statement_ticket_ttl = Duration::from_millis(50);

    let server = tokio::spawn(async move { serve(config).await });
    let mut first_client = connect_with_retry(addr).await;
    let mut second_client = connect_with_retry(addr).await;

    let first_info = first_client.execute("select 1 as one".to_string(), None).await.unwrap();
    let first_ticket = first_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();

    let second_info =
        tokio::time::timeout(Duration::from_secs(5), second_client.execute("select 2 as two".to_string(), None))
            .await
            .expect("second GetFlightInfo should proceed after the first ticket expires")
            .unwrap();
    let second_ticket = second_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let batches = second_client.do_get(second_ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(int_value(batches[0].column(0)), Some(2));

    let error = first_client.do_get(first_ticket).await.unwrap_err();

    server.abort();

    let FlightError::Tonic(status) = error else {
        panic!("expected tonic status, got {error:?}");
    };
    assert_eq!(status.code(), Code::InvalidArgument);
    assert!(status.message().contains("expired"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn get_sql_info_advertises_core_server_capabilities() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
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
    assert_eq!(sql_info_string(batch, SqlInfo::FlightSqlServerName), "datafusion-cudf");
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
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.get_catalogs().await.unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    let names = collect_string_column(&batches, "catalog_name");
    assert!(names.contains(&"datafusion".to_string()), "expected default catalog, got {names:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_db_schemas_lists_default_schemas() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
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
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    // Seed both a real base table and a view so we can assert that the
    // metadata layer reports each with the correct Flight SQL table_type.
    let _ = execute_batches(
        &mut client,
        "CREATE TABLE metadata_probe_base AS \
         SELECT * FROM (VALUES (1::INT, 'a'), (2::INT, 'b')) AS t(id, label)",
    )
    .await;
    let _ = execute_batches(
        &mut client,
        "CREATE VIEW metadata_probe_view AS SELECT id FROM metadata_probe_base",
    )
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
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let _ = execute_batches(
        &mut client,
        "CREATE TABLE table_types_filter_base AS SELECT 1 AS x",
    )
    .await;
    let _ = execute_batches(
        &mut client,
        "CREATE VIEW table_types_filter_view AS SELECT x FROM table_types_filter_base",
    )
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
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
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
            (0..bytes.len())
                .map(|i| arrow::ipc::convert::try_schema_from_ipc_buffer(bytes.value(i)).unwrap())
                .next()
        })
        .collect();
    assert_eq!(schemas.len(), 1, "expected exactly one table_schema row");
    let field_names: Vec<String> = schemas[0].fields().iter().map(|f| f.name().clone()).collect();
    assert_eq!(field_names, vec!["id".to_string(), "label".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_table_types_returns_table_and_view() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let flight_info = client.get_table_types().await.unwrap();
    let batches = collect_metadata_batches(&mut client, &flight_info).await;

    server.abort();

    let types = collect_string_column(&batches, "table_type");
    assert_eq!(types, vec!["TABLE".to_string(), "VIEW".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_round_trip_executes_select() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let mut prepared = client.prepare("select 1 as one".to_string(), None).await.unwrap();
    assert_eq!(prepared.dataset_schema().unwrap().fields().len(), 1, "dataset_schema must reflect projected columns");

    // Execute twice to confirm the prepared handle is not consumed.
    for expected_iteration in 0..2 {
        let flight_info = prepared.execute().await.unwrap();
        let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
        let batches = client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap();
        assert_eq!(batches.len(), 1, "iteration {expected_iteration} should return one batch");
        assert_eq!(int_value(batches[0].column(0)), Some(1));
    }
    prepared.close().await.unwrap();

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_rejects_transaction_id() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let txn_id = prost::bytes::Bytes::from_static(b"fake-txn");
    let error = client.prepare("select 1".to_string(), Some(txn_id)).await.unwrap_err();
    let status = unwrap_flight_status(error);
    assert_eq!(status.code(), Code::Unimplemented);

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_close_is_idempotent_via_raw_action() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    // Send a Close action for a handle that was never created. The server
    // must treat this as a no-op (Ok) rather than UNIMPLEMENTED or an error.
    use arrow_flight::sql::{ActionClosePreparedStatementRequest, ProstMessageExt};
    use prost::Message;
    let action_body =
        ActionClosePreparedStatementRequest { prepared_statement_handle: prost::bytes::Bytes::from_static(b"ghost") };
    let action = arrow_flight::Action {
        r#type: "ClosePreparedStatement".to_string(),
        body: action_body.as_any().encode_to_vec().into(),
    };
    let mut stream = client.do_action(action).await.unwrap();
    while stream.message().await.unwrap().is_some() {}

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_statement_invalid_sql_fails_at_create() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let error = client.prepare("select from where".to_string(), None).await.unwrap_err();
    let status = unwrap_flight_status(error);
    // Logical planning failures map through datafusion_error_to_status; the
    // exact code is engine-defined but must not be OK / UNIMPLEMENTED.
    assert_ne!(status.code(), Code::Ok);
    assert_ne!(status.code(), Code::Unimplemented);

    server.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn flight_sql_get_primary_keys_returns_empty_with_schema() {
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
    let mut client = connect_with_retry(addr).await;

    let cmd = CommandGetPrimaryKeys {
        catalog: None,
        db_schema: None,
        table: "any_table".to_string(),
    };
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
    let addr = free_local_addr();
    let mut config = ServerConfig::from_env_get(|_| None).unwrap();
    config.bind_addr = addr;

    let server = tokio::spawn(async move { serve(config).await });
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

fn unwrap_flight_status(error: FlightError) -> tonic::Status {
    match error {
        FlightError::Tonic(status) => *status,
        other => panic!("expected tonic Status, got {other:?}"),
    }
}

async fn collect_metadata_batches(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    flight_info: &arrow_flight::FlightInfo,
) -> Vec<arrow::record_batch::RecordBatch> {
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap()
}

/// Like `collect_metadata_batches`, but also returns the stream's schema
/// extracted from the IPC schema-header message. `try_collect::<Vec<_>>`
/// drops the schema and yields only RecordBatch payloads, so for empty-
/// result metadata RPCs (`get_primary_keys`, foreign-key RPCs) the batch
/// vector is empty and the schema would otherwise be unreachable.
async fn collect_metadata_schema_and_batches(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    flight_info: &arrow_flight::FlightInfo,
) -> (arrow::datatypes::SchemaRef, Vec<arrow::record_batch::RecordBatch>) {
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    let mut stream = client.do_get(ticket).await.unwrap();
    let mut batches = Vec::new();
    while let Some(batch) = stream.try_next().await.unwrap() {
        batches.push(batch);
    }
    let schema = stream.schema().cloned().expect("metadata stream must expose a schema");
    (schema, batches)
}

fn collect_name_and_type(batches: &[arrow::record_batch::RecordBatch]) -> Vec<(String, String)> {
    let names = collect_string_column(batches, "table_name");
    let types = collect_string_column(batches, "table_type");
    assert_eq!(names.len(), types.len(), "GetTables name/type columns desynced");
    names.into_iter().zip(types).collect()
}

fn collect_string_column(batches: &[arrow::record_batch::RecordBatch], column: &str) -> Vec<String> {
    let mut out = Vec::new();
    for batch in batches {
        let Some(index) = batch.schema().index_of(column).ok() else {
            continue;
        };
        let array =
            batch.column(index).as_any().downcast_ref::<StringArray>().expect("metadata column must be Utf8");
        for i in 0..array.len() {
            if array.is_valid(i) {
                out.push(array.value(i).to_string());
            }
        }
    }
    out
}

fn free_local_addr() -> std::net::SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
}

async fn connect_with_retry(addr: std::net::SocketAddr) -> FlightSqlServiceClient<tonic::transport::Channel> {
    let uri = format!("http://{addr}");
    let mut last_error = None;
    for _ in 0..50 {
        match Endpoint::from_shared(uri.clone()).unwrap().connect().await {
            Ok(channel) => return FlightSqlServiceClient::new(channel),
            Err(error) => {
                last_error = Some(error);
                tokio::time::sleep(Duration::from_millis(50)).await;
            },
        }
    }
    panic!("Flight SQL server did not accept connections at {addr}: {last_error:?}");
}

async fn execute_batches(
    client: &mut FlightSqlServiceClient<tonic::transport::Channel>,
    sql: &str,
) -> Vec<arrow::record_batch::RecordBatch> {
    let flight_info = client.execute(sql.to_string(), None).await.unwrap();
    let ticket = flight_info.endpoint.first().and_then(|endpoint| endpoint.ticket.clone()).unwrap();
    client.do_get(ticket).await.unwrap().try_collect::<Vec<_>>().await.unwrap()
}

fn int_value(array: &arrow::array::ArrayRef) -> Option<i64> {
    array
        .as_any()
        .downcast_ref::<Int64Array>()
        .map(|array| array.value(0))
        .or_else(|| array.as_any().downcast_ref::<Int32Array>().map(|array| i64::from(array.value(0))))
}

fn sql_info_string(batch: &arrow::record_batch::RecordBatch, info: SqlInfo) -> String {
    let (values, row) = sql_info_values(batch, info);
    assert_eq!(values.type_id(row), 0, "SqlInfo {info:?} must be encoded as a string");
    values
        .child(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("SqlInfo string child must be StringArray")
        .value(values.value_offset(row))
        .to_string()
}

fn sql_info_bool(batch: &arrow::record_batch::RecordBatch, info: SqlInfo) -> bool {
    let (values, row) = sql_info_values(batch, info);
    assert_eq!(values.type_id(row), 1, "SqlInfo {info:?} must be encoded as a bool");
    values
        .child(1)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("SqlInfo bool child must be BooleanArray")
        .value(values.value_offset(row))
}

fn sql_info_i32(batch: &arrow::record_batch::RecordBatch, info: SqlInfo) -> i32 {
    let (values, row) = sql_info_values(batch, info);
    assert_eq!(values.type_id(row), 3, "SqlInfo {info:?} must be encoded as an i32");
    values
        .child(3)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("SqlInfo i32 child must be Int32Array")
        .value(values.value_offset(row))
}

fn sql_info_values(batch: &arrow::record_batch::RecordBatch, info: SqlInfo) -> (&UnionArray, usize) {
    let names =
        batch.column(0).as_any().downcast_ref::<UInt32Array>().expect("SqlInfo info_name column must be UInt32Array");
    let values =
        batch.column(1).as_any().downcast_ref::<UnionArray>().expect("SqlInfo value column must be UnionArray");
    let expected = info as u32;
    let row = (0..names.len())
        .find(|row| names.value(*row) == expected)
        .unwrap_or_else(|| panic!("missing SqlInfo {info:?}"));
    (values, row)
}

fn sql_info_bitmask(values: impl IntoIterator<Item = i32>) -> i32 {
    values.into_iter().fold(0, |mask, value| mask | (1_i32 << value))
}

fn assert_error_metadata(status: &tonic::Status, code: &str, kind: &str) {
    let actual_code = status.metadata().get(ERROR_CODE_METADATA_KEY).expect("error code metadata must be present");
    assert_eq!(actual_code.to_str().expect("error code metadata must be ASCII"), code);
    let actual_kind = status.metadata().get(ERROR_KIND_METADATA_KEY).expect("error kind metadata must be present");
    assert_eq!(actual_kind.to_str().expect("error kind metadata must be ASCII"), kind);
}
