//! Explicit E2E for Iceberg REST and Glue catalogs served over Flight SQL.
//!
//! The runner needs the local services of
//! `crates/tools/fixtures/iceberg-local/compose.yaml`: RustFS, the Iceberg
//! REST fixture and moto as Glue. It writes tables with iceberg-rust through
//! the same catalog connections the server makes, starts the Flight SQL
//! server in process the way the `algeon_server` launcher does with the
//! `iceberg` feature, and checks the catalogs through a Flight SQL client:
//! the catalog, namespace and table listings, a native aggregate over a
//! REST table and a Glue table, and the rejection of a snapshot with a
//! position-delete file. Every namespace, table and object it wrote is
//! removed at the end.

use std::{
    collections::{BTreeMap, HashMap},
    env,
    error::Error,
    fs,
    net::{SocketAddr, TcpListener},
    path::Path,
    str::FromStr,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use algeon_datafusion::{
    backend::GpuMemoryOwnership,
    server::{
        ALGEON_SERVER_CONFIG_FILE_ENV, ERROR_CODE_METADATA_KEY, FlightSqlServer, REASON_METADATA_KEY, ServerConfig,
        ServerShutdown, backend_from_server_config, build_session_context_with_backend,
    },
};
use algeon_iceberg::{connect_catalog, register_iceberg_catalogs};
use arrow::{
    array::{Array, ArrayRef, Int64Array, RecordBatch, StringArray},
    compute::cast,
    datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef},
};
use arrow_flight::{
    FlightInfo,
    error::FlightError,
    sql::{CommandGetDbSchemas, CommandGetTables, client::FlightSqlServiceClient},
};
use futures::TryStreamExt;
use iceberg::{
    Catalog, MetadataLocation, NamespaceIdent, TableCreation, TableIdent,
    io::FileIO,
    spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, MAIN_BRANCH, ManifestListWriter,
        ManifestWriterBuilder, NestedField, Operation, PrimitiveType, Schema, Snapshot, Struct, Summary,
        TableMetadataBuilder, Type,
    },
    table::Table,
    transaction::{ApplyTransactionAction, Transaction},
    writer::{
        IcebergWriter, IcebergWriterBuilder,
        base_writer::data_file_writer::DataFileWriterBuilder,
        file_writer::{
            ParquetWriterBuilder,
            location_generator::{DefaultFileNameGenerator, DefaultLocationGenerator},
            rolling_writer::RollingFileWriterBuilder,
        },
    },
};
use parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY};
use tokio_util::sync::CancellationToken;
use tonic::{Code, Status, transport::Channel};

use crate::connection_retry::{ConnectionRetryConfig, connect_with_retry};

type E2eResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

const S3_ENDPOINT: &str = "http://127.0.0.1:9000";
const REGION: &str = "us-east-1";
const REST_URI: &str = "http://127.0.0.1:8181";
const REST_WAREHOUSE: &str = "s3://algeon-e2e/warehouse/";
const GLUE_ENDPOINT: &str = "http://127.0.0.1:5001";
const GLUE_WAREHOUSE: &str = "s3://algeon-e2e/glue-warehouse";
const REST_CATALOG: &str = "rest";
const GLUE_CATALOG: &str = "glue";
const TABLE: &str = "events";
const DELETES_TABLE: &str = "events_with_deletes";
const GROUPS: i64 = 7;
/// Ids of the two data files of the REST table.
const REST_FILES: [std::ops::Range<i64>; 2] = [0..1_000, 1_000..2_500];
const GLUE_IDS: std::ops::Range<i64> = 0..700;
/// Iceberg's reserved field ids of the position-delete file columns.
const DELETE_FILE_PATH_FIELD_ID: i32 = 2_147_483_546;
const DELETE_POS_FIELD_ID: i32 = 2_147_483_545;
const CONNECT_BUDGET: Duration = Duration::from_secs(30);
const CONNECT_BACKOFF: Duration = Duration::from_millis(100);

/// Catalog objects the run created, removed in reverse order at the end.
#[derive(Default)]
struct Written {
    tables: Vec<(Arc<dyn Catalog>, TableIdent)>,
    namespaces: Vec<(Arc<dyn Catalog>, NamespaceIdent)>,
    locations: Vec<(FileIO, String)>,
}

pub async fn run(args: Vec<String>) -> E2eResult<()> {
    if let Some(arg) = args.first() {
        if arg == "-h" || arg == "--help" {
            print_usage();
            return Ok(());
        }
        return Err(format!("unknown iceberg-local-e2e argument: {arg}").into());
    }
    let secs = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let namespace = format!("e2e_{}_{secs}", std::process::id());
    let config_dir = env::temp_dir().join(format!("algeon-iceberg-local-e2e-{namespace}"));
    let config = server_config(&config_dir)?;
    let rest = connect_catalog(&config.iceberg_catalogs[0], config.s3_storage()).await?;
    let glue = connect_catalog(&config.iceberg_catalogs[1], config.s3_storage()).await?;

    let mut written = Written::default();
    let result = async {
        write_tables(&rest, &glue, &namespace, &mut written).await?;
        serve_and_check(config, &namespace).await
    }
    .await;
    let cleanup = remove(written).await;
    fs::remove_dir_all(&config_dir)?;
    result?;
    cleanup?;
    println!("cleanup ok");
    println!("iceberg_local_e2e ok");
    Ok(())
}

fn print_usage() {
    println!(
        "Usage: iceberg-local-e2e\n\n\
         Writes Iceberg tables through a REST catalog and a Glue catalog, serves both from an in-process\n\
         Flight SQL server in native_required mode, checks the listings, two native aggregates and the\n\
         rejection of a snapshot with a position-delete file, and removes what it wrote. It needs the\n\
         services of crates/tools/fixtures/iceberg-local/compose.yaml: RustFS at {S3_ENDPOINT}, the REST\n\
         catalog at {REST_URI} and moto as Glue at {GLUE_ENDPOINT}. S3 and Glue credentials come from the\n\
         default AWS chain.\n\n\
           docker compose -f crates/tools/fixtures/iceberg-local/compose.yaml up -d --wait\n\
           AWS_ACCESS_KEY_ID=algeon-e2e AWS_SECRET_ACCESS_KEY=algeon-e2e-secret \\\n\
             flock /tmp/cudf-gpu.lock cargo run -p tools --features iceberg-local-e2e -- iceberg-local-e2e\n\
           docker compose -f crates/tools/fixtures/iceberg-local/compose.yaml down -v"
    );
}

/// The server configuration, loaded from a TOML file like the launcher's.
/// The REST entry comes first and the Glue entry second.
fn server_config(dir: &Path) -> E2eResult<ServerConfig> {
    fs::create_dir_all(dir)?;
    let path = dir.join("server.toml");
    let toml = format!(
        "[native]\nexecution_mode = \"native_required\"\n\n\
         [s3]\nendpoint = \"{S3_ENDPOINT}\"\nregion = \"{REGION}\"\npath_style = true\n\n\
         [[iceberg.catalogs]]\nname = \"{REST_CATALOG}\"\nkind = \"rest\"\nuri = \"{REST_URI}\"\nwarehouse = \"{REST_WAREHOUSE}\"\n\n\
         [[iceberg.catalogs]]\nname = \"{GLUE_CATALOG}\"\nkind = \"glue\"\nwarehouse = \"{GLUE_WAREHOUSE}\"\n\
         region = \"{REGION}\"\nendpoint = \"{GLUE_ENDPOINT}\"\n\n\
         [[admission.device_profiles]]\ndevice_ordinal = 0\n"
    );
    fs::write(&path, toml)?;
    let config = ServerConfig::from_vars([(ALGEON_SERVER_CONFIG_FILE_ENV, path.to_string_lossy().into_owned())])?;
    Ok(config.with_gpu_memory_ownership([(0, GpuMemoryOwnership::WholeDeviceExclusive)]))
}

/// `id`, `grp` and `val`, all required longs.
fn events_schema() -> E2eResult<Schema> {
    Ok(Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "grp", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(3, "val", Type::Primitive(PrimitiveType::Long)).into(),
        ])
        .build()?)
}

/// REST: `events` with two data files from one `fast_append`, and
/// `events_with_deletes`, a snapshot of `events` plus a position delete.
/// Glue: `events` with one data file.
async fn write_tables(
    rest: &Arc<dyn Catalog>,
    glue: &Arc<dyn Catalog>,
    namespace: &str,
    written: &mut Written,
) -> E2eResult<()> {
    let rest_events = create_table(rest, namespace, written).await?;
    let mut files = Vec::new();
    for (index, ids) in REST_FILES.iter().enumerate() {
        files.extend(write_events(&rest_events, &format!("part-{index}"), ids.clone()).await?);
    }
    if files.len() != REST_FILES.len() {
        return Err(format!("expected {} REST data files, wrote {}", REST_FILES.len(), files.len()).into());
    }
    let target = files[0].file_path().to_string();
    let rest_events = append(rest, &rest_events, files).await?;
    let deletes_ident = TableIdent::new(NamespaceIdent::new(namespace.to_string()), DELETES_TABLE.to_string());
    commit_position_delete(rest, &rest_events, &deletes_ident, &target).await?;
    written.tables.push((Arc::clone(rest), deletes_ident));
    println!("rest tables written: {TABLE} ({} data files), {DELETES_TABLE} (1 position delete)", REST_FILES.len());

    let glue_events = create_table(glue, namespace, written).await?;
    let files = write_events(&glue_events, "part-0", GLUE_IDS).await?;
    append(glue, &glue_events, files).await?;
    println!("glue table written: {TABLE} (1 data file)");
    Ok(())
}

async fn create_table(catalog: &Arc<dyn Catalog>, namespace: &str, written: &mut Written) -> E2eResult<Table> {
    let namespace = NamespaceIdent::new(namespace.to_string());
    catalog.create_namespace(&namespace, HashMap::new()).await?;
    written.namespaces.push((Arc::clone(catalog), namespace.clone()));
    let creation = TableCreation::builder().name(TABLE.to_string()).schema(events_schema()?).build();
    let table = catalog.create_table(&namespace, creation).await?;
    written.tables.push((Arc::clone(catalog), table.identifier().clone()));
    written.locations.push((table.file_io().clone(), table.metadata().location().to_string()));
    Ok(table)
}

fn events_batch(schema: &ArrowSchemaRef, ids: std::ops::Range<i64>) -> E2eResult<RecordBatch> {
    let ids = ids.collect::<Vec<_>>();
    Ok(RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(Int64Array::from(ids.clone())) as ArrayRef,
            Arc::new(Int64Array::from(ids.iter().map(|id| id % GROUPS).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(ids.iter().map(|id| id * 3).collect::<Vec<_>>())),
        ],
    )?)
}

/// Writes `ids` with Iceberg's Parquet writer, which records field ids and
/// column bounds, under the table's data directory.
async fn write_events(table: &Table, prefix: &str, ids: std::ops::Range<i64>) -> E2eResult<Vec<DataFile>> {
    let iceberg_schema = Arc::clone(table.metadata().current_schema());
    let arrow_schema: ArrowSchemaRef = Arc::new(iceberg_schema.as_ref().try_into()?);
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(Default::default(), iceberg_schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata())?,
        DefaultFileNameGenerator::new(prefix.to_string(), None, DataFileFormat::Parquet),
    );
    let mut writer = DataFileWriterBuilder::new(rolling).build(None).await?;
    writer.write(events_batch(&arrow_schema, ids)?).await?;
    Ok(writer.close().await?)
}

async fn append(catalog: &Arc<dyn Catalog>, table: &Table, files: Vec<DataFile>) -> E2eResult<Table> {
    let transaction = Transaction::new(table);
    let transaction = transaction.fast_append().add_data_files(files).apply(transaction)?;
    Ok(transaction.commit(catalog.as_ref()).await?)
}

/// Registers `ident` as `table` plus a snapshot whose delete manifest holds
/// one position delete for the data file at `target`. Iceberg's
/// transactions cannot commit delete files, so the delete file, manifest,
/// manifest list and metadata are written by hand.
async fn commit_position_delete(
    catalog: &Arc<dyn Catalog>,
    table: &Table,
    ident: &TableIdent,
    target: &str,
) -> E2eResult<()> {
    let metadata = table.metadata();
    let parent = metadata.current_snapshot().ok_or("the table has no snapshot to delete from")?;
    let snapshot_id = parent.snapshot_id() + 1;
    let sequence_number = metadata.last_sequence_number() + 1;
    let file_io = table.file_io();
    let metadata_dir = format!("{}/metadata", metadata.location());

    let delete_path = format!("{}/data/deletes-{snapshot_id}.parquet", metadata.location());
    let delete_bytes = position_delete_parquet(target)?;
    let delete_size = u64::try_from(delete_bytes.len())?;
    file_io.new_output(&delete_path)?.write(delete_bytes.into()).await?;
    let mut delete_file = DataFileBuilder::default();
    delete_file
        .content(DataContentType::PositionDeletes)
        .file_path(delete_path)
        .file_format(DataFileFormat::Parquet)
        .partition(Struct::empty())
        .record_count(1)
        .file_size_in_bytes(delete_size)
        .referenced_data_file(Some(target.to_string()));
    let manifest_output = file_io.new_output(format!("{metadata_dir}/deletes-{snapshot_id}.avro"))?;
    let mut manifest = ManifestWriterBuilder::new(
        manifest_output,
        Some(snapshot_id),
        Arc::clone(metadata.current_schema()),
        metadata.default_partition_spec().as_ref().clone(),
    )
    .build_v2_deletes();
    manifest.add_file(delete_file.build()?, sequence_number)?;
    let delete_manifest = manifest.write_manifest_file().await?;

    let existing = table.manifest_list_reader(parent).load().await?.consume_entries();
    let manifest_list = format!("{metadata_dir}/snap-{snapshot_id}.avro");
    let mut list = ManifestListWriter::v2(
        file_io.new_output(&manifest_list)?.writer().await?,
        snapshot_id,
        Some(parent.snapshot_id()),
        sequence_number,
    );
    list.add_manifests(existing.into_iter().chain([delete_manifest]))?;
    list.close().await?;

    let snapshot = Snapshot::builder()
        .with_snapshot_id(snapshot_id)
        .with_parent_snapshot_id(Some(parent.snapshot_id()))
        .with_sequence_number(sequence_number)
        .with_timestamp_ms(parent.timestamp_ms() + 1)
        .with_manifest_list(manifest_list)
        .with_summary(Summary { operation: Operation::Delete, additional_properties: HashMap::new() })
        .with_schema_id(metadata.current_schema_id())
        .build();
    let new_metadata = TableMetadataBuilder::new_from_metadata(metadata.clone(), None)
        .set_branch_snapshot(snapshot, MAIN_BRANCH)?
        .build()?
        .metadata;
    // The next version gets a new file name; the current file stays the
    // metadata of `table`.
    let location = MetadataLocation::from_str(table.metadata_location().ok_or("the table has no metadata location")?)?
        .with_next_version()
        .try_with_new_metadata(&new_metadata)?;
    new_metadata.write_to(file_io, &location).await?;
    catalog.register_table(ident, location.to_string()).await?;
    Ok(())
}

/// A position-delete file that deletes row 0 of `target`.
fn position_delete_parquet(target: &str) -> E2eResult<Vec<u8>> {
    let field = |name: &str, data_type, id: i32| {
        Field::new(name, data_type, false)
            .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())]))
    };
    let schema = Arc::new(ArrowSchema::new(vec![
        field("file_path", DataType::Utf8, DELETE_FILE_PATH_FIELD_ID),
        field("pos", DataType::Int64, DELETE_POS_FIELD_ID),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(StringArray::from(vec![target])) as ArrayRef, Arc::new(Int64Array::from(vec![0_i64]))],
    )?;
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(bytes)
}

/// Drops the tables and namespaces and deletes every object under the table
/// locations. Every step runs; the first failure is returned.
async fn remove(written: Written) -> E2eResult<()> {
    let mut first_error: Option<Box<dyn Error + Send + Sync>> = None;
    let mut record = |result: E2eResult<()>| {
        if let Err(error) = result
            && first_error.is_none()
        {
            first_error = Some(error);
        }
    };
    for (catalog, ident) in written.tables.iter().rev() {
        record(catalog.drop_table(ident).await.map_err(Into::into));
    }
    for (catalog, namespace) in written.namespaces.iter().rev() {
        record(catalog.drop_namespace(namespace).await.map_err(Into::into));
    }
    for (file_io, location) in &written.locations {
        record(file_io.delete_prefix(location).await.map_err(Into::into));
    }
    first_error.map_or(Ok(()), Err)
}

/// Builds the server as the launcher does: the backend and shared session
/// from `config`, the Iceberg catalogs registered on that session, then the
/// Flight SQL server around both.
async fn serve_and_check(config: ServerConfig, namespace: &str) -> E2eResult<()> {
    let backend = backend_from_server_config(&config)?;
    let ctx = build_session_context_with_backend(&config, &backend).await?;
    register_iceberg_catalogs(&ctx, &config).await?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    let shutdown = CancellationToken::new();
    let server = FlightSqlServer::builder()
        .config(config)
        .backend(backend)
        .session_context(Arc::new(ctx))
        .listener(listener)
        .shutdown(ServerShutdown::Token(shutdown.clone()))
        .build()
        .await?;
    let serving = tokio::spawn(server.serve());
    let checks = check_server(addr, namespace).await;
    shutdown.cancel();
    let served = serving.await?;
    checks?;
    served?;
    Ok(())
}

async fn check_server(addr: SocketAddr, namespace: &str) -> E2eResult<()> {
    let url = format!("http://{addr}");
    let mut client = connect_with_retry(
        &url,
        ConnectionRetryConfig::time_budget(CONNECT_BUDGET, CONNECT_BACKOFF),
        FlightSqlServiceClient::new,
        |error| format!("no Flight SQL server accepted connections at {url}: {error}"),
    )
    .await?;

    let info = client.get_catalogs().await?;
    let catalogs = string_column(&fetch(&mut client, info).await?, "catalog_name")?;
    for name in [REST_CATALOG, GLUE_CATALOG] {
        if !catalogs.iter().any(|catalog| catalog == name) {
            return Err(format!("GetCatalogs returned {catalogs:?}, missing {name}").into());
        }
    }
    println!("catalogs listed ok: {REST_CATALOG}, {GLUE_CATALOG}");

    for catalog in [REST_CATALOG, GLUE_CATALOG] {
        let info = client
            .get_db_schemas(CommandGetDbSchemas { catalog: Some(catalog.into()), db_schema_filter_pattern: None })
            .await?;
        let schemas = string_column(&fetch(&mut client, info).await?, "db_schema_name")?;
        if !schemas.iter().any(|schema| schema == namespace) {
            return Err(format!("GetDbSchemas for {catalog} returned {schemas:?}, missing {namespace}").into());
        }
        println!("{catalog} namespace listed ok: {namespace}");
    }

    for (catalog, expected) in [(REST_CATALOG, vec![TABLE, DELETES_TABLE]), (GLUE_CATALOG, vec![TABLE])] {
        let info = client
            .get_tables(CommandGetTables {
                catalog: Some(catalog.into()),
                db_schema_filter_pattern: Some(namespace.into()),
                table_name_filter_pattern: None,
                table_types: vec![],
                include_schema: false,
            })
            .await?;
        let mut tables = string_column(&fetch(&mut client, info).await?, "table_name")?;
        tables.sort();
        let mut expected = expected.into_iter().map(str::to_string).collect::<Vec<_>>();
        expected.sort();
        if tables != expected {
            return Err(
                format!("GetTables for {catalog}.{namespace} returned {tables:?}, expected {expected:?}").into()
            );
        }
        println!("{catalog} tables listed ok: {}", tables.join(", "));
    }

    let rest_ids = REST_FILES.iter().flat_map(Clone::clone).collect::<Vec<_>>();
    for (catalog, ids) in [(REST_CATALOG, rest_ids), (GLUE_CATALOG, GLUE_IDS.collect())] {
        let query = format!(
            "SELECT grp, COUNT(*) AS n, SUM(val) AS total FROM {catalog}.{namespace}.{TABLE} GROUP BY grp ORDER BY grp"
        );
        let (gpu_path, reason_code, detail) = explain_gpu(&mut client, &query).await?;
        if gpu_path != "native" {
            return Err(format!(
                "EXPLAIN GPU over {catalog}.{namespace}.{TABLE} reports gpu_path {gpu_path}, reason {reason_code}: {detail}"
            )
            .into());
        }
        let info = client.execute(query, None).await?;
        let rows = aggregate_rows(&fetch(&mut client, info).await?)?;
        let expected = expected_rows(&ids);
        if rows != expected {
            return Err(format!("{catalog}.{namespace}.{TABLE} returned {rows:?}, expected {expected:?}").into());
        }
        println!("{catalog} aggregate ok: native plan, {} rows over {} ids", rows.len(), ids.len());
    }

    let query = format!("SELECT COUNT(*) FROM {REST_CATALOG}.{namespace}.{DELETES_TABLE}");
    let status = match execute(&mut client, &query).await {
        Ok(_) => return Err(format!("{query} succeeded; the snapshot has a position-delete file").into()),
        Err(FlightError::Tonic(status)) => *status,
        Err(other) => return Err(format!("{query} failed without a gRPC status: {other}").into()),
    };
    let code = metadata_value(&status, ERROR_CODE_METADATA_KEY);
    let reason = metadata_value(&status, REASON_METADATA_KEY);
    if status.code() != Code::Unimplemented
        || code != Some("unsupported_table_feature")
        || reason != Some("iceberg_delete_files")
    {
        return Err(format!(
            "{query} failed with {:?}, {ERROR_CODE_METADATA_KEY}={code:?}, {REASON_METADATA_KEY}={reason:?}: {}",
            status.code(),
            status.message()
        )
        .into());
    }
    println!("rest delete-file table rejected ok: UNIMPLEMENTED, unsupported_table_feature, iceberg_delete_files");
    Ok(())
}

/// Runs `sql` and collects its batches. A planning or execution failure
/// keeps its gRPC status, whether GetFlightInfo or DoGet returns it.
async fn execute(client: &mut FlightSqlServiceClient<Channel>, sql: &str) -> Result<Vec<RecordBatch>, FlightError> {
    let info = client.execute(sql.to_string(), None).await?;
    fetch(client, info).await
}

async fn fetch(
    client: &mut FlightSqlServiceClient<Channel>,
    info: FlightInfo,
) -> Result<Vec<RecordBatch>, FlightError> {
    let mut batches = Vec::new();
    for endpoint in info.endpoint {
        let Some(ticket) = endpoint.ticket else {
            continue;
        };
        batches.extend(client.do_get(ticket).await?.try_collect::<Vec<_>>().await?);
    }
    Ok(batches)
}

/// The `gpu_path`, `reason_code` and `detail_json` of the summary row
/// `EXPLAIN GPU` returns for `query`.
async fn explain_gpu(client: &mut FlightSqlServiceClient<Channel>, query: &str) -> E2eResult<(String, String, String)> {
    let batches = execute(client, &format!("EXPLAIN GPU {query}")).await?;
    let row_kinds = string_column(&batches, "row_kind")?;
    let gpu_paths = string_column(&batches, "gpu_path")?;
    let reason_codes = string_column(&batches, "reason_code")?;
    let details = string_column(&batches, "detail_json")?;
    let summary = row_kinds.iter().position(|kind| kind == "summary").ok_or("EXPLAIN GPU returned no summary row")?;
    Ok((gpu_paths[summary].clone(), reason_codes[summary].clone(), details[summary].clone()))
}

fn string_column(batches: &[RecordBatch], name: &str) -> E2eResult<Vec<String>> {
    let mut values = Vec::new();
    for batch in batches {
        let column = cast(batch.column(batch.schema().index_of(name)?), &DataType::Utf8)?;
        let column = column.as_any().downcast_ref::<StringArray>().ok_or("cast to Utf8 returned another array")?;
        values.extend((0..column.len()).map(|row| column.value(row).to_string()));
    }
    Ok(values)
}

fn aggregate_rows(batches: &[RecordBatch]) -> E2eResult<Vec<(i64, i64, i64)>> {
    let mut rows = Vec::new();
    for batch in batches {
        let column = |index: usize| -> E2eResult<&Int64Array> {
            Ok(batch.column(index).as_any().downcast_ref::<Int64Array>().ok_or("result column is not Int64")?)
        };
        let (grp, count, total) = (column(0)?, column(1)?, column(2)?);
        for row in 0..batch.num_rows() {
            rows.push((grp.value(row), count.value(row), total.value(row)));
        }
    }
    Ok(rows)
}

fn expected_rows(ids: &[i64]) -> Vec<(i64, i64, i64)> {
    let mut groups = BTreeMap::<i64, (i64, i64)>::new();
    for id in ids {
        let entry = groups.entry(id % GROUPS).or_default();
        entry.0 += 1;
        entry.1 += id * 3;
    }
    groups.into_iter().map(|(grp, (count, total))| (grp, count, total)).collect()
}

fn metadata_value<'a>(status: &'a Status, key: &str) -> Option<&'a str> {
    status.metadata().get(key).and_then(|value| value.to_str().ok())
}
