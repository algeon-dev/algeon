#![allow(dead_code, reason = "shared test helpers used across independent integration test targets")]

//! In-process Iceberg tables: a memory catalog over a local warehouse, data
//! files written by Iceberg's Parquet writer or by hand, and snapshots
//! committed with `fast_append` or assembled from hand-written manifests.

use std::{
    collections::HashMap,
    fs::{self, File},
    path::Path,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use algeon_datafusion::error::{DataFusionErrorExt, Error as AlgeonError, ErrorCode};
use algeon_iceberg::IcebergCatalogProvider;
use datafusion::{
    arrow::{
        array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray},
        datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef},
    },
    common::{DataFusionError, Result as DataFusionResult},
    datasource::{physical_plan::FileScanConfig, source::DataSourceExec},
    parquet::arrow::{ArrowWriter, PARQUET_FIELD_ID_META_KEY},
    physical_plan::ExecutionPlan,
    prelude::SessionContext,
};
use iceberg::{
    Catalog, CatalogBuilder, MetadataLocation, NamespaceIdent, TableCreation, TableIdent,
    io::LocalFsStorageFactory,
    memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder},
    spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, MAIN_BRANCH, ManifestListWriter,
        ManifestWriterBuilder, NestedField, Operation, PrimitiveType, Schema, Snapshot, Struct, Summary,
        TableMetadataBuilder, Type, UnboundPartitionSpec,
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
use tempfile::TempDir;

pub(crate) const NAMESPACE: &str = "ns";

static FILE_PREFIX: AtomicUsize = AtomicUsize::new(0);

/// A memory catalog whose warehouse is a temporary local directory.
pub(crate) struct Warehouse {
    pub(crate) dir: TempDir,
    pub(crate) catalog: Arc<dyn Catalog>,
}

impl Warehouse {
    pub(crate) async fn new() -> Self {
        Self::with_properties(HashMap::new()).await
    }

    /// A warehouse whose catalog FileIO also receives `properties`.
    pub(crate) async fn with_properties(properties: HashMap<String, String>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut catalog_properties = properties;
        catalog_properties.insert(MEMORY_CATALOG_WAREHOUSE.to_string(), format!("file://{}", dir.path().display()));
        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load("test", catalog_properties)
            .await
            .unwrap();
        catalog.create_namespace(&NamespaceIdent::new(NAMESPACE.to_string()), HashMap::new()).await.unwrap();
        Self { dir, catalog: Arc::new(catalog) }
    }

    pub(crate) fn ident(name: &str) -> TableIdent {
        TableIdent::new(NamespaceIdent::new(NAMESPACE.to_string()), name.to_string())
    }

    pub(crate) async fn create_table(
        &self,
        name: &str,
        schema: Schema,
        partition_spec: Option<UnboundPartitionSpec>,
        properties: HashMap<String, String>,
    ) -> Table {
        let creation = TableCreation::builder()
            .name(name.to_string())
            .schema(schema)
            .partition_spec_opt(partition_spec)
            .properties(properties)
            .build();
        self.catalog.create_table(&NamespaceIdent::new(NAMESPACE.to_string()), creation).await.unwrap()
    }

    pub(crate) async fn append(&self, table: &Table, files: Vec<DataFile>) -> Table {
        let transaction = Transaction::new(table);
        let transaction = transaction.fast_append().add_data_files(files).apply(transaction).unwrap();
        transaction.commit(self.catalog.as_ref()).await.unwrap()
    }

    /// Registers table `name`: `table` plus a snapshot whose delete manifest
    /// holds one position delete for the data file at `target`. Iceberg's
    /// transactions cannot commit delete files, so the manifest, manifest
    /// list and metadata are written by hand.
    pub(crate) async fn commit_position_delete(&self, table: &Table, name: &str, target: &str) -> Table {
        let metadata = table.metadata();
        let parent = metadata.current_snapshot().unwrap();
        let snapshot_id = parent.snapshot_id() + 1;
        let sequence_number = metadata.last_sequence_number() + 1;
        let file_io = table.file_io();
        let metadata_dir = format!("{}/metadata", metadata.location());

        let mut delete_file = DataFileBuilder::default();
        delete_file
            .content(DataContentType::PositionDeletes)
            .file_path(format!("{metadata_dir}/../data/deletes.parquet"))
            .file_format(DataFileFormat::Parquet)
            .partition(Struct::empty())
            .record_count(1)
            .file_size_in_bytes(64)
            .referenced_data_file(Some(target.to_string()));
        let manifest_output = file_io.new_output(format!("{metadata_dir}/deletes-{snapshot_id}.avro")).unwrap();
        let mut manifest = ManifestWriterBuilder::new(
            manifest_output,
            Some(snapshot_id),
            Arc::clone(metadata.current_schema()),
            metadata.default_partition_spec().as_ref().clone(),
        )
        .build_v2_deletes();
        manifest.add_file(delete_file.build().unwrap(), sequence_number).unwrap();
        let delete_manifest = manifest.write_manifest_file().await.unwrap();

        let existing = table.manifest_list_reader(parent).load().await.unwrap().consume_entries();
        let manifest_list = format!("{metadata_dir}/snap-{snapshot_id}.avro");
        let mut list = ManifestListWriter::v2(
            file_io.new_output(&manifest_list).unwrap().writer().await.unwrap(),
            snapshot_id,
            Some(parent.snapshot_id()),
            sequence_number,
        );
        list.add_manifests(existing.into_iter().chain([delete_manifest])).unwrap();
        list.close().await.unwrap();

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
            .set_branch_snapshot(snapshot, MAIN_BRANCH)
            .unwrap()
            .build()
            .unwrap()
            .metadata;
        let location = MetadataLocation::from_str(table.metadata_location().unwrap())
            .unwrap()
            .with_next_version()
            .try_with_new_metadata(&new_metadata)
            .unwrap();
        new_metadata.write_to(file_io, &location).await.unwrap();
        self.catalog.register_table(&Self::ident(name), location.to_string()).await.unwrap()
    }

    /// A path under the warehouse directory for a hand-written file.
    pub(crate) fn local_path(&self, name: &str) -> String {
        self.dir.path().join(name).display().to_string()
    }

    /// A session with every top-level namespace of the catalog as a schema
    /// of the `iceberg` catalog.
    pub(crate) async fn session(&self) -> SessionContext {
        let ctx = SessionContext::new();
        self.register(&ctx).await;
        ctx
    }

    pub(crate) async fn register(&self, ctx: &SessionContext) {
        let provider = IcebergCatalogProvider::try_new(Arc::clone(&self.catalog), None).await.unwrap();
        ctx.register_catalog("iceberg", Arc::new(provider));
    }
}

/// `id long required, name string optional, year int optional`.
pub(crate) fn people_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "year", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()
        .unwrap()
}

/// A batch of `people_schema` rows with ids `ids`; `year` is the id
/// divided by 100 plus 2000.
pub(crate) fn people_batch(schema: &ArrowSchemaRef, ids: &[i64]) -> RecordBatch {
    let names = ids.iter().map(|id| format!("person-{id}")).collect::<Vec<_>>();
    let years = ids.iter().map(|id| 2000 + i32::try_from(id / 100).unwrap()).collect::<Vec<_>>();
    RecordBatch::try_new(
        Arc::clone(schema),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())) as ArrayRef,
            Arc::new(StringArray::from(names)) as ArrayRef,
            Arc::new(Int32Array::from(years)) as ArrayRef,
        ],
    )
    .unwrap()
}

/// Writes `ids` through Iceberg's Parquet writer, which records field ids
/// and column bounds.
pub(crate) async fn write_people(table: &Table, ids: &[i64]) -> Vec<DataFile> {
    let iceberg_schema = Arc::clone(table.metadata().current_schema());
    let arrow_schema: ArrowSchemaRef = Arc::new(iceberg_schema.as_ref().try_into().unwrap());
    let rolling = RollingFileWriterBuilder::new_with_default_file_size(
        ParquetWriterBuilder::new(Default::default(), iceberg_schema),
        table.file_io().clone(),
        DefaultLocationGenerator::new(table.metadata()).unwrap(),
        // A generator numbers files from zero, so each call needs its own prefix.
        DefaultFileNameGenerator::new(
            format!("data-{}", FILE_PREFIX.fetch_add(1, Ordering::Relaxed)),
            None,
            DataFileFormat::Parquet,
        ),
    );
    let mut writer = DataFileWriterBuilder::new(rolling).build(None).await.unwrap();
    writer.write(people_batch(&arrow_schema, ids)).await.unwrap();
    writer.close().await.unwrap()
}

/// An Arrow field with a Parquet field id.
pub(crate) fn field_with_id(name: &str, data_type: DataType, nullable: bool, id: i32) -> Field {
    Field::new(name, data_type, nullable)
        .with_metadata(HashMap::from([(PARQUET_FIELD_ID_META_KEY.to_string(), id.to_string())]))
}

/// Writes `batch` as a Parquet file at `path` with Arrow's writer, which
/// keeps whatever field ids the batch schema carries.
pub(crate) fn write_parquet_bytes(batch: &RecordBatch) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, batch.schema(), None).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    bytes
}

pub(crate) fn write_parquet(path: &Path, batch: &RecordBatch) -> u64 {
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    fs::metadata(path).unwrap().len()
}

/// A manifest entry for a hand-written data file at `location`.
pub(crate) fn data_file(location: &str, size: u64, rows: u64) -> DataFileBuilder {
    let mut builder = DataFileBuilder::default();
    builder
        .content(DataContentType::Data)
        .file_path(location.to_string())
        .file_format(DataFileFormat::Parquet)
        .partition(Struct::empty())
        .record_count(rows)
        .file_size_in_bytes(size);
    builder
}

/// `people_schema` columns as an Arrow schema whose fields carry the field
/// ids in `ids`, or none when `ids` is `None`.
pub(crate) fn people_arrow_schema(ids: Option<[i32; 3]>) -> ArrowSchemaRef {
    let fields = [("id", DataType::Int64, false), ("name", DataType::Utf8, true), ("year", DataType::Int32, true)];
    Arc::new(ArrowSchema::new(
        fields
            .into_iter()
            .enumerate()
            .map(|(index, (name, data_type, nullable))| match ids {
                Some(ids) => field_with_id(name, data_type, nullable, ids[index]),
                None => Field::new(name, data_type, nullable),
            })
            .collect::<Vec<_>>(),
    ))
}

/// The file scan of a plan's only `DataSourceExec`.
pub(crate) fn file_scan(plan: &Arc<dyn ExecutionPlan>) -> FileScanConfig {
    fn find(plan: &Arc<dyn ExecutionPlan>, found: &mut Vec<FileScanConfig>) {
        if let Some(source) = plan.downcast_ref::<DataSourceExec>()
            && let Some(config) = source.data_source().downcast_ref::<FileScanConfig>()
        {
            found.push(config.clone());
        }
        for child in plan.children() {
            find(child, found);
        }
    }
    let mut found = Vec::new();
    find(plan, &mut found);
    assert_eq!(found.len(), 1, "the plan must hold one file scan");
    found.pop().unwrap()
}

pub(crate) async fn plan(ctx: &SessionContext, sql: &str) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
    ctx.sql(sql).await?.create_physical_plan().await
}

/// The code and facts of an Algeon error.
#[derive(Debug)]
pub(crate) struct Failure {
    pub(crate) code: ErrorCode,
    pub(crate) facts: HashMap<String, String>,
}

impl Failure {
    pub(crate) fn fact(&self, key: &str) -> Option<&str> {
        self.facts.get(key).map(String::as_str)
    }
}

pub(crate) async fn plan_failure(ctx: &SessionContext, sql: &str) -> Failure {
    let error = plan(ctx, sql).await.expect_err("planning must fail");
    failure(&error)
}

pub(crate) fn failure(error: &DataFusionError) -> Failure {
    let error: &AlgeonError = error.find_algeon_error().unwrap_or_else(|| panic!("not an Algeon error: {error}"));
    Failure {
        code: error.code(),
        facts: error.facts().iter().map(|(key, value)| (key.to_string(), value.to_string())).collect(),
    }
}

pub(crate) async fn sorted_ids(ctx: &SessionContext, sql: &str) -> Vec<i64> {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let mut ids = batches
        .iter()
        .flat_map(|batch| {
            let column = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            column.iter().map(Option::unwrap).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}
