//! [`IcebergTableProvider`]: an Iceberg table planned as a Parquet scan.

use std::{fmt, sync::Arc};

use algeon_datafusion::{
    error::{Error, ErrorCode, Result},
    native_exec::S3SourceStorageProfile,
};
use async_trait::async_trait;
use datafusion::{
    arrow::datatypes::{Schema, SchemaRef},
    catalog::{Session, TableProvider},
    common::{DataFusionError, Result as DataFusionResult, Statistics, stats::Precision},
    datasource::{
        TableType,
        file_format::{FileFormat, parquet::ParquetFormat},
        listing::PartitionedFile,
        physical_plan::{FileGroup, FileScanConfigBuilder, parquet::metadata::DFParquetMetadata},
        table_schema::TableSchema,
    },
    execution::{cache::cache_manager::FileMetadataCache, object_store::ObjectStoreUrl},
    logical_expr::{Expr, TableProviderFilterPushDown},
    object_store::{ObjectMeta, ObjectStore, ObjectStoreExt, path::Path as ObjectPath},
    parquet::arrow::parquet_to_arrow_schema,
    physical_plan::{ExecutionPlan, empty::EmptyExec},
};
use futures::{StreamExt, TryStreamExt, stream};
use iceberg::{
    Catalog, TableIdent,
    scan::FileScanTask,
    spec::{DataFileFormat, NameMapping, SchemaRef as IcebergSchemaRef},
    table::Table,
};

use crate::{
    binding::{FileBinding, identity_partition_sources},
    error::{SCAN_OPERATION, UnsupportedReason, iceberg_error, unsupported},
    predicate::filters_to_predicate,
    schema::table_arrow_schema,
    storage::mismatched_property,
};

/// Data files whose object metadata and footer one scan reads at a time.
const FOOTER_FETCH_CONCURRENCY: usize = 32;

const LOAD_OPERATION: &str = "algeon_iceberg.load_table";

/// An Iceberg table that DataFusion scans as Parquet.
///
/// Every [`TableProvider::scan`] loads the table again and plans the files
/// of its current snapshot. The scan is a `DataSourceExec` over a stock
/// `ParquetSource` with one file per Iceberg task, which Algeon's native
/// lowering reads like a listing table. A snapshot whose files that reader
/// would read incorrectly fails the scan with
/// [`ErrorCode::UnsupportedTableFeature`] in every execution mode; the
/// `reason` fact names the feature.
pub struct IcebergTableProvider {
    catalog: Arc<dyn Catalog>,
    ident: TableIdent,
    storage: Option<S3SourceStorageProfile>,
    schema: SchemaRef,
}

impl fmt::Debug for IcebergTableProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IcebergTableProvider")
            .field("ident", &self.ident)
            .field("storage", &self.storage)
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl IcebergTableProvider {
    /// Loads `ident` from `catalog` and takes its schema.
    ///
    /// `storage` is the S3 storage the session reads data files with. When
    /// set, a table whose FileIO properties name another endpoint, region or
    /// addressing style is rejected with reason
    /// `iceberg_storage_properties_mismatch`. A missing table fails with
    /// [`ErrorCode::ObjectNotFound`].
    pub async fn try_new(
        catalog: Arc<dyn Catalog>,
        ident: TableIdent,
        storage: Option<S3SourceStorageProfile>,
    ) -> Result<Self> {
        let mut provider = Self { catalog, ident, storage, schema: Arc::new(Schema::empty()) };
        let table = provider.load().await?;
        provider.schema = table_arrow_schema(scan_schema(&table)?.as_ref())?;
        Ok(provider)
    }

    /// The table identifier in its catalog.
    pub fn ident(&self) -> &TableIdent {
        &self.ident
    }

    async fn load(&self) -> Result<Table> {
        let table = self.catalog.load_table(&self.ident).await.map_err(|source| {
            iceberg_error(LOAD_OPERATION, "failed to load the Iceberg table", source).with_fact("table", &self.ident)
        })?;
        if let Some(storage) = &self.storage
            && let Some(property) = mismatched_property(table.file_io().config().props(), storage)
        {
            return Err(unsupported(
                UnsupportedReason::StoragePropertiesMismatch,
                &self.ident,
                table.metadata().current_snapshot_id(),
                format!("the Iceberg table's FileIO property `{property}` differs from the server's [s3] storage"),
            )
            .with_fact("file_io_property", property));
        }
        Ok(table)
    }

    async fn scan_plan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let projected_schema = match projection {
            Some(indices) => Arc::new(self.schema.project(indices)?),
            None => Arc::clone(&self.schema),
        };
        let table = self.load().await?;
        let Some(snapshot) = table.metadata().current_snapshot() else {
            return Ok(Arc::new(EmptyExec::new(projected_schema)));
        };
        let snapshot_id = snapshot.snapshot_id();
        let iceberg_schema = scan_schema(&table)?;
        if table_arrow_schema(&iceberg_schema)? != self.schema {
            return Err(Error::new(
                ErrorCode::SourceTemporaryFailure,
                SCAN_OPERATION,
                "the Iceberg table schema changed after the query resolved the table; run the query again",
            )
            .with_fact("table", &self.ident)
            .with_fact("snapshot_id", snapshot_id)
            .into());
        }

        let mut scan = table.scan().snapshot_id(snapshot_id);
        scan = match projection {
            Some(indices) => scan.select(indices.iter().map(|index| self.schema.field(*index).name().clone())),
            None => scan.select_all(),
        };
        if let Some(predicate) = filters_to_predicate(filters, &iceberg_schema) {
            scan = scan.with_filter(predicate);
        }
        let plan_error = |source| iceberg_error(SCAN_OPERATION, "failed to plan the Iceberg scan", source);
        let tasks = scan
            .build()
            .map_err(plan_error)?
            .plan_files()
            .await
            .map_err(plan_error)?
            .try_collect::<Vec<_>>()
            .await
            .map_err(plan_error)?;

        let context = ScanContext { ident: &self.ident, snapshot_id, name_mapping: table_name_mapping(&table)? };
        let mut store_url = None;
        let mut located = Vec::with_capacity(tasks.len());
        for task in tasks {
            let (url, path) = context.check_task(&task)?;
            match &store_url {
                None => store_url = Some(url),
                Some(first) if *first != url => {
                    return Err(context
                        .rejection(
                            UnsupportedReason::MultipleBuckets,
                            "the snapshot's data files are in more than one bucket",
                        )
                        .into());
                },
                Some(_) => {},
            }
            located.push((task, path));
        }
        let Some(store_url) = store_url else {
            return Ok(Arc::new(EmptyExec::new(projected_schema)));
        };

        let runtime_env = state.runtime_env();
        let store = runtime_env.object_store(&store_url)?;
        let metadata_cache = runtime_env.cache_manager.get_file_metadata_cache();
        let metadata_size_hint = state.table_options().parquet.global.metadata_size_hint;
        let files = stream::iter(located)
            .map(|(task, path)| {
                let store = Arc::clone(&store);
                let metadata_cache = Arc::clone(&metadata_cache);
                let context = &context;
                let table_schema = &self.schema;
                async move {
                    context
                        .partitioned_file(store.as_ref(), metadata_cache, metadata_size_hint, &task, path, table_schema)
                        .await
                }
            })
            .buffered(FOOTER_FETCH_CONCURRENCY)
            .try_collect::<Vec<_>>()
            .await?;

        let total_rows = files
            .iter()
            .map(|file| file.statistics.as_ref().map_or(Precision::Absent, |stats| stats.num_rows))
            .fold(Precision::Exact(0), |total, rows| total.add(&rows));
        let statistics = Statistics { num_rows: total_rows, ..Statistics::new_unknown(&self.schema) };
        let file_groups = FileGroup::new(files).split_files(state.config().target_partitions().max(1));
        let format = ParquetFormat::default().with_options(state.table_options().parquet.clone());
        let source = format.file_source(TableSchema::from(Arc::clone(&self.schema)));
        let config = FileScanConfigBuilder::new(store_url, source)
            .with_file_groups(file_groups)
            .with_statistics(statistics)
            .with_projection_indices(projection.cloned())?
            .with_limit(limit)
            .build();
        format.create_physical_plan(state, config).await
    }
}

/// The schema `TableScan` binds against: the current snapshot's schema, or
/// the table's current schema when it has no snapshot.
fn scan_schema(table: &Table) -> Result<IcebergSchemaRef> {
    match table.metadata().current_snapshot() {
        Some(snapshot) => snapshot
            .schema(table.metadata())
            .map_err(|source| iceberg_error(LOAD_OPERATION, "the Iceberg snapshot names an unknown schema", source)),
        None => Ok(Arc::clone(table.metadata().current_schema())),
    }
}

fn table_name_mapping(table: &Table) -> Result<Option<NameMapping>> {
    table
        .metadata()
        .table_properties()
        .default_name_mapping()
        .map_err(|source| iceberg_error(SCAN_OPERATION, "the Iceberg table's name mapping is invalid", source))
}

struct ScanContext<'a> {
    ident: &'a TableIdent,
    snapshot_id: i64,
    name_mapping: Option<NameMapping>,
}

impl ScanContext<'_> {
    fn rejection(&self, reason: UnsupportedReason, message: impl Into<String>) -> Error {
        unsupported(reason, self.ident, Some(self.snapshot_id), message)
    }

    /// Rejects a task the Parquet scan cannot read and returns the store
    /// URL and in-store path of its data file.
    fn check_task(&self, task: &FileScanTask) -> Result<(ObjectStoreUrl, ObjectPath)> {
        let location = task.data_file_path();
        let file_rejection = |reason, message: &str| self.rejection(reason, message).with_fact("path", location);
        if !task.deletes().is_empty() {
            return Err(file_rejection(UnsupportedReason::DeleteFiles, "a data file of the snapshot has delete files")
                .with_fact("delete_files", task.deletes().len()));
        }
        if task.data_file_format() != DataFileFormat::Parquet {
            return Err(file_rejection(
                UnsupportedReason::NonParquetFile,
                "a data file of the snapshot is not Parquet",
            )
            .with_fact("file_format", task.data_file_format()));
        }
        if task.key_metadata().is_some() {
            return Err(file_rejection(UnsupportedReason::EncryptedFile, "a data file of the snapshot is encrypted"));
        }
        split_location(location).ok_or_else(|| {
            file_rejection(
                UnsupportedReason::UnsupportedLocationScheme,
                "a data file of the snapshot is outside s3:// storage",
            )
        })
    }

    /// Reads the object metadata and footer of a task's data file, checks
    /// its column binding, and returns it as a scan file. The footer goes
    /// into the session's file metadata cache, where native lowering reads
    /// it.
    async fn partitioned_file(
        &self,
        store: &dyn ObjectStore,
        metadata_cache: Arc<FileMetadataCache>,
        metadata_size_hint: Option<usize>,
        task: &FileScanTask,
        path: ObjectPath,
        table_schema: &SchemaRef,
    ) -> DataFusionResult<PartitionedFile> {
        let object = store.head(&path).await.map_err(|error| DataFusionError::ObjectStore(Box::new(error)))?;
        if object.size != task.file_size_in_bytes() {
            return Err(Error::new(
                ErrorCode::SourceDependencyFailure,
                SCAN_OPERATION,
                "a data file's size differs from the size its Iceberg manifest records",
            )
            .with_fact("table", self.ident)
            .with_fact("snapshot_id", self.snapshot_id)
            .with_fact("path", task.data_file_path())
            .with_fact("object_size", object.size)
            .with_fact("manifest_size", task.file_size_in_bytes())
            .into());
        }
        let footer = DFParquetMetadata::new(store, &object)
            .with_metadata_size_hint(metadata_size_hint)
            .with_file_metadata_cache(Some(metadata_cache))
            .fetch_metadata()
            .await?;
        let file_metadata = footer.file_metadata();
        let file_schema = parquet_to_arrow_schema(file_metadata.schema_descr(), file_metadata.key_value_metadata())?;
        let binding = FileBinding {
            identity_partition_sources: task
                .partition_spec()
                .map(|spec| identity_partition_sources(spec))
                .unwrap_or_default(),
            name_mapping: task
                .name_mapping()
                .map(|mapping| mapping.fields())
                .or_else(|| self.name_mapping.as_ref().map(NameMapping::fields)),
        };
        binding.check(task.schema(), task.project_field_ids(), file_schema.fields()).map_err(|rejection| {
            let message = match rejection.reason {
                UnsupportedReason::FieldIdMismatch => "a data file binds a column by field id to another name",
                UnsupportedReason::MissingFieldIds => {
                    "a data file without field ids has no name mapping that resolves a column to its current name"
                },
                UnsupportedReason::PartitionColumnAbsent => {
                    "a data file lacks an identity partition column, whose values are only in partition data"
                },
                _ => "a data file lacks a required column",
            };
            self.rejection(rejection.reason, message)
                .with_fact("path", task.data_file_path())
                .with_fact("column", rejection.field)
        })?;
        Ok(file_with_row_count(object, task.record_count(), table_schema))
    }
}

fn file_with_row_count(object: ObjectMeta, record_count: Option<u64>, table_schema: &SchemaRef) -> PartitionedFile {
    let num_rows = record_count.and_then(|rows| usize::try_from(rows).ok()).map_or(Precision::Absent, Precision::Exact);
    let statistics = Statistics { num_rows, ..Statistics::new_unknown(table_schema) };
    PartitionedFile::new_from_meta(object).with_statistics(Arc::new(statistics))
}

/// The object store URL and in-store path of a data file location.
///
/// Iceberg's FileIO treats `s3a://` and `s3n://` as `s3://`; the native
/// reader accepts only `s3://`, so the URL is rewritten. `file://` serves
/// local warehouses.
fn split_location(location: &str) -> Option<(ObjectStoreUrl, ObjectPath)> {
    let (scheme, rest) = location.split_once("://")?;
    let (url, key) = match scheme {
        "s3" | "s3a" | "s3n" => {
            let (bucket, key) = rest.split_once('/')?;
            if bucket.is_empty() {
                return None;
            }
            (ObjectStoreUrl::parse(format!("s3://{bucket}")).ok()?, key)
        },
        "file" => (ObjectStoreUrl::local_filesystem(), rest),
        _ => return None,
    };
    let path = ObjectPath::parse(key.trim_start_matches('/')).ok()?;
    Some((url, path))
}

#[async_trait]
impl TableProvider for IcebergTableProvider {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(&self, filters: &[&Expr]) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        self.scan_plan(state, projection, filters, limit).await
    }
}

#[cfg(test)]
mod tests;
