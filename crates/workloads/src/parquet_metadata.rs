use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use arrow::datatypes::Schema;
use parquet::{
    arrow::arrow_reader::ParquetRecordBatchReaderBuilder,
    file::reader::{FileReader, SerializedFileReader},
};

use crate::{WorkloadError, WorkloadResult, metadata::DatasetFingerprint};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParquetFileMetadata {
    pub(crate) path: PathBuf,
    pub(crate) relative_path: String,
    pub(crate) schema: String,
    pub(crate) byte_size: u64,
    pub(crate) row_count: u64,
    pub(crate) row_group_count: usize,
    pub(crate) row_group_rows: Vec<u64>,
    pub(crate) row_group_bytes: Vec<u64>,
}

pub(crate) fn read_file_metadata(root: &Path, path: &Path) -> WorkloadResult<ParquetFileMetadata> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(WorkloadError::InvalidDataset(format!("expected {} to be a Parquet file", path.display())));
    }

    let byte_size = metadata.len();
    let reader_builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let schema = canonical_schema(reader_builder.schema().as_ref());
    let file = File::open(path)?;
    let reader = SerializedFileReader::new(file)?;
    let parquet_metadata = reader.metadata();
    let row_group_count = parquet_metadata.num_row_groups();
    let mut row_group_rows = Vec::with_capacity(row_group_count);
    let mut row_group_bytes = Vec::with_capacity(row_group_count);
    for row_group in parquet_metadata.row_groups() {
        let rows = u64::try_from(row_group.num_rows()).map_err(|_| {
            WorkloadError::InvalidDataset(format!("negative row count in Parquet metadata for {}", path.display()))
        })?;
        let bytes = u64::try_from(row_group.compressed_size()).map_err(|_| {
            WorkloadError::InvalidDataset(format!(
                "negative compressed byte size in Parquet metadata for {}",
                path.display()
            ))
        })?;
        row_group_rows.push(rows);
        row_group_bytes.push(bytes);
    }
    let row_count = row_group_rows.iter().sum();
    Ok(ParquetFileMetadata {
        path: path.to_path_buf(),
        relative_path: relative_dataset_path(root, path),
        schema,
        byte_size,
        row_count,
        row_group_count,
        row_group_rows,
        row_group_bytes,
    })
}

pub(crate) fn collect_parquet_files(path: &Path, files: &mut Vec<PathBuf>) -> WorkloadResult<()> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_parquet_files(&entry_path, files)?;
        } else if file_type.is_file() && entry_path.extension().is_some_and(|extension| extension == "parquet") {
            files.push(entry_path);
        }
    }
    Ok(())
}

pub(crate) fn dataset_content_fingerprint<'a, Tables, Files>(domain_tag: &str, tables: Tables) -> WorkloadResult<String>
where
    Tables: IntoIterator<Item = (&'a str, Files)>,
    Files: IntoIterator<Item = (&'a str, u64, &'a Path)>,
{
    let mut hasher = DatasetFingerprint::new();
    hasher.update_field(domain_tag);
    let mut buffer = vec![0_u8; 1024 * 1024];
    for (table, files) in tables {
        hasher.update_field(table);
        for (relative_path, byte_size, path) in files {
            hasher.update_field(relative_path);
            hasher.update_field(&byte_size.to_string());
            let mut reader = File::open(path)?;
            loop {
                let bytes_read = reader.read(&mut buffer)?;
                if bytes_read == 0 {
                    break;
                }
                hasher.update_bytes(&buffer[..bytes_read]);
            }
            hasher.update_bytes(&[0]);
        }
    }
    Ok(format!("content-fnv1a64:{:016x}", hasher.finish()))
}

pub(crate) fn canonical_schema(schema: &Schema) -> String {
    schema
        .fields()
        .iter()
        .map(|field| {
            format!(
                "{}:{:?}:nullable={}:metadata={:?}",
                field.name(),
                field.data_type(),
                field.is_nullable(),
                field.metadata()
            )
        })
        .collect::<Vec<_>>()
        .join("|")
}

pub(crate) fn relative_dataset_path(root: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(root).expect("discovered dataset path must be under root");
    relative.iter().map(|part| part.to_string_lossy()).collect::<Vec<_>>().join("/")
}
