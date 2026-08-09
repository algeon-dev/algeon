use super::*;
use crate::projection_outputs;

mod fixture_local_parquet_source {
    use super::*;

    pub(super) fn new(
        path: impl Into<PathBuf>,
        schema: Arc<Schema>,
        read_columns: Vec<String>,
    ) -> NativeLocalParquetSource {
        let path = path.into();
        if path.is_file() {
            NativeLocalParquetSource::from_files(vec![fixture_local_parquet_file(&path)], schema, read_columns)
        } else {
            NativeLocalParquetSource::new(path, schema, read_columns)
        }
    }

    pub(super) fn from_files(
        files: Vec<LocalParquetFile>,
        schema: Arc<Schema>,
        read_columns: Vec<String>,
    ) -> NativeLocalParquetSource {
        NativeLocalParquetSource::from_files(
            files.into_iter().map(enrich_fixture_local_parquet_file).collect(),
            schema,
            read_columns,
        )
    }
}

use fixture_local_parquet_source as LocalParquetSource;

// Plan fixtures are grouped by admitted shape and included at this module
// level so existing `use fixtures_plans::*` consumers keep working.
include!("fixtures_plans/join_roots.rs");
include!("fixtures_plans/root_full_sort.rs");
include!("fixtures_plans/rollup_rank_topk.rs");
include!("fixtures_plans/pipelines.rs");
include!("fixtures_plans/root_topk.rs");
include!("fixtures_plans/projection_global_limit.rs");
include!("fixtures_plans/projection_scalar_setop.rs");
