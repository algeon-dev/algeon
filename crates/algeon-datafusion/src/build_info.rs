//! Build identity of this library: version, commit, optional engines, and
//! the DataFusion release it links.
//!
//! [`build_info`] is the one source for `algeon_server --version`, the
//! server startup log, and the `algeon_version()` SQL table function that
//! every session installed through [`crate::backend::AlgeonGpuBackend::install_on`]
//! carries.

#[cfg(feature = "server")]
use std::any::TypeId;
use std::sync::Arc;

use arrow::{
    array::{ArrayRef, StringArray},
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
};
use async_trait::async_trait;
use datafusion::{
    catalog::{Session, TableFunction, TableFunctionImpl, TableProvider},
    common::{DataFusionError, Result as DataFusionResult},
    datasource::{MemTable, TableType},
    logical_expr::Expr,
    physical_plan::ExecutionPlan,
};
#[cfg(feature = "server")]
use datafusion::{datasource::source_as_provider, logical_expr::TableSource};

use crate::{physical_plan::cpu_physical_boundary, planner::AlgeonGpuExecutionMode};

/// Name of the table function that returns [`build_info`] as one row.
pub const ALGEON_VERSION_FUNCTION: &str = "algeon_version";

/// Version, commit, and optional engines of this `algeon-datafusion` build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct BuildInfo {
    /// The crate version, `CARGO_PKG_VERSION`.
    pub version: &'static str,
    /// Abbreviated commit the crate was built from: `ALGEON_GIT_COMMIT` at
    /// build time, else `git rev-parse --short=12 HEAD`, else `None` when the
    /// source tree is not a git checkout.
    pub git_commit: Option<&'static str>,
    /// Enabled optional engine features, sorted: a subset of `cugraph`,
    /// `cuml`, `cuvs` and `nvml`.
    pub features: &'static [&'static str],
    /// The linked DataFusion release.
    pub datafusion_version: &'static str,
}

const ENABLED_FEATURES: &[&str] = &[
    #[cfg(feature = "cugraph")]
    "cugraph",
    #[cfg(feature = "cuml")]
    "cuml",
    #[cfg(feature = "cuvs")]
    "cuvs",
    #[cfg(feature = "nvml")]
    "nvml",
];

/// Build identity of this library.
pub const fn build_info() -> BuildInfo {
    BuildInfo {
        version: env!("CARGO_PKG_VERSION"),
        git_commit: option_env!("ALGEON_BUILD_GIT_COMMIT"),
        features: ENABLED_FEATURES,
        datafusion_version: datafusion::DATAFUSION_VERSION,
    }
}

pub(crate) fn algeon_version_table_function(execution_mode: AlgeonGpuExecutionMode) -> Arc<TableFunction> {
    Arc::new(TableFunction::new(
        ALGEON_VERSION_FUNCTION.to_string(),
        Arc::new(AlgeonVersionFunction { execution_mode }),
    ))
}

#[derive(Debug)]
struct AlgeonVersionFunction {
    execution_mode: AlgeonGpuExecutionMode,
}

impl TableFunctionImpl for AlgeonVersionFunction {
    fn call(&self, args: &[Expr]) -> DataFusionResult<Arc<dyn TableProvider>> {
        if !args.is_empty() {
            return Err(DataFusionError::Plan(format!("{ALGEON_VERSION_FUNCTION} does not accept arguments")));
        }
        Ok(Arc::new(AlgeonVersionProvider { execution_mode: self.execution_mode }))
    }
}

#[derive(Debug)]
struct AlgeonVersionProvider {
    execution_mode: AlgeonGpuExecutionMode,
}

#[async_trait]
impl TableProvider for AlgeonVersionProvider {
    fn schema(&self) -> SchemaRef {
        version_schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let info = build_info();
        let column = |value: Option<&str>| Arc::new(StringArray::from(vec![value])) as ArrayRef;
        let batch = RecordBatch::try_new(
            version_schema(),
            vec![
                column(Some(info.version)),
                column(info.git_commit),
                column(Some(&info.features.join(","))),
                column(Some(info.datafusion_version)),
                column(Some(self.execution_mode.as_str())),
            ],
        )?;
        let input =
            MemTable::try_new(version_schema(), vec![vec![batch]])?.scan(state, projection, filters, limit).await?;
        Ok(cpu_physical_boundary(input))
    }
}

/// Whether `source` is the `algeon_version()` provider, which reads no table
/// and runs no native work; the Flight SQL server plans such statements as
/// metadata.
#[cfg(feature = "server")]
pub(crate) fn is_algeon_version_table_source(source: &Arc<dyn TableSource>) -> bool {
    source_as_provider(source)
        .is_ok_and(|provider| provider.as_ref().type_id() == TypeId::of::<AlgeonVersionProvider>())
}

fn version_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("version", DataType::Utf8, false),
        Field::new("git_commit", DataType::Utf8, true),
        Field::new("features", DataType::Utf8, false),
        Field::new("datafusion_version", DataType::Utf8, false),
        Field::new("execution_mode", DataType::Utf8, false),
    ]))
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, AsArray};
    use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};

    use super::*;
    use crate::{backend::install_backend_optimizer_config, planner::AlgeonNativeOptimizerConfig};

    #[tokio::test]
    async fn backend_sessions_report_the_build_and_the_backend_execution_mode() {
        let optimizer_config =
            AlgeonNativeOptimizerConfig::default().with_execution_mode(AlgeonGpuExecutionMode::FunctionsOnly);
        let state =
            install_backend_optimizer_config(SessionStateBuilder::new_with_default_features(), &optimizer_config)
                .expect("install the backend session integration")
                .build();
        let ctx = SessionContext::new_with_state(state);

        let batches = ctx.sql("SELECT * FROM algeon_version()").await.unwrap().collect().await.unwrap();

        assert_eq!(batches.len(), 1);
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 1);
        let cell = |name: &str| {
            let column = batch.column_by_name(name).unwrap_or_else(|| panic!("missing column {name}"));
            let column = column.as_string::<i32>();
            column.is_valid(0).then(|| column.value(0).to_string())
        };
        let info = build_info();
        assert_eq!(cell("version").as_deref(), Some(env!("CARGO_PKG_VERSION")));
        assert_eq!(cell("git_commit").as_deref(), info.git_commit);
        assert_eq!(cell("features"), Some(info.features.join(",")));
        assert_eq!(cell("datafusion_version").as_deref(), Some(datafusion::DATAFUSION_VERSION));
        assert_eq!(cell("execution_mode").as_deref(), Some("functions_only"));
    }
}
