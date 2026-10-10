//! Rejection of table functions that carry a GPU provider prefix but name no
//! installed GPU function.

use std::sync::Arc;

use datafusion::{
    common::Result as DataFusionResult,
    logical_expr::planner::{RelationPlanner, RelationPlannerContext, RelationPlanning},
    sql::sqlparser::ast::TableFactor,
};

use super::{
    GPU_DESCRIBE_FUNCTION_FUNCTION, GPU_LIST_FUNCTIONS_FUNCTION, GPU_VALIDATE_CALL_FUNCTION, GpuFunctionCatalog,
    dense_relation::table_function_name,
};
use crate::{
    error::{Error, ErrorCode},
    name_suggestion::closest_name,
};

/// Table-function name prefixes reserved for Algeon GPU functions. A family
/// that is not installed keeps its prefix reserved, so its names are reported
/// as unknown rather than reaching DataFusion's argument parser.
const GPU_FUNCTION_PREFIXES: [&str; 4] = ["cugraph_", "cuml_", "cuvs_", "gpu_"];

const METADATA_FUNCTIONS: [&str; 3] =
    [GPU_LIST_FUNCTIONS_FUNCTION, GPU_DESCRIBE_FUNCTION_FUNCTION, GPU_VALIDATE_CALL_FUNCTION];

/// The last relation planner of a session with GPU families: it rejects a
/// prefixed call that no family planner recognized.
///
/// Without it, DataFusion's default table-function path parses the arguments
/// before it looks up the name and reports `name => value` as an unsupported
/// argument type.
#[derive(Debug)]
pub(crate) struct UnknownGpuFunctionPlanner {
    catalog: Arc<GpuFunctionCatalog>,
}

impl UnknownGpuFunctionPlanner {
    pub(crate) fn new(catalog: Arc<GpuFunctionCatalog>) -> Self {
        Self { catalog }
    }
}

impl RelationPlanner for UnknownGpuFunctionPlanner {
    fn plan_relation(
        &self,
        relation: TableFactor,
        context: &mut dyn RelationPlannerContext,
    ) -> DataFusionResult<RelationPlanning> {
        match table_function_name(&relation, context) {
            Some(name)
                if GPU_FUNCTION_PREFIXES.iter().any(|prefix| name.starts_with(prefix))
                    && self.catalog.get(&name).is_none()
                    && !METADATA_FUNCTIONS.contains(&name.as_str()) =>
            {
                Err(unknown_gpu_function_error(&self.catalog, &name).into())
            },
            _ => Ok(RelationPlanning::Original(Box::new(relation))),
        }
    }
}

pub(crate) fn unknown_gpu_function_error(catalog: &GpuFunctionCatalog, name: &str) -> Error {
    Error::new(
        ErrorCode::ObjectNotFound,
        "algeon_datafusion.gpu_functions.unknown_function",
        unknown_gpu_function_message(catalog, name),
    )
    .with_log_safe_fact("reason", "unknown_gpu_function")
}

pub(crate) fn unknown_gpu_function_message(catalog: &GpuFunctionCatalog, name: &str) -> String {
    let prefix = GPU_FUNCTION_PREFIXES.iter().copied().find(|prefix| name.starts_with(prefix)).unwrap_or_default();
    let registered = catalog
        .entries()
        .iter()
        .map(|entry| entry.descriptor.function_name.as_str())
        .chain(METADATA_FUNCTIONS)
        .collect::<Vec<_>>();
    let family_installed = registered.iter().any(|candidate| candidate.starts_with(prefix));
    match closest_name(name, prefix, registered.iter().copied()) {
        Some(closest) => format!(
            "unknown GPU function {name}; closest registered name: {closest}; {GPU_LIST_FUNCTIONS_FUNCTION}() lists every installed function"
        ),
        None if !family_installed => format!(
            "unknown GPU function {name}; no {prefix}* functions are installed in this session; {GPU_LIST_FUNCTIONS_FUNCTION}() lists every installed function"
        ),
        None => format!("unknown GPU function {name}; {GPU_LIST_FUNCTIONS_FUNCTION}() lists every installed function"),
    }
}
