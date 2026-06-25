mod descriptions;
mod json_schema;
mod provider;
mod validate;

pub(crate) use provider::cugraph_metadata_table_functions;

use super::{CUGRAPH_OPERATION_REGISTRY, CugraphAlgorithm};

const ERROR_UNKNOWN_ALGORITHM: &str = "unknown_algorithm";
const ERROR_ALGORITHM_DISABLED: &str = "algorithm_disabled";
const ERROR_INVALID_CALL_JSON: &str = "invalid_call_json";
const ERROR_INVALID_ARGUMENT: &str = "invalid_argument";
const ERROR_TABLE_NOT_FOUND: &str = "table_not_found";
const ERROR_COLUMN_NOT_FOUND: &str = "column_not_found";
const ERROR_UNSUPPORTED_VERTEX_DTYPE: &str = "unsupported_vertex_dtype";
const ERROR_VERTEX_DTYPE_MISMATCH: &str = "vertex_dtype_mismatch";
const ERROR_UNSUPPORTED_WEIGHT_DTYPE: &str = "unsupported_weight_dtype";
const ERROR_UNSUPPORTED_OPTION: &str = "unsupported_option";

pub(super) fn algorithm_for_metadata_name(name: &str) -> Option<CugraphAlgorithm> {
    CUGRAPH_OPERATION_REGISTRY
        .iter()
        .find(|definition| {
            definition.sql_function_name == name || definition.native_algorithm.definition().name == name
        })
        .map(|definition| definition.algorithm)
}
