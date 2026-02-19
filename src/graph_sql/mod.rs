//! SQL-facing cuGraph table-function contracts.

mod exec;
mod parser;
mod table_function;
mod types;

use datafusion::common::DataFusionError;
pub use exec::CugraphAlgoExec;
pub(crate) use exec::gpu_resident_producer;
pub use parser::parse_cugraph_algorithm_spec;
pub use table_function::{CugraphAlgoTableProvider, cugraph_table_functions, register_cugraph_udtfs};
pub use types::{
    CUGRAPH_BFS_FUNCTION, CUGRAPH_LOUVAIN_FUNCTION, CUGRAPH_PAGERANK_FUNCTION, CugraphAlgorithm,
    CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsParams, CugraphGraphConstructionPolicy,
    CugraphGraphOptions, CugraphLouvainParams, CugraphPageRankParams, CugraphSqlConfig, CugraphTableReference,
    stable_result_schema,
};

fn plan_error(message: impl Into<String>) -> DataFusionError {
    DataFusionError::Plan(message.into())
}
