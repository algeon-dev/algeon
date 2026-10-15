use std::time::Instant;

use arrow::datatypes::Schema;
use cudf::{Exec, ExecutionBudgetToken, Series, interop::arrow_host::cudf_dtype_from_arrow_type};
use query_planning::{
    dynamic_filter::{DynamicFilter, dynamic_filter_predicate_for_schema},
    error::{QueryError, QueryErrorCode, Result},
    plan::{FilterNode, NodeId},
};
use query_runtime::runtime::{ExecutionContext, resident::GpuDataFrame};

use super::{attach_budget_token_failure_facts, execute_filter_with_budget};
use crate::exec::{ExecutionMetrics, cudf_query_error};

/// A route bound once at its consumer boundary. Series clones retain the same
/// query-domain allocation; cuDF orders each consumer stream against its producer.
#[derive(Clone)]
pub(in crate::exec) enum PreparedDynamicFilter {
    Expression(FilterNode),
    Membership { column: String, values: Series },
}

impl PreparedDynamicFilter {
    pub(in crate::exec) fn bind(
        schema: &Schema,
        column: &str,
        input: NodeId,
        filter: &DynamicFilter,
        values: Option<&Series>,
    ) -> Result<Option<Self>> {
        let Some(values) = values else {
            if matches!(filter, DynamicFilter::ResidentKeysetI64 { .. }) {
                return Err(QueryError::new(
                    QueryErrorCode::InvalidPlan,
                    "query_engine.dynamic_filter.consumer_keyset",
                    "resident dynamic-filter payload is missing its retained keys",
                ));
            }
            return Ok(dynamic_filter_predicate_for_schema(schema, column, filter)
                .map(|predicate| Self::Expression(FilterNode { input, predicate })));
        };
        let Some((min, max)) = filter.i64_range() else {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                "query_engine.dynamic_filter.consumer_keyset",
                "resident dynamic-filter keys have no integer bounds",
            ));
        };
        if dynamic_filter_predicate_for_schema(schema, column, &DynamicFilter::RangeI64 { min, max }).is_none() {
            return Ok(None);
        }
        let data_type = schema.field_with_name(column).expect("validated consumer column").data_type();
        let dtype = cudf_dtype_from_arrow_type(data_type).map_err(|error| {
            cudf_query_error("query_engine.dynamic_filter.consumer_keyset", "invalid consumer key type", error)
        })?;
        let values = if values.dtype().map_err(|error| {
            cudf_query_error("query_engine.dynamic_filter.consumer_keyset", "failed to inspect key type", error)
        })? == dtype
        {
            values.clone()
        } else {
            values.on(Exec::RETAINED).astype(dtype).map_err(|error| {
                cudf_query_error("query_engine.dynamic_filter.consumer_keyset", "failed to cast resident keys", error)
            })?
        };
        Ok(Some(Self::Membership { column: column.to_string(), values }))
    }

    pub(in crate::exec) fn execute(
        &self,
        context: &ExecutionContext,
        input: GpuDataFrame,
        token: &ExecutionBudgetToken,
        metrics: &ExecutionMetrics,
    ) -> Result<GpuDataFrame> {
        match self {
            Self::Expression(node) => execute_filter_with_budget(context, node, input, token, metrics),
            Self::Membership { column, values } => {
                let timer = Instant::now();
                let output = input.frame().on(token.exec()).filter_isin(column, values).map_err(|error| {
                    attach_budget_token_failure_facts(
                        cudf_query_error(
                            "query_engine.filter.execute",
                            "failed to execute resident dynamic-filter membership",
                            error,
                        ),
                        token,
                    )
                })?;
                metrics.add_operator_time(timer.elapsed());
                GpuDataFrame::try_new_with_source_residency(output, &input)
            },
        }
    }
}
