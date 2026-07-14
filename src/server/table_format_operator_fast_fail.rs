//! Logical-plan fast-fail for table-format operator semantics that native
//! execution cannot model yet.

#[cfg(feature = "iceberg")]
use datafusion::datasource::DefaultTableSource;
use datafusion::{
    common::{Column, DFSchemaRef, TableReference},
    logical_expr::{Distinct, Expr, ExprSchemable, LogicalPlan},
};

use crate::{
    error::{Error, ErrorCode},
    policy::FallbackReason,
};

const TABLE_FORMAT_OPERATOR_FAST_FAIL_OPERATION: &str =
    "datafusion_nexus.server.flight.fast_fail_table_format_operator";

#[derive(Debug, Clone, PartialEq, Eq)]
struct UnsupportedTableFormatOperatorColumn {
    operator: &'static str,
    column: String,
    data_type: String,
    role: &'static str,
}

pub(super) fn fast_fail_unsupported_table_format_operator_plan(plan: &LogicalPlan) -> Option<Error> {
    unsupported_table_format_operator_column(plan).map(|column| {
        Error::new(
            ErrorCode::NativeCandidateRejected,
            TABLE_FORMAT_OPERATOR_FAST_FAIL_OPERATION,
            format!(
                "table-format query requires unsupported native operator type; operator={} column={} type={}",
                column.operator, column.column, column.data_type
            ),
        )
        .with_fact("fallback_reason", FallbackReason::UnsupportedType.code())
        .with_fact("fast_fail_stage", "logical_plan")
        .with_fact("scan_planning_skipped", true)
        .with_fact("table_format", "iceberg")
        .with_fact("unsupported_operator", column.operator)
        .with_fact("unsupported_column", column.column)
        .with_fact("unsupported_column_type", column.data_type)
        .with_fact("unsupported_column_role", column.role)
    })
}

fn unsupported_table_format_operator_column(plan: &LogicalPlan) -> Option<UnsupportedTableFormatOperatorColumn> {
    let local = match plan {
        LogicalPlan::Distinct(distinct) => unsupported_distinct_column(distinct),
        LogicalPlan::Aggregate(aggregate) => unsupported_aggregate_column(aggregate),
        _ => None,
    };
    local.or_else(|| plan.inputs().into_iter().find_map(unsupported_table_format_operator_column))
}

fn unsupported_distinct_column(distinct: &Distinct) -> Option<UnsupportedTableFormatOperatorColumn> {
    match distinct {
        Distinct::All(input) => input.schema().fields().iter().enumerate().find_map(|(index, field)| {
            (native_table_format_operator_type_is_unsupported(field.data_type())
                && logical_plan_output_field_is_from_gpu_only_table_format_scan(input, index))
            .then(|| unsupported_operator_column("distinct", field.name(), field.data_type(), "operator_key"))
        }),
        Distinct::On(distinct_on) => distinct_on.on_expr.iter().find_map(|expr| {
            unsupported_logical_expr_operator_column(
                "distinct",
                expr,
                &distinct_on.input,
                distinct_on.input.schema(),
                "operator_key",
            )
        }),
    }
}

fn unsupported_aggregate_column(
    aggregate: &datafusion::logical_expr::Aggregate,
) -> Option<UnsupportedTableFormatOperatorColumn> {
    aggregate
        .group_expr
        .iter()
        .find_map(|expr| {
            unsupported_logical_expr_operator_column(
                "group_by",
                expr,
                &aggregate.input,
                aggregate.input.schema(),
                "operator_key",
            )
        })
        .or_else(|| {
            aggregate.aggr_expr.iter().find_map(|expr| match expr {
                Expr::AggregateFunction(function) if function.params.distinct => {
                    function.params.args.iter().find_map(|arg| {
                        unsupported_logical_expr_operator_column(
                            "aggregate_distinct",
                            arg,
                            &aggregate.input,
                            aggregate.input.schema(),
                            "operator_value",
                        )
                    })
                },
                _ => None,
            })
        })
}

fn unsupported_logical_expr_operator_column(
    operator: &'static str,
    expr: &Expr,
    input_plan: &LogicalPlan,
    input_schema: &DFSchemaRef,
    role: &'static str,
) -> Option<UnsupportedTableFormatOperatorColumn> {
    let data_type = expr.get_type(input_schema).ok()?;
    (native_table_format_operator_type_is_unsupported(&data_type)
        && logical_expr_is_from_gpu_only_table_format_scan(expr, input_plan))
    .then(|| unsupported_operator_column(operator, logical_expr_column_name(expr), &data_type, role))
}

fn unsupported_operator_column(
    operator: &'static str,
    column: impl Into<String>,
    data_type: &arrow::datatypes::DataType,
    role: &'static str,
) -> UnsupportedTableFormatOperatorColumn {
    UnsupportedTableFormatOperatorColumn {
        operator,
        column: column.into(),
        data_type: nexus_query_engine::datatypes::diagnostic_arrow_type(data_type),
        role,
    }
}

fn logical_expr_column_name(expr: &Expr) -> String {
    match expr {
        Expr::Column(column) => column.name.clone(),
        _ => expr.schema_name().to_string(),
    }
}

fn native_table_format_operator_type_is_unsupported(data_type: &arrow::datatypes::DataType) -> bool {
    !nexus_query_engine::datatypes::is_supported_arrow_type(data_type)
        || nexus_query_engine::datatypes::is_supported_list_carry_type(data_type)
        || nexus_query_engine::datatypes::is_timestamp_with_timezone(data_type)
}

fn logical_expr_is_from_gpu_only_table_format_scan(expr: &Expr, input_plan: &LogicalPlan) -> bool {
    let columns = expr.column_refs();
    !columns.is_empty()
        && columns.iter().all(|column| logical_plan_column_ref_is_from_gpu_only_table_format_scan(input_plan, column))
}

fn logical_plan_column_ref_is_from_gpu_only_table_format_scan(plan: &LogicalPlan, column: &Column) -> bool {
    plan.schema()
        .maybe_index_of_column(column)
        .is_some_and(|index| logical_plan_output_field_is_from_gpu_only_table_format_scan(plan, index))
}

fn logical_plan_output_field_is_from_gpu_only_table_format_scan(plan: &LogicalPlan, output_index: usize) -> bool {
    match plan {
        LogicalPlan::TableScan(scan) => table_scan_uses_gpu_only_table_format_provider(&scan.source),
        LogicalPlan::Projection(projection) => projection
            .expr
            .get(output_index)
            .is_some_and(|expr| logical_expr_is_from_gpu_only_table_format_scan(expr, &projection.input)),
        LogicalPlan::SubqueryAlias(alias) => {
            output_field_matches_input_field(plan, &alias.input, output_index, output_index)
                && logical_plan_output_field_is_from_gpu_only_table_format_scan(&alias.input, output_index)
        },
        _ => {
            let inputs = plan.inputs();
            if inputs.is_empty() {
                return false;
            }
            let Some((output_qualifier, output_field)) = plan.schema().iter().nth(output_index) else {
                return false;
            };
            let mut matches = inputs.iter().filter_map(|input| {
                unique_matching_input_field_index(input, output_qualifier, output_field)
                    .map(|input_index| logical_plan_output_field_is_from_gpu_only_table_format_scan(input, input_index))
            });
            let Some(first_match) = matches.next() else {
                return false;
            };
            matches.next().is_none() && first_match
        },
    }
}

fn unique_matching_input_field_index(
    input: &LogicalPlan,
    output_qualifier: Option<&TableReference>,
    output_field: &arrow::datatypes::Field,
) -> Option<usize> {
    let mut matches = input.schema().iter().enumerate().filter_map(|(index, (input_qualifier, input_field))| {
        qualified_fields_match(output_qualifier, output_field, input_qualifier, input_field).then_some(index)
    });
    let first_match = matches.next()?;
    matches.next().is_none().then_some(first_match)
}

fn output_field_matches_input_field(
    plan: &LogicalPlan,
    input: &LogicalPlan,
    output_index: usize,
    input_index: usize,
) -> bool {
    let Some(output_field) = plan.schema().fields().get(output_index) else {
        return false;
    };
    let Some(input_field) = input.schema().fields().get(input_index) else {
        return false;
    };
    output_field.name() == input_field.name() && output_field.data_type() == input_field.data_type()
}

fn qualified_fields_match(
    output_qualifier: Option<&TableReference>,
    output_field: &arrow::datatypes::Field,
    input_qualifier: Option<&TableReference>,
    input_field: &arrow::datatypes::Field,
) -> bool {
    output_field.name() == input_field.name()
        && output_field.data_type() == input_field.data_type()
        && match (output_qualifier, input_qualifier) {
            (Some(output_qualifier), Some(input_qualifier)) => output_qualifier.resolved_eq(input_qualifier),
            (None, None) => true,
            _ => false,
        }
}

#[cfg(feature = "iceberg")]
fn table_scan_uses_gpu_only_table_format_provider(
    source: &std::sync::Arc<dyn datafusion::logical_expr::TableSource>,
) -> bool {
    let source = source.as_ref() as &dyn std::any::Any;
    source.downcast_ref::<DefaultTableSource>().is_some_and(|source| {
        let provider = source.table_provider.as_ref() as &dyn std::any::Any;
        provider.is::<crate::table_format::iceberg::CudfIcebergTableProvider>()
    })
}

#[cfg(not(feature = "iceberg"))]
fn table_scan_uses_gpu_only_table_format_provider(
    _source: &std::sync::Arc<dyn datafusion::logical_expr::TableSource>,
) -> bool {
    false
}
