use cudf_query_engine::{Expr as NativeExpr, SortExpr as NativeSortExpr, SortSpec as NativeSortSpec};
use datafusion::{
    physical_expr::{PhysicalSortExpr, expressions::Column},
    physical_plan::sorts::sort::SortExec,
};

use super::LoweredNativeInput;
use crate::policy::FallbackReason;

pub(super) fn lower_sort_spec_with_input(
    sort: &SortExec,
    input: &LoweredNativeInput,
) -> Result<NativeSortSpec, FallbackReason> {
    let expressions = sort
        .expr()
        .iter()
        .map(|sort_expr| lower_sort_expr_with_input(sort_expr, input))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(NativeSortSpec::new(expressions))
}

fn lower_sort_expr_with_input(
    sort_expr: &PhysicalSortExpr,
    input: &LoweredNativeInput,
) -> Result<NativeSortExpr, FallbackReason> {
    let Some(column) = sort_expr.expr.as_any().downcast_ref::<Column>() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    let binding = input.binding_for_column(column)?;
    Ok(NativeSortExpr::new(
        NativeExpr::column(binding.native_name.as_str()),
        sort_expr.options.descending,
        sort_expr.options.nulls_first,
    ))
}
