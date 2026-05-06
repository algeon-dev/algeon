use cudf_query_engine::{Expr as NativeExpr, SortExpr as NativeSortExpr, SortSpec as NativeSortSpec};
use datafusion::{
    physical_expr::{PhysicalSortExpr, expressions::Column},
    physical_plan::sorts::sort::SortExec,
};

use crate::policy::FallbackReason;

pub(super) fn lower_sort_spec(sort: &SortExec) -> Result<NativeSortSpec, FallbackReason> {
    let expressions = sort.expr().iter().map(lower_sort_expr).collect::<Result<Vec<_>, _>>()?;
    Ok(NativeSortSpec::new(expressions))
}

pub(super) fn lower_sort_expr(sort_expr: &PhysicalSortExpr) -> Result<NativeSortExpr, FallbackReason> {
    let Some(column) = sort_expr.expr.as_any().downcast_ref::<Column>() else {
        return Err(FallbackReason::UnsupportedExpression);
    };
    Ok(NativeSortExpr::new(
        NativeExpr::column(column.name()),
        sort_expr.options.descending,
        sort_expr.options.nulls_first,
    ))
}
