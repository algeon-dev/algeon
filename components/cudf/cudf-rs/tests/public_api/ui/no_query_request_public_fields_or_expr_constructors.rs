use cudf::{
    Expr,
    ops::expr::col,
    query::{
        NamedExpr, QueryAggregateExpr, QueryAggregateFunction, QueryAggregateSpec, QueryBinaryOp, QueryLiteral,
        QuerySortExpr, QuerySortSpec,
    },
};

fn main() {
    let named = NamedExpr::new(col("x"), "x");
    let _ = named.name;
    let _manual_named = NamedExpr { expr: col("x"), name: "x".to_string() };

    let aggregate = QueryAggregateExpr::new(QueryAggregateFunction::Sum, col("x"), "sum_x");
    let _ = aggregate.output_name;
    let _manual_aggregate = QueryAggregateExpr {
        function: QueryAggregateFunction::Max,
        input: col("x"),
        output_name: "max_x".to_string(),
    };

    let aggregate_spec = QueryAggregateSpec::new([named], [aggregate]);
    let _ = aggregate_spec.aggr_exprs;
    let _manual_aggregate_spec = QueryAggregateSpec { group_exprs: Vec::new(), aggr_exprs: Vec::new() };

    let sort_key = QuerySortExpr::ascending(col("x"));
    let _ = sort_key.descending;
    let _manual_sort_key = QuerySortExpr { expr: col("x"), descending: false, nulls_first: false };

    let sort = QuerySortSpec::new([sort_key]);
    let _ = sort.stable;
    let _manual_sort = QuerySortSpec { expressions: Vec::new(), stable: false };

    let _ = Expr::column("x");
    let _ = Expr::typed_literal(QueryLiteral::Int32(1));
    let _ = Expr::binary(col("x"), QueryBinaryOp::Add, QueryLiteral::Int32(1));
}
