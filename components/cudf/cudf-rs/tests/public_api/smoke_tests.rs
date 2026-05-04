#[path = "../core/debug_tests.rs"]
mod core_debug_tests;
#[path = "../frame/dataframe/debug_tests.rs"]
mod dataframe_debug_tests;
#[path = "../frame/dataframe/display_tests.rs"]
mod dataframe_display_tests;
#[path = "../frame/contracts/high_level_reexport_tests.rs"]
mod high_level_reexport_tests;
#[path = "../runtime/debug_tests.rs"]
mod runtime_debug_tests;
#[path = "../frame/series/display_tests.rs"]
mod series_display_tests;

#[test]
fn join_indices_and_hash_join_slice6_surface_is_publicly_importable() {
    use cudf::ops::join::{
        HashJoin, HashJoinOptions, JoinIndices, NullEquality, full_join_indices, full_join_indices_in,
        inner_join_indices, inner_join_indices_in, left_join_indices, left_join_indices_in,
    };

    let _options = HashJoinOptions::new().null_equality(NullEquality::Equal);
    let _: Option<HashJoin> = None;
    let _: Option<JoinIndices> = None;
    let _ = (
        HashJoin::new,
        HashJoin::new_in,
        HashJoin::inner_join_indices,
        HashJoin::inner_join_indices_in,
        HashJoin::left_join_indices,
        HashJoin::left_join_indices_in,
        HashJoin::full_join_indices,
        HashJoin::full_join_indices_in,
    );
    let _ = (
        inner_join_indices,
        inner_join_indices_in,
        left_join_indices,
        left_join_indices_in,
        full_join_indices,
        full_join_indices_in,
    );
}

#[test]
fn slice7_join_and_copying_surface_is_publicly_importable() {
    use cudf::ops::{
        copying::{scatter, scatter_in, scatter_unchecked, scatter_unchecked_in},
        join::{
            SingleSideJoinIndices, cross_join, cross_join_in, left_anti_join_indices, left_anti_join_indices_in,
            left_semi_join_indices, left_semi_join_indices_in,
        },
    };

    let _: Option<SingleSideJoinIndices> = None;
    let _ = (
        left_semi_join_indices,
        left_semi_join_indices_in,
        left_anti_join_indices,
        left_anti_join_indices_in,
        cross_join,
        cross_join_in,
        scatter,
        scatter_in,
        scatter_unchecked,
        scatter_unchecked_in,
    );
}

#[test]
fn slice8_ast_conditional_and_mixed_join_surface_is_publicly_importable() {
    use cudf::ops::{
        ast::{AstExpression, AstOperator},
        join::{
            conditional_full_join_indices, conditional_full_join_indices_in, conditional_inner_join_indices,
            conditional_inner_join_indices_in, conditional_left_anti_join_indices,
            conditional_left_anti_join_indices_in, conditional_left_join_indices, conditional_left_join_indices_in,
            conditional_left_semi_join_indices, conditional_left_semi_join_indices_in, filter_join_indices,
            filter_join_indices_in, mixed_full_join_indices, mixed_full_join_indices_in, mixed_inner_join_indices,
            mixed_inner_join_indices_in, mixed_left_anti_join_indices, mixed_left_anti_join_indices_in,
            mixed_left_join_indices, mixed_left_join_indices_in, mixed_left_semi_join_indices,
            mixed_left_semi_join_indices_in,
        },
    };

    let _: Option<AstExpression> = None;
    let _operator = AstOperator::Less;
    let _ = (
        conditional_inner_join_indices,
        conditional_inner_join_indices_in,
        conditional_left_join_indices,
        conditional_left_join_indices_in,
        conditional_full_join_indices,
        conditional_full_join_indices_in,
        conditional_left_semi_join_indices,
        conditional_left_semi_join_indices_in,
        conditional_left_anti_join_indices,
        conditional_left_anti_join_indices_in,
        filter_join_indices,
        filter_join_indices_in,
        mixed_inner_join_indices,
        mixed_inner_join_indices_in,
        mixed_left_join_indices,
        mixed_left_join_indices_in,
        mixed_full_join_indices,
        mixed_full_join_indices_in,
        mixed_left_semi_join_indices,
        mixed_left_semi_join_indices_in,
        mixed_left_anti_join_indices,
        mixed_left_anti_join_indices_in,
    );
}

#[test]
fn slice9_multi_file_parquet_surface_is_publicly_importable() {
    use std::path::PathBuf;

    use cudf::{DataFrame, Runtime, frame::dataframe::MultiParquetReadOptions};

    fn assert_reader(_: fn(MultiParquetReadOptions) -> cudf::Result<DataFrame>) {}

    let rt = Runtime::builder().build().unwrap();
    let paths = [PathBuf::from("part-0.parquet"), PathBuf::from("part-1.parquet")];
    let _options = MultiParquetReadOptions::builder().filepaths(&paths).columns(&["id"]).with_runtime(&rt).build();
    assert!(DataFrame::read_parquet_many(&rt, Vec::<PathBuf>::new()).is_err());

    assert_reader(DataFrame::read_parquet_many_with_options);
}

#[test]
fn query_expression_surface_is_publicly_importable() {
    use cudf::{
        Expr,
        query::{
            NamedExpr, QueryAggregateExpr, QueryAggregateFunction, QueryAggregateSpec, QueryBinaryOp, QueryLiteral,
            QuerySortExpr, QuerySortSpec,
        },
    };

    let expr = Expr::binary(Expr::column("x"), QueryBinaryOp::Add, QueryLiteral::Int32(1));
    let _named = NamedExpr::new(expr.clone(), "x_plus_one");
    let _aggregate = QueryAggregateSpec::new(
        [NamedExpr::new(Expr::column("k"), "key")],
        [QueryAggregateExpr::new(QueryAggregateFunction::Sum, expr, "total")],
    );
    let _sort = QuerySortSpec::new([QuerySortExpr::descending(Expr::column("x"))]).with_stable(true);
    let _null = Expr::typed_literal(QueryLiteral::Null(cudf::DataType::new(cudf::TypeId::Int32)));
}
