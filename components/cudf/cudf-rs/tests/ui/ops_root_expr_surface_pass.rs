use cudf::ops::{Expr, col, lit, when};

fn main() {
    let base: Expr = col("value") + lit(1i32);
    let branch = when(col("value").gt(lit(0i32))).then("positive").otherwise("non-positive");

    let _ = (base.alias("bumped"), branch.alias("label"));
}
