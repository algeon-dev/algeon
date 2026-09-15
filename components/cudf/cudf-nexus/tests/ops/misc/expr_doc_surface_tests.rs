const EXPR_RS: &str = include_str!("../../../src/ops/expr.rs");
const EXPR_AGG_RS: &str = include_str!("../../../src/ops/expr/agg.rs");
const EXPR_CORE_RS: &str = include_str!("../../../src/ops/expr/core.rs");
const EXPR_STRINGS_RS: &str = include_str!("../../../src/ops/expr/strings.rs");
const EXPR_WHEN_RS: &str = include_str!("../../../src/ops/expr/when.rs");
const OPS_MOD_RS: &str = include_str!("../../../src/ops/mod.rs");

fn expr_surface_source() -> String {
    [EXPR_RS, EXPR_CORE_RS, EXPR_AGG_RS, EXPR_STRINGS_RS, EXPR_WHEN_RS].join("\n")
}

fn normalized(source: &str) -> String {
    source
        .lines()
        .map(|line| {
            line.trim_start()
                .strip_prefix("//!")
                .or_else(|| line.trim_start().strip_prefix("///"))
                .unwrap_or(line)
                .trim()
        })
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn expr_module_docs_center_dsl_entry_points() {
    let expr_source = expr_surface_source();
    let expr_rs = normalized(&expr_source);

    assert!(expr_rs.contains("DSL entry points"));
    assert!(expr_rs.contains("conditional expression chain"));
    assert!(!expr_source.contains("Free functions: col(), lit(), when()"));
}

fn expr_method_family_docs_avoid_namespace_jargon() {
    let expr_source = expr_surface_source();
    let expr_rs = normalized(&expr_source);

    assert!(expr_rs.contains("aggregation expression methods"));
    assert!(expr_rs.contains("string expression methods"));
    assert!(!expr_source.contains("Enter the aggregation namespace"));
    assert!(!expr_source.contains("Enter the string namespace"));
}

fn when_chain_docs_describe_staged_construction() {
    let expr_source = expr_surface_source();
    let expr_rs = normalized(&expr_source);

    assert!(expr_rs.contains("First stage of a `when(...).then(...).otherwise(...)` chain."));
    assert!(expr_rs.contains("Second stage of a `when(...).then(...).otherwise(...)` chain."));
    assert!(!expr_source.contains("Intermediate builder: holds the condition"));
}

fn ops_module_docs_align_with_expression_dsl_surface() {
    let ops_mod_rs = normalized(OPS_MOD_RS);

    assert!(ops_mod_rs.contains("expression DSL surface"));
    assert!(ops_mod_rs.contains("broadcast aggregate expressions for row-aligned use"));
    assert!(ops_mod_rs.contains("string-specific expression methods"));
}

#[test]
fn expr_doc_surface_tests_cases() {
    crate::common::run_named_cases(&[
        ("expr_module_docs_center_dsl_entry_points", expr_module_docs_center_dsl_entry_points),
        ("expr_method_family_docs_avoid_namespace_jargon", expr_method_family_docs_avoid_namespace_jargon),
        ("when_chain_docs_describe_staged_construction", when_chain_docs_describe_staged_construction),
        ("ops_module_docs_align_with_expression_dsl_surface", ops_module_docs_align_with_expression_dsl_surface),
    ]);
}
