const EXPR_RS: &str = include_str!("../../../src/ops/expr.rs");
const OPS_MOD_RS: &str = include_str!("../../../src/ops/mod.rs");

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

#[test]
fn expr_module_docs_center_dsl_entry_points() {
    let expr_rs = normalized(EXPR_RS);

    assert!(expr_rs.contains("DSL entry points"));
    assert!(expr_rs.contains("conditional expression chain"));
    assert!(!EXPR_RS.contains("Free functions: col(), lit(), when()"));
}

#[test]
fn expr_method_family_docs_avoid_namespace_jargon() {
    let expr_rs = normalized(EXPR_RS);

    assert!(expr_rs.contains("aggregation expression methods"));
    assert!(expr_rs.contains("string expression methods"));
    assert!(!EXPR_RS.contains("Enter the aggregation namespace"));
    assert!(!EXPR_RS.contains("Enter the string namespace"));
}

#[test]
fn when_chain_docs_describe_staged_construction() {
    let expr_rs = normalized(EXPR_RS);

    assert!(expr_rs.contains("First stage of a `when(...).then(...).otherwise(...)` chain."));
    assert!(expr_rs.contains("Second stage of a `when(...).then(...).otherwise(...)` chain."));
    assert!(!EXPR_RS.contains("Intermediate builder: holds the condition"));
}

#[test]
fn ops_module_docs_align_with_expression_dsl_surface() {
    let ops_mod_rs = normalized(OPS_MOD_RS);

    assert!(ops_mod_rs.contains("expression DSL surface"));
    assert!(ops_mod_rs.contains("broadcast aggregate expressions for row-aligned use"));
    assert!(ops_mod_rs.contains("string-specific expression methods"));
}
