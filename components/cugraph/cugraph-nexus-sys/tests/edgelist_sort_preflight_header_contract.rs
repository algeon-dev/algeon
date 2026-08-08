mod support;

use support::{include_dir, manifest_dir, strip_c_comments};

const SYMBOL: &str = "cugraph_edgelist_sort_workspace_preflight";

fn normalized_declaration(source: &str) -> String {
    let source = strip_c_comments(source);
    let symbol_offset = source.find(SYMBOL).unwrap_or_else(|| panic!("missing {SYMBOL}"));
    let declaration_start = source[..symbol_offset]
        .rfind("CUGRAPH_EXPORT")
        .unwrap_or_else(|| panic!("missing CUGRAPH_EXPORT declaration for {SYMBOL}"));
    let declaration_end = source[symbol_offset..]
        .find(';')
        .map(|offset| symbol_offset + offset + 1)
        .unwrap_or_else(|| panic!("missing declaration terminator for {SYMBOL}"));

    source[declaration_start..declaration_end].chars().filter(|character| !character.is_whitespace()).collect()
}

#[test]
fn wrapper_preflight_declaration_matches_cugraph_graph_header() {
    let wrapper = std::fs::read_to_string(manifest_dir().join("wrapper.h")).expect("read wrapper.h");
    let graph_header = std::fs::read_to_string(include_dir().join("cugraph_c/graph.h")).expect("read graph.h");

    assert_eq!(
        normalized_declaration(&wrapper),
        normalized_declaration(&graph_header),
        "wrapper declaration drift detected for {SYMBOL}"
    );
}
