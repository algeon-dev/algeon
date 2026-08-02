use std::{collections::HashSet, fs, path::PathBuf};

fn parse_wrapper_includes(wrapper: &str) -> HashSet<String> {
    wrapper
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            line.strip_prefix("#include <cugraph_c/").and_then(|rest| rest.strip_suffix('>')).map(|s| s.to_string())
        })
        .collect()
}

#[test]
fn wrapper_includes_aggregated_algorithms_header() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let wrapper = fs::read_to_string(manifest_dir.join("wrapper.h")).expect("read wrapper.h");

    let includes = parse_wrapper_includes(&wrapper);
    assert!(
        includes.contains("algorithms.h"),
        "wrapper.h should include cugraph_c/algorithms.h to cover algorithm API additions"
    );
}
