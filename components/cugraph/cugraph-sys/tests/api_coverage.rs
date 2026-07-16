use std::{
    collections::{BTreeSet, HashSet, VecDeque},
    fs,
    path::{Path, PathBuf},
};

mod support;

// Shared helper still delegates to build_support::resolve_include_dir.
use support::{include_dir, manifest_dir, strip_c_comments};

fn bindings_rs_path() -> PathBuf {
    PathBuf::from(env!("CUGRAPH_SYS_BINDINGS_RS"))
}

fn parse_cugraph_includes(source: &str) -> Vec<String> {
    source
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            line.strip_prefix("#include <cugraph_c/").and_then(|rest| rest.strip_suffix('>')).map(|s| s.to_string())
        })
        .collect()
}

fn gather_transitive_headers(include_dir: &Path, roots: &[String]) -> BTreeSet<PathBuf> {
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::new();

    for root in roots {
        queue.push_back(root.clone());
    }

    while let Some(rel) = queue.pop_front() {
        let path = include_dir.join("cugraph_c").join(&rel);
        if !visited.insert(path.clone()) {
            continue;
        }

        let Ok(content) = fs::read_to_string(&path) else {
            continue;
        };

        for inc in parse_cugraph_includes(&content) {
            queue.push_back(inc);
        }
    }

    visited
}

fn strip_disabled_if0_blocks(source: &str) -> String {
    let mut out = String::new();
    let mut stack: Vec<bool> = Vec::new();

    for line in source.lines() {
        let trimmed = line.trim_start();

        if trimmed.starts_with("#if") {
            let parent_disabled = stack.iter().any(|s| *s);
            let is_if0 = trimmed.starts_with("#if 0") || trimmed.starts_with("#if(0)");
            stack.push(parent_disabled || is_if0);
            continue;
        }

        if trimmed.starts_with("#else") || trimmed.starts_with("#elif") {
            if !stack.is_empty() {
                let last = stack.len() - 1;
                let parent_disabled = stack[..last].iter().any(|s| *s);
                let current = stack[last];
                stack[last] = if parent_disabled { true } else { !current };
            }
            continue;
        }

        if trimmed.starts_with("#endif") {
            let _ = stack.pop();
            continue;
        }

        if !stack.iter().any(|s| *s) {
            out.push_str(line);
            out.push('\n');
        }
    }

    out
}

fn extract_cugraph_function_names(source: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let bytes = source.as_bytes();
    let mut i = 0;

    while i + "cugraph_".len() <= bytes.len() {
        if &bytes[i..i + "cugraph_".len()] != b"cugraph_" {
            i += 1;
            continue;
        }

        let start = i;
        i += "cugraph_".len();

        while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
            i += 1;
        }

        let end = i;
        let mut j = i;
        while j < bytes.len() && bytes[j].is_ascii_whitespace() {
            j += 1;
        }

        if j < bytes.len() && bytes[j] == b'(' {
            names.insert(source[start..end].to_string());
        }
    }

    names
}

fn expected_api_from_wrapper_headers() -> BTreeSet<String> {
    let wrapper = fs::read_to_string(manifest_dir().join("wrapper.h")).expect("read wrapper.h");
    let roots = parse_cugraph_includes(&wrapper);

    let include_dir = include_dir();
    let headers = gather_transitive_headers(&include_dir, &roots);

    let mut expected = BTreeSet::new();

    for header in headers {
        let content = fs::read_to_string(&header)
            .unwrap_or_else(|_| panic!("failed to read header for coverage test: {}", header.display()));
        let content = strip_disabled_if0_blocks(&content);
        let content = strip_c_comments(&content);

        for name in extract_cugraph_function_names(&content) {
            expected.insert(name);
        }
    }

    expected
}

fn actual_api_from_generated_bindings() -> BTreeSet<String> {
    let bindings_path = bindings_rs_path();
    let content = fs::read_to_string(&bindings_path)
        .unwrap_or_else(|_| panic!("failed to read generated bindings for coverage test: {}", bindings_path.display()));

    let mut actual = BTreeSet::new();
    for line in content.lines() {
        if let Some(rest) = line.trim().strip_prefix("pub fn cugraph_")
            && let Some(name) = rest.split('(').next()
        {
            actual.insert(format!("cugraph_{name}"));
        }
    }

    actual
}

#[test]
fn bindings_rs_path_points_to_existing_file() {
    let path = bindings_rs_path();
    assert!(path.is_file(), "bindings_rs_path() must point to a file, got: {}", path.display());
}

#[test]
fn generated_bindings_cover_all_wrapper_declared_functions() {
    let expected = expected_api_from_wrapper_headers();
    let actual = actual_api_from_generated_bindings();

    let mut ignore = HashSet::new();
    // These are intentionally excluded from cuGraph headers with #if 0.
    ignore.insert("cugraph_type_erased_device_array_release".to_string());
    ignore.insert("cugraph_type_erased_host_array_release".to_string());

    let missing: Vec<_> = expected.difference(&actual).filter(|name| !ignore.contains(*name)).cloned().collect();

    assert!(
        missing.is_empty(),
        "generated bindings are missing wrapper-declared APIs (showing up to 30):\n{}",
        missing.iter().take(30).map(String::as_str).collect::<Vec<_>>().join("\n")
    );
}
