use std::collections::{BTreeMap, BTreeSet};

mod support;

// Shared helper still delegates to build_support::resolve_include_dir.
use support::{include_dir, manifest_dir, strip_c_comments};

const LAYOUT_FUNCTIONS: &[&str] = &[
    "cugraph_force_atlas2",
    "cugraph_layout_result_get_vertices",
    "cugraph_layout_result_get_x_axis",
    "cugraph_layout_result_get_y_axis",
    "cugraph_layout_result_free",
];

fn tokenize(fragment: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars: Vec<_> = fragment.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }

        if chars[i].is_ascii_alphanumeric() || chars[i] == '_' {
            let start = i;
            i += 1;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        tokens.push(chars[i].to_string());
        i += 1;
    }

    tokens
}

fn split_top_level_params(params: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut depth = 0i32;

    for (idx, ch) in params.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(params[start..idx].trim());
                start = idx + 1;
            },
            _ => {},
        }
    }

    let tail = params[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    }

    parts
}

fn normalize_param_type(param: &str) -> String {
    let mut tokens = tokenize(param);
    if tokens.last().is_some_and(|last| last.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_')) {
        tokens.pop();
    }
    tokens.join(" ")
}

fn normalize_function_declaration(declaration: &str, name: &str) -> String {
    let needle = format!("{name}(");
    let name_start = declaration.find(&needle).unwrap_or_else(|| panic!("failed to locate declaration for {name}"));
    let return_type = declaration[..name_start].trim();
    let params_start = name_start + name.len() + 1;
    let params_end = declaration.rfind(')').unwrap_or_else(|| panic!("failed to parse parameter list for {name}"));
    let params = &declaration[params_start..params_end];

    let normalized_params =
        split_top_level_params(params).into_iter().map(normalize_param_type).collect::<Vec<_>>().join(", ");

    format!("{} {}({})", tokenize(return_type).join(" "), name, normalized_params)
}

fn extract_named_declarations(source: &str) -> BTreeMap<String, String> {
    let mut declarations = BTreeMap::new();
    let cleaned = strip_c_comments(source);

    let filtered = cleaned
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.starts_with('#')
                && trimmed != "{"
                && trimmed != "}"
                && !trimmed.starts_with("extern ")
                && !trimmed.is_empty()
        })
        .collect::<Vec<_>>()
        .join("\n");

    for statement in filtered.split(';') {
        let declaration = statement.trim();
        if declaration.is_empty() {
            continue;
        }

        for name in LAYOUT_FUNCTIONS {
            let needle = format!("{name}(");
            if declaration.contains(&needle) {
                declarations.insert((*name).to_string(), normalize_function_declaration(declaration, name));
            }
        }
    }

    declarations
}

#[test]
fn wrapper_layout_declarations_match_upstream_layout_header() {
    let wrapper = std::fs::read_to_string(manifest_dir().join("wrapper.h")).expect("read wrapper.h");
    let layout = std::fs::read_to_string(include_dir().join("cugraph_c/layout_algorithms.h"))
        .expect("read cugraph_c/layout_algorithms.h");

    let wrapper_declarations = extract_named_declarations(&wrapper);
    let layout_declarations = extract_named_declarations(&layout);
    let expected_names = LAYOUT_FUNCTIONS.iter().map(|name| (*name).to_string()).collect::<BTreeSet<_>>();

    assert_eq!(
        wrapper_declarations.keys().cloned().collect::<BTreeSet<_>>(),
        expected_names,
        "wrapper.h should declare exactly the manually mirrored layout functions"
    );
    assert_eq!(
        layout_declarations.keys().cloned().collect::<BTreeSet<_>>(),
        expected_names,
        "layout_algorithms.h should expose the expected layout functions for drift comparison"
    );

    for name in LAYOUT_FUNCTIONS {
        assert_eq!(
            wrapper_declarations.get(*name),
            layout_declarations.get(*name),
            "layout declaration drift detected for {name}"
        );
    }
}
