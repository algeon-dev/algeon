use std::{
    collections::{BTreeMap, BTreeSet},
    fs, panic,
    path::{Path, PathBuf},
};

struct Definition {
    path: PathBuf,
    calls: BTreeSet<String>,
}

fn tokens(source: &str) -> Vec<String> {
    let bytes = source.as_bytes();
    let mut result = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index].is_ascii_whitespace() {
            index += 1;
        } else if bytes[index..].starts_with(b"//") {
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
        } else if bytes[index..].starts_with(b"/*") {
            index += 2;
            while index < bytes.len() && !bytes[index..].starts_with(b"*/") {
                index += 1;
            }
            index = (index + 2).min(bytes.len());
        } else if matches!(bytes[index], b'"' | b'\'') {
            let quote = bytes[index];
            index += 1;
            while index < bytes.len() {
                if bytes[index] == b'\\' {
                    index = (index + 2).min(bytes.len());
                } else if bytes[index] == quote {
                    index += 1;
                    break;
                } else {
                    index += 1;
                }
            }
        } else if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
            let start = index;
            index += 1;
            while index < bytes.len() && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_') {
                index += 1;
            }
            result.push(source[start..index].to_owned());
        } else {
            result.push(char::from(bytes[index]).to_string());
            index += 1;
        }
    }
    result
}

fn group_end(tokens: &[String], start: usize, open: &str, close: &str) -> usize {
    let mut depth = 0;
    for (index, token) in tokens.iter().enumerate().skip(start) {
        if token == open {
            depth += 1;
        } else if token == close {
            depth -= 1;
            if depth == 0 {
                return index + 1;
            }
        }
    }
    panic!("unterminated native {open} group");
}

fn add_definitions(path: &Path, source: &str, definitions: &mut BTreeMap<String, Vec<Definition>>) {
    let tokens = tokens(source);
    for index in 0..tokens.len().saturating_sub(3) {
        if tokens[index] != "cudf_status_t" || tokens[index + 2] != "(" {
            continue;
        }
        let name = &tokens[index + 1];
        let body_start = group_end(&tokens, index + 2, "(", ")");
        let next = tokens.get(body_start).map(String::as_str);
        if name.starts_with("cudf_") && name.ends_with("_ex") {
            assert!(matches!(next, Some("{" | ";")), "{}: unsupported _ex definition form for {name}", path.display());
        }
        if next != Some("{") {
            continue;
        }
        let body_end = group_end(&tokens, body_start, "{", "}");
        let calls =
            tokens[body_start..body_end].windows(2).filter(|pair| pair[1] == "(").map(|pair| pair[0].clone()).collect();
        definitions.entry(name.clone()).or_default().push(Definition { path: path.to_owned(), calls });
    }
}

fn load_definitions(directory: &Path, definitions: &mut BTreeMap<String, Vec<Definition>>) {
    let mut entries = fs::read_dir(directory).unwrap().map(|entry| entry.unwrap().path()).collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            load_definitions(&path, definitions);
        } else if matches!(path.extension().and_then(|extension| extension.to_str()), Some("cpp" | "cu" | "hpp")) {
            add_definitions(&path, &fs::read_to_string(&path).unwrap(), definitions);
        }
    }
}

fn calls_resolver(
    name: &str,
    target: &str,
    definitions: &BTreeMap<String, Vec<Definition>>,
    seen: &mut BTreeSet<String>,
) -> bool {
    if name == target {
        return true;
    }
    if !seen.insert(name.to_owned()) {
        return false;
    }
    definitions.get(name).is_some_and(|entries| {
        entries
            .iter()
            .any(|definition| definition.calls.iter().any(|call| calls_resolver(call, target, definitions, seen)))
    })
}

fn violations(definitions: &BTreeMap<String, Vec<Definition>>) -> Vec<String> {
    let mut failures = Vec::new();
    for (name, entries) in definitions {
        if !name.starts_with("cudf_") || !name.ends_with("_ex") {
            continue;
        }
        assert_eq!(entries.len(), 1, "duplicate native _ex definition: {name}");
        let required = calls_resolver(name, "cudf_resolve_required_exec_options", definitions, &mut BTreeSet::new());
        let ambient = ["cudf_resolve_exec_options", "cudf_make_exec_context"]
            .iter()
            .any(|resolver| calls_resolver(name, resolver, definitions, &mut BTreeSet::new()));
        if !required || ambient {
            failures.push(format!("{}: {name} requires strict explicit execution options", entries[0].path.display()));
        }
    }
    failures
}

#[test]
fn every_native_ex_entry_resolves_required_execution_options() {
    let mut definitions = BTreeMap::new();
    load_definitions(&Path::new(env!("CARGO_MANIFEST_DIR")).join("native/src"), &mut definitions);
    assert!(
        definitions.keys().any(|name| name.starts_with("cudf_") && name.ends_with("_ex")),
        "no native _ex definitions found"
    );
    let failures = violations(&definitions);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn resolver_scan_follows_delegates_and_rejects_comment_and_string_decoys() {
    let mut definitions = BTreeMap::new();
    add_definitions(
        Path::new("fixture.cpp"),
        r#"
        cudf_status_t delegated_impl(cudf_exec_options_t* options) {
            return cudf_resolve_required_exec_options(options, &exec);
        }
        cudf_status_t cudf_delegated_ex(cudf_exec_options_t* options) {
            return delegated_impl(options);
        }
        cudf_status_t cudf_missing_ex(cudf_exec_options_t* options) {
            return CUDF_SUCCESS;
        }
        cudf_status_t cudf_ambient_ex(cudf_exec_options_t* options) {
            // cudf_resolve_required_exec_options(options, &exec)
            set_error("cudf_resolve_required_exec_options(options, &exec)");
            return cudf_resolve_exec_options(options, &exec);
        }
    "#,
        &mut definitions,
    );
    let failures = violations(&definitions);
    assert_eq!(failures.len(), 2);
    assert!(failures.iter().any(|failure| failure.contains("cudf_ambient_ex")));
    assert!(failures.iter().any(|failure| failure.contains("cudf_missing_ex")));
    add_definitions(
        Path::new("duplicate.cpp"),
        "cudf_status_t cudf_delegated_ex(cudf_exec_options_t* options) { return CUDF_SUCCESS; }",
        &mut definitions,
    );
    assert!(panic::catch_unwind(|| violations(&definitions)).is_err());
}
