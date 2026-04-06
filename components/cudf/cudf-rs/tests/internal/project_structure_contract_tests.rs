//! Source-level contracts for the simplified project structure.
//!
//! These tests lock in the post-shim policy:
//! - `src/` root contains only `lib.rs`
//! - in-repo code uses crate-root or organized domain-module imports instead of
//!   historical root-module paths
//! - the top-level domain folders stay present

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

const LIB_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs"));
const INTERNAL_MOD_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/internal/mod.rs"));
const INTEROP_MOD_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/interop/mod.rs"));
const DATAFRAME_OWNED_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/frame/dataframe/owned.rs"));
const SERIES_OWNED_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/frame/series/owned.rs"));
const JOIN_DATAFRAME_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/ops/join/dataframe.rs"));
const REMOVED_ROOT_MODULES: &[&str] = &[
    "arrow",
    "binary",
    "column",
    "concatenate",
    "copying",
    "dataframe",
    "dataframe_view",
    "dictionary",
    "error",
    "execution_context",
    "expr",
    "filter",
    "groupby",
    "hashing",
    "join",
    "memory",
    "reduction",
    "replace",
    "scalar",
    "search",
    "series",
    "series_view",
    "sorting",
    "stream",
    "stream_compaction",
    "table",
    "types",
    "unary",
];

fn rust_source_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                files.push(path);
            }
        }
    }

    files.sort();
    files
}

fn direct_rust_child_files(dir: &Path) -> Vec<String> {
    let mut files = Vec::new();

    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            files.push(entry.file_name().to_string_lossy().into_owned());
        }
    }

    files.sort();
    files
}

fn direct_child_dirs(dir: &Path) -> Vec<String> {
    let mut dirs = Vec::new();

    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            dirs.push(entry.file_name().to_string_lossy().into_owned());
        }
    }

    dirs.sort();
    dirs
}

fn split_top_level_entries(group: &str) -> Vec<&str> {
    let mut entries = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;

    for (idx, ch) in group.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                entries.push(&group[start..idx]);
                start = idx + 1;
            },
            _ => {},
        }
    }

    if start < group.len() {
        entries.push(&group[start..]);
    }

    entries.into_iter().filter(|entry| !entry.is_empty()).collect()
}

fn cudf_brace_groups(normalized: &str) -> Vec<String> {
    let mut groups = Vec::new();
    let mut offset = 0usize;

    while let Some(start) = normalized[offset..].find("cudf::{") {
        let content_start = offset + start + "cudf::{".len();
        let mut depth = 1usize;
        let mut end = content_start;

        for (idx, ch) in normalized[content_start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = content_start + idx;
                        break;
                    }
                },
                _ => {},
            }
        }

        if end > content_start {
            groups.push(normalized[content_start..end].to_string());
            offset = end + 1;
        } else {
            break;
        }
    }

    groups
}

fn removed_root_path_hits(source: &str) -> Vec<&'static str> {
    let normalized: String = source.chars().filter(|ch| !ch.is_whitespace()).collect();
    let mut hits = BTreeSet::new();

    for module in REMOVED_ROOT_MODULES {
        if normalized.contains(&format!("cudf::{module}::")) {
            hits.insert(*module);
        }
    }

    for group in cudf_brace_groups(&normalized) {
        for entry in split_top_level_entries(&group) {
            for module in REMOVED_ROOT_MODULES {
                if entry == *module || entry.starts_with(&format!("{module}::")) {
                    hits.insert(*module);
                }
            }
        }
    }

    hits.into_iter().collect()
}

#[test]
fn lib_rs_no_longer_declares_legacy_root_modules() {
    for module in REMOVED_ROOT_MODULES {
        let exact = format!("pub mod {module};");
        assert!(!LIB_RS.contains(&exact), "expected legacy root module `{module}` to be removed from lib.rs");
    }
}

#[test]
fn repo_internal_code_does_not_use_removed_root_module_paths() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut offenders = Vec::new();

    for root in ["src", "tests"] {
        for path in rust_source_files(&manifest_dir.join(root)) {
            let relative = path.strip_prefix(manifest_dir).unwrap().to_string_lossy().replace('\\', "/");
            let hits = removed_root_path_hits(&fs::read_to_string(&path).unwrap());
            if !hits.is_empty() {
                offenders.push(format!("{relative}: {}", hits.join(", ")));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "expected in-repo code to stop using removed root module paths:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn frame_test_domain_does_not_leave_flat_rs_files_at_its_root() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let offenders = direct_rust_child_files(&manifest_dir.join("tests/frame"));

    assert!(
        offenders.is_empty(),
        "expected tests/frame/ to organize Rust tests into subdomains instead of flat files:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn src_root_contains_only_lib_rs() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let actual = direct_rust_child_files(&manifest_dir.join("src"));
    assert_eq!(actual, vec!["lib.rs".to_string()], "expected src/ root to contain only lib.rs");
}

#[test]
fn target_top_level_source_domains_exist() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let actual = direct_child_dirs(&manifest_dir.join("src"));
    let expected = vec![
        "core".to_string(),
        "frame".to_string(),
        "internal".to_string(),
        "interop".to_string(),
        "io".to_string(),
        "ops".to_string(),
        "runtime".to_string(),
    ];

    assert_eq!(actual, expected, "expected src/ to keep the spec-defined top-level domain folders");
}

#[test]
fn interop_domain_stays_split_across_projection_and_table_export_submodules() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let interop_dir = manifest_dir.join("src/interop");
    let mod_rs = fs::read_to_string(interop_dir.join("mod.rs")).unwrap();

    assert!(interop_dir.join("projection/mod.rs").is_file(), "expected src/interop/projection/mod.rs to exist");
    assert!(interop_dir.join("projection/device.rs").is_file(), "expected src/interop/projection/device.rs to exist");
    assert!(interop_dir.join("projection/host.rs").is_file(), "expected src/interop/projection/host.rs to exist");
    assert!(interop_dir.join("table_export.rs").is_file(), "expected src/interop/table_export.rs to exist");
    assert!(mod_rs.contains("mod projection;"), "expected interop::mod.rs to declare the projection submodule");
    assert!(mod_rs.contains("mod table_export;"), "expected interop::mod.rs to declare the table_export submodule");
    assert!(
        mod_rs.lines().count() < 2200,
        "expected interop::mod.rs to stay below 2200 lines after splitting heavy projection and table-export logic",
    );
}

#[test]
fn crate_root_docs_do_not_point_users_at_removed_root_modules() {
    assert!(
        !LIB_RS.contains("see the [`stream`] module"),
        "expected public crate docs to send users to visible domain modules instead of removed root modules"
    );
}

#[test]
fn interop_module_docs_reflect_consumer_import_support() {
    assert!(
        INTEROP_MOD_RS.contains("consumer-side import from the shared borrowed"),
        "expected interop module docs to mention borrowed consumer-side import support"
    );
    assert!(
        INTEROP_MOD_RS.contains("zero-copy consumer-side import from"),
        "expected interop module docs to mention zero-copy owned-transfer consumer-side import support"
    );
}

#[test]
fn repo_does_not_leave_stale_arrow_feature_gates() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut offenders = Vec::new();

    for root in ["src", "tests"] {
        for path in rust_source_files(&manifest_dir.join(root)) {
            let relative = path.strip_prefix(manifest_dir).unwrap().to_string_lossy().replace('\\', "/");
            let source = fs::read_to_string(&path).unwrap();
            if source.contains("cfg(feature = \"arrow\")") || source.contains("cfg(feature=\"arrow\")") {
                offenders.push(relative);
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "expected Arrow to stay unconditional after becoming a required dependency:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn internal_temp_table_module_stays_crate_private() {
    assert!(
        INTERNAL_MOD_RS.contains("pub(crate) mod temp_table;"),
        "expected internal::temp_table to stay crate-private",
    );
    assert!(
        !INTERNAL_MOD_RS.contains("pub mod temp_table;"),
        "expected internal::temp_table to avoid wider visibility than lib.rs::internal",
    );
}

#[test]
fn dataframe_owned_fields_do_not_leak_crate_wide_visibility() {
    for field in ["pub(crate) columns:", "pub(crate) row_count:", "pub(crate) domain:"] {
        assert!(
            !DATAFRAME_OWNED_RS.contains(field),
            "expected DataFrame field visibility to stay inside the frame domain, not `{field}`",
        );
    }
}

#[test]
fn series_owned_fields_do_not_leak_crate_wide_visibility() {
    for field in ["pub(crate) name:", "pub(crate) column:", "pub(crate) cached_len:", "pub(crate) cached_null_count:"] {
        assert!(
            !SERIES_OWNED_RS.contains(field),
            "expected Series field visibility to stay inside the series domain, not `{field}`",
        );
    }
}

#[test]
fn join_dataframe_glue_uses_dataframe_accessors_instead_of_fields() {
    for field_access in [
        "&self.columns",
        "&right.columns",
        "&left.columns",
        "self.columns.into_values",
        "right.columns.into_values",
        "left.columns.into_values",
        "self.columns.keys",
        "right.columns.keys",
        "left.columns.keys",
        "self.row_count",
        "right.row_count",
        "left.row_count",
        "&self.domain",
        "&right.domain",
        "&left.domain",
        "Arc::clone(&self.domain)",
        "Arc::clone(&right.domain)",
        "Arc::clone(&left.domain)",
    ] {
        assert!(
            !JOIN_DATAFRAME_RS.contains(field_access),
            "expected ops::join::dataframe glue to avoid direct DataFrame field access `{field_access}`",
        );
    }
}
