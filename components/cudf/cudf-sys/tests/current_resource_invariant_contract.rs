//! Device-domain current-resource invariant (inverts the retired guard
//! contract).
//!
//! Each CUDA device gets exactly one memory domain: cuda mr → pool →
//! reservation-aware adaptor, installed as RMM's per-device current resource
//! when the domain is created and never changed afterwards. libcudf
//! temporaries deliberately resolve through that long-lived adaptor and are
//! accounted per stream, so no operation ever needs to swap the current
//! resource — and none may: a per-op swap is process-visible mutable state
//! that forces global serialization and reintroduces the 2026-06-07
//! dangling-resource race.
//!
//! These tests therefore forbid the guard/swap machinery everywhere instead
//! of allowlisting its call sites.

use std::{fs, path::Path, path::PathBuf};

fn native_source_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    for dir in ["native/src", "native/include"] {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(dir);
        for entry in fs::read_dir(&dir).expect("native source dir must be readable") {
            let path = entry.expect("native source entry must be readable").path();
            let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
                continue;
            };
            if matches!(ext, "cpp" | "hpp" | "h" | "cu" | "cuh") {
                files.push(path);
            }
        }
    }
    assert!(files.len() > 20, "expected to scan the full native source tree, found {} files", files.len());
    files
}

fn file_name(path: &Path) -> &str {
    path.file_name().and_then(|name| name.to_str()).expect("native source file must have a UTF-8 name")
}

#[test]
fn shim_contains_no_current_resource_guard_or_swap_machinery() {
    let mut violations = Vec::new();

    for path in native_source_files() {
        let source = fs::read_to_string(&path).expect("native source file must be readable");
        for (index, line) in source.lines().enumerate() {
            for forbidden in [
                "cudf_current_device_resource_guard",
                "cudf_current_device_resource_mutex",
                "install_current_device_resource",
                // Reading the ambient current resource inside an operation is
                // how swap-dependent routing sneaks back in: under the old
                // guard these reads observed the swapped-in resource, and
                // without the swap they silently observe the device domain
                // instead. Operations must route through explicit handles.
                "get_current_device_resource",
            ] {
                if line.contains(forbidden) {
                    violations.push(format!("{}:{} forbidden `{}`:\n{}", file_name(&path), index + 1, forbidden, line));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "per-operation current-resource swapping was removed by the device-domain concurrency model \
         (docs/plan/2026-06-10-p0-6-device-domain-concurrency.md); do not reintroduce it.\n{}",
        violations.join("\n\n")
    );
}

#[test]
fn current_resource_installation_is_confined_to_device_domain_setup() {
    // The only file allowed to reference the install primitives is rmm.cpp
    // (definition, lazy default install, manager-creation install, and the
    // test hooks), plus the declaration in execution_context_internal.hpp.
    // No operation path may install or swap a device's current resource.
    let mut violations = Vec::new();

    for path in native_source_files() {
        let name = file_name(&path).to_string();
        let source = fs::read_to_string(&path).expect("native source file must be readable");
        for (index, line) in source.lines().enumerate() {
            let references_install = line.contains("cudf_set_current_device_resource_raw")
                || line.contains("set_current_device_resource_raw_locked")
                || line.contains("rmm::mr::set_current_device_resource");
            if !references_install {
                continue;
            }
            let allowed = name == "rmm.cpp"
                || (name == "execution_context_internal.hpp"
                    && line.contains("void cudf_set_current_device_resource_raw("));
            if !allowed {
                violations.push(format!("{}:{}\n{}", name, index + 1, line));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "installing a device's current resource is reserved for memory-domain creation and test hooks \
         in rmm.cpp; operation paths must pass explicit stream/mr instead.\n{}",
        violations.join("\n\n")
    );
}

#[test]
fn rmm_install_primitive_is_a_single_locked_choke_point() {
    let rmm_cpp = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("native/src/rmm.cpp"))
        .expect("rmm.cpp must be readable");

    // Exactly one call into rmm's per-device current-resource setter, inside
    // the identity-map-locked helper, keeps tracking and installation atomic.
    assert_eq!(
        rmm_cpp.matches("rmm::mr::set_current_device_resource(").count(),
        1,
        "rmm-level current-resource installation must stay confined to set_current_device_resource_raw_locked"
    );
    assert!(
        rmm_cpp.contains("std::mutex& current_memory_resource_identity_mutex()"),
        "the current-resource identity map must keep its own small lock (never spanning libcudf calls)"
    );
}

#[test]
fn rmm_device_buffer_stream_allocations_use_explicit_memory_resource() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("native/src");
    let mut violations = Vec::new();

    for entry in fs::read_dir(&src_dir).expect("native/src must be readable") {
        let path = entry.expect("native/src entry must be readable").path();
        let is_join_common = path.file_name().and_then(|name| name.to_str()) == Some("join_common.hpp");
        if path.extension().and_then(|ext| ext.to_str()) != Some("cpp") && !is_join_common {
            continue;
        }

        let source = fs::read_to_string(&path).expect("native source file must be readable");
        let lines: Vec<_> = source.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            if !line.contains("rmm::device_buffer") {
                continue;
            }
            let statement = statement_from(&lines, index);
            if statement.contains("stream")
                && !statement.contains("*exec->mr")
                && !statement.contains("*exec.mr")
                && !statement.contains("*mr_")
            {
                let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("<unknown>");
                violations.push(format!(
                    "{}:{} rmm::device_buffer stream allocation must pass an explicit memory resource:\n{}",
                    name,
                    index + 1,
                    statement
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "rmm::device_buffer(size, stream) uses RMM current resource; pass the caller's explicit \
         execution memory resource instead.\n{}",
        violations.join("\n\n")
    );
}

fn statement_from(lines: &[&str], index: usize) -> String {
    let mut out = Vec::new();
    for line in &lines[index..] {
        out.push(*line);
        if line.contains(';') {
            break;
        }
    }
    out.join("\n")
}
