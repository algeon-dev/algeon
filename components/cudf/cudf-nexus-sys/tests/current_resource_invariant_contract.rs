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

use std::{
    fs,
    path::{Path, PathBuf},
};

fn native_source_files() -> Vec<PathBuf> {
    let mut files = Vec::new();
    for dir in ["native/src", "native/include"] {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(dir);
        collect_native_source_files(&dir, &mut files);
    }
    assert!(files.len() > 20, "expected to scan the full native source tree, found {} files", files.len());
    files
}

fn collect_native_source_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("native source dir must be readable") {
        let path = entry.expect("native source entry must be readable").path();
        if path.is_dir() {
            collect_native_source_files(&path, files);
            continue;
        }
        let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
            continue;
        };
        if matches!(ext, "cpp" | "hpp" | "h" | "cu" | "cuh") {
            files.push(path);
        }
    }
}

fn file_name(path: &Path) -> &str {
    path.file_name().and_then(|name| name.to_str()).expect("native source file must have a UTF-8 name")
}

#[test]
fn memory_runtime_split_keeps_owner_state_local() {
    let native_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("native/src");
    let memory_dir = native_root.join("memory");
    assert!(!native_root.join("rmm.cpp").exists(), "the retired monolithic memory runtime must not return");

    let mut memory_units = fs::read_dir(&memory_dir)
        .expect("memory runtime source directory must be readable")
        .map(|entry| {
            entry
                .expect("memory runtime source entry must be readable")
                .file_name()
                .into_string()
                .expect("memory runtime source name must be UTF-8")
        })
        .collect::<Vec<_>>();
    memory_units.sort();
    assert_eq!(
        memory_units,
        ["internal.hpp", "manager.cpp", "query_domain.cpp", "reservation.cpp", "resource.cpp", "test_support.cpp"],
        "the memory runtime must remain a shallow ownership-oriented split"
    );

    for (symbol, owner) in [
        ("memory_resource_registry(", "resource.cpp"),
        ("memory_manager_registry(", "manager.cpp"),
        ("current_memory_resource_identities(", "manager.cpp"),
        ("default_memory_manager_options(", "manager.cpp"),
        ("query_domain_route_pause_state", "test_support.cpp"),
    ] {
        let defining_units = memory_units
            .iter()
            .filter(|unit| unit.ends_with(".cpp"))
            .filter(|unit| {
                fs::read_to_string(memory_dir.join(unit))
                    .expect("memory runtime source must be readable")
                    .contains(symbol)
            })
            .collect::<Vec<_>>();
        assert_eq!(defining_units, [owner], "{symbol} must be owned only by {owner}");
    }
}

#[test]
fn query_domain_teardown_activates_its_device_before_pool_destroy() {
    let query_domain_cpp =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("native/src/memory/query_domain.cpp"))
            .expect("memory/query_domain.cpp must be readable");
    let destructor = query_domain_cpp
        .split("query_domain_state::~query_domain_state()")
        .nth(1)
        .and_then(|tail| tail.split("\n}").next())
        .expect("query_domain_state destructor must be defined in query_domain.cpp");

    let activation = "if (device >= 0 && guard.activate(device) == CUDF_SUCCESS) {";
    let activation_offset = destructor
        .find(activation)
        .expect("query-domain teardown must activate its recorded device before destruction");
    let destroy_offset =
        destructor.find("cudaMemPoolDestroy(pool)").expect("query-domain teardown must destroy its private CUDA pool");
    assert!(
        activation_offset < destroy_offset,
        "query-domain teardown must not destroy a CUDA pool before activating its recorded device"
    );
    assert!(
        destructor.contains(concat!(
            "if (device >= 0 && guard.activate(device) == CUDF_SUCCESS) {\n",
            "      if (cudaMemPoolDestroy(pool) != cudaSuccess) {"
        )),
        "private-pool destruction must remain nested below successful activation of the recorded device"
    );
    assert_eq!(
        destructor.matches("cudaMemPoolDestroy(pool)").count(),
        1,
        "query-domain teardown must have one device-guarded private-pool destroy"
    );
    assert!(
        destructor[activation_offset..].contains("pool = nullptr;"),
        "activation failure must fail safe by releasing the handle without destroying against an ambient device"
    );
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
    // `memory/manager.cpp` owns the device-domain state machine; its one typed
    // install call lives in the private bridge so the version-scripted shim cannot
    // instantiate RMM's header-defined per-device singleton. No operation
    // path may install or swap a device's current resource.
    let mut violations = Vec::new();

    for path in native_source_files() {
        let name = file_name(&path).to_string();
        let source = fs::read_to_string(&path).expect("native source file must be readable");
        for (index, line) in source.lines().enumerate() {
            let permitted = if line.contains("rmm::mr::set_current_device_resource") {
                name == "rmm_interop_bridge.cpp"
            } else if line.contains("cudf_rust_rmm_set_current_device_resource") {
                path.ends_with("native/src/memory/manager.cpp")
                    || name == "rmm_interop_bridge.cpp"
                    || name == "rmm_interop_bridge.hpp"
            } else if line.contains("cudf_set_current_device_resource_raw")
                || line.contains("set_current_device_resource_raw_locked")
            {
                path.ends_with("native/src/memory/manager.cpp")
                    || path.ends_with("native/src/memory/test_support.cpp")
                    || (name == "execution_context_internal.hpp"
                        && line.contains("void cudf_set_current_device_resource_raw("))
            } else {
                continue;
            };
            if permitted {
                continue;
            }
            violations.push(format!("{}:{}\n{}", name, index + 1, line));
        }
    }

    assert!(
        violations.is_empty(),
        "installing a device's current resource is reserved for memory-domain creation and test hooks \
         in memory/manager.cpp plus test hooks and its narrow private RMM bridge; operation paths must pass explicit stream/mr instead.\n{}",
        violations.join("\n\n")
    );
}

#[test]
fn rmm_install_primitive_is_a_single_locked_choke_point() {
    let manager_cpp = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("native/src/memory/manager.cpp"))
        .expect("memory/manager.cpp must be readable");
    let bridge_cpp =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("native/src/rmm_interop_bridge.cpp"))
            .expect("rmm_interop_bridge.cpp must be readable");

    // Exactly one typed call into RMM's per-device current-resource setter is
    // isolated in the private bridge. The main shim must only cross that
    // boundary from the identity-map-locked helper, keeping tracking and
    // installation atomic without duplicating RMM's singleton.
    assert_eq!(
        bridge_cpp.matches("rmm::mr::set_current_device_resource(").count(),
        1,
        "the private bridge must contain exactly one typed RMM current-resource installation"
    );
    assert!(
        !manager_cpp.contains("rmm::mr::set_current_device_resource"),
        "the version-scripted shim must not instantiate RMM's per-device singleton"
    );
    let locked_helper = manager_cpp
        .split("set_current_device_resource_raw_locked(")
        .nth(1)
        .and_then(|tail| tail.split("\n}").next())
        .expect("memory/manager.cpp must define set_current_device_resource_raw_locked");
    assert_eq!(
        locked_helper.matches("cudf_rust_rmm_set_current_device_resource(target);").count(),
        1,
        "the locked install helper must make exactly one call through the private RMM bridge"
    );
    assert!(
        manager_cpp.contains("current_memory_resource_identity_mutex()"),
        "the current-resource identity map must keep its own small lock (never spanning libcudf calls)"
    );
    assert!(
        !bridge_cpp.contains("dlsym") && !bridge_cpp.contains("_Z"),
        "the private bridge must use a typed local call, not dynamic or mangled-symbol lookup"
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
