use std::{
    collections::BTreeSet,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");
const NATIVE_CMAKELISTS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/CMakeLists.txt"));
const MASTER_HEADER: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/cudf_rust.h"));
const C_HEADER_SMOKE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/c_header_smoke.c"));
const CPP_HEADER_SMOKE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/cpp_header_smoke.cpp"));
const MEMORY_MANAGER_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/memory/manager.cpp"));
const RMM_INTEROP_BRIDGE_CPP: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/rmm_interop_bridge.cpp"));

fn production_shim_library() -> PathBuf {
    let Some(path) = option_env!("CUDF_RUST_PRODUCTION_SHIM") else {
        panic!("CUDF_RUST_PRODUCTION_SHIM must be set when the native ABI contract runs");
    };
    let library = PathBuf::from(path);
    assert!(
        library.is_file(),
        "production shim {} must exist; build.rs builds it for the current Cargo unit",
        library.display()
    );
    library
}

fn defined_dynamic_symbols(library: &Path) -> BTreeSet<String> {
    let output = Command::new("nm")
        .args(["-D", "--defined-only"])
        .arg(library)
        .output()
        .expect("nm must be available for the native ABI contract");
    assert!(
        output.status.success(),
        "nm failed for {}: {}",
        library.display(),
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout)
        .expect("nm output must be UTF-8")
        .lines()
        .filter_map(|line| line.split_whitespace().last())
        .map(|name| name.split('@').next().unwrap_or(name).to_owned())
        .collect()
}

fn dynamic_dependencies(library: &Path) -> BTreeSet<String> {
    let output = Command::new("readelf")
        .arg("-d")
        .arg(library)
        .output()
        .expect("readelf must be available for the native ABI contract");
    assert!(
        output.status.success(),
        "readelf failed for {}: {}",
        library.display(),
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout)
        .expect("readelf output must be UTF-8")
        .lines()
        .filter_map(|line| line.split_once("Shared library: [").map(|(_, remainder)| remainder))
        .filter_map(|remainder| remainder.split_once(']').map(|(name, _)| name.to_owned()))
        .collect()
}

fn resolved_dynamic_dependency(library: &Path, dependency: &str) -> PathBuf {
    let output = Command::new("ldd").arg(library).output().expect("ldd must be available for the native ABI contract");
    assert!(
        output.status.success(),
        "ldd failed for {}: {}",
        library.display(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .find(|line| line.trim_start().starts_with(&format!("{dependency} =>")))
        .unwrap_or_else(|| panic!("{} must resolve {dependency}", library.display()));
    let resolved = line
        .split_once("=>")
        .and_then(|(_, remainder)| remainder.split_whitespace().next())
        .unwrap_or_else(|| panic!("{dependency} resolution for {} is malformed: {line}", library.display()));
    assert_ne!(resolved, "not", "{dependency} is not resolved for {}", library.display());
    fs::canonicalize(resolved).unwrap_or_else(|error| {
        panic!("resolved {dependency} path {resolved} for {} is invalid: {error}", library.display())
    })
}

fn collect_headers(dir: &Path, headers: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("public header directory must be readable") {
        let entry = entry.expect("public header entry must be readable");
        let path = entry.path();
        if path.is_dir() {
            collect_headers(&path, headers);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("h") {
            headers.push(path);
        }
    }
}

fn production_header_declarations() -> BTreeSet<String> {
    let public_headers = Path::new(MANIFEST_DIR).join("native/include/cudf_rust");
    let mut headers = Vec::new();
    collect_headers(&public_headers, &mut headers);
    headers.sort();

    let mut declarations = BTreeSet::new();
    for header in headers {
        if header.file_name().and_then(|name| name.to_str()) == Some("test_support.h") {
            continue;
        }
        let text = fs::read_to_string(&header).expect("public header must be readable");
        let mut remainder = text.as_str();
        while let Some(start) = remainder.find("cudf_") {
            remainder = &remainder[start..];
            let end = remainder
                .char_indices()
                .take_while(|(_, character)| character.is_ascii_alphanumeric() || *character == '_')
                .last()
                .map(|(index, character)| index + character.len_utf8())
                .expect("cudf_ token must have an identifier");
            let name = &remainder[..end];
            if remainder[end..].trim_start().starts_with('(') {
                declarations.insert(name.to_owned());
            }
            remainder = &remainder[end..];
        }
    }
    declarations
}

#[test]
fn installed_header_layout_and_smoke_consumers_are_namespaced() {
    let include_root = Path::new(MANIFEST_DIR).join("native/include");
    let public_root = include_root.join("cudf_rust");
    assert!(public_root.join("cudf_rust.h").is_file(), "master header must be namespaced");
    assert!(
        MASTER_HEADER.contains("#include \"column.h\""),
        "master header must include its sibling production headers"
    );
    assert!(!MASTER_HEADER.contains("test_support.h"), "production master header must exclude test hooks");
    assert!(
        fs::read_dir(&include_root).unwrap().all(|entry| entry.unwrap().path() == public_root),
        "generic top-level public headers must not remain"
    );

    assert!(C_HEADER_SMOKE.starts_with("#include \"cudf_rust/cudf_rust.h\""));
    assert!(
        C_HEADER_SMOKE.contains("static int cudf_rust_c_header_smoke"),
        "C smoke entrypoint must be translation-unit-local"
    );
    assert!(CPP_HEADER_SMOKE.starts_with("#include \"cudf_rust/cudf_rust.h\""));
    let compact_cpp_smoke = CPP_HEADER_SMOKE.split_ascii_whitespace().collect::<String>();
    for assertion in [
        "cudf_column_handle, cudf_table_handle",
        "cudf_column_handle, cudf_scalar_handle",
        "cudf_stream_handle, cudf_event_handle",
        "cudf_memory_resource_handle, cudf_memory_manager_handle",
    ] {
        let compact_assertion = assertion.split_ascii_whitespace().collect::<String>();
        assert!(
            compact_cpp_smoke.contains(&compact_assertion),
            "C++ smoke must prove distinct opaque handle types: {assertion}"
        );
    }

    assert!(NATIVE_CMAKELISTS.contains("src/c_header_smoke.c"));
    assert!(NATIVE_CMAKELISTS.contains("src/cpp_header_smoke.cpp"));
    assert!(NATIVE_CMAKELISTS.contains("$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"));
    assert!(NATIVE_CMAKELISTS.contains("$<INSTALL_INTERFACE:include>"));
    assert!(NATIVE_CMAKELISTS.contains("\"${CMAKE_CURRENT_SOURCE_DIR}/src\""));
    assert!(NATIVE_CMAKELISTS.contains("install(DIRECTORY include/cudf_rust/ DESTINATION include/cudf_rust"));
    assert!(!NATIVE_CMAKELISTS.contains("install(DIRECTORY include/ DESTINATION include"));

    let production_library = production_shim_library();
    let installed_include = production_library
        .parent()
        .and_then(Path::parent)
        .expect("production shim must live below its CMake install prefix")
        .join("include");
    let installed_entries = fs::read_dir(&installed_include)
        .expect("installed include root must be readable")
        .map(|entry| entry.expect("installed include entry must be readable").file_name())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        installed_entries,
        BTreeSet::from(["cudf_rust".into()]),
        "the actual production install must contain only the namespaced public header directory"
    );
    assert!(
        !installed_include.join("cudf_rust/test_support.h").exists(),
        "the actual production install must exclude the test-support header"
    );
}

#[test]
fn production_dynamic_exports_match_master_header_declarations() {
    let library = production_shim_library();
    let symbols = defined_dynamic_symbols(&library);
    let dynamic_cudf_symbols =
        symbols.iter().filter(|symbol| symbol.starts_with("cudf_")).cloned().collect::<BTreeSet<_>>();
    let declarations = production_header_declarations();

    assert_eq!(
        dynamic_cudf_symbols,
        declarations,
        "production cudf_* definitions must exactly match production master-header declarations in {}",
        library.display()
    );
    assert!(!symbols.contains("cudf_rust_c_header_smoke"), "smoke entrypoint must not be exported");
    assert!(
        !symbols.iter().any(|symbol| symbol.starts_with("cudf_test_")),
        "production shim must not export test hooks"
    );
    assert!(
        !symbols.iter().any(|symbol| symbol.contains("set_error")
            || symbol.contains("translate_exception")
            || symbol.contains("move_arrow_")),
        "production shim must hide internal error and Arrow helper symbols"
    );
    assert!(
        !symbols.iter().any(|symbol| symbol.starts_with("_Z")),
        "production shim must not export mangled C++ implementation symbols"
    );
}

#[test]
fn rmm_singleton_bridge_is_private_and_resolved_by_the_shim() {
    let library = production_shim_library();
    let bridge_name = "libcudf_rust_rmm_interop.so";
    let bridge = library.parent().expect("shim library must have a parent directory").join(bridge_name);

    assert!(
        dynamic_dependencies(&library).contains(bridge_name),
        "the version-scripted shim must depend on the private RMM singleton bridge"
    );
    assert!(bridge.is_file(), "the private RMM singleton bridge must ship beside the shim");

    let ldd = Command::new("ldd").arg(&library).output().expect("ldd must be available for the native ABI contract");
    assert!(ldd.status.success(), "ldd failed for {}: {}", library.display(), String::from_utf8_lossy(&ldd.stderr));
    assert!(
        String::from_utf8_lossy(&ldd.stdout)
            .lines()
            .any(|line| line.contains(bridge_name) && !line.contains("not found")),
        "the shim RUNPATH must resolve its private RMM singleton bridge"
    );

    assert!(MEMORY_MANAGER_CPP.contains("cudf_rust_rmm_set_current_device_resource(target);"));
    assert!(
        !MEMORY_MANAGER_CPP.contains("rmm::mr::set_current_device_resource"),
        "the version-scripted shim must not instantiate RMM's per-device singleton"
    );
    assert!(RMM_INTEROP_BRIDGE_CPP.contains("rmm::mr::set_current_device_resource"));
    assert!(RMM_INTEROP_BRIDGE_CPP.contains("rmm::mr::get_current_device_resource_ref()"));
    assert!(
        !RMM_INTEROP_BRIDGE_CPP.contains("dlsym") && !RMM_INTEROP_BRIDGE_CPP.contains("_Z"),
        "the bridge must use a typed local call, not dynamic or mangled-symbol lookup"
    );
    assert!(NATIVE_CMAKELISTS.contains("add_library(cudf_rust_rmm_interop SHARED"));
    assert!(NATIVE_CMAKELISTS.contains("PRIVATE cudf_rust_rmm_interop"));
    assert!(NATIVE_CMAKELISTS.contains("install(TARGETS cudf_rust cudf_rust_rmm_interop LIBRARY DESTINATION lib)"));
}

#[test]
fn libcudf_and_both_shims_resolve_one_candidate_rmm() {
    let shim = production_shim_library();
    let bridge = shim.parent().expect("shim library must have a parent directory").join("libcudf_rust_rmm_interop.so");
    let libcudf = resolved_dynamic_dependency(&shim, "libcudf.so");
    let candidate = resolved_dynamic_dependency(&libcudf, "librmm.so");

    assert!(dynamic_dependencies(&libcudf).contains("librmm.so"));
    for library in [&libcudf, &shim, &bridge] {
        assert_eq!(
            resolved_dynamic_dependency(library, "librmm.so"),
            candidate,
            "{} must resolve the candidate RMM",
            library.display()
        );
    }
}

#[test]
fn public_handles_are_nominal_opaque_pointer_types() {
    let public_headers = Path::new(MANIFEST_DIR).join("native/include/cudf_rust");
    let mut headers = Vec::new();
    collect_headers(&public_headers, &mut headers);
    for header in headers {
        let text = fs::read_to_string(&header).expect("public header must be readable");
        assert!(
            !text.contains("typedef void* cudf_") && !text.contains("typedef void *cudf_"),
            "{} must not expose void-pointer handles",
            header.display()
        );
    }

    let compact_headers = fs::read_dir(&public_headers)
        .unwrap()
        .map(|entry| fs::read_to_string(entry.unwrap().path()).unwrap())
        .collect::<String>()
        .split_ascii_whitespace()
        .collect::<String>();
    for declaration in [
        "typedef struct cudf_column_handle_state *cudf_column_handle;",
        "typedef struct cudf_table_handle_state *cudf_table_handle;",
        "typedef struct cudf_scalar_handle_state *cudf_scalar_handle;",
        "typedef struct cudf_stream_handle_state *cudf_stream_handle;",
        "typedef struct cudf_event_handle_state *cudf_event_handle;",
        "typedef struct cudf_memory_resource_handle_state *cudf_memory_resource_handle;",
    ] {
        let compact_declaration = declaration.split_ascii_whitespace().collect::<String>();
        assert!(
            compact_headers.contains(&compact_declaration),
            "missing nominal public handle declaration `{declaration}`"
        );
    }
}
