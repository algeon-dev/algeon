mod source_contract_support;

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

const GPU_OWNING_PUBLIC_DESTROY_FUNCTIONS: &[&str] =
    &["cudf_column_destroy", "cudf_table_destroy", "cudf_scalar_destroy", "cudf_device_buffer_destroy"];

#[test]
fn gpu_owning_public_destroy_functions_use_provenance_helper_instead_of_naked_delete() {
    let sources = NativeSources::load();
    let mut violations = Vec::new();

    for function in GPU_OWNING_PUBLIC_DESTROY_FUNCTIONS {
        let region = sources.function_body(function);
        if source_contract_support::contains_normalized(region.body, "delete static_cast<") {
            violations.push(format!(
                "{} in {} still performs a naked `delete static_cast<...>(handle)`",
                function, region.file_name
            ));
        }

        let helpers = destroy_provenance_identifiers(region.body);
        if helpers.is_empty() {
            violations.push(format!(
                "{} in {} must route destruction through a provenance-aware helper, \
                 for example `destroy_under_provenance(...)` or another \
                 `destroy_*provenance` helper",
                function, region.file_name
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "GPU-owning public destroy functions must release under captured handle provenance.\n{}",
        violations.join("\n")
    );
}

#[test]
fn gpu_owning_handle_owners_carry_destroy_provenance() {
    let sources = NativeSources::load();
    let combined = sources.combined();
    let mut violations = Vec::new();

    if !combined.contains("cudf_handle_provenance") {
        violations.push("native/src must define a shared `cudf_handle_provenance` type".to_string());
    }
    let provenance_body = type_body(combined, "cudf_handle_provenance");
    if !provenance_body.contains("mr_lifetime") || !provenance_body.contains("shared_ptr") {
        violations.push(
            "`cudf_handle_provenance` must retain shim-owned memory-resource lifetime, \
             not just a raw `device_memory_resource*`"
                .to_string(),
        );
    }

    for owner in ["native_column_handle_state", "native_table_handle_state"] {
        let body = type_body(combined, owner);
        if !body.contains("cudf_handle_provenance") {
            violations.push(format!("{owner} must store or otherwise retain `cudf_handle_provenance` for destroy"));
        }
    }

    for owner in ["scalar", "device_buffer"] {
        if !contains_markers_near(combined, owner, "cudf_handle_provenance", 1_500) {
            violations.push(format!("{owner} owners/helpers must carry `cudf_handle_provenance` for destroy"));
        }
    }

    for releaser in ["cudf_release_column_handle", "cudf_release_table_handle"] {
        if !has_signature_with_marker(combined, releaser, "cudf_handle_provenance") {
            violations.push(format!("{releaser} must have a provenance-carrying overload or equivalent release path"));
        }
    }

    assert!(
        violations.is_empty(),
        "GPU-owning native handle owners must retain destroy-time provenance.\n{}",
        violations.join("\n")
    );
}

#[test]
fn gpu_resources_are_declared_after_retained_provenance_so_they_drop_first() {
    let sources = NativeSources::load();
    let combined = sources.combined();
    let mut violations = Vec::new();

    for (owner, resource) in [
        ("native_column_handle_state", "_column"),
        ("native_table_handle_state", "_table"),
        ("cudf_scalar_owner", "scalar"),
        ("cudf_device_buffer_owner", "buffer"),
        ("adopted_device_buffer_column_handle_state", "_data_owner"),
        ("prepared_groupby_owner", "groupby"),
        ("arrow_device_imported_column_handle_state", "_view"),
        ("arrow_device_imported_table_root_state", "view"),
    ] {
        let body = type_body(combined, owner);
        let provenance_position =
            body.rfind("provenance").unwrap_or_else(|| panic!("expected {owner} to retain provenance"));
        let resource_position =
            body.rfind(resource).unwrap_or_else(|| panic!("expected {owner} to contain resource `{resource}`"));
        if provenance_position > resource_position {
            violations.push(format!(
                "{owner} must declare retained provenance before `{resource}` so reverse field destruction releases the GPU resource first"
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "GPU resource owners must outlive neither their stream/MR provenance.\n{}",
        violations.join("\n")
    );
}

#[test]
fn cached_join_owners_retain_stream_lifetime_and_destroy_under_captured_mr() {
    let sources = NativeSources::load();
    let combined = sources.combined();
    let mut violations = Vec::new();

    for (owner, first_gpu_member) in [
        ("distinct_hash_join_owner", "join"),
        ("hash_join_owner", "join"),
        ("filtered_join_owner", "observed_mr"),
        ("filtered_join_probe_owner", "observed_mr"),
        ("conditional_full_join_size_data_owner", "observed_mr"),
        ("mixed_join_size_data_owner", "observed_mr"),
    ] {
        let body = type_body(combined, owner);
        let stream_position = body.find("stream_lifetime");
        let mr_position = body.find("upstream_mr_lifetime");
        let gpu_member_position = body
            .find(first_gpu_member)
            .unwrap_or_else(|| panic!("expected {owner} to contain member `{first_gpu_member}`"));

        if stream_position.is_none() {
            violations.push(format!("{owner} must retain the creation stream lifetime"));
        }
        if mr_position.is_none() {
            violations.push(format!("{owner} must retain the creation memory-resource lifetime"));
        }
        if stream_position.is_some_and(|position| position > gpu_member_position) {
            violations.push(format!(
                "{owner} must declare stream_lifetime before GPU-owning members so it is destroyed after them"
            ));
        }
        if mr_position.is_some_and(|position| position > gpu_member_position) {
            violations.push(format!(
                "{owner} must declare upstream_mr_lifetime before GPU-owning members so it is destroyed after them"
            ));
        }
    }

    for function in [
        "cudf_distinct_hash_join_create_ex",
        "cudf_hash_join_create_ex",
        "conditional_full_join_size_data_impl",
        "mixed_join_size_data_impl",
        "filtered_join_create_impl",
        "filtered_join_begin_probe_impl",
    ] {
        let body = sources.function_body(function).body;
        if !source_contract_support::contains_normalized(body, "stream_lifetime = exec->stream_lifetime") {
            violations.push(format!("{function} must copy exec->stream_lifetime into the cached owner"));
        }
    }

    let destroy_helper = sources.function_body("cudf_destroy_under_provenance").body;
    // Device-domain model: destruction must not swap the device's current
    // resource. Retained GPU state releases through the resource refs captured
    // inside each buffer; the provenance only pins the device and lifetimes.
    if destroy_helper.contains("cudf_current_device_resource_guard") {
        violations.push(
            "cudf_destroy_under_provenance must not swap the device's current resource while destroying retained GPU state"
                .to_string(),
        );
    }
    if !source_contract_support::contains_normalized(destroy_helper, "std::forward<Callable>(cleanup)();") {
        violations
            .push("cudf_destroy_under_provenance must run cleanup inside the captured provenance scope".to_string());
    }

    assert!(
        violations.is_empty(),
        "cached join artifacts must retain CUDA stream/RMM lifetimes and destroy under captured MR.\n{}",
        violations.join("\n")
    );
}

#[test]
fn hash_join_create_preserves_pre_build_rejection_instead_of_building_unknown_retained_state() {
    let sources = NativeSources::load();
    let body = sources.function_body("cudf_hash_join_create_ex").body;
    let mut violations = Vec::new();

    if !source_contract_support::contains_normalized(
        body,
        "auto const reservation_size = cudf::hash_join::pre_build_reservation_size(*build_view, load_factor, exec->stream, *exec->mr);",
    ) {
        violations.push(
            "cudf_hash_join_create_ex must compute the retained-size contract before constructing the join".to_string(),
        );
    }
    if !source_contract_support::contains_normalized(body, "owner->device_allocated_size_bytes = reservation_size;") {
        violations.push("cudf_hash_join_create_ex must publish the exact pre-build reservation size".to_string());
    }
    for forbidden in ["reservation_size = std::nullopt", "value_or(0)", "catch (...)"] {
        if body.contains(forbidden) {
            violations.push(format!("cudf_hash_join_create_ex must not swallow pre-build rejection via `{forbidden}`"));
        }
    }

    assert!(
        violations.is_empty(),
        "hash join creation must not build retained state after libcudf rejects exact pre-build sizing.\n{}",
        violations.join("\n")
    );
}

#[test]
fn hash_join_pre_build_sizing_forwards_explicit_execution_context() {
    let sources = NativeSources::load();
    let body = sources.function_body("cudf_hash_join_pre_build_reservation_size_ex").body;

    assert!(
        source_contract_support::contains_normalized(
            body,
            "*out = cudf::hash_join::pre_build_reservation_size(*build_view, load_factor, exec->stream, *exec->mr);",
        ),
        "hash-join pre-build sizing must inspect nullable string metadata on the resolved stream/MR context"
    );
}

#[test]
fn arrow_device_array_release_carries_provenance_without_global_side_map() {
    let sources = NativeSources::load();
    let combined = sources.combined();
    let array_destroy = sources.function_body("cudf_arrow_device_array_destroy");
    let mut violations = Vec::new();

    if combined.contains("unordered_map<ArrowDeviceArray*, cudf_handle_provenance>") {
        violations.push(
            "ArrowDeviceArray provenance must not live in a global raw-pointer side map; \
             raw address reuse can attach stale provenance"
                .to_string(),
        );
    }
    if !combined.contains("release_arrow_device_array_under_provenance") {
        violations.push(
            "ArrowDeviceArray release must wrap the ArrowArray release callback so direct \
             Arrow ABI release uses captured provenance"
                .to_string(),
        );
    }
    if !array_destroy.body.contains("destroy_arrow_device_array") {
        violations.push("cudf_arrow_device_array_destroy must delegate to the Arrow release path".to_string());
    }

    assert!(
        violations.is_empty(),
        "Arrow device array release must carry captured provenance in the ABI object.\n{}",
        violations.join("\n")
    );
}

#[test]
fn destroy_provenance_cleanup_returns_status_and_never_terminates() {
    let sources = NativeSources::load();
    let combined = sources.combined();
    let mut violations = Vec::new();

    if combined.contains("std::terminate()") {
        violations.push("native teardown must never call std::terminate()".to_string());
    }
    if combined.contains("cudf_provenance_cleanup_result") {
        violations.push(
            "provenance teardown must expose cudf_status_t directly, not an internal completion enum".to_string(),
        );
    }

    let helper = sources.function_body("cudf_destroy_under_provenance").body;
    for marker in
        ["return injected", "return status", "return CUDF_SUCCESS", "return cudf_translate_cleanup_exception_noexcept"]
    {
        if !helper.contains(marker) {
            violations.push(format!("cudf_destroy_under_provenance must contain `{marker}`"));
        }
    }

    for function in GPU_OWNING_PUBLIC_DESTROY_FUNCTIONS {
        if !combined.contains(&format!("cudf_status_t {function}")) {
            violations.push(format!("{function} must expose cudf_status_t in the native ABI"));
        }
        let body = sources.function_body(function).body;
        if !body.contains("return") {
            violations.push(format!("{function} must return the observable teardown status"));
        }
    }

    assert!(
        violations.is_empty(),
        "destroy provenance cleanup must be status-returning and never terminate.\n{}",
        violations.join("\n")
    );
}

#[test]
fn prepared_replace_execute_consumes_failures_under_owner_provenance() {
    let sources = NativeSources::load();
    let execute = sources.function_body("cudf_strings_execute_replace_with_backrefs_ex").body;
    let cleanup = sources.function_body("destroy_consumed_prepared_replace_owner").body;
    let mut violations = Vec::new();

    if execute.contains("std::unique_ptr<prepared_replace_with_backrefs_owner>") {
        violations.push(
            "prepared replace execute must not install a default unique_ptr before execution options are validated"
                .to_string(),
        );
    }
    for marker in [
        "cleanup_consumed_prepared_replace_owner_before_return(owner, exec_status)",
        "exec->device != owner->provenance.device",
        "cleanup_consumed_prepared_replace_owner_before_return( owner, CUDF_ERROR_INVALID_OPERATION)",
    ] {
        if !source_contract_support::contains_normalized(execute, marker) {
            violations.push(format!("prepared replace execute must contain `{marker}`"));
        }
    }
    let mismatch = execute.find("exec->device != owner->provenance.device");
    let libcudf_execute = execute.find("cudf::strings::execute_replace_with_backrefs");
    if mismatch.zip(libcudf_execute).is_none_or(|(mismatch, execute)| mismatch >= execute) {
        violations.push("prepared replace execute must reject device mismatch before calling libcudf".to_string());
    }
    if !source_contract_support::contains_normalized(
        cleanup,
        "return cudf_destroy_under_provenance(provenance, [owner] { delete owner; });",
    ) {
        violations.push(
            "consumed prepared replace cleanup must destroy or quarantine through captured provenance".to_string(),
        );
    }

    assert!(
        violations.is_empty(),
        "prepared replace failure cleanup must be provenance-aware and consume exactly once.\n{}",
        violations.join("\n")
    );
}

#[test]
fn table_release_columns_preserves_existing_column_provenance_from_table_state() {
    let sources = NativeSources::load();
    let table_release = sources.function_body("cudf_table_release_columns");
    let native_table_release_columns = sources.method_body("native_table_handle_state", "release_columns");
    let mut violations = Vec::new();

    if !source_contract_support::contains_normalized(table_release.body, "cols = table_state.release_columns();") {
        violations.push(
            "cudf_table_release_columns must obtain column handle states from \
             the existing table state before publishing output handles"
                .to_string(),
        );
    }
    for marker in [
        "cudf_make_exec_context",
        "cudf_resolve_exec_options",
        "get_current_device_resource",
        "cudf_current_device_resource_guard",
        "cudf_current_device_resource_mutex",
    ] {
        if table_release.body.contains(marker) {
            violations
                .push(format!("cudf_table_release_columns must not recapture caller current MR/device via `{marker}`"));
        }
    }

    if !source_contract_support::contains_normalized(
        sources.combined(),
        "std::vector<std::unique_ptr<cudf_column_handle_state>> release_columns()",
    ) {
        violations.push("table release_columns() must return column handle states".to_string());
    }

    if !native_table_release_columns.contains("std::make_unique<native_column_handle_state>") {
        violations
            .push("native_table_handle_state::release_columns must rebuild native column handle states".to_string());
    }

    if !native_table_release_columns.contains("provenance") {
        violations.push(
            "native_table_handle_state::release_columns must pass retained table/column provenance \
             into each native_column_handle_state instead of recapturing caller current MR"
                .to_string(),
        );
    }

    if !native_table_release_columns.contains("cudf_invoke_under_provenance") {
        violations.push(
            "native_table_handle_state::release_columns must perform release-and-wrap work under \
             retained provenance so exception cleanup of raw cudf::column values uses the right context"
                .to_string(),
        );
    }

    assert!(
        violations.is_empty(),
        "table release_columns must preserve existing handle provenance.\n{}",
        violations.join("\n")
    );
}

#[test]
fn table_create_preserves_input_column_handle_provenance() {
    let sources = NativeSources::load();
    let table_create = sources.function_body("cudf_table_create");
    let mut violations = Vec::new();

    if table_create.body.contains("release_native_column") {
        violations.push(
            "cudf_table_create must not flatten input column states into a native table; \
             doing so loses each column handle's retained destroy provenance"
                .to_string(),
        );
    }

    if source_contract_support::contains_normalized(table_create.body, "cudf_release_table_handle(std::move(native))") {
        violations.push(
            "cudf_table_create must not re-release an all-native table by recapturing caller \
             current provenance"
                .to_string(),
        );
    }

    if !source_contract_support::contains_normalized(
        table_create.body,
        "cudf_make_table_handle_state_from_columns(expected_rows, std::move(states))",
    ) {
        violations.push(
            "cudf_table_create must retain the consumed column handle states in a table state \
             so later table destroy/release preserves their original provenance"
                .to_string(),
        );
    }

    assert!(
        violations.is_empty(),
        "table creation must preserve input column handle provenance.\n{}",
        violations.join("\n")
    );
}

struct NativeSources {
    files: Vec<NativeSource>,
    combined: String,
}

struct NativeSource {
    file_name: String,
    code: String,
}

struct FunctionRegion<'a> {
    file_name: &'a str,
    body: &'a str,
}

impl NativeSources {
    fn load() -> Self {
        let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("native/src");
        let mut paths = native_source_paths(&src_dir);
        paths.sort();

        let mut files = Vec::new();
        let mut combined = String::new();
        for path in paths {
            let source = fs::read_to_string(&path).expect("native source file must be readable");
            let code = strip_comments_and_literals(&source);
            let file_name = path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("native source file must have UTF-8 name")
                .to_string();
            combined.push_str("\n// ");
            combined.push_str(&file_name);
            combined.push('\n');
            combined.push_str(&code);
            files.push(NativeSource { file_name, code });
        }

        Self { files, combined }
    }

    fn combined(&self) -> &str {
        &self.combined
    }

    fn function_body(&self, name: &str) -> FunctionRegion<'_> {
        for source in &self.files {
            if let Some(body) = find_function_body(&source.code, name) {
                return FunctionRegion { file_name: &source.file_name, body };
            }
        }
        panic!("expected to find native function body `{name}` in cudf-nexus-sys/native/src");
    }

    fn method_body(&self, owner: &str, method: &str) -> &str {
        let owner_body = type_body(&self.combined, owner);
        if let Some(body) = find_function_body(owner_body, method) {
            return body;
        }

        let scoped_name = format!("{owner}::{method}");
        find_function_body(&self.combined, &scoped_name)
            .unwrap_or_else(|| panic!("expected method `{scoped_name}` to have a native body"))
    }
}

fn native_source_paths(src_dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(src_dir)
        .expect("native/src must be readable")
        .map(|entry| entry.expect("native/src entry must be readable").path())
        .filter(|path| {
            matches!(path.extension().and_then(|ext| ext.to_str()), Some("c") | Some("cpp") | Some("h") | Some("hpp"))
        })
        .collect()
}

fn strip_comments_and_literals(source: &str) -> String {
    #[derive(Clone, Copy)]
    enum State {
        Code,
        LineComment,
        BlockComment,
        StringLiteral { escaped: bool },
        CharLiteral { escaped: bool },
    }

    let mut out = String::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    let mut state = State::Code;

    while let Some(ch) = chars.next() {
        match state {
            State::Code => match ch {
                '/' if chars.peek() == Some(&'/') => {
                    chars.next();
                    out.push(' ');
                    out.push(' ');
                    state = State::LineComment;
                },
                '/' if chars.peek() == Some(&'*') => {
                    chars.next();
                    out.push(' ');
                    out.push(' ');
                    state = State::BlockComment;
                },
                '"' => {
                    out.push(' ');
                    state = State::StringLiteral { escaped: false };
                },
                '\'' => {
                    out.push(' ');
                    state = State::CharLiteral { escaped: false };
                },
                _ => out.push(ch),
            },
            State::LineComment => {
                if ch == '\n' {
                    out.push('\n');
                    state = State::Code;
                } else {
                    out.push(' ');
                }
            },
            State::BlockComment => {
                if ch == '*' && chars.peek() == Some(&'/') {
                    chars.next();
                    out.push(' ');
                    out.push(' ');
                    state = State::Code;
                } else if ch == '\n' {
                    out.push('\n');
                } else {
                    out.push(' ');
                }
            },
            State::StringLiteral { escaped } => {
                if ch == '\n' {
                    out.push('\n');
                    state = State::Code;
                } else {
                    out.push(' ');
                    state = match (ch, escaped) {
                        ('\\', false) => State::StringLiteral { escaped: true },
                        ('"', false) => State::Code,
                        _ => State::StringLiteral { escaped: false },
                    };
                }
            },
            State::CharLiteral { escaped } => {
                if ch == '\n' {
                    out.push('\n');
                    state = State::Code;
                } else {
                    out.push(' ');
                    state = match (ch, escaped) {
                        ('\\', false) => State::CharLiteral { escaped: true },
                        ('\'', false) => State::Code,
                        _ => State::CharLiteral { escaped: false },
                    };
                }
            },
        }
    }

    out
}

fn find_function_body<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let mut search_start = 0;
    while let Some(relative_position) = source[search_start..].find(name) {
        let name_start = search_start + relative_position;
        let name_end = name_start + name.len();
        search_start = name_end;

        if !has_identifier_boundaries(source, name_start, name_end) {
            continue;
        }

        let open_paren = skip_ascii_whitespace(source, name_end);
        if source.as_bytes().get(open_paren) != Some(&b'(') {
            continue;
        }
        let close_paren = matching_delimiter(source, open_paren, b'(', b')');
        let after_signature = skip_ascii_whitespace(source, close_paren + 1);
        let Some((body_start, _)) = source[after_signature..].char_indices().find(|(_, ch)| *ch == '{' || *ch == ';')
        else {
            continue;
        };
        let body_start = after_signature + body_start;
        if source.as_bytes()[body_start] == b';' {
            continue;
        }

        let body_end = matching_delimiter(source, body_start, b'{', b'}');
        return Some(&source[body_start + 1..body_end]);
    }

    None
}

fn type_body<'a>(source: &'a str, type_name: &str) -> &'a str {
    let start = ["class", "struct"]
        .iter()
        .find_map(|keyword| source.find(&format!("{keyword} {type_name}")))
        .unwrap_or_else(|| panic!("expected native owner/helper `{type_name}` to exist"));
    let open_brace = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("expected native owner/helper `{type_name}` to have a body"));
    let close_brace = matching_delimiter(source, open_brace, b'{', b'}');
    &source[open_brace + 1..close_brace]
}

fn matching_delimiter(source: &str, open_position: usize, open: u8, close: u8) -> usize {
    assert_eq!(
        source.as_bytes().get(open_position),
        Some(&open),
        "matching_delimiter must start at the opening delimiter"
    );

    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open_position..].iter().enumerate() {
        if *byte == open {
            depth += 1;
        } else if *byte == close {
            depth -= 1;
            if depth == 0 {
                return open_position + offset;
            }
        }
    }

    panic!("unmatched delimiter in native source");
}

fn has_identifier_boundaries(source: &str, start: usize, end: usize) -> bool {
    let before = start.checked_sub(1).and_then(|index| source.as_bytes().get(index)).copied();
    let after = source.as_bytes().get(end).copied();
    !before.is_some_and(is_identifier_byte) && !after.is_some_and(is_identifier_byte)
}

fn skip_ascii_whitespace(source: &str, start: usize) -> usize {
    source[start..]
        .char_indices()
        .find_map(|(offset, ch)| (!ch.is_ascii_whitespace()).then_some(start + offset))
        .unwrap_or(source.len())
}

fn destroy_provenance_identifiers(source: &str) -> BTreeSet<String> {
    identifiers(source)
        .filter(|identifier| identifier.contains("destroy") && identifier.contains("provenance"))
        .map(ToString::to_string)
        .collect()
}

fn identifiers(source: &str) -> impl Iterator<Item = &str> {
    IdentifierIter { source, offset: 0 }
}

struct IdentifierIter<'a> {
    source: &'a str,
    offset: usize,
}

impl<'a> Iterator for IdentifierIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.source.as_bytes();
        while self.offset < bytes.len() && !is_identifier_start_byte(bytes[self.offset]) {
            self.offset += 1;
        }
        if self.offset >= bytes.len() {
            return None;
        }

        let start = self.offset;
        self.offset += 1;
        while self.offset < bytes.len() && is_identifier_byte(bytes[self.offset]) {
            self.offset += 1;
        }

        Some(&self.source[start..self.offset])
    }
}

fn is_identifier_start_byte(byte: u8) -> bool {
    byte == b'_' || byte.is_ascii_alphabetic()
}

fn is_identifier_byte(byte: u8) -> bool {
    is_identifier_start_byte(byte) || byte.is_ascii_digit()
}

fn contains_markers_near(source: &str, left: &str, right: &str, max_distance: usize) -> bool {
    let mut search_start = 0;
    while let Some(relative_position) = source[search_start..].find(left) {
        let position = search_start + relative_position;
        let window_start = position.saturating_sub(max_distance);
        let window_end = (position + left.len() + max_distance).min(source.len());
        if source[window_start..window_end].contains(right) {
            return true;
        }
        search_start = position + left.len();
    }
    false
}

fn has_signature_with_marker(source: &str, function_name: &str, marker: &str) -> bool {
    let mut search_start = 0;
    while let Some(relative_position) = source[search_start..].find(function_name) {
        let name_start = search_start + relative_position;
        let name_end = name_start + function_name.len();
        search_start = name_end;

        if !has_identifier_boundaries(source, name_start, name_end) {
            continue;
        }

        let Some(signature_end_offset) =
            source[name_end..].char_indices().find_map(|(offset, ch)| (ch == ';' || ch == '{').then_some(offset))
        else {
            continue;
        };
        let signature = &source[name_start..name_end + signature_end_offset];
        if signature.contains(marker) {
            return true;
        }
    }
    false
}
