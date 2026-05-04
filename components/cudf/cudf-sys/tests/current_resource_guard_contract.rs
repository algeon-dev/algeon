use std::{collections::BTreeSet, fs, path::Path};

const EXPECTED_GUARDS: &[&str] = &[
    "copying::gather",
    "copying::scatter",
    "column::list_sliced_child",
    "dictionary::decode",
    "dictionary::encode",
    "dictionary::match_dictionaries",
    "dictionary::set_keys",
    "filter::apply_boolean_mask",
    "filter::row_predicate_final_apply_boolean_mask",
    "groupby::aggregate",
    "io_csv::write_csv",
    "io_json::write_json",
    "io_parquet::write_parquet",
    "join::distinct_hash_join_create",
    "join::distinct_hash_join_destroy",
    "join::distinct_hash_join_left_join",
    "join::cross_join",
    "join::left_anti_join_indices",
    "join::left_semi_join_indices",
    "join::conditional_pair_join_indices",
    "join::conditional_single_side_join_indices",
    "join::hash_join_create",
    "join::hash_join_destroy",
    "join::hash_join_full_join",
    "join::hash_join_inner_join",
    "join::hash_join_left_join",
    "join::full_join_indices",
    "join::inner_join_indices",
    "join::left_join_indices",
    "join::mixed_pair_join_indices",
    "join::mixed_single_side_join_indices",
    "replace::replace_nulls_policy",
    "sorting::is_sorted",
    "sorting::sort",
    "sorting::sort_by_key",
    "sorting::sorted_order",
    "sorting::stable_sort",
    "sorting::stable_sort_by_key",
    "stream_compaction::distinct",
    "stream_compaction::drop_nulls",
    "strings::like",
    "strings::slice",
];

#[test]
fn current_device_resource_guard_stays_on_explicit_allowlist() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("native/src");
    let mut seen = BTreeSet::new();
    let mut violations = Vec::new();

    for entry in fs::read_dir(&src_dir).expect("native/src must be readable") {
        let path = entry.expect("native/src entry must be readable").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("cpp") {
            continue;
        }

        let source = fs::read_to_string(&path).expect("native source file must be readable");
        let lines: Vec<_> = source.lines().collect();
        for (index, line) in lines.iter().enumerate() {
            if !line.contains("cudf_current_device_resource_guard current_mr") {
                continue;
            }

            let file_name =
                path.file_name().and_then(|name| name.to_str()).expect("native source file must have UTF-8 name");
            let context = context_around(&lines, index);
            match classify_allowed_guard(file_name, &context) {
                Some(label) if seen.insert(label) => {},
                Some(label) => violations.push(format!(
                    "{}:{} duplicate allowlisted guard context `{}`",
                    file_name,
                    index + 1,
                    label
                )),
                None => violations.push(format!(
                    "{}:{} guard is outside the explicit allowlist:\n{}",
                    file_name,
                    index + 1,
                    context
                )),
            }
        }
    }

    let expected: BTreeSet<_> = EXPECTED_GUARDS.iter().copied().collect();
    let missing: Vec<_> = expected.difference(&seen).copied().collect();
    if !violations.is_empty() || !missing.is_empty() {
        panic!(
            "cudf_current_device_resource_guard must stay scoped to libcudf APIs without an \
             explicit memory_resource parameter, or APIs proven to use RMM's current resource \
             for temporaries despite an explicit memory_resource argument.\nviolations:\n{}\nmissing expected guards: {:?}",
            violations.join("\n\n"),
            missing
        );
    }
}

fn context_around(lines: &[&str], index: usize) -> String {
    let start = index.saturating_sub(14);
    let end = (index + 8).min(lines.len());
    lines[start..end].join("\n")
}

fn classify_allowed_guard(file_name: &str, context: &str) -> Option<&'static str> {
    match file_name {
        "copying.cpp" if context.contains("cudf::gather(*view, *map_view, policy, exec.stream, exec.mr)") => {
            Some("copying::gather")
        },
        "copying.cpp"
            if context.contains("cudf::scatter(*source_view, *map_view, *target_view, exec.stream, exec.mr)") =>
        {
            Some("copying::scatter")
        },
        "column.cpp" if context.contains("lists.get_sliced_child(exec.stream)") => Some("column::list_sliced_child"),
        "dictionary.cpp" if context.contains("cudf::dictionary::decode(dict_view, exec.stream, exec.mr)") => {
            Some("dictionary::decode")
        },
        "dictionary.cpp" if context.contains("cudf::dictionary::encode(*view, indices_type, exec.stream, exec.mr)") => {
            Some("dictionary::encode")
        },
        "dictionary.cpp"
            if context.contains("cudf::dictionary::match_dictionaries(dict_views, exec.stream, exec.mr)") =>
        {
            Some("dictionary::match_dictionaries")
        },
        "dictionary.cpp"
            if context.contains("cudf::dictionary::set_keys(dict_view, *keys_view, exec.stream, exec.mr)") =>
        {
            Some("dictionary::set_keys")
        },
        "filter.cpp" if context.contains("cudf::apply_boolean_mask(*view, *mask_view, exec.stream, exec.mr)") => {
            Some("filter::apply_boolean_mask")
        },
        "filter.cpp" if context.contains("cudf::apply_boolean_mask(*view, mask->view(), exec.stream, exec.mr)") => {
            Some("filter::row_predicate_final_apply_boolean_mask")
        },
        "groupby.cpp" if context.contains("cudf::groupby::groupby gb(*keys_view, null_policy)") => {
            Some("groupby::aggregate")
        },
        "io_csv.cpp" if context.contains("cudf::io::write_csv(options, exec.stream)") => Some("io_csv::write_csv"),
        "io_json.cpp" if context.contains("cudf::io::write_json(options, exec.stream)") => Some("io_json::write_json"),
        "io_parquet.cpp" if context.contains("cudf::io::write_parquet(options, exec.stream)") => {
            Some("io_parquet::write_parquet")
        },
        "join.cpp" if context.contains("std::make_unique<cudf::distinct_hash_join>") => {
            Some("join::distinct_hash_join_create")
        },
        "join.cpp" if context.contains("void cudf_distinct_hash_join_destroy") && context.contains("delete owner") => {
            Some("join::distinct_hash_join_destroy")
        },
        "join.cpp" if context.contains("distinct_join->left_join(*probe_view, exec.stream, exec.mr)") => {
            Some("join::distinct_hash_join_left_join")
        },
        "join.cpp" if context.contains("cudf::cross_join(*left_view, *right_view, exec.stream, exec.mr)") => {
            Some("join::cross_join")
        },
        "join.cpp" if context.contains("std::make_unique<cudf::hash_join>") => Some("join::hash_join_create"),
        "join.cpp" if context.contains("void cudf_hash_join_destroy") && context.contains("delete owner") => {
            Some("join::hash_join_destroy")
        },
        "join.cpp" if context.contains("cudf::full_join(*lk, *rk, null_eq, exec.stream, exec.mr)") => {
            Some("join::full_join_indices")
        },
        "join.cpp" if context.contains("cudf::inner_join(*lk, *rk, null_eq, exec.stream, exec.mr)") => {
            Some("join::inner_join_indices")
        },
        "join.cpp" if context.contains("cudf::left_join(*lk, *rk, null_eq, exec.stream, exec.mr)") => {
            Some("join::left_join_indices")
        },
        "join.cpp" if context.contains("join.anti_join(*lk, exec.stream, exec.mr)") => {
            Some("join::left_anti_join_indices")
        },
        "join.cpp" if context.contains("join.semi_join(*lk, exec.stream, exec.mr)") => {
            Some("join::left_semi_join_indices")
        },
        "join.cpp" if context.contains("cudf::conditional_inner_join(") => Some("join::conditional_pair_join_indices"),
        "join.cpp" if context.contains("cudf::conditional_left_semi_join(") => {
            Some("join::conditional_single_side_join_indices")
        },
        "join.cpp" if context.contains("cudf::mixed_inner_join(") => Some("join::mixed_pair_join_indices"),
        "join.cpp" if context.contains("cudf::mixed_left_semi_join(") => Some("join::mixed_single_side_join_indices"),
        "join.cpp" if context.contains("hash_join->full_join(*probe_view, {}, exec.stream, exec.mr)") => {
            Some("join::hash_join_full_join")
        },
        "join.cpp" if context.contains("hash_join->inner_join(*probe_view, {}, exec.stream, exec.mr)") => {
            Some("join::hash_join_inner_join")
        },
        "join.cpp" if context.contains("hash_join->left_join(*probe_view, {}, exec.stream, exec.mr)") => {
            Some("join::hash_join_left_join")
        },
        "replace.cpp" if context.contains("cudf::replace_nulls(*view, rp, exec.stream, exec.mr)") => {
            Some("replace::replace_nulls_policy")
        },
        "sorting.cpp" if context.contains("cudf::is_sorted(*view, orders, null_orders, exec.stream)") => {
            Some("sorting::is_sorted")
        },
        "sorting.cpp" if context.contains("cudf::sort(*view, orders, null_orders, exec.stream, exec.mr)") => {
            Some("sorting::sort")
        },
        "sorting.cpp"
            if context.contains("cudf::sort_by_key(*vals, *ks, orders, null_orders, exec.stream, exec.mr)") =>
        {
            Some("sorting::sort_by_key")
        },
        "sorting.cpp" if context.contains("cudf::sorted_order(*view, orders, null_orders, exec.stream, exec.mr)") => {
            Some("sorting::sorted_order")
        },
        "sorting.cpp" if context.contains("cudf::stable_sort(*view, orders, null_orders, exec.stream, exec.mr)") => {
            Some("sorting::stable_sort")
        },
        "sorting.cpp"
            if context.contains("cudf::stable_sort_by_key(*vals, *ks, orders, null_orders, exec.stream, exec.mr)") =>
        {
            Some("sorting::stable_sort_by_key")
        },
        "stream_compaction.cpp"
            if context
                .contains("cudf::distinct(*view, key_vec, keep_opt, nulls_eq, nans_eq, exec.stream, exec.mr)") =>
        {
            Some("stream_compaction::distinct")
        },
        "stream_compaction.cpp" if context.contains("cudf::drop_nulls(") => Some("stream_compaction::drop_nulls"),
        "strings.cpp" if context.contains("cudf::strings::like(") => Some("strings::like"),
        "strings.cpp" if context.contains("cudf::strings::slice_strings(") => Some("strings::slice"),
        _ => None,
    }
}
