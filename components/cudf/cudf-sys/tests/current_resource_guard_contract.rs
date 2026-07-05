use std::{collections::BTreeSet, fs, path::Path};

const EXPECTED_GUARDS: &[&str] = &[
    "binary::binary_operation_sv",
    "binary::binary_operation_vs",
    "binary::binary_operation_vv",
    "concatenate::columns",
    "concatenate::tables",
    "copying::copy_if_else",
    "copying::gather",
    "copying::gather_device_allocated_size",
    "copying::scatter",
    "column::copy_bitmask",
    "column::from_scalar",
    "column::list_sliced_child",
    "datetime::add_calendrical_months_scalar",
    "datetime::extract_component",
    "datetime::floor",
    "dictionary::decode",
    "dictionary::encode",
    "dictionary::match_dictionaries",
    "dictionary::set_keys",
    "filter::apply_boolean_mask",
    "filter::row_predicate_compute_mask_and_final_apply_boolean_mask",
    "groupby::aggregate",
    "groupby::scan",
    "hashing::murmurhash3_x86_32",
    "hashing::xxhash_64",
    "io_csv::read_csv",
    "io_csv::write_csv",
    "io_json::read_json",
    "io_json::read_json_with_metadata",
    "io_json::write_json",
    "io_parquet::read_parquet",
    "io_parquet::read_parquet_with_metadata",
    "io_parquet::read_parquet_with_metadata_filter",
    "io_parquet::write_parquet",
    "join::distinct_hash_join_create",
    "join::distinct_hash_join_left_join",
    "join::cross_join",
    "join::projected_cross_join",
    "join::projected_cross_join_device_allocated_size",
    "join::left_anti_join_indices",
    "join::left_semi_join_indices",
    "join::conditional_pair_join_indices",
    "join::conditional_single_side_join_indices",
    "join::conditional_join_size",
    "join::conditional_full_join_size_data_create",
    "join::conditional_full_join_size_data_materialize_indices",
    "join::hash_join_create",
    "join::hash_join_full_join",
    "join::hash_join_full_join_with_output_size",
    "join::hash_join_inner_join",
    "join::hash_join_inner_join_with_output_size",
    "join::hash_join_left_join",
    "join::hash_join_left_join_with_output_size",
    "join::hash_join_size",
    "join::full_join",
    "join::full_join_indices",
    "join::inner_join",
    "join::inner_join_indices",
    "join::left_join",
    "join::left_join_indices",
    "join::filter_join_indices",
    "join::filtered_join_begin_probe",
    "join::filtered_join_create",
    "join::filtered_join_indices_with_output_size",
    "join::filtered_join_probe_materialize_indices",
    "join::filtered_join_size",
    "join::mixed_pair_join_indices",
    "join::mixed_join_size_data_create",
    "join::mixed_join_size_data_materialize_pair_indices",
    "join::mixed_join_size_data_materialize_single_indices",
    "join::mixed_single_side_join_indices",
    "partitioning::hash_partition",
    "prepared_import::column_copy_from_prepared_device_export",
    "prepared_import::table_copy_from_prepared_device_export",
    "reduction::minmax",
    "reduction::reduce",
    "reduction::scan",
    "replace::replace_nulls_column",
    "replace::replace_nulls_policy",
    "replace::replace_nulls_scalar",
    "search::contains",
    "sorting::is_sorted",
    "sorting::sort",
    "sorting::sort_by_key",
    "sorting::sorted_order",
    "sorting::stable_sort",
    "sorting::stable_sort_by_key",
    "stream_compaction::distinct",
    "stream_compaction::drop_nulls",
    "strings::contains",
    "strings::concatenate",
    "strings::count_bytes",
    "strings::count_characters",
    "strings::ends_with",
    "strings::like",
    "strings::replace_with_backrefs",
    "strings::slice",
    "strings::starts_with",
    "strings::strip",
    "strings::to_lower",
    "strings::to_upper",
    "strings::try_to_integers",
    "unary::cast",
    "unary::is_nan",
    "unary::is_not_nan",
    "unary::is_null",
    "unary::is_valid",
    "unary::round_half_up",
    "unary::unary_operation",
];

#[test]
fn current_device_resource_guard_stays_on_explicit_allowlist() {
    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("native/src");
    let mut seen = BTreeSet::new();
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
            if !line.contains("cudf_current_device_resource_guard current_mr") {
                continue;
            }

            let file_name =
                path.file_name().and_then(|name| name.to_str()).expect("native source file must have UTF-8 name");
            let context = context_around(&lines, index);
            let normalized_context = context.replace("*exec->mr", "exec->mr");
            match classify_allowed_guard(file_name, &normalized_context) {
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

fn is_join_source(file_name: &str) -> bool {
    matches!(
        file_name,
        "join_common.hpp" | "join_conditional.cpp" | "join_filter.cpp" | "join_hash.cpp" | "join_mixed.cpp"
    )
}

fn contains_normalized(haystack: &str, needle: &str) -> bool {
    let normalized_haystack = haystack.split_whitespace().collect::<Vec<_>>().join(" ");
    let normalized_needle = needle.split_whitespace().collect::<Vec<_>>().join(" ");
    normalized_haystack.contains(&normalized_needle)
}

fn classify_allowed_guard(file_name: &str, context: &str) -> Option<&'static str> {
    match file_name {
        "binary.cpp"
            if context
                .contains("cudf::binary_operation(*lhs_view, *rhs_view, cudf_op, *dtype, exec->stream, exec->mr)") =>
        {
            Some("binary::binary_operation_vv")
        },
        "binary.cpp"
            if context
                .contains("cudf::binary_operation(*lhs_view, rhs_scalar, cudf_op, *dtype, exec->stream, exec->mr)") =>
        {
            Some("binary::binary_operation_vs")
        },
        "binary.cpp"
            if context
                .contains("cudf::binary_operation(lhs_scalar, *rhs_view, cudf_op, *dtype, exec->stream, exec->mr)") =>
        {
            Some("binary::binary_operation_sv")
        },
        "concatenate.cpp"
            if context.contains("cudf::concatenate(views, exec->stream, exec->mr)")
                && context.contains("cudf_release_column_handle") =>
        {
            Some("concatenate::columns")
        },
        "concatenate.cpp"
            if context.contains("cudf::concatenate(views, exec->stream, exec->mr)")
                && context.contains("cudf_release_table_handle") =>
        {
            Some("concatenate::tables")
        },
        "copying.cpp" if context.contains("cudf::gather(*view, *map_view, policy, exec->stream, exec->mr)") => {
            if context.contains("result->alloc_size()") {
                Some("copying::gather_device_allocated_size")
            } else {
                Some("copying::gather")
            }
        },
        "copying.cpp"
            if context.contains("cudf::scatter(*source_view, *map_view, *target_view, exec->stream, exec->mr)") =>
        {
            Some("copying::scatter")
        },
        "copying.cpp"
            if contains_normalized(
                context,
                "cudf::copy_if_else(*lhs_view, *rhs_view, *mask_view, exec->stream, exec->mr)",
            ) =>
        {
            Some("copying::copy_if_else")
        },
        "copying.cpp" if context.contains("cudf::shift(*input_view,") => Some("copying::shift"),
        "column.cpp" if contains_normalized(context, "cudf::copy_bitmask(*view, exec->stream, exec->mr)") => {
            Some("column::copy_bitmask")
        },
        "column.cpp" if context.contains("cudf::make_column_from_scalar(") => Some("column::from_scalar"),
        "column.cpp" if contains_normalized(context, "lists.get_sliced_child(exec->stream)") => {
            Some("column::list_sliced_child")
        },
        "datetime.cpp" if context.contains("cudf::datetime::extract_datetime_component(") => {
            Some("datetime::extract_component")
        },
        "datetime.cpp" if context.contains("cudf::datetime::floor_datetimes(") => Some("datetime::floor"),
        "datetime.cpp" if context.contains("cudf::datetime::add_calendrical_months(") => {
            Some("datetime::add_calendrical_months_scalar")
        },
        "dictionary.cpp" if context.contains("cudf::dictionary::decode(dict_view, exec->stream, exec->mr)") => {
            Some("dictionary::decode")
        },
        "dictionary.cpp"
            if context.contains("cudf::dictionary::encode(*view, indices_type, exec->stream, exec->mr)") =>
        {
            Some("dictionary::encode")
        },
        "dictionary.cpp"
            if context.contains("cudf::dictionary::match_dictionaries(dict_views, exec->stream, exec->mr)") =>
        {
            Some("dictionary::match_dictionaries")
        },
        "dictionary.cpp"
            if context.contains("cudf::dictionary::set_keys(dict_view, *keys_view, exec->stream, exec->mr)") =>
        {
            Some("dictionary::set_keys")
        },
        "filter.cpp" if context.contains("cudf::apply_boolean_mask(*view, *mask_view, exec->stream, exec->mr)") => {
            Some("filter::apply_boolean_mask")
        },
        "filter.cpp"
            if context.contains(
                "cudf::compute_column(*view, resolve_operand(lowered, root_index), exec->stream, exec->mr)",
            ) =>
        {
            Some("filter::row_predicate_compute_mask_and_final_apply_boolean_mask")
        },
        "groupby.cpp" if context.contains("cudf::groupby::groupby gb(*keys_view, null_policy)") => {
            Some("groupby::aggregate")
        },
        "groupby.cpp" if context.contains("cudf::groupby::scan_request") => Some("groupby::scan"),
        "groupby.cpp" if context.contains("gb.shift(") => Some("groupby::shift"),
        "hashing.cpp" if context.contains("cudf::hashing::murmurhash3_x86_32(") => Some("hashing::murmurhash3_x86_32"),
        "hashing.cpp" if context.contains("cudf::hashing::xxhash_64(") => Some("hashing::xxhash_64"),
        "io_csv.cpp" if contains_normalized(context, "cudf::io::read_csv(options, exec->stream, exec->mr)") => {
            Some("io_csv::read_csv")
        },
        "io_csv.cpp" if contains_normalized(context, "cudf::io::write_csv(options, exec->stream)") => {
            Some("io_csv::write_csv")
        },
        "io_json.cpp"
            if contains_normalized(
                context,
                "auto [tbl, metadata] = cudf::io::read_json(options, exec->stream, exec->mr)",
            ) =>
        {
            Some("io_json::read_json")
        },
        "io_json.cpp"
            if contains_normalized(context, "result = cudf::io::read_json(options, exec->stream, exec->mr)") =>
        {
            Some("io_json::read_json_with_metadata")
        },
        "io_json.cpp" if contains_normalized(context, "cudf::io::write_json(options, exec->stream)") => {
            Some("io_json::write_json")
        },
        "io_parquet.cpp"
            if contains_normalized(
                context,
                "auto [tbl, metadata] = cudf::io::read_parquet(options, exec->stream, exec->mr)",
            ) =>
        {
            Some("io_parquet::read_parquet")
        },
        "io_parquet.cpp"
            if contains_normalized(context, "result = cudf::io::read_parquet(options, exec->stream, exec->mr)")
                && context.contains("builder.filter") =>
        {
            Some("io_parquet::read_parquet_with_metadata_filter")
        },
        "io_parquet.cpp"
            if contains_normalized(context, "result = cudf::io::read_parquet(options, exec->stream, exec->mr)") =>
        {
            Some("io_parquet::read_parquet_with_metadata")
        },
        "io_parquet.cpp" if contains_normalized(context, "cudf::io::write_parquet(options, exec->stream)") => {
            Some("io_parquet::write_parquet")
        },
        file_name if is_join_source(file_name) && context.contains("std::make_unique<cudf::distinct_hash_join>") => {
            Some("join::distinct_hash_join_create")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("distinct_join->left_join(*probe_view, exec->stream, exec->mr)") =>
        {
            Some("join::distinct_hash_join_left_join")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("cudf::cross_join(*left_view, *right_view, exec->stream, exec->mr)") =>
        {
            Some("join::cross_join")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("projected_cross_join_table(")
                && context.contains("cudf_release_table_handle") =>
        {
            Some("join::projected_cross_join")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("projected_cross_join_table(")
                && context.contains("result->alloc_size()") =>
        {
            Some("join::projected_cross_join_device_allocated_size")
        },
        file_name if is_join_source(file_name) && context.contains("std::make_unique<cudf::hash_join>") => {
            Some("join::hash_join_create")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("cudf::full_join(*lk, *rk, null_eq, exec->stream, exec->mr)")
                && context.contains("gather_table(") =>
        {
            Some("join::full_join")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("cudf::full_join(*lk, *rk, null_eq, exec->stream, exec->mr)") =>
        {
            Some("join::full_join_indices")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("cudf::inner_join(*lk, *rk, null_eq, exec->stream, exec->mr)")
                && context.contains("gather_table(") =>
        {
            Some("join::inner_join")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("cudf::inner_join(*lk, *rk, null_eq, exec->stream, exec->mr)") =>
        {
            Some("join::inner_join_indices")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("cudf::left_join(*lk, *rk, null_eq, exec->stream, exec->mr)")
                && context.contains("gather_table(") =>
        {
            Some("join::left_join")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("cudf::left_join(*lk, *rk, null_eq, exec->stream, exec->mr)") =>
        {
            Some("join::left_join_indices")
        },
        file_name if is_join_source(file_name) && context.contains("join.anti_join(*lk, exec->stream, exec->mr)") => {
            Some("join::left_anti_join_indices")
        },
        file_name if is_join_source(file_name) && context.contains("join.semi_join(*lk, exec->stream, exec->mr)") => {
            Some("join::left_semi_join_indices")
        },
        file_name if is_join_source(file_name) && context.contains("cudf::conditional_inner_join(") => {
            Some("join::conditional_pair_join_indices")
        },
        file_name if is_join_source(file_name) && context.contains("cudf::conditional_left_semi_join(") => {
            Some("join::conditional_single_side_join_indices")
        },
        file_name if is_join_source(file_name) && context.contains("cudf::conditional_inner_join_size(") => {
            Some("join::conditional_join_size")
        },
        file_name if is_join_source(file_name) && context.contains("cudf::conditional_full_join_size(") => {
            Some("join::conditional_full_join_size_data_create")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("conditional full join size data missing state")
                && contains_normalized(context, "owner->size_data->materialize_indices(exec->stream, exec->mr)") =>
        {
            Some("join::conditional_full_join_size_data_materialize_indices")
        },
        file_name if is_join_source(file_name) && context.contains("cudf::filter_join_indices(") => {
            Some("join::filter_join_indices")
        },
        file_name if is_join_source(file_name) && context.contains("std::make_unique<cudf::filtered_join>") => {
            Some("join::filtered_join_create")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("*out = indices->size()")
                && context.contains("owner->join->semi_join") =>
        {
            Some("join::filtered_join_size")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("filtered join output_size does not match produced index count")
                && context.contains("owner->join->semi_join") =>
        {
            Some("join::filtered_join_indices_with_output_size")
        },
        file_name if is_join_source(file_name) && context.contains("owner->join->begin_left_semi_probe") => {
            Some("join::filtered_join_begin_probe")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("filtered join probe is null")
                && contains_normalized(context, "owner->probe->materialize_indices(exec->stream, exec->mr)") =>
        {
            Some("join::filtered_join_probe_materialize_indices")
        },
        file_name if is_join_source(file_name) && context.contains("cudf::mixed_inner_join(") => {
            Some("join::mixed_pair_join_indices")
        },
        file_name if is_join_source(file_name) && context.contains("cudf::mixed_left_semi_join(") => {
            Some("join::mixed_single_side_join_indices")
        },
        file_name if is_join_source(file_name) && context.contains("cudf::mixed_inner_join_size(") => {
            Some("join::mixed_join_size_data_create")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("join_index_pair indices")
                && context.contains("mixed join size data missing left equality view") =>
        {
            Some("join::mixed_join_size_data_materialize_pair_indices")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("mixed semi/anti join size data missing state")
                && contains_normalized(
                    context,
                    "owner->single_size_data->materialize_indices(exec->stream, exec->mr)",
                ) =>
        {
            Some("join::mixed_join_size_data_materialize_single_indices")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("hash_join->full_join(*probe_view, {}, exec->stream, exec->mr)") =>
        {
            Some("join::hash_join_full_join")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("hash_join->full_join(")
                && context.contains("std::optional<std::size_t>{output_size}, exec->stream, exec->mr") =>
        {
            Some("join::hash_join_full_join_with_output_size")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("hash_join->inner_join(*probe_view, {}, exec->stream, exec->mr)") =>
        {
            Some("join::hash_join_inner_join")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("hash_join->inner_join(")
                && context.contains("std::optional<std::size_t>{output_size}, exec->stream, exec->mr") =>
        {
            Some("join::hash_join_inner_join_with_output_size")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("hash_join->left_join(*probe_view, {}, exec->stream, exec->mr)") =>
        {
            Some("join::hash_join_left_join")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("hash_join->left_join(")
                && context.contains("std::optional<std::size_t>{output_size}, exec->stream, exec->mr") =>
        {
            Some("join::hash_join_left_join_with_output_size")
        },
        file_name
            if is_join_source(file_name)
                && context.contains("hash_join->inner_join_size(*probe_view, exec->stream)") =>
        {
            Some("join::hash_join_size")
        },
        "partitioning.cpp"
            if context.contains("cudf::hash_partition(")
                && context.contains("cudf::hash_id::HASH_MURMUR3")
                && context.contains("exec->stream,")
                && context.contains("exec->mr") =>
        {
            Some("partitioning::hash_partition")
        },
        "prepared_import.cpp" if context.contains("std::make_unique<cudf::column>(view, exec->stream, exec->mr)") => {
            Some("prepared_import::column_copy_from_prepared_device_export")
        },
        "prepared_import.cpp"
            if context.contains("std::make_unique<cudf::table>(cudf::table_view(columns),")
                && context.contains("exec->stream, exec->mr") =>
        {
            Some("prepared_import::table_copy_from_prepared_device_export")
        },
        "reduction.cpp" if context.contains("cudf::reduce(*view, *reduce_agg, *dtype, exec->stream, exec->mr)") => {
            Some("reduction::reduce")
        },
        "reduction.cpp" if context.contains("cudf::minmax(*view, exec->stream, exec->mr)") => Some("reduction::minmax"),
        "reduction.cpp"
            if context
                .contains("cudf::scan(*view, *scan_agg, cudf_scan_type, cudf_null_policy, exec->stream, exec->mr)") =>
        {
            Some("reduction::scan")
        },
        "replace.cpp" if contains_normalized(context, "cudf::replace_nulls(*view, scalar, exec->stream, exec->mr)") => {
            Some("replace::replace_nulls_scalar")
        },
        "replace.cpp"
            if contains_normalized(
                context,
                "cudf::replace_nulls(*view_input, *view_replacement, exec->stream, exec->mr)",
            ) =>
        {
            Some("replace::replace_nulls_column")
        },
        "replace.cpp" if context.contains("cudf::replace_nulls(*view, rp, exec->stream, exec->mr)") => {
            Some("replace::replace_nulls_policy")
        },
        "search.cpp" if context.contains("cudf::contains(*haystack_view, *needles_view, exec->stream, exec->mr)") => {
            Some("search::contains")
        },
        "sorting.cpp" if contains_normalized(context, "cudf::is_sorted(*view, orders, null_orders, exec->stream)") => {
            Some("sorting::is_sorted")
        },
        "sorting.cpp"
            if contains_normalized(context, "cudf::sort(*view, orders, null_orders, exec->stream, exec->mr)") =>
        {
            Some("sorting::sort")
        },
        "sorting.cpp"
            if contains_normalized(
                context,
                "cudf::sort_by_key(*vals, *ks, orders, null_orders, exec->stream, exec->mr)",
            ) =>
        {
            Some("sorting::sort_by_key")
        },
        "sorting.cpp"
            if contains_normalized(
                context,
                "cudf::sorted_order(*view, orders, null_orders, exec->stream, exec->mr)",
            ) =>
        {
            Some("sorting::sorted_order")
        },
        "sorting.cpp"
            if contains_normalized(
                context,
                "cudf::stable_sort(*view, orders, null_orders, exec->stream, exec->mr)",
            ) =>
        {
            Some("sorting::stable_sort")
        },
        "sorting.cpp"
            if contains_normalized(
                context,
                "cudf::stable_sort_by_key(*vals, *ks, orders, null_orders, exec->stream, exec->mr)",
            ) =>
        {
            Some("sorting::stable_sort_by_key")
        },
        "stream_compaction.cpp"
            if context
                .contains("cudf::distinct(*view, key_vec, keep_opt, nulls_eq, nans_eq, exec->stream, exec->mr)") =>
        {
            Some("stream_compaction::distinct")
        },
        "stream_compaction.cpp" if context.contains("cudf::drop_nulls(") => Some("stream_compaction::drop_nulls"),
        "unary.cpp" if context.contains("cudf::unary_operation(*view, cudf_op, exec->stream, exec->mr)") => {
            Some("unary::unary_operation")
        },
        "unary.cpp"
            if context.contains("cudf::round(*view, decimal_places, method, exec->stream, exec->mr)")
                || contains_normalized(
                    context,
                    "cudf::round_decimal(*view, decimal_places, method, exec->stream, exec->mr)",
                ) =>
        {
            Some("unary::round_half_up")
        },
        "unary.cpp" if context.contains("cudf::is_null(*view, exec->stream, exec->mr)") => Some("unary::is_null"),
        "unary.cpp" if context.contains("cudf::is_valid(*view, exec->stream, exec->mr)") => Some("unary::is_valid"),
        "unary.cpp" if context.contains("cudf::cast(*view, *dtype, exec->stream, exec->mr)") => Some("unary::cast"),
        "unary.cpp" if context.contains("cudf::is_nan(*view, exec->stream, exec->mr)") => Some("unary::is_nan"),
        "unary.cpp" if context.contains("cudf::is_not_nan(*view, exec->stream, exec->mr)") => Some("unary::is_not_nan"),
        "strings.cpp" if context.contains("cudf::strings::to_lower(") => Some("strings::to_lower"),
        "strings.cpp" if context.contains("cudf::strings::to_upper(") => Some("strings::to_upper"),
        "strings.cpp" if context.contains("cudf::strings::contains(") => Some("strings::contains"),
        "strings.cpp" if context.contains("cudf::strings::starts_with(") => Some("strings::starts_with"),
        "strings.cpp" if context.contains("cudf::strings::ends_with(") => Some("strings::ends_with"),
        "strings.cpp" if context.contains("cudf::strings::concatenate(") => Some("strings::concatenate"),
        "strings.cpp" if context.contains("cudf::strings::like(") => Some("strings::like"),
        "strings.cpp" if context.contains("cudf::strings::replace_with_backrefs(") => {
            Some("strings::replace_with_backrefs")
        },
        "strings.cpp" if context.contains("cudf::strings::slice_strings(") => Some("strings::slice"),
        "strings.cpp" if context.contains("cudf::strings::strip(") => Some("strings::strip"),
        "strings.cpp" if context.contains("cudf::strings::count_characters(") => Some("strings::count_characters"),
        "strings.cpp" if context.contains("cudf::strings::count_bytes(") => Some("strings::count_bytes"),
        "strings.cpp" if context.contains("TRY_CAST semantics") => Some("strings::try_to_integers"),
        _ => None,
    }
}
