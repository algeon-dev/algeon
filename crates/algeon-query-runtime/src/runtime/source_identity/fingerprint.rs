//! Byte-stable local-Parquet cache fingerprint construction.

use std::fmt::Write;

use arrow::datatypes::SchemaRef;
use query_planning::{
    source::{
        LocalParquetSource, ParquetEqualityDeleteFile, ParquetPositionDeleteFile, ParquetReaderDecimalType,
        ResolvedParquetSource, SourceColumnRef,
    },
    util::{escape_kv_delimited, fnv1a64},
};

pub(super) fn push_schema(output: &mut String, schema: &SchemaRef) {
    push_token(output, "schema_field_count", schema.fields().len().to_string());
    for (index, field) in schema.fields().iter().enumerate() {
        push_token(output, format!("schema.{index}.name"), field.name());
        push_token(output, format!("schema.{index}.data_type"), format!("{:?}", field.data_type()));
        push_token(output, format!("schema.{index}.nullable"), field.is_nullable().to_string());
        for (key, value) in field.metadata() {
            push_token(output, format!("schema.{index}.metadata.{}", escape_token(key)), value);
        }
    }
}

pub(super) fn push_source_scan_contract(output: &mut String, source: &LocalParquetSource) {
    push_schema(output, &source.schema);
    push_token(output, "count_star_probe", source.is_count_star_probe_source().to_string());
    match &source.output_schema_override {
        Some(schema) => {
            push_token(output, "output_schema_override", "some");
            push_schema(output, schema);
        },
        None => push_token(output, "output_schema_override", "none"),
    }
    push_string_vec(output, "read_columns", &source.read_columns);
    push_usize_vec(output, "read_column_file_indices", &source.read_column_file_indices);
    push_string_vec(output, "projected_output_columns", &source.projected_output_columns);
    push_source_refs(output, "projected_output_column_source_refs", &source.projected_output_column_source_refs);
    push_string_vec(output, "pure_filter_columns", &source.pure_filter_columns);
    push_source_refs(output, "pure_filter_column_source_refs", &source.pure_filter_column_source_refs);
    push_token(
        output,
        "reader_predicate",
        source.reader_predicate.as_ref().map_or_else(|| "none".to_string(), ToString::to_string),
    );
}

pub(super) fn push_resolved_source_contract(
    output: &mut String,
    label: impl AsRef<str>,
    resolved_source: &ResolvedParquetSource,
) {
    let label = label.as_ref();
    push_optional_string(
        output,
        format!("{label}.original_location"),
        Some(resolved_source.original_location.as_str()),
    );
    push_optional_string(output, format!("{label}.reader_location"), Some(resolved_source.reader_location()));
    push_optional_string(output, format!("{label}.storage_domain"), resolved_source.storage_domain.as_deref());
    push_optional_string(output, format!("{label}.content_identity"), resolved_source.content_identity.as_deref());
    push_optional_string(output, format!("{label}.storage_profile_id"), resolved_source.storage_profile_id());
    push_token(output, format!("{label}.read_strategy"), resolved_source.read_strategy().as_str());
    push_token(output, format!("{label}.credential_mode"), resolved_source.credential_mode().as_str());
    push_token(output, format!("{label}.lakehouse_cache_status"), resolved_source.cache_status.as_str());
    push_token(
        output,
        format!("{label}.lakehouse_cache_read_mode"),
        resolved_source.lakehouse_cache_read_mode.as_str(),
    );
    push_optional_string(
        output,
        format!("{label}.lakehouse_cache_root"),
        resolved_source.lakehouse_cache_root.as_deref(),
    );
    push_optional_string(
        output,
        format!("{label}.lakehouse_cache_path"),
        resolved_source.lakehouse_cache_path.as_deref(),
    );
    push_optional_u64(output, format!("{label}.lakehouse_cache_max_bytes"), resolved_source.lakehouse_cache_max_bytes);
    push_token(
        output,
        format!("{label}.lakehouse_cache_fill_mode"),
        resolved_source.lakehouse_cache_fill_mode.as_str(),
    );
}

pub(super) fn push_position_delete_files(
    output: &mut String,
    label: impl AsRef<str>,
    delete_files: &[ParquetPositionDeleteFile],
) {
    let label = label.as_ref();
    push_token(output, format!("{label}.len"), delete_files.len().to_string());
    for (index, delete_file) in delete_files.iter().enumerate() {
        let label = format!("{label}.{index}");
        push_token(output, format!("{label}.id"), delete_file.id.as_str());
        push_token(output, format!("{label}.location"), delete_file.location.reader_path());
        push_resolved_source_contract(output, format!("{label}.resolved"), &delete_file.resolved_source);
        push_token(output, format!("{label}.stable_id"), delete_file.stable_id.as_str());
        push_optional_u64(output, format!("{label}.size"), delete_file.file_size_bytes);
        push_optional_u64(output, format!("{label}.rows"), delete_file.row_count);
        push_optional_string(
            output,
            format!("{label}.referenced_data_file"),
            delete_file.referenced_data_file.as_deref(),
        );
        push_optional_u64(output, format!("{label}.position_lower_bound"), delete_file.position_lower_bound);
        push_optional_u64(output, format!("{label}.position_upper_bound"), delete_file.position_upper_bound);
    }
}

pub(super) fn push_equality_delete_files(
    output: &mut String,
    label: impl AsRef<str>,
    delete_files: &[ParquetEqualityDeleteFile],
) {
    let label = label.as_ref();
    push_token(output, format!("{label}.len"), delete_files.len().to_string());
    for (index, delete_file) in delete_files.iter().enumerate() {
        let label = format!("{label}.{index}");
        push_token(output, format!("{label}.id"), delete_file.id.as_str());
        push_token(output, format!("{label}.location"), delete_file.location.reader_path());
        push_resolved_source_contract(output, format!("{label}.resolved"), &delete_file.resolved_source);
        push_token(output, format!("{label}.stable_id"), delete_file.stable_id.as_str());
        push_i32_vec(output, format!("{label}.equality_ids"), &delete_file.equality_ids);
        push_string_vec(output, format!("{label}.equality_columns"), &delete_file.equality_columns);
        push_optional_u64(output, format!("{label}.size"), delete_file.file_size_bytes);
        push_optional_u64(output, format!("{label}.rows"), delete_file.row_count);
    }
}

pub(super) fn redact_source_cache_fingerprint(output: &mut String, source: &LocalParquetSource) {
    for (index, file) in source.files.iter().enumerate() {
        replace_token_value(
            output,
            format!("file.{index}.location"),
            file.location.reader_path().as_str(),
            file.location.diagnostic_location().as_str(),
        );
        replace_token_value(
            output,
            format!("file.{index}.stable_id"),
            file.stable_id.as_str(),
            file.location.diagnostic_stable_id(file.stable_id.as_str()).as_str(),
        );
        redact_resolved_source_contract(output, format!("file.{index}.resolved"), &file.resolved_source);
        for (delete_index, delete_file) in file.position_delete_files.iter().enumerate() {
            let label = format!("file.{index}.position_delete_files.{delete_index}");
            replace_token_value(
                output,
                format!("{label}.location"),
                delete_file.location.reader_path().as_str(),
                delete_file.location.diagnostic_location().as_str(),
            );
            replace_token_value(
                output,
                format!("{label}.stable_id"),
                delete_file.stable_id.as_str(),
                delete_file.location.diagnostic_stable_id(delete_file.stable_id.as_str()).as_str(),
            );
            if let Some(referenced_data_file) = delete_file.referenced_data_file.as_deref() {
                replace_token_value(
                    output,
                    format!("{label}.referenced_data_file"),
                    referenced_data_file,
                    redacted_position_delete_referenced_data_file(referenced_data_file).as_str(),
                );
            }
            redact_resolved_source_contract(output, format!("{label}.resolved"), &delete_file.resolved_source);
        }
        for (delete_index, delete_file) in file.equality_delete_files.iter().enumerate() {
            let label = format!("file.{index}.equality_delete_files.{delete_index}");
            replace_token_value(
                output,
                format!("{label}.location"),
                delete_file.location.reader_path().as_str(),
                delete_file.location.diagnostic_location().as_str(),
            );
            replace_token_value(
                output,
                format!("{label}.stable_id"),
                delete_file.stable_id.as_str(),
                delete_file.location.diagnostic_stable_id(delete_file.stable_id.as_str()).as_str(),
            );
            redact_resolved_source_contract(output, format!("{label}.resolved"), &delete_file.resolved_source);
        }
    }
}

pub(super) fn redacted_position_delete_referenced_data_file(path: &str) -> String {
    format!("stable_id_fnv64:{:016x}", fnv1a64(path.as_bytes()))
}

pub(super) fn redact_resolved_source_contract(
    output: &mut String,
    label: impl AsRef<str>,
    resolved_source: &ResolvedParquetSource,
) {
    let label = label.as_ref();
    replace_token_value(
        output,
        format!("{label}.original_location"),
        resolved_source.original_location.as_str(),
        stable_hashed_string(resolved_source.original_location.as_str()).as_str(),
    );
    replace_token_value(
        output,
        format!("{label}.reader_location"),
        resolved_source.reader_location(),
        stable_hashed_string(resolved_source.reader_location()).as_str(),
    );
    replace_optional_token_value(output, format!("{label}.storage_domain"), resolved_source.storage_domain.as_deref());
    replace_optional_token_value(
        output,
        format!("{label}.content_identity"),
        resolved_source.content_identity.as_deref(),
    );
    replace_optional_token_value(
        output,
        format!("{label}.lakehouse_cache_root"),
        resolved_source.lakehouse_cache_root.as_deref(),
    );
    replace_optional_token_value(
        output,
        format!("{label}.lakehouse_cache_path"),
        resolved_source.lakehouse_cache_path.as_deref(),
    );
}

pub(super) fn replace_optional_token_value(output: &mut String, key: impl AsRef<str>, value: Option<&str>) {
    if let Some(value) = value {
        replace_token_value(output, key, value, stable_hashed_string(value).as_str());
    }
}

pub(super) fn replace_token_value(output: &mut String, key: impl AsRef<str>, current_value: &str, new_value: &str) {
    let key = escape_token(key.as_ref());
    let current_value = escape_token(current_value);
    let new_value = escape_token(new_value);
    let needle = format!("{key}={current_value};");
    let replacement = format!("{key}={new_value};");
    *output = output.replace(&needle, replacement.as_str());
}

pub(super) fn push_source_refs(output: &mut String, label: impl AsRef<str>, refs: &[SourceColumnRef]) {
    let values = refs.iter().map(format_source_ref).collect::<Vec<_>>();
    push_string_vec(output, label, &values);
}

pub(super) fn format_source_ref(source_ref: &SourceColumnRef) -> String {
    match source_ref {
        SourceColumnRef::File(index) => format!("file:{index}"),
        SourceColumnRef::Partition(index) => format!("partition:{index}"),
        SourceColumnRef::MissingNull(index) => format!("missing_null:{index}"),
    }
}

pub(super) fn push_string_vec(output: &mut String, label: impl AsRef<str>, values: &[String]) {
    push_token(output, format!("{}.len", label.as_ref()), values.len().to_string());
    for (index, value) in values.iter().enumerate() {
        push_token(output, format!("{}.{}", label.as_ref(), index), value);
    }
}

pub(super) fn push_usize_vec(output: &mut String, label: impl AsRef<str>, values: &[usize]) {
    push_token(output, format!("{}.len", label.as_ref()), values.len().to_string());
    for (index, value) in values.iter().enumerate() {
        push_token(output, format!("{}.{}", label.as_ref(), index), value.to_string());
    }
}

pub(super) fn push_i32_vec(output: &mut String, label: impl AsRef<str>, values: &[i32]) {
    push_token(output, format!("{}.len", label.as_ref()), values.len().to_string());
    for (index, value) in values.iter().enumerate() {
        push_token(output, format!("{}.{}", label.as_ref(), index), value.to_string());
    }
}

pub(super) fn push_reader_decimal_types(
    output: &mut String,
    label: impl AsRef<str>,
    values: &[ParquetReaderDecimalType],
) {
    let label = label.as_ref();
    push_token(output, format!("{label}.len"), values.len().to_string());
    for (index, value) in values.iter().enumerate() {
        push_token(output, format!("{label}.{index}.column_name"), &value.column_name);
        push_token(output, format!("{label}.{index}.type_id"), value.type_id.as_str());
        push_token(output, format!("{label}.{index}.cudf_scale"), value.cudf_scale.to_string());
    }
}

pub(super) fn push_optional_usize_vec(output: &mut String, label: impl AsRef<str>, values: Option<&[usize]>) {
    match values {
        Some(values) => push_usize_vec(output, label, values),
        None => push_token(output, label, "none"),
    }
}

pub(super) fn push_optional_u64(output: &mut String, label: impl AsRef<str>, value: Option<u64>) {
    match value {
        Some(value) => push_token(output, label, value.to_string()),
        None => push_token(output, label, "none"),
    }
}

pub(super) fn push_optional_u64_vec(output: &mut String, label: impl AsRef<str>, values: Option<&[u64]>) {
    match values {
        Some(values) => {
            push_token(output, format!("{}.len", label.as_ref()), values.len().to_string());
            for (index, value) in values.iter().enumerate() {
                push_token(output, format!("{}.{}", label.as_ref(), index), value.to_string());
            }
        },
        None => push_token(output, label, "none"),
    }
}

pub(super) fn push_optional_string(output: &mut String, label: impl AsRef<str>, value: Option<&str>) {
    match value {
        Some(value) => push_token(output, label, value),
        None => push_token(output, label, "none"),
    }
}

pub(super) fn stable_hashed_string(value: &str) -> String {
    format!("fnv64:{:016x}:len:{}", fnv1a64(value.as_bytes()), value.len())
}

pub(super) fn push_token(output: &mut String, key: impl AsRef<str>, value: impl ToString) {
    let _ = write!(output, "{}={};", escape_token(key.as_ref()), escape_token(&value.to_string()));
}

pub(super) fn escape_token(value: &str) -> String {
    escape_kv_delimited(value)
}
