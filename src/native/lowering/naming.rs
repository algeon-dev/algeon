pub(super) fn sanitize_field_name(name: &str) -> String {
    let sanitized =
        name.chars().map(|ch| if ch.is_ascii_alphanumeric() || ch == '_' { ch } else { '_' }).collect::<String>();
    if sanitized.is_empty() { "field".to_string() } else { sanitized }
}

pub(super) fn unique_name_with_suffix(base: &str, mut is_available: impl FnMut(&str) -> bool) -> String {
    nexus_query_engine::naming::unique_alias_with_suffix(base, |candidate| !is_available(candidate))
}
