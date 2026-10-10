pub fn normalize_whitespace(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

pub fn contains_normalized(haystack: &str, needle: &str) -> bool {
    normalize_whitespace(haystack).contains(&normalize_whitespace(needle))
}
