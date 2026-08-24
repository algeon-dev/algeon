pub fn normalize_whitespace(source: &str) -> String {
    source.chars().filter(|ch| !ch.is_whitespace()).collect()
}

pub fn contains_normalized(haystack: &str, needle: &str) -> bool {
    normalize_whitespace(haystack).contains(&normalize_whitespace(needle))
}

#[cfg(test)]
mod tests {
    use super::{contains_normalized, normalize_whitespace};

    #[test]
    fn normalize_whitespace_collapses_runs_and_newlines() {
        assert_eq!(normalize_whitespace("a  b\n\tc"), "abc");
    }

    #[test]
    fn contains_normalized_ignores_formatting_only() {
        assert!(contains_normalized("auto const x =\n  foo(bar);", "auto const x = foo(bar);"));
        assert!(!contains_normalized("auto const x = foo(baz);", "auto const x = foo(bar);"));
    }
}
