pub(crate) fn comma_separated_usize(values: &[usize]) -> String {
    let mut output = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output
}

pub(crate) fn escape_kv_delimited(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '=' => escaped.push_str("\\="),
            ';' => escaped.push_str("\\;"),
            other => escaped.push(other),
        }
    }
    escaped
}

#[cfg(test)]
mod tests {
    use super::{comma_separated_usize, escape_kv_delimited};

    #[test]
    fn comma_separated_usize_formats_stable_lists() {
        assert_eq!(comma_separated_usize(&[]), "");
        assert_eq!(comma_separated_usize(&[0]), "0");
        assert_eq!(comma_separated_usize(&[1, 2, 10]), "1,2,10");
    }

    #[test]
    fn escape_kv_delimited_preserves_contract_escape_bytes() {
        assert_eq!(escape_kv_delimited("plain"), "plain");
        assert_eq!(escape_kv_delimited(r"a\b=c;d"), r"a\\b\=c\;d");
    }
}
