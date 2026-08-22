use std::collections::HashMap;

#[doc(hidden)]
pub fn max_option<T: Ord>(left: Option<T>, right: Option<T>) -> Option<T> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct StableFnv1a64 {
    state: u64,
}

impl StableFnv1a64 {
    pub const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    pub const PRIME: u64 = 0x100000001b3;
    pub const FIELD_SEPARATOR: u8 = 0xff;

    pub const fn new() -> Self {
        Self { state: Self::OFFSET_BASIS }
    }

    pub fn update_field(&mut self, value: &str) {
        self.update_bytes(value.as_bytes());
        self.update_bytes(&[Self::FIELD_SEPARATOR]);
    }

    pub fn update_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= u64::from(*byte);
            self.state = self.state.wrapping_mul(Self::PRIME);
        }
    }

    pub fn update_metadata(&mut self, metadata: &HashMap<String, String>) {
        self.update_field(&metadata.len().to_string());
        let mut keys = metadata.keys().collect::<Vec<_>>();
        keys.sort();
        for key in keys {
            self.update_field(key);
            self.update_field(metadata.get(key).expect("metadata key collected from map must exist"));
        }
    }

    pub const fn finish(&self) -> u64 {
        self.state
    }
}

impl Default for StableFnv1a64 {
    fn default() -> Self {
        Self::new()
    }
}

#[doc(hidden)]
pub fn comma_separated_usize(values: &[usize]) -> String {
    let mut output = String::new();
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output
}

#[doc(hidden)]
pub fn escape_kv_delimited(value: &str) -> String {
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

#[doc(hidden)]
pub fn log_safe_fingerprint(value: &str) -> String {
    format!("blake3:{}", blake3::hash(value.as_bytes()).to_hex())
}

#[cfg(test)]
mod tests {
    use super::{StableFnv1a64, comma_separated_usize, escape_kv_delimited};

    #[test]
    fn stable_fnv1a64_matches_known_field_hashes() {
        let mut hasher = StableFnv1a64::new();
        hasher.update_bytes(b"hello");
        assert_eq!(hasher.finish(), 0xa430_d846_80aa_bd0b);

        let mut hasher = StableFnv1a64::new();
        hasher.update_field("hello");
        assert_eq!(hasher.finish(), 0xa9bd_73cc_a220_c59c);

        let mut hasher = StableFnv1a64::new();
        hasher.update_field("alpha");
        hasher.update_field("beta");
        assert_eq!(hasher.finish(), 0x72bc_ef35_e1dc_68a3);
    }

    #[test]
    fn stable_fnv1a64_metadata_is_key_sorted() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("b".to_string(), "2".to_string());
        metadata.insert("a".to_string(), "1".to_string());

        let mut hasher = StableFnv1a64::new();
        hasher.update_metadata(&metadata);

        assert_eq!(hasher.finish(), 0x0cc4_6075_c437_d482);
    }

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
