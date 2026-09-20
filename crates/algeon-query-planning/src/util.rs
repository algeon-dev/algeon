use std::{collections::HashMap, hash::Hasher};

use fnv::FnvHasher;

/// Incremental 64-bit FNV-1a hasher whose output depends only on the bytes
/// fed to it, so fingerprints stay equal across processes, hosts, and Rust
/// releases; `std`'s `DefaultHasher` leaves its algorithm unspecified. Report
/// fingerprints, schedule projections, and benchmark result checksums compare
/// values produced by different runs, which is why they use this type.
///
/// [`update_field`](Self::update_field) terminates each field with the
/// separator byte `0xff`. That byte never occurs in UTF-8, so `("ab", "c")`
/// and `("a", "bc")` hash differently.
/// [`update_metadata`](Self::update_metadata) hashes the entry count and then
/// the entries in key order, which makes the result independent of `HashMap`
/// iteration order.
///
/// Not collision resistant; use [`log_safe_fingerprint`] where a value must
/// not be recoverable or forgeable.
#[derive(Debug, Clone)]
pub struct StableFnv1a64 {
    state: u64,
}

impl StableFnv1a64 {
    pub(crate) const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    pub(crate) const FIELD_SEPARATOR: u8 = 0xff;

    pub const fn new() -> Self {
        Self { state: Self::OFFSET_BASIS }
    }

    pub fn update_field(&mut self, value: &str) {
        self.update_bytes(value.as_bytes());
        self.update_bytes(&[Self::FIELD_SEPARATOR]);
    }

    pub fn update_bytes(&mut self, bytes: &[u8]) {
        let mut hasher = FnvHasher::with_key(self.state);
        hasher.write(bytes);
        self.state = hasher.finish();
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

#[doc(hidden)]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hasher = FnvHasher::default();
    hasher.write(bytes);
    hasher.finish()
}

/// Sums known byte or row counts; `None` when any input is unknown or the sum
/// overflows `u64`, so an overflowed total is never reported as known.
#[doc(hidden)]
pub fn sum_optional_u64(values: impl IntoIterator<Item = Option<u64>>) -> Option<u64> {
    values.into_iter().try_fold(0_u64, |sum, value| sum.checked_add(value?))
}

impl Default for StableFnv1a64 {
    fn default() -> Self {
        Self::new()
    }
}

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
