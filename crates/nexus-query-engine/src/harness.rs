//! Shared helper primitives for report and benchmark harnesses.

use std::collections::HashMap;

/// Fixed FNV-1a 64-bit hasher with `0xff` field separators.
///
/// This helper is for report/parity fingerprints that already use the
/// `field-bytes + 0xff` domain. Do not use it for dataset fingerprints or
/// stable ids that intentionally use a different domain separator or label.
#[derive(Debug, Clone)]
pub struct StableFnv1a64 {
    state: u64,
}

impl StableFnv1a64 {
    /// FNV-1a 64-bit offset basis.
    pub const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    /// FNV-1a 64-bit prime.
    pub const PRIME: u64 = 0x100000001b3;
    /// Field separator used by output/parity fingerprints.
    pub const FIELD_SEPARATOR: u8 = 0xff;

    /// Creates a new stable hasher.
    pub const fn new() -> Self {
        Self { state: Self::OFFSET_BASIS }
    }

    /// Updates the hash with a UTF-8 field followed by the `0xff` separator.
    pub fn update_field(&mut self, value: &str) {
        self.update_bytes(value.as_bytes());
        self.update_bytes(&[Self::FIELD_SEPARATOR]);
    }

    /// Updates the hash with raw bytes and no implicit separator.
    pub fn update_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= u64::from(*byte);
            self.state = self.state.wrapping_mul(Self::PRIME);
        }
    }

    /// Updates the hash with a stable, key-sorted metadata map encoding.
    pub fn update_metadata(&mut self, metadata: &HashMap<String, String>) {
        self.update_field(&metadata.len().to_string());
        let mut keys = metadata.keys().collect::<Vec<_>>();
        keys.sort();
        for key in keys {
            self.update_field(key);
            self.update_field(metadata.get(key).expect("metadata key collected from map must exist"));
        }
    }

    /// Returns the current hash state.
    pub const fn finish(&self) -> u64 {
        self.state
    }
}

impl Default for StableFnv1a64 {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns the maximum present value from two options.
pub fn max_option<T: Ord>(left: Option<T>, right: Option<T>) -> Option<T> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{StableFnv1a64, max_option};

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
        let mut metadata = HashMap::new();
        metadata.insert("b".to_string(), "2".to_string());
        metadata.insert("a".to_string(), "1".to_string());

        let mut hasher = StableFnv1a64::new();
        hasher.update_metadata(&metadata);

        assert_eq!(hasher.finish(), 0x0cc4_6075_c437_d482);
    }

    #[test]
    fn max_option_returns_max_present_value() {
        assert_eq!(max_option(Some(1usize), Some(3)), Some(3));
        assert_eq!(max_option(Some(4usize), None), Some(4));
        assert_eq!(max_option(None, Some(5usize)), Some(5));
        assert_eq!(max_option::<usize>(None, None), None);
    }
}
