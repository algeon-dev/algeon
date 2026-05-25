//! Fast CPU-side hash collections for internal metadata and planning paths.
//!
//! These aliases use a non-cryptographic hasher and are intended for trusted,
//! in-process keys such as column names, schema metadata, and native execution
//! bookkeeping. Do not use them for hash-flooding-exposed request surfaces.

/// Hash map optimized for trusted CPU-side metadata keys.
pub type FastHashMap<K, V> = ahash::HashMap<K, V>;

/// Hash set optimized for trusted CPU-side metadata keys.
pub type FastHashSet<T> = ahash::HashSet<T>;

/// Insertion-order-preserving hash map optimized for trusted CPU-side metadata keys.
pub type FastIndexMap<K, V> = indexmap::IndexMap<K, V, ahash::RandomState>;

/// Creates a [`FastHashMap`] with the requested capacity.
pub fn fast_hash_map_with_capacity<K, V>(capacity: usize) -> FastHashMap<K, V> {
    FastHashMap::with_capacity_and_hasher(capacity, ahash::RandomState::default())
}

/// Creates a [`FastHashSet`] with the requested capacity.
pub fn fast_hash_set_with_capacity<T>(capacity: usize) -> FastHashSet<T> {
    FastHashSet::with_capacity_and_hasher(capacity, ahash::RandomState::default())
}

/// Creates an empty [`FastIndexMap`].
pub fn fast_index_map<K, V>() -> FastIndexMap<K, V> {
    FastIndexMap::with_hasher(ahash::RandomState::default())
}

/// Creates a [`FastIndexMap`] with the requested capacity.
pub fn fast_index_map_with_capacity<K, V>(capacity: usize) -> FastIndexMap<K, V> {
    FastIndexMap::with_capacity_and_hasher(capacity, ahash::RandomState::default())
}
