//! Internal fast-hash aliases for CPU-side schema and metadata paths.

pub(crate) type FastHashMap<K, V> = ahash::HashMap<K, V>;
pub(crate) type FastHashSet<T> = ahash::HashSet<T>;
pub(crate) type FastIndexMap<K, V> = indexmap::IndexMap<K, V, ahash::RandomState>;

pub(crate) fn fast_hash_map_with_capacity<K, V>(capacity: usize) -> FastHashMap<K, V> {
    FastHashMap::with_capacity_and_hasher(capacity, ahash::RandomState::default())
}

pub(crate) fn fast_hash_set_with_capacity<T>(capacity: usize) -> FastHashSet<T> {
    FastHashSet::with_capacity_and_hasher(capacity, ahash::RandomState::default())
}

pub(crate) fn fast_index_map<K, V>() -> FastIndexMap<K, V> {
    FastIndexMap::with_hasher(ahash::RandomState::default())
}

pub(crate) fn fast_index_map_with_capacity<K, V>(capacity: usize) -> FastIndexMap<K, V> {
    FastIndexMap::with_capacity_and_hasher(capacity, ahash::RandomState::default())
}
