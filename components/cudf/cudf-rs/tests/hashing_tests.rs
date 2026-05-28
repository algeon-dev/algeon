//! Tests for hashing functions.

use cudf::column::Column;
use cudf::hashing;
use cudf::table::Table;
use cudf::types::TypeId;

#[test]
fn test_murmurhash3_i32_column() {
    let col = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let hashes = hashing::murmurhash3_x86_32(&view, 0).unwrap();
    let hash_view = hashes.view().unwrap();

    assert_eq!(hash_view.len().unwrap(), 5);
    assert_eq!(hash_view.data_type().unwrap().id(), TypeId::UInt32);

    let values = hash_view.to_vec::<u32>().unwrap();
    assert_eq!(values.len(), 5);

    // Different input values should (almost certainly) produce different hashes.
    // This is probabilistic but extremely unlikely to fail.
    let unique: std::collections::HashSet<u32> = values.iter().copied().collect();
    assert!(
        unique.len() >= 4,
        "Expected mostly unique hashes, got {:?}",
        values
    );
}

#[test]
fn test_murmurhash3_deterministic() {
    let col = Column::from_slice(&[42i32, 99, 7]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let hashes1 = hashing::murmurhash3_x86_32(&view, 0).unwrap();
    let hashes2 = hashing::murmurhash3_x86_32(&view, 0).unwrap();

    let v1 = hashes1.view().unwrap().to_vec::<u32>().unwrap();
    let v2 = hashes2.view().unwrap().to_vec::<u32>().unwrap();
    assert_eq!(v1, v2, "Same input with same seed must produce same hashes");
}

#[test]
fn test_murmurhash3_different_seeds() {
    let col = Column::from_slice(&[42i32, 99, 7]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let hashes_seed0 = hashing::murmurhash3_x86_32(&view, 0).unwrap();
    let hashes_seed1 = hashing::murmurhash3_x86_32(&view, 12345).unwrap();

    let v0 = hashes_seed0.view().unwrap().to_vec::<u32>().unwrap();
    let v1 = hashes_seed1.view().unwrap().to_vec::<u32>().unwrap();
    assert_ne!(v0, v1, "Different seeds should produce different hashes");
}

#[test]
fn test_murmurhash3_device_ptr() {
    // Verify the hash result column has a non-null device pointer.
    let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let hashes = hashing::murmurhash3_x86_32(&view, 0).unwrap();
    let hash_view = hashes.view().unwrap();

    let ptr = hash_view.device_data_ptr().unwrap();
    assert!(
        !ptr.is_null(),
        "hash column device pointer must be non-null"
    );
}

#[test]
fn test_xxhash_64() {
    let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    let table = Table::from_columns(vec![col]).unwrap();
    let view = table.view().unwrap();

    let hashes = hashing::xxhash_64(&view, 0).unwrap();
    let hash_view = hashes.view().unwrap();

    assert_eq!(hash_view.len().unwrap(), 3);
    assert_eq!(hash_view.data_type().unwrap().id(), TypeId::UInt64);

    let values = hash_view.to_vec::<u64>().unwrap();
    assert_eq!(values.len(), 3);
}

#[test]
fn test_murmurhash3_multi_column() {
    let col1 = Column::from_slice(&[1i32, 2, 3]).unwrap();
    let col2 = Column::from_slice(&[10.0f64, 20.0, 30.0]).unwrap();
    let table = Table::from_columns(vec![col1, col2]).unwrap();
    let view = table.view().unwrap();

    let hashes = hashing::murmurhash3_x86_32(&view, 0).unwrap();
    let hash_view = hashes.view().unwrap();

    assert_eq!(hash_view.len().unwrap(), 3);
    assert_eq!(hash_view.data_type().unwrap().id(), TypeId::UInt32);
}
