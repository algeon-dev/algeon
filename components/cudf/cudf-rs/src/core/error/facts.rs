//! Typed canonical facts for the truthful error contract.
//!
//! Rules (see spec §5):
//!
//! - Facts are machine-readable values that downstreams match on for policy.
//! - Each `FactKeyName` binds to exactly one `FactValue` variant; the pairing
//!   is enforced at the writer-API type layer via [`FactKey<T>`].
//! - Duplicate insertion is a programmer error and is rejected in all build
//!   profiles — `debug_assert!` is explicitly *not* acceptable.
//! - Public downstream API is read-only. Internal writers use the typed
//!   `with_fact(FactKey<T>, T)` shape.

use std::{borrow::Cow, fmt, marker::PhantomData};

use indexmap::IndexMap;

use crate::core::types::TypeId;

/// Stable canonical keys under which machine-readable facts are indexed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FactKeyName {
    ColumnName,
    ColumnIndex,
    ExpectedRows,
    ActualRows,
    ExpectedCount,
    ActualCount,
    LeftTypeId,
    RightTypeId,
    LeftDevice,
    RightDevice,
    TableRowCount,
    TableColumnCount,
    KeysCount,
    DistinctKeysCount,
    TypeId,
}

impl fmt::Display for FactKeyName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_snake_case())
    }
}

/// Type-checked handle used by the internal writer API.
///
/// The phantom `T` encodes the legal value type for `name` so
/// `with_fact(FactKey<Usize>, "not-a-number")` is a compile error.
#[derive(Debug)]
pub(crate) struct FactKey<T> {
    name: FactKeyName,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Clone for FactKey<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for FactKey<T> {}

impl<T> FactKey<T> {
    pub(crate) const fn new(name: FactKeyName) -> Self {
        Self { name, _marker: PhantomData }
    }

    pub(crate) const fn name(&self) -> FactKeyName {
        self.name
    }
}

/// Typed machine-readable fact value.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FactValue {
    Str(Cow<'static, str>),
    Usize(usize),
    I64(i64),
    TypeId(TypeId),
    DeviceId(i32),
}

impl fmt::Display for FactValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FactValue::Str(s) => f.write_str(s),
            FactValue::Usize(v) => write!(f, "{v}"),
            FactValue::I64(v) => write!(f, "{v}"),
            FactValue::TypeId(t) => write!(f, "{t:?}"),
            FactValue::DeviceId(d) => write!(f, "{d}"),
        }
    }
}

/// Tag identifying the legal `FactValue` variant for a given `FactKeyName`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FactValueVariant {
    Str,
    Usize,
    I64,
    TypeId,
    DeviceId,
}

impl FactValueVariant {
    pub(crate) const fn of_value(value: &FactValue) -> Self {
        match value {
            FactValue::Str(_) => FactValueVariant::Str,
            FactValue::Usize(_) => FactValueVariant::Usize,
            FactValue::I64(_) => FactValueVariant::I64,
            FactValue::TypeId(_) => FactValueVariant::TypeId,
            FactValue::DeviceId(_) => FactValueVariant::DeviceId,
        }
    }
}

impl FactKeyName {
    /// Returns the canonical snake_case identity for this key.
    ///
    /// This mirrors `Display` and is used by reverse-lookup helpers (e.g.,
    /// logging, test helpers) that need to resolve a `FactKeyName` from its
    /// stable string form.
    pub const fn as_snake_case(self) -> &'static str {
        match self {
            FactKeyName::ColumnName => "column_name",
            FactKeyName::ColumnIndex => "column_index",
            FactKeyName::ExpectedRows => "expected_rows",
            FactKeyName::ActualRows => "actual_rows",
            FactKeyName::ExpectedCount => "expected_count",
            FactKeyName::ActualCount => "actual_count",
            FactKeyName::LeftTypeId => "left_type_id",
            FactKeyName::RightTypeId => "right_type_id",
            FactKeyName::LeftDevice => "left_device",
            FactKeyName::RightDevice => "right_device",
            FactKeyName::TableRowCount => "table_row_count",
            FactKeyName::TableColumnCount => "table_column_count",
            FactKeyName::KeysCount => "keys_count",
            FactKeyName::DistinctKeysCount => "distinct_keys_count",
            FactKeyName::TypeId => "type_id",
        }
    }

    /// Parses the canonical snake_case identity back into a `FactKeyName`.
    ///
    /// Returns `None` for any string that is not a known canonical key.
    pub fn from_snake_case(name: &str) -> Option<Self> {
        Some(match name {
            "column_name" => FactKeyName::ColumnName,
            "column_index" => FactKeyName::ColumnIndex,
            "expected_rows" => FactKeyName::ExpectedRows,
            "actual_rows" => FactKeyName::ActualRows,
            "expected_count" => FactKeyName::ExpectedCount,
            "actual_count" => FactKeyName::ActualCount,
            "left_type_id" => FactKeyName::LeftTypeId,
            "right_type_id" => FactKeyName::RightTypeId,
            "left_device" => FactKeyName::LeftDevice,
            "right_device" => FactKeyName::RightDevice,
            "table_row_count" => FactKeyName::TableRowCount,
            "table_column_count" => FactKeyName::TableColumnCount,
            "keys_count" => FactKeyName::KeysCount,
            "distinct_keys_count" => FactKeyName::DistinctKeysCount,
            "type_id" => FactKeyName::TypeId,
            _ => return None,
        })
    }

    /// Returns the unique `FactValue` variant legal for this key.
    ///
    /// Adding a new `FactKeyName` without extending this is a compile error.
    pub(crate) const fn legal_variant(self) -> FactValueVariant {
        match self {
            FactKeyName::ColumnName => FactValueVariant::Str,
            FactKeyName::ColumnIndex => FactValueVariant::Usize,
            FactKeyName::ExpectedRows => FactValueVariant::Usize,
            FactKeyName::ActualRows => FactValueVariant::Usize,
            FactKeyName::ExpectedCount => FactValueVariant::Usize,
            FactKeyName::ActualCount => FactValueVariant::Usize,
            FactKeyName::LeftTypeId => FactValueVariant::TypeId,
            FactKeyName::RightTypeId => FactValueVariant::TypeId,
            FactKeyName::LeftDevice => FactValueVariant::DeviceId,
            FactKeyName::RightDevice => FactValueVariant::DeviceId,
            FactKeyName::TableRowCount => FactValueVariant::Usize,
            FactKeyName::TableColumnCount => FactValueVariant::Usize,
            FactKeyName::KeysCount => FactValueVariant::Usize,
            FactKeyName::DistinctKeysCount => FactValueVariant::Usize,
            FactKeyName::TypeId => FactValueVariant::TypeId,
        }
    }
}

/// Sealed trait implemented only for the concrete value types of each
/// `FactKey<T>`. This is how typed key/value pairing is enforced at compile
/// time.
pub(crate) trait IntoFactValue: sealed::Sealed {
    fn into_fact_value(self) -> FactValue;
}

mod sealed {
    pub trait Sealed {}
}

impl sealed::Sealed for &'static str {}
impl IntoFactValue for &'static str {
    fn into_fact_value(self) -> FactValue {
        FactValue::Str(Cow::Borrowed(self))
    }
}

impl sealed::Sealed for String {}
impl IntoFactValue for String {
    fn into_fact_value(self) -> FactValue {
        FactValue::Str(Cow::Owned(self))
    }
}

impl sealed::Sealed for Cow<'static, str> {}
impl IntoFactValue for Cow<'static, str> {
    fn into_fact_value(self) -> FactValue {
        FactValue::Str(self)
    }
}

impl sealed::Sealed for usize {}
impl IntoFactValue for usize {
    fn into_fact_value(self) -> FactValue {
        FactValue::Usize(self)
    }
}

impl sealed::Sealed for i64 {}
impl IntoFactValue for i64 {
    fn into_fact_value(self) -> FactValue {
        FactValue::I64(self)
    }
}

impl sealed::Sealed for TypeId {}
impl IntoFactValue for TypeId {
    fn into_fact_value(self) -> FactValue {
        FactValue::TypeId(self)
    }
}

/// Wrapper around `i32` for device-id facts so that arbitrary `i32` integers
/// do not accidentally satisfy `FactKey<DeviceId>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DeviceIdFact(pub(crate) i32);

impl sealed::Sealed for DeviceIdFact {}
impl IntoFactValue for DeviceIdFact {
    fn into_fact_value(self) -> FactValue {
        FactValue::DeviceId(self.0)
    }
}

// -- Canonical fact keys ----------------------------------------------------

pub(crate) const FACT_COLUMN_NAME: FactKey<Cow<'static, str>> = FactKey::new(FactKeyName::ColumnName);
pub(crate) const FACT_COLUMN_INDEX: FactKey<usize> = FactKey::new(FactKeyName::ColumnIndex);
pub(crate) const FACT_EXPECTED_ROWS: FactKey<usize> = FactKey::new(FactKeyName::ExpectedRows);
pub(crate) const FACT_ACTUAL_ROWS: FactKey<usize> = FactKey::new(FactKeyName::ActualRows);
pub(crate) const FACT_EXPECTED_COUNT: FactKey<usize> = FactKey::new(FactKeyName::ExpectedCount);
pub(crate) const FACT_ACTUAL_COUNT: FactKey<usize> = FactKey::new(FactKeyName::ActualCount);
pub(crate) const FACT_LEFT_TYPE_ID: FactKey<TypeId> = FactKey::new(FactKeyName::LeftTypeId);
pub(crate) const FACT_RIGHT_TYPE_ID: FactKey<TypeId> = FactKey::new(FactKeyName::RightTypeId);
pub(crate) const FACT_LEFT_DEVICE: FactKey<DeviceIdFact> = FactKey::new(FactKeyName::LeftDevice);
pub(crate) const FACT_RIGHT_DEVICE: FactKey<DeviceIdFact> = FactKey::new(FactKeyName::RightDevice);
pub(crate) const FACT_TABLE_ROW_COUNT: FactKey<usize> = FactKey::new(FactKeyName::TableRowCount);
pub(crate) const FACT_TABLE_COLUMN_COUNT: FactKey<usize> = FactKey::new(FactKeyName::TableColumnCount);
pub(crate) const FACT_KEYS_COUNT: FactKey<usize> = FactKey::new(FactKeyName::KeysCount);
pub(crate) const FACT_DISTINCT_KEYS_COUNT: FactKey<usize> = FactKey::new(FactKeyName::DistinctKeysCount);
pub(crate) const FACT_TYPE_ID: FactKey<TypeId> = FactKey::new(FactKeyName::TypeId);

// -- Facts container --------------------------------------------------------

/// Error returned when a fact key is inserted twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DuplicateFactError {
    pub(crate) key: FactKeyName,
}

impl fmt::Display for DuplicateFactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "duplicate fact key: {}", self.key)
    }
}

impl std::error::Error for DuplicateFactError {}

/// Container for canonical facts. Keys are unique.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Facts {
    inner: IndexMap<FactKeyName, FactValue>,
}

impl Facts {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn get(&self, key: FactKeyName) -> Option<&FactValue> {
        self.inner.get(&key)
    }

    pub fn contains(&self, key: FactKeyName) -> bool {
        self.inner.contains_key(&key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (FactKeyName, &FactValue)> {
        self.inner.iter().map(|(k, v)| (*k, v))
    }

    /// Fallible insertion — returns `Err` if `key` is already present.
    ///
    /// Also asserts the key/value variant pairing to catch any bulk-insertion
    /// miswiring at runtime (defense in depth; the typed `with_fact` API
    /// already makes this unrepresentable).
    pub(crate) fn try_insert(&mut self, key: FactKeyName, value: FactValue) -> Result<(), DuplicateFactError> {
        assert_eq!(
            FactValueVariant::of_value(&value),
            key.legal_variant(),
            "cudf-rs internal error: {key} bound to wrong FactValue variant",
        );
        if self.inner.contains_key(&key) {
            return Err(DuplicateFactError { key });
        }
        self.inner.insert(key, value);
        Ok(())
    }

    /// Panicking insertion — used by typed `with_fact(FactKey<T>, T)`.
    pub(crate) fn insert(&mut self, key: FactKeyName, value: FactValue) {
        if let Err(err) = self.try_insert(key, value) {
            panic!("cudf-rs internal error: {err}");
        }
    }
}

impl fmt::Display for Facts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{ ")?;
        let mut first = true;
        for (key, value) in self.iter() {
            if !first {
                f.write_str(", ")?;
            }
            first = false;
            write!(f, "{key}: {value}")?;
        }
        f.write_str(" }")
    }
}

#[cfg(test)]
mod tests {
    //! Crate-internal tests that lock the `Facts` writer contract.
    //!
    //! These were previously in `tests/core/error_truthful_contract_tests.rs`.
    //! They moved here when the writer surface became `pub(crate)` per spec
    //! §2–§3: duplicate-insert and wrong-variant behavior is a crate-internal
    //! invariant, so the tests must live inside the crate too.

    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::{FactKeyName, FactValue, Facts};

    #[test]
    fn try_insert_rejects_duplicate_key_without_overwriting_first_value() {
        let mut facts = Facts::new();
        facts.try_insert(FactKeyName::ExpectedRows, FactValue::Usize(3)).expect("first insert should succeed");

        let duplicate = facts
            .try_insert(FactKeyName::ExpectedRows, FactValue::Usize(99))
            .expect_err("duplicate fact insertion must fail");

        assert_eq!(duplicate.key, FactKeyName::ExpectedRows);
        assert_eq!(facts.get(FactKeyName::ExpectedRows), Some(&FactValue::Usize(3)));
    }

    #[test]
    fn try_insert_panics_on_wrong_value_variant() {
        let panic = catch_unwind(AssertUnwindSafe(|| {
            let mut facts = Facts::new();
            let _ = facts.try_insert(FactKeyName::ExpectedRows, FactValue::Str("3".into()));
        }));

        assert!(panic.is_err(), "wrong FactValue variant must panic instead of smuggling numbers through strings");
    }
}
