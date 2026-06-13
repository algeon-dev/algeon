// Spec §2–§3: the `Facts` container is a read-only downstream-visible type.
// `Facts::new`, `Facts::try_insert`, and `Facts::insert` are crate-internal
// writer entry points and must not compile from outside the crate.

use cudf::core::error::{FactKeyName, FactValue, Facts};

fn main() {
    let mut facts = Facts::new();
    let _ = facts.try_insert(FactKeyName::ExpectedRows, FactValue::Usize(1));
    facts.insert(FactKeyName::ActualRows, FactValue::Usize(1));
}
