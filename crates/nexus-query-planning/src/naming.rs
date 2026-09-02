//! Stable name-generation helpers shared across native planning code.

/// Prefix reserved for hidden columns that preserve DataFusion row ordering.
#[doc(hidden)]
pub const DATAFUSION_ORDER_COLUMN_PREFIX: &str = "__nexus_native_df_order_";

/// Returns whether `name` belongs to the hidden DataFusion ordering contract.
#[doc(hidden)]
pub fn is_datafusion_order_column_name(name: &str) -> bool {
    name.starts_with(DATAFUSION_ORDER_COLUMN_PREFIX)
}

/// Returns `base` when it is not taken, otherwise appends the first available
/// numeric suffix using the `{base}_{suffix}` convention.
pub fn unique_alias_with_suffix(base: &str, mut taken: impl FnMut(&str) -> bool) -> String {
    unique_name_with_suffix_from(base, 1, &mut taken)
}

/// Returns `base` when it is not taken, otherwise appends the first available
/// numeric suffix at or after `first_suffix` using `{base}_{suffix}`.
pub fn unique_name_with_suffix_from(base: &str, first_suffix: usize, mut taken: impl FnMut(&str) -> bool) -> String {
    if !taken(base) {
        return base.to_string();
    }
    let suffix = first_available_numeric_suffix_from(first_suffix, |suffix| {
        let candidate = format!("{base}_{suffix}");
        taken(candidate.as_str())
    });
    format!("{base}_{suffix}")
}

/// Returns the first numeric suffix at or after `first_suffix` that is not
/// considered taken by the caller.
#[doc(hidden)]
pub fn first_available_numeric_suffix_from(first_suffix: usize, mut taken: impl FnMut(usize) -> bool) -> usize {
    for suffix in first_suffix.. {
        if !taken(suffix) {
            return suffix;
        }
    }
    unreachable!("usize suffix space is finite but practically inexhaustible")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::unique_alias_with_suffix;

    #[test]
    fn unique_alias_with_suffix_preserves_base_when_available() {
        let taken = BTreeSet::<String>::new();
        assert_eq!(unique_alias_with_suffix("alias", |candidate| taken.contains(candidate)), "alias");
    }

    #[test]
    fn unique_alias_with_suffix_uses_first_available_numeric_suffix() {
        let taken = ["alias", "alias_1", "alias_3"].into_iter().map(str::to_string).collect::<BTreeSet<_>>();
        assert_eq!(unique_alias_with_suffix("alias", |candidate| taken.contains(candidate)), "alias_2");
    }

    #[test]
    fn unique_name_with_suffix_from_preserves_zero_based_callers() {
        let taken = ["alias", "alias_0"].into_iter().map(str::to_string).collect::<BTreeSet<_>>();
        assert_eq!(super::unique_name_with_suffix_from("alias", 0, |candidate| taken.contains(candidate)), "alias_1");
    }

    #[test]
    fn first_available_numeric_suffix_from_returns_first_open_slot() {
        assert_eq!(super::first_available_numeric_suffix_from(2, |suffix| suffix != 4), 4);
    }
}
