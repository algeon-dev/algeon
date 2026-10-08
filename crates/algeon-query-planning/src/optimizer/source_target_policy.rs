//! Size limits that bound which sources a pushdown rewrite may target.
//!
//! Semijoin reduction and join-filter reassociation read these limits so a
//! pushed keyset or predicate lands on a dimension-sized source. A filter
//! pushed onto a fact scan would add a pass over the large input for rows the
//! join removes later. Rows and bytes are alternatives: either bound satisfied
//! makes a target dimension-sized, and a target with no estimate on either is
//! not.

/// Internal source-target gates shared by optimizer rewrites.
///
/// These limits keep predicate/keyset pushdown rewrites focused on
/// dimension-sized targets. They are intentionally not part of public optimizer
/// options because the current report evidence is still rule-local.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::optimizer) struct OptimizerSourceTargetPolicy {
    dimension_target_max_rows: u64,
    dimension_target_max_bytes: u64,
    residual_scalar_range_max_rows: u64,
}

impl OptimizerSourceTargetPolicy {
    pub(in crate::optimizer) const DEFAULT_DIMENSION_TARGET_MAX_ROWS: u64 = 1_000_000;
    pub(in crate::optimizer) const DEFAULT_DIMENSION_TARGET_MAX_BYTES: u64 = 64 * 1024 * 1024;
    pub(in crate::optimizer) const DEFAULT_RESIDUAL_SCALAR_RANGE_MAX_ROWS: u64 = 1_000_000;

    pub(in crate::optimizer) fn target_is_dimension_sized(
        self,
        target_rows: Option<u64>,
        target_bytes: Option<u64>,
    ) -> bool {
        target_rows.is_some_and(|rows| rows <= self.dimension_target_max_rows)
            || target_bytes.is_some_and(|bytes| bytes <= self.dimension_target_max_bytes)
    }

    /// Largest filtered subtree whose scalar value range a residual pushdown
    /// may derive from footer statistics; above it the derivation is skipped.
    pub(in crate::optimizer) fn residual_scalar_range_max_rows(self) -> u64 {
        self.residual_scalar_range_max_rows
    }
}

impl Default for OptimizerSourceTargetPolicy {
    fn default() -> Self {
        Self {
            dimension_target_max_rows: Self::DEFAULT_DIMENSION_TARGET_MAX_ROWS,
            dimension_target_max_bytes: Self::DEFAULT_DIMENSION_TARGET_MAX_BYTES,
            residual_scalar_range_max_rows: Self::DEFAULT_RESIDUAL_SCALAR_RANGE_MAX_ROWS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_sized_target_accepts_small_rows_or_bytes() {
        let policy = OptimizerSourceTargetPolicy::default();

        assert!(
            policy
                .target_is_dimension_sized(Some(OptimizerSourceTargetPolicy::DEFAULT_DIMENSION_TARGET_MAX_ROWS), None)
        );
        assert!(
            policy
                .target_is_dimension_sized(None, Some(OptimizerSourceTargetPolicy::DEFAULT_DIMENSION_TARGET_MAX_BYTES))
        );
        assert!(!policy.target_is_dimension_sized(
            Some(OptimizerSourceTargetPolicy::DEFAULT_DIMENSION_TARGET_MAX_ROWS + 1),
            Some(OptimizerSourceTargetPolicy::DEFAULT_DIMENSION_TARGET_MAX_BYTES + 1),
        ));
        assert!(!policy.target_is_dimension_sized(None, None));
    }

    #[test]
    fn residual_scalar_range_max_rows_preserves_limit() {
        assert_eq!(
            OptimizerSourceTargetPolicy::default().residual_scalar_range_max_rows(),
            OptimizerSourceTargetPolicy::DEFAULT_RESIDUAL_SCALAR_RANGE_MAX_ROWS
        );
    }
}
