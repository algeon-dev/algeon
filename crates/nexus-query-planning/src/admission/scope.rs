//! Plan-scope and output-schema identity carried by admitted capability
//! evidence.
//!
//! These facts answer "is the plan about to execute the plan that was
//! admitted?". They deliberately carry no bytes: allocation limits belong to
//! the query domain's allocator, not to a planning-time manifest.

use std::collections::HashSet;

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    plan::{QueryPlan, canonical_execution_scope_fingerprint},
};

/// One producer occurrence bound to an admitted consumer scope.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AttemptProducerOccurrence {
    occurrence_id: String,
    execution_scope_fingerprint: String,
    output_schema_fingerprint: String,
}

impl AttemptProducerOccurrence {
    #[doc(hidden)]
    pub fn new(
        occurrence_id: impl Into<String>,
        execution_scope_fingerprint: impl Into<String>,
        output_schema_fingerprint: impl Into<String>,
    ) -> Self {
        Self {
            occurrence_id: occurrence_id.into(),
            execution_scope_fingerprint: execution_scope_fingerprint.into(),
            output_schema_fingerprint: output_schema_fingerprint.into(),
        }
    }

    #[doc(hidden)]
    pub fn occurrence_id(&self) -> &str {
        &self.occurrence_id
    }

    #[doc(hidden)]
    pub fn execution_scope_fingerprint(&self) -> &str {
        &self.execution_scope_fingerprint
    }

    #[doc(hidden)]
    pub fn output_schema_fingerprint(&self) -> &str {
        &self.output_schema_fingerprint
    }
}

/// Immutable scope and output-schema identity for one admitted plan fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AttemptExecutionScope {
    execution_scope_fingerprint: String,
    output_schema_fingerprint: String,
    producer_occurrences: Vec<AttemptProducerOccurrence>,
}

impl AttemptExecutionScope {
    #[doc(hidden)]
    pub fn for_query_plan(plan: &QueryPlan) -> Self {
        Self {
            execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan),
            output_schema_fingerprint: crate::admission::log_safe_fingerprint(&format!("{:?}", plan.output_schema())),
            producer_occurrences: Vec::new(),
        }
    }

    #[doc(hidden)]
    pub fn execution_scope_fingerprint(&self) -> &str {
        &self.execution_scope_fingerprint
    }

    #[doc(hidden)]
    pub fn output_schema_fingerprint(&self) -> &str {
        &self.output_schema_fingerprint
    }

    #[doc(hidden)]
    pub fn producer_occurrences(&self) -> &[AttemptProducerOccurrence] {
        &self.producer_occurrences
    }

    #[doc(hidden)]
    pub fn push_producer_occurrence(&mut self, occurrence: AttemptProducerOccurrence) {
        self.producer_occurrences.push(occurrence);
    }

    #[doc(hidden)]
    pub fn matches(&self, plan: &QueryPlan) -> bool {
        self.execution_scope_fingerprint == canonical_execution_scope_fingerprint(plan)
    }

    #[doc(hidden)]
    pub fn matches_execution_scope_fingerprint(&self, execution_scope_fingerprint: &str) -> bool {
        self.execution_scope_fingerprint == execution_scope_fingerprint
    }

    pub(crate) fn producer_occurrence_scope_count(&self, execution_scope_fingerprint: &str) -> usize {
        self.producer_occurrences
            .iter()
            .filter(|occurrence| occurrence.execution_scope_fingerprint == execution_scope_fingerprint)
            .count()
    }

    /// Rejects a manifest whose occurrence identity is incomplete or
    /// ambiguous. A duplicate or empty occurrence id would let one admitted
    /// scope stand in for another at execution time.
    #[doc(hidden)]
    pub fn validate_manifest_core(&self) -> Result<()> {
        let mut occurrence_ids = HashSet::with_capacity(self.producer_occurrences.len());
        for occurrence in &self.producer_occurrences {
            if occurrence.occurrence_id.is_empty() {
                return Err(manifest_error("missing_producer_occurrence_id", None));
            }
            if !occurrence_ids.insert(&occurrence.occurrence_id) {
                return Err(manifest_error("duplicate_producer_occurrence_id", Some(&occurrence.occurrence_id)));
            }
            if occurrence.execution_scope_fingerprint.is_empty() {
                return Err(manifest_error("missing_producer_occurrence_scope", Some(&occurrence.occurrence_id)));
            }
            if occurrence.output_schema_fingerprint.is_empty() {
                return Err(manifest_error("missing_producer_occurrence_schema", Some(&occurrence.occurrence_id)));
            }
        }
        if self.execution_scope_fingerprint.is_empty() {
            return Err(manifest_error("missing_consumer_scope", None));
        }
        Ok(())
    }
}

fn manifest_error(reason: &'static str, occurrence_id: Option<&str>) -> QueryError {
    let error = QueryError::new(
        QueryErrorCode::UnsupportedPlanShape,
        "nexus_query_engine.admission.execution_scope",
        "an admitted attempt lacks complete plan-scope identity",
    )
    .with_fact("reason", reason)
    .with_fact("validation_stage", "manifest");
    match occurrence_id {
        Some(occurrence_id) => error.with_fact("occurrence_id", occurrence_id),
        None => error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scope_with_occurrences(occurrences: Vec<AttemptProducerOccurrence>) -> AttemptExecutionScope {
        AttemptExecutionScope {
            execution_scope_fingerprint: "consumer".to_string(),
            output_schema_fingerprint: "consumer-schema".to_string(),
            producer_occurrences: occurrences,
        }
    }

    #[test]
    fn occurrence_manifest_rejects_duplicate_runtime_identity() {
        let scope = scope_with_occurrences(vec![
            AttemptProducerOccurrence::new("left", "producer-a", "schema"),
            AttemptProducerOccurrence::new("left", "producer-b", "schema"),
        ]);

        let error = scope.validate_manifest_core().expect_err("duplicate occurrence identity must be ambiguous");
        assert_eq!(error.fact_value("reason"), Some("duplicate_producer_occurrence_id"));
        assert_eq!(error.fact_value("occurrence_id"), Some("left"));
    }

    #[test]
    fn occurrence_manifest_rejects_missing_identity_scope_and_schema() {
        for (occurrence, reason) in [
            (AttemptProducerOccurrence::new("", "producer", "schema"), "missing_producer_occurrence_id"),
            (AttemptProducerOccurrence::new("input", "", "schema"), "missing_producer_occurrence_scope"),
            (AttemptProducerOccurrence::new("input", "producer", ""), "missing_producer_occurrence_schema"),
        ] {
            let error = scope_with_occurrences(vec![occurrence])
                .validate_manifest_core()
                .expect_err("incomplete occurrence identity must reject");
            assert_eq!(error.fact_value("reason"), Some(reason));
            assert_eq!(error.fact_value("validation_stage"), Some("manifest"));
        }
    }
}
