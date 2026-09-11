//! Plan-scope and output-schema identity carried by admitted capability
//! evidence.
//!
//! These facts answer "is the plan about to execute the plan that was
//! admitted?". They deliberately carry no bytes: allocation limits belong to
//! the query domain's allocator, not to a planning-time manifest.

use std::collections::HashSet;

use crate::{
    error::{QueryError, QueryErrorCode, Result},
    plan::{DeferredGraphExecutionScopeIdentity, QueryPlan, canonical_execution_scope_fingerprint},
};

/// Host boundary crossed by one GPU function occurrence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum AttemptExecutionBoundary {
    /// The GPU function materializes Arrow for a DataFusion consumer.
    HostArrow,
}

/// One GPU function occurrence in a DataFusion-owned execution graph.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AttemptExecutionOccurrence {
    occurrence_id: String,
    producer_identity: String,
    execution_scope_fingerprint: String,
    deferred_execution_scope_identity: Option<DeferredGraphExecutionScopeIdentity>,
    output_schema_fingerprint: String,
    boundary: AttemptExecutionBoundary,
    predecessors: Vec<String>,
}

impl AttemptExecutionOccurrence {
    #[doc(hidden)]
    pub fn host_arrow(
        occurrence_id: impl Into<String>,
        plan: &QueryPlan,
        predecessors: Vec<String>,
        deferred_graph_facts: bool,
    ) -> Self {
        let occurrence_id = occurrence_id.into();
        Self::host_arrow_for_producer(occurrence_id.clone(), occurrence_id, plan, predecessors, deferred_graph_facts)
    }

    /// Creates one invocation occurrence for a producer execution identity.
    /// A shared physical producer may appear more than once in the final plan,
    /// so its stable producer identity is distinct from the occurrence id.
    #[doc(hidden)]
    pub fn host_arrow_for_producer(
        occurrence_id: impl Into<String>,
        producer_identity: impl Into<String>,
        plan: &QueryPlan,
        predecessors: Vec<String>,
        deferred_graph_facts: bool,
    ) -> Self {
        Self {
            occurrence_id: occurrence_id.into(),
            producer_identity: producer_identity.into(),
            execution_scope_fingerprint: canonical_execution_scope_fingerprint(plan),
            deferred_execution_scope_identity: deferred_graph_facts
                .then(|| DeferredGraphExecutionScopeIdentity::for_plan(plan)),
            output_schema_fingerprint: crate::admission::log_safe_fingerprint(&format!("{:?}", plan.output_schema())),
            boundary: AttemptExecutionBoundary::HostArrow,
            predecessors,
        }
    }

    #[doc(hidden)]
    pub fn occurrence_id(&self) -> &str {
        &self.occurrence_id
    }

    #[doc(hidden)]
    pub fn producer_identity(&self) -> &str {
        &self.producer_identity
    }

    #[doc(hidden)]
    pub fn predecessors(&self) -> &[String] {
        &self.predecessors
    }

    #[doc(hidden)]
    pub fn matches_plan(&self, plan: &QueryPlan) -> bool {
        let exact = canonical_execution_scope_fingerprint(plan);
        self.execution_scope_fingerprint == exact
            || self.deferred_execution_scope_identity.as_ref().is_some_and(|expected| expected.matches(plan))
    }

    #[doc(hidden)]
    pub fn matches_schema(&self, plan: &QueryPlan) -> bool {
        self.output_schema_fingerprint == crate::admission::log_safe_fingerprint(&format!("{:?}", plan.output_schema()))
    }

    #[doc(hidden)]
    pub fn is_host_arrow(&self) -> bool {
        self.boundary == AttemptExecutionBoundary::HostArrow
    }
}

/// Topologically ordered GPU occurrences for one DataFusion invocation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[doc(hidden)]
pub struct AttemptExecutionManifest {
    occurrences: Vec<AttemptExecutionOccurrence>,
}

impl AttemptExecutionManifest {
    #[doc(hidden)]
    pub fn new(occurrences: Vec<AttemptExecutionOccurrence>) -> Result<Self> {
        let manifest = Self { occurrences };
        manifest.validate()?;
        Ok(manifest)
    }

    #[doc(hidden)]
    pub fn occurrences(&self) -> &[AttemptExecutionOccurrence] {
        &self.occurrences
    }

    #[doc(hidden)]
    pub fn occurrence(&self, occurrence_id: &str) -> Option<&AttemptExecutionOccurrence> {
        self.occurrences.iter().find(|occurrence| occurrence.occurrence_id == occurrence_id)
    }

    #[doc(hidden)]
    pub fn occurrence_for_plan(&self, plan: &QueryPlan) -> Option<&AttemptExecutionOccurrence> {
        self.occurrences.iter().find(|occurrence| occurrence.matches_plan(plan) && occurrence.matches_schema(plan))
    }

    /// Validates the exact host-materializing occurrence about to execute and
    /// returns its already-validated predecessor identities.
    #[doc(hidden)]
    pub fn validate_host_occurrence(&self, occurrence_id: &str, plan: &QueryPlan) -> Result<Vec<String>> {
        self.validate_host_occurrence_for_producer(occurrence_id, occurrence_id, plan)
    }

    /// Validates the concrete invocation occurrence and its stable producer
    /// identity, then returns its predecessor occurrence ids.
    #[doc(hidden)]
    pub fn validate_host_occurrence_for_producer(
        &self,
        occurrence_id: &str,
        producer_identity: &str,
        plan: &QueryPlan,
    ) -> Result<Vec<String>> {
        let occurrence = self
            .occurrence(occurrence_id)
            .ok_or_else(|| manifest_error("execution_occurrence_not_admitted", Some(occurrence_id)))?;
        if occurrence.producer_identity != producer_identity {
            return Err(manifest_error("execution_occurrence_producer_mismatch", Some(occurrence_id)));
        }
        if occurrence.boundary != AttemptExecutionBoundary::HostArrow {
            return Err(manifest_error("execution_occurrence_boundary_mismatch", Some(occurrence_id)));
        }
        if !occurrence.matches_plan(plan) {
            return Err(manifest_error("execution_occurrence_plan_mismatch", Some(occurrence_id)));
        }
        if !occurrence.matches_schema(plan) {
            return Err(manifest_error("execution_occurrence_schema_mismatch", Some(occurrence_id)));
        }
        Ok(occurrence.predecessors.clone())
    }

    /// Validates that a producer execution identity has at least one matching
    /// host occurrence in this invocation. Runtime claims the concrete unique
    /// occurrence when execution begins.
    #[doc(hidden)]
    pub fn validate_host_producer(&self, producer_identity: &str, plan: &QueryPlan) -> Result<()> {
        let candidates = self
            .occurrences
            .iter()
            .filter(|occurrence| occurrence.producer_identity == producer_identity)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Err(manifest_error("execution_producer_not_admitted", Some(producer_identity)));
        }
        if !candidates.iter().any(|occurrence| occurrence.is_host_arrow()) {
            return Err(manifest_error("execution_occurrence_boundary_mismatch", Some(producer_identity)));
        }
        if !candidates.iter().any(|occurrence| occurrence.matches_plan(plan)) {
            return Err(manifest_error("execution_occurrence_plan_mismatch", Some(producer_identity)));
        }
        if !candidates.iter().any(|occurrence| occurrence.matches_plan(plan) && occurrence.matches_schema(plan)) {
            return Err(manifest_error("execution_occurrence_schema_mismatch", Some(producer_identity)));
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        let mut seen = HashSet::with_capacity(self.occurrences.len());
        for occurrence in &self.occurrences {
            if occurrence.occurrence_id.is_empty() {
                return Err(manifest_error("missing_execution_occurrence_id", None));
            }
            if occurrence.producer_identity.is_empty() {
                return Err(manifest_error("missing_execution_producer_identity", Some(&occurrence.occurrence_id)));
            }
            if !seen.insert(occurrence.occurrence_id.as_str()) {
                return Err(manifest_error("duplicate_execution_occurrence_id", Some(&occurrence.occurrence_id)));
            }
            if occurrence.execution_scope_fingerprint.is_empty() || occurrence.output_schema_fingerprint.is_empty() {
                return Err(manifest_error(
                    "incomplete_execution_occurrence_identity",
                    Some(&occurrence.occurrence_id),
                ));
            }
            for predecessor in &occurrence.predecessors {
                if predecessor == &occurrence.occurrence_id || !seen.contains(predecessor.as_str()) {
                    return Err(manifest_error("invalid_execution_predecessor", Some(&occurrence.occurrence_id)));
                }
            }
        }
        Ok(())
    }
}

/// One producer occurrence bound to an admitted consumer scope.
#[derive(Debug, Clone, PartialEq, Eq)]
#[doc(hidden)]
pub struct AttemptProducerOccurrence {
    occurrence_id: String,
    execution_scope_fingerprint: String,
    deferred_execution_scope_identity: Option<DeferredGraphExecutionScopeIdentity>,
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
            deferred_execution_scope_identity: None,
            output_schema_fingerprint: output_schema_fingerprint.into(),
        }
    }

    #[doc(hidden)]
    pub fn with_deferred_graph_identity(mut self, identity: Option<DeferredGraphExecutionScopeIdentity>) -> Self {
        self.deferred_execution_scope_identity = identity;
        self
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

    #[doc(hidden)]
    pub fn matches_execution_plan(&self, plan: &QueryPlan) -> bool {
        self.execution_scope_fingerprint == canonical_execution_scope_fingerprint(plan)
            || self.deferred_execution_scope_identity.as_ref().is_some_and(|identity| identity.matches(plan))
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
        "query_engine.admission.execution_scope",
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
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;
    use crate::{
        plan::QueryPlanBuilder,
        source::{GpuResidentSource, SourceNode},
    };

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

    fn resident_plan(name: &str, row_bound: Option<u64>) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("vertex", DataType::Int64, false)]));
        let mut source = GpuResidentSource::new(name, schema);
        if let Some(row_bound) = row_bound {
            source = source.with_row_count_upper_bound(row_bound);
        }
        let mut builder = QueryPlanBuilder::new();
        let root = builder.add_source(SourceNode::GpuResident(source));
        builder.finish_gpu_resident(root)
    }

    #[test]
    fn execution_manifest_rejects_duplicate_and_forward_predecessors() {
        let plan = resident_plan("edges", None);
        let duplicate = AttemptExecutionManifest::new(vec![
            AttemptExecutionOccurrence::host_arrow("gpu-0", &plan, vec![], true),
            AttemptExecutionOccurrence::host_arrow("gpu-0", &plan, vec![], true),
        ])
        .expect_err("duplicate occurrence ids must reject");
        assert_eq!(duplicate.fact_value("reason"), Some("duplicate_execution_occurrence_id"));

        let forward = AttemptExecutionManifest::new(vec![AttemptExecutionOccurrence::host_arrow(
            "gpu-1",
            &plan,
            vec!["gpu-0".to_string()],
            true,
        )])
        .expect_err("predecessors must already exist in topological order");
        assert_eq!(forward.fact_value("reason"), Some("invalid_execution_predecessor"));
    }

    #[test]
    fn host_occurrence_validation_requires_id_plan_schema_and_declared_refinement() {
        let admitted = resident_plan("edges", None);
        let manifest = AttemptExecutionManifest::new(vec![AttemptExecutionOccurrence::host_arrow(
            "gpu-0",
            &admitted,
            vec![],
            true,
        )])
        .unwrap();

        assert!(manifest.validate_host_occurrence("gpu-0", &resident_plan("edges", Some(12))).is_ok());
        let wrong_id = manifest.validate_host_occurrence("gpu-1", &admitted).unwrap_err();
        assert_eq!(wrong_id.fact_value("reason"), Some("execution_occurrence_not_admitted"));
        let wrong_plan = manifest.validate_host_occurrence("gpu-0", &resident_plan("other", Some(12))).unwrap_err();
        assert_eq!(wrong_plan.fact_value("reason"), Some("execution_occurrence_plan_mismatch"));
    }
}
