//! Runtime-owned execution-occurrence lifecycle for one invocation round.

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use query_planning::{
    admission::AttemptExecutionManifest,
    error::{QueryError, QueryErrorCode, Result},
    plan::QueryPlan,
};

/// Runtime state for the compiled GPU occurrence schedule in one execution
/// round. The adapter reports DataFusion boundary events through this type;
/// dependency and duplicate-execution authority remains in the runtime.
#[derive(Debug, Clone)]
pub struct AttemptExecutionRound {
    manifest: Option<Arc<AttemptExecutionManifest>>,
    state: Arc<Mutex<AttemptExecutionRoundState>>,
}

#[derive(Debug, Default)]
struct AttemptExecutionRoundState {
    active: HashSet<String>,
    completed: HashSet<String>,
}

/// Claim for one concrete occurrence selected from the immutable schedule.
/// Dropping an incomplete claim releases it for retry without satisfying its
/// dependants.
#[derive(Debug)]
pub struct AttemptHostOccurrence {
    round: AttemptExecutionRound,
    occurrence_id: String,
    completed: bool,
}

impl AttemptExecutionRound {
    /// Starts an independent invocation round over one immutable manifest.
    #[doc(hidden)]
    pub fn new(manifest: Option<AttemptExecutionManifest>) -> Self {
        Self { manifest: manifest.map(Arc::new), state: Arc::new(Mutex::new(AttemptExecutionRoundState::default())) }
    }

    /// Claims the concrete final-plan occurrence selected by the adapter's
    /// execution-path lineage.
    #[doc(hidden)]
    pub fn begin_host_occurrence(
        &self,
        occurrence_id: &str,
        producer_identity: &str,
        plan: &QueryPlan,
    ) -> Result<Option<AttemptHostOccurrence>> {
        let Some(manifest) = self.manifest.as_deref() else {
            return Ok(None);
        };
        let predecessors = manifest.validate_host_occurrence_for_producer(occurrence_id, producer_identity, plan)?;
        let mut state = self.state.lock().map_err(|_| occurrence_error("execution_round_lock_poisoned", None))?;
        if let Some(predecessor) = predecessors.iter().find(|predecessor| !state.completed.contains(*predecessor)) {
            return Err(occurrence_error("host_occurrence_predecessor_incomplete", Some(occurrence_id))
                .with_fact("predecessor_occurrence_id", predecessor));
        }
        if state.completed.contains(occurrence_id) || !state.active.insert(occurrence_id.to_string()) {
            return Err(occurrence_error("host_occurrence_duplicate_execution", Some(occurrence_id)));
        }
        Ok(Some(AttemptHostOccurrence {
            round: self.clone(),
            occurrence_id: occurrence_id.to_string(),
            completed: false,
        }))
    }

    fn complete(&self, occurrence_id: &str) {
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        state.active.remove(occurrence_id);
        state.completed.insert(occurrence_id.to_string());
    }

    fn abandon(&self, occurrence_id: &str) {
        self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner).active.remove(occurrence_id);
    }
}

impl AttemptHostOccurrence {
    /// Marks successful production of the admitted host boundary result.
    #[doc(hidden)]
    pub fn complete(mut self) {
        self.round.complete(&self.occurrence_id);
        self.completed = true;
    }

    /// Returns the unique invocation occurrence selected for this producer.
    #[cfg(test)]
    pub(crate) fn occurrence_id(&self) -> &str {
        &self.occurrence_id
    }
}

impl Drop for AttemptHostOccurrence {
    fn drop(&mut self) {
        if !self.completed {
            self.round.abandon(&self.occurrence_id);
        }
    }
}

fn occurrence_error(reason: &'static str, occurrence_id: Option<&str>) -> QueryError {
    let error = QueryError::new(
        QueryErrorCode::InvalidPlan,
        "query_engine.attempt.execution_round",
        "a GPU function occurrence violates its admitted execution schedule",
    )
    .with_fact("reason", reason);
    match occurrence_id {
        Some(occurrence_id) => error.with_fact("occurrence_id", occurrence_id),
        None => error,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use query_planning::{
        admission::{AttemptExecutionManifest, AttemptExecutionOccurrence},
        plan::QueryPlanBuilder,
        source::{GpuResidentSource, SourceNode},
    };

    use super::*;

    fn resident_plan(name: &str) -> QueryPlan {
        let schema = Arc::new(Schema::new(vec![Field::new("value", DataType::Int64, false)]));
        let mut builder = QueryPlanBuilder::new();
        let source = builder.add_source(SourceNode::GpuResident(GpuResidentSource::new(name, schema)));
        builder.finish_with_host_sink(source).unwrap()
    }

    #[test]
    fn shared_producer_claims_unique_invocation_occurrences() {
        let plan = resident_plan("shared");
        let manifest = AttemptExecutionManifest::new(vec![
            AttemptExecutionOccurrence::host_arrow_for_producer("shared#0", "shared", &plan, vec![], false),
            AttemptExecutionOccurrence::host_arrow_for_producer("shared#1", "shared", &plan, vec![], false),
        ])
        .unwrap();
        let round = AttemptExecutionRound::new(Some(manifest));

        let first = round.begin_host_occurrence("shared#0", "shared", &plan).unwrap().unwrap();
        assert_eq!(first.occurrence_id(), "shared#0");
        first.complete();
        let second = round.begin_host_occurrence("shared#1", "shared", &plan).unwrap().unwrap();
        assert_eq!(second.occurrence_id(), "shared#1");
        second.complete();
        let error = round.begin_host_occurrence("shared#1", "shared", &plan).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("host_occurrence_duplicate_execution"));
    }

    #[test]
    fn dependency_requires_successful_predecessor_production() {
        let first_plan = resident_plan("first");
        let second_plan = resident_plan("second");
        let manifest = AttemptExecutionManifest::new(vec![
            AttemptExecutionOccurrence::host_arrow("first", &first_plan, vec![], false),
            AttemptExecutionOccurrence::host_arrow("second", &second_plan, vec!["first".to_string()], false),
        ])
        .unwrap();
        let round = AttemptExecutionRound::new(Some(manifest));

        let error = round.begin_host_occurrence("second", "second", &second_plan).unwrap_err();
        assert_eq!(error.fact_value("reason"), Some("host_occurrence_predecessor_incomplete"));
        drop(round.begin_host_occurrence("first", "first", &first_plan).unwrap().unwrap());
        assert_eq!(
            round.begin_host_occurrence("second", "second", &second_plan).unwrap_err().fact_value("reason"),
            Some("host_occurrence_predecessor_incomplete")
        );
        round.begin_host_occurrence("first", "first", &first_plan).unwrap().unwrap().complete();
        round.begin_host_occurrence("second", "second", &second_plan).unwrap().unwrap().complete();
    }

    #[test]
    fn shared_producer_lineage_satisfies_the_selected_branch() {
        let shared_plan = resident_plan("shared");
        let left_plan = resident_plan("left-consumer");
        let right_plan = resident_plan("right-consumer");
        let manifest = AttemptExecutionManifest::new(vec![
            AttemptExecutionOccurrence::host_arrow_for_producer("shared-left", "shared", &shared_plan, vec![], false),
            AttemptExecutionOccurrence::host_arrow("left-consumer", &left_plan, vec!["shared-left".to_string()], false),
            AttemptExecutionOccurrence::host_arrow_for_producer("shared-right", "shared", &shared_plan, vec![], false),
            AttemptExecutionOccurrence::host_arrow(
                "right-consumer",
                &right_plan,
                vec!["shared-right".to_string()],
                false,
            ),
        ])
        .unwrap();
        let round = AttemptExecutionRound::new(Some(manifest));

        round.begin_host_occurrence("shared-right", "shared", &shared_plan).unwrap().unwrap().complete();
        round.begin_host_occurrence("right-consumer", "right-consumer", &right_plan).unwrap().unwrap().complete();
        let left = round.begin_host_occurrence("left-consumer", "left-consumer", &left_plan).unwrap_err();
        assert_eq!(left.fact_value("predecessor_occurrence_id"), Some("shared-left"));
    }
}
