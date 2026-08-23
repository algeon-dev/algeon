use super::{BoundedExecutionAttempt, CommittedPartial};

// Ownership action selected for one output slot of a top-k merge level.
pub(super) enum TopKLevelItem {
    Merged(Box<CommittedPartial>),
    Carry { partial_index: usize },
}

pub(super) fn commit_topk_merge_level(
    execution: &mut BoundedExecutionAttempt<'_>,
    candidates: Vec<CommittedPartial>,
    level_output: Vec<TopKLevelItem>,
) -> Vec<CommittedPartial> {
    let mut candidates = candidates.into_iter().map(Some).collect::<Vec<_>>();
    let mut next_candidates = Vec::with_capacity(level_output.len());

    for item in level_output {
        match item {
            TopKLevelItem::Merged(partial) => next_candidates.push(*partial),
            TopKLevelItem::Carry { partial_index } => {
                let partial = candidates[partial_index].take().expect("carry index references a top-k candidate");
                next_candidates.push(partial);
            },
        }
    }

    for partial in candidates.into_iter().flatten() {
        execution.release(partial.reservation);
    }

    next_candidates
}

pub(super) fn release_topk_level_items(execution: &mut BoundedExecutionAttempt<'_>, items: Vec<TopKLevelItem>) {
    for item in items {
        if let TopKLevelItem::Merged(partial) = item {
            let partial = *partial;
            execution.release(partial.reservation);
        }
    }
}

pub(super) fn release_committed_partials(execution: &mut BoundedExecutionAttempt<'_>, partials: Vec<CommittedPartial>) {
    for partial in partials {
        release_committed_partial(execution, partial);
    }
}

pub(super) fn release_committed_partial(execution: &mut BoundedExecutionAttempt<'_>, partial: CommittedPartial) {
    execution.release(partial.reservation);
}
