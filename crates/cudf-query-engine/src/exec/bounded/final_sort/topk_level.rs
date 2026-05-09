use super::*;

pub(super) enum TopKLevelItem {
    Merged(CommittedPartial),
    Carry { partial_index: usize },
}

pub(super) fn commit_topk_merge_level(
    candidates: Vec<CommittedPartial>,
    level_output: Vec<TopKLevelItem>,
    reservations: &mut ReservationTracker,
) -> Vec<CommittedPartial> {
    let mut candidates = candidates.into_iter().map(Some).collect::<Vec<_>>();
    let mut next_candidates = Vec::with_capacity(level_output.len());

    for item in level_output {
        match item {
            TopKLevelItem::Merged(partial) => next_candidates.push(partial),
            TopKLevelItem::Carry { partial_index } => {
                let partial = candidates[partial_index].take().expect("carry index references a top-k candidate");
                next_candidates.push(partial);
            },
        }
    }

    for partial in candidates.into_iter().flatten() {
        reservations.release(partial.reservation);
    }

    next_candidates
}

pub(super) fn release_topk_level_items(items: Vec<TopKLevelItem>, reservations: &mut ReservationTracker) {
    for item in items {
        if let TopKLevelItem::Merged(partial) = item {
            reservations.release(partial.reservation);
        }
    }
}

pub(super) fn release_committed_partials(partials: Vec<CommittedPartial>, reservations: &mut ReservationTracker) {
    for partial in partials {
        reservations.release(partial.reservation);
    }
}
