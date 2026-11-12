//! Hash tables a streamed partitioned hash join keeps over its build partitions, and the probe
//! slices it defers for the partitions that keep none.
//!
//! A streamed probe pairs every probe piece with every build partition. Built per pair, the table of
//! a build partition would be built once per probe piece, and a downstream partitioned join would
//! multiply that again by its own piece count. The probe owner therefore builds each partition's
//! table once, when the first probe piece arrives, and joins every later piece against it.
//!
//! Kept tables count against two limits. The attempt's retained total stays within the share that
//! `BoundedPartitionTargetPolicy::partitioned_join_retained_hash_table_bytes` sets
//! (`BoundedExecutionAttempt::charge_retained_join_table`), with each table charged at its
//! pre-build reservation size plus the key columns the table copies. The query domain's allocator
//! still enforces the grant: a kept table is an ordinary allocation of the attempt. A partition
//! that does not fit under the share, or whose table the allocator rejects, keeps no table. The
//! allocator's rejection also ends retention for the partitions after it, since memory is already
//! short.
//!
//! A partition without a kept table builds one table per batch of probe input instead of one per
//! piece. The probe owner copies that partition's rows out of each piece and holds them
//! (`DeferredProbePartitions`). The slices all joins of the attempt hold stay within one
//! partitioned-join target (`BoundedExecutionAttempt::deferred_join_probe_charge`). When a piece's
//! slices would pass it, the partitions with the most held bytes are joined first, each as one
//! concatenated batch against a table built for it and dropped after it, until the rest fits. The
//! slices still held when the probe input ends are joined then. Inner joins and left joins that
//! preserve the probe side are the only streamed shapes, and each probe row's output depends only
//! on its build partition, so the order in which slices are joined does not change the result.
//! Execution lanes hold nothing: they return their slices to the coordinator, which holds and joins
//! them (`DeferredProbeSlices`).
//!
//! The tables are built on the coordinator's stream and are only read afterwards, so execution
//! lanes probe them concurrently on their own streams (`RetainedPartitionHashTables`). They are
//! released, and their charges returned, with the probe state that owns the build partitions.

use std::sync::Arc;

use query_runtime::runtime::resident::GpuHashJoinBuildHandle;

use super::{
    BoundedExecutionAttempt, BoundedJoinSideFrame, GpuDataFrame, JoinNode, JoinSide, PartitionedJoinPartitionOutput,
    PartitionedJoinSide, Result, StreamedPartitionedJoinSummary, bounded_invariant_violation,
    compiled_keyed_equality_hash_join_options, cudf_query_error, join_keys_for_side, observe_device_memory_bytes,
};
use crate::exec::{
    bounded::{DeferredJoinProbeCharge, RetainedJoinTableCharge, join::hash::build_bounded_hash_join},
    retry::{is_retryable_resource_error, record_retryable_resource_error},
};

/// The kept partition tables of one partitioned hash join and their charges against the attempt's
/// retained share. Undecided until the first probe piece; after that the set does not change.
pub(in crate::exec::bounded::join::compiled) struct PartitionedJoinHashTables {
    tables: Option<RetainedPartitionHashTables>,
    charges: Vec<RetainedJoinTableCharge>,
}

/// Read-only view of the kept tables, indexed by build partition. `None` marks a partition joined
/// with a table built for one batch of probe input. Clones share the tables.
#[derive(Clone)]
pub(super) struct RetainedPartitionHashTables(Arc<[Option<GpuHashJoinBuildHandle>]>);

impl RetainedPartitionHashTables {
    pub(super) fn table(&self, partition: usize) -> Option<&GpuHashJoinBuildHandle> {
        self.0.get(partition).and_then(Option::as_ref)
    }
}

impl PartitionedJoinHashTables {
    pub(super) fn undecided() -> Self {
        Self { tables: None, charges: Vec::new() }
    }

    /// A decided set over tables another owner keeps, for a lane that probes them. It holds no
    /// charge.
    pub(super) fn shared(tables: RetainedPartitionHashTables) -> Self {
        Self { tables: Some(tables), charges: Vec::new() }
    }

    /// Returns the kept tables, deciding them on the first call.
    ///
    /// With `retain` false, which a probe of one piece passes, no table is kept. Otherwise each
    /// partition in order is sized, charged and built while it fits. The builds and the kept bytes
    /// are recorded in `summary`. A retryable allocation failure keeps the tables built so far and
    /// keeps none after it; any other failure is returned and leaves the set undecided.
    pub(super) fn decide(
        &mut self,
        execution: &mut BoundedExecutionAttempt<'_>,
        node: &JoinNode,
        build_side: JoinSide,
        build_partitioned: &PartitionedJoinSide,
        retain: bool,
        summary: &mut StreamedPartitionedJoinSummary,
    ) -> Result<RetainedPartitionHashTables> {
        if let Some(tables) = &self.tables {
            return Ok(tables.clone());
        }
        let PartitionedJoinSide::Split { partitions, .. } = build_partitioned else {
            return Err(bounded_invariant_violation(
                "partitioned hash join keeps hash tables only over split build partitions",
                "partitioned_join_build_not_split",
            ));
        };
        let mut tables = vec![None; partitions.len()];
        let mut charges = Vec::new();
        if retain {
            let build_keys = join_keys_for_side(node, build_side);
            let hash_options = compiled_keyed_equality_hash_join_options(node);
            let exec_runtime = execution.context().runtime()?;
            let overrides = exec_runtime.exec();
            for (partition, frame) in partitions.iter().enumerate() {
                let table_bytes = frame
                    .on(overrides)
                    .hash_join_pre_build_reservation_size_bytes(&build_keys, hash_options)
                    .map_err(|error| {
                        cudf_query_error(
                            "query_engine.bounded.join.partition_hash_table_size",
                            "failed to size a partitioned join build partition hash table",
                            error,
                        )
                        .with_fact("partition", partition)
                        .with_fact("build_side", build_side)
                    })?;
                let key_frame = frame.select(&build_keys).map_err(|error| {
                    cudf_query_error(
                        "query_engine.bounded.join.partition_hash_table_keys",
                        "failed to select the key columns of a partitioned join build partition",
                        error,
                    )
                    .with_fact("partition", partition)
                })?;
                let key_bytes = observe_device_memory_bytes(&GpuDataFrame::new(key_frame), execution.metrics())?;
                let Some(charge) = execution.charge_retained_join_table((table_bytes as u64).saturating_add(key_bytes))
                else {
                    continue;
                };
                match build_bounded_hash_join(
                    execution,
                    node,
                    frame,
                    &build_keys,
                    hash_options,
                    build_side,
                    overrides,
                    true,
                ) {
                    Ok((hash_join, hash_table_bytes)) => {
                        summary.record_retained_hash_table(hash_table_bytes);
                        tables[partition] = Some(hash_join);
                        charges.push(charge);
                    },
                    Err(error) if is_retryable_resource_error(&error) => {
                        record_retryable_resource_error(&error, execution.metrics());
                        break;
                    },
                    Err(error) => return Err(error.with_fact_if_absent("partition", partition)),
                }
            }
        }
        let tables = RetainedPartitionHashTables(tables.into());
        self.tables = Some(tables.clone());
        self.charges = charges;
        Ok(tables)
    }
}

/// Probe slices of the partitions without a kept table, indexed by build partition. Each slice is a
/// copy of one probe piece's rows of that partition, so the piece itself can be released.
pub(super) struct DeferredProbeSlices {
    partitions: Vec<Vec<BoundedJoinSideFrame>>,
}

impl DeferredProbeSlices {
    pub(super) fn new(partition_count: usize) -> Self {
        Self { partitions: (0..partition_count).map(|_| Vec::new()).collect() }
    }

    /// Adds one slice of `partition`. A slice without rows adds nothing to the join's output and
    /// is dropped.
    pub(super) fn push(&mut self, partition: usize, slice: BoundedJoinSideFrame) {
        if slice.rows > 0 {
            self.partitions[partition].push(slice);
        }
    }

    pub(super) fn append(&mut self, other: Self) {
        for (held, arriving) in self.partitions.iter_mut().zip(other.partitions) {
            held.extend(arriving);
        }
    }

    fn partition_bytes(&self, partition: usize) -> u64 {
        self.partitions[partition].iter().map(|slice| slice.bytes).fold(0, u64::saturating_add)
    }
}

/// The deferred probe slices a partitioned hash join holds, with their charge against the
/// attempt's deferred total.
pub(super) struct DeferredProbePartitions {
    held: DeferredProbeSlices,
    charge: DeferredJoinProbeCharge,
}

/// Arriving slices whose batches `DeferredProbePartitions::admit` joined, to be committed once the
/// caller has kept the joined output.
pub(super) struct DeferredProbeAdmission {
    arriving: DeferredProbeSlices,
    joined: Vec<usize>,
    held_bytes: u64,
}

impl DeferredProbePartitions {
    /// An empty set for `partition_count` partitions, or `None` on a lane attempt, which defers
    /// nothing of its own.
    pub(super) fn new(execution: &BoundedExecutionAttempt<'_>, partition_count: usize) -> Option<Self> {
        let charge = execution.deferred_join_probe_charge()?;
        Some(Self { held: DeferredProbeSlices::new(partition_count), charge })
    }

    /// Joins the batches that `arriving` makes too large to hold, and returns their outputs with
    /// the admission that `commit` applies.
    ///
    /// The partitions are chosen by `deferred_partitions_to_join` against what this join's charge
    /// can grow to. `join_partition` joins one partition's held and arriving slices as one batch.
    /// The held slices are shared, not moved, so a failure leaves the set as it was and drops only
    /// the arriving slices.
    pub(super) fn admit<F>(
        &self,
        arriving: DeferredProbeSlices,
        mut join_partition: F,
    ) -> Result<(Vec<PartitionedJoinPartitionOutput>, DeferredProbeAdmission)>
    where
        F: FnMut(usize, Vec<BoundedJoinSideFrame>) -> Result<Vec<PartitionedJoinPartitionOutput>>,
    {
        let combined_bytes = (0..self.held.partitions.len())
            .map(|partition| self.held.partition_bytes(partition).saturating_add(arriving.partition_bytes(partition)))
            .collect::<Vec<_>>();
        let (joined, held_bytes) = deferred_partitions_to_join(&combined_bytes, self.charge.limit_bytes());
        let mut outputs = Vec::new();
        for &partition in &joined {
            let batch = self.held.partitions[partition]
                .iter()
                .chain(&arriving.partitions[partition])
                .map(share_side_frame)
                .collect();
            outputs.extend(join_partition(partition, batch)?);
        }
        Ok((outputs, DeferredProbeAdmission { arriving, joined, held_bytes }))
    }

    /// Drops the joined partitions' slices and holds the other arriving slices.
    pub(super) fn commit(&mut self, admission: DeferredProbeAdmission) -> Result<()> {
        let DeferredProbeAdmission { mut arriving, joined, held_bytes } = admission;
        // `admit` sized the held total against this charge's limit, and only this join's probe
        // owner changes the charge in between.
        if !self.charge.try_resize(held_bytes) {
            return Err(bounded_invariant_violation(
                "deferred partitioned join probe slices no longer fit the attempt's deferred share",
                "partitioned_join_deferred_probe_over_share",
            )
            .with_fact("deferred_probe_bytes", held_bytes)
            .with_fact("deferred_probe_limit_bytes", self.charge.limit_bytes()));
        }
        for partition in joined {
            self.held.partitions[partition].clear();
            arriving.partitions[partition].clear();
        }
        self.held.append(arriving);
        Ok(())
    }

    /// Joins every partition that still holds slices, at the end of the probe input. The charge
    /// returns to the attempt with the set.
    pub(super) fn join_all<F>(self, mut join_partition: F) -> Result<Vec<PartitionedJoinPartitionOutput>>
    where F: FnMut(usize, Vec<BoundedJoinSideFrame>) -> Result<Vec<PartitionedJoinPartitionOutput>> {
        let mut outputs = Vec::new();
        for (partition, slices) in self.held.partitions.into_iter().enumerate() {
            if !slices.is_empty() {
                outputs.extend(join_partition(partition, slices)?);
            }
        }
        Ok(outputs)
    }
}

/// Partitions to join now, largest first, so that the slices left held total at most
/// `limit_bytes`; returned with that total. `combined_bytes[p]` is what partition `p` would hold
/// with the arriving slices. A partition joined now takes its arriving slices with it, so a single
/// slice above the limit is joined at once.
fn deferred_partitions_to_join(combined_bytes: &[u64], limit_bytes: u64) -> (Vec<usize>, u64) {
    let mut held_bytes = combined_bytes.iter().copied().fold(0, u64::saturating_add);
    let mut candidates = combined_bytes.iter().copied().enumerate().filter(|(_, bytes)| *bytes > 0).collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.1.cmp(&left.1).then(left.0.cmp(&right.0)));
    let mut joined = Vec::new();
    for (partition, bytes) in candidates {
        if held_bytes <= limit_bytes {
            break;
        }
        joined.push(partition);
        held_bytes = held_bytes.saturating_sub(bytes);
    }
    (joined, held_bytes)
}

/// A handle on the same device columns. It adds no device memory.
fn share_side_frame(side: &BoundedJoinSideFrame) -> BoundedJoinSideFrame {
    BoundedJoinSideFrame { frame: side.frame.clone(), rows: side.rows, bytes: side.bytes }
}

#[cfg(test)]
mod tests {
    use super::deferred_partitions_to_join;

    #[test]
    fn deferred_probe_joins_the_largest_partitions_until_the_rest_fits() {
        assert_eq!(deferred_partitions_to_join(&[30, 0, 50, 20], 100), (Vec::new(), 100));
        assert_eq!(deferred_partitions_to_join(&[30, 0, 60, 20], 100), (vec![2], 50));
        assert_eq!(deferred_partitions_to_join(&[40, 0, 40, 40], 30), (vec![0, 2, 3], 0));
        assert_eq!(deferred_partitions_to_join(&[0, 5, 0], 0), (vec![1], 0));
    }
}
