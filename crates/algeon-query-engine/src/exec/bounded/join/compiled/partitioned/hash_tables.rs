//! Hash tables a streamed partitioned hash join keeps over its build partitions.
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
//! that does not fit under the share, or whose table the allocator rejects, keeps no table, and
//! each probe piece builds one for it and drops it after the pair. The allocator's rejection also
//! ends retention for the partitions after it, since memory is already short.
//!
//! The tables are built on the coordinator's stream and are only read afterwards, so execution
//! lanes probe them concurrently on their own streams (`RetainedPartitionHashTables`). They are
//! released, and their charges returned, with the probe state that owns the build partitions.

use std::sync::Arc;

use query_runtime::runtime::resident::GpuHashJoinBuildHandle;

use super::{
    BoundedExecutionAttempt, GpuDataFrame, JoinNode, JoinSide, PartitionedJoinSide, Result,
    StreamedPartitionedJoinSummary, bounded_invariant_violation, compiled_keyed_equality_hash_join_options,
    cudf_query_error, join_keys_for_side, observe_device_memory_bytes,
};
use crate::exec::{
    bounded::{RetainedJoinTableCharge, join::hash::build_bounded_hash_join},
    retry::{is_retryable_resource_error, record_retryable_resource_error},
};

/// The kept partition tables of one partitioned hash join and their charges against the attempt's
/// retained share. Undecided until the first probe piece; after that the set does not change.
pub(in crate::exec::bounded::join::compiled) struct PartitionedJoinHashTables {
    tables: Option<RetainedPartitionHashTables>,
    charges: Vec<RetainedJoinTableCharge>,
}

/// Read-only view of the kept tables, indexed by build partition. `None` marks a partition joined
/// with a table built for one probe piece. Clones share the tables.
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
