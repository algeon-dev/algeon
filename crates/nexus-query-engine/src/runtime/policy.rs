//! Runtime compatibility re-exports for execution policy contracts.
//!
//! The canonical definitions live below the runtime so metrics can project
//! them without depending on runtime resource ownership.

pub use crate::execution_contracts::{
    DevicePlacement, MemoryControlPolicy, MemoryResourcePolicy, MemorySpaceId, ReservationKind, SourceChunkPlanningMode,
};
