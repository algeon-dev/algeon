//! Native query result sink boundary.
//!
//! Sinks define the final residency contract for a query plan: host Arrow or a
//! GPU-resident handle. They do not imply intermediate host materialization.

use super::{NodeId, SchemaRef};

/// Explicit result boundary for a native query plan.
///
/// A sink is part of the IR so callers can choose whether the final result stays
/// GPU-resident or crosses back to Arrow on the host. Operators before the sink
/// should not materialize host output merely because they are the plan root.
#[derive(Debug, Clone, PartialEq)]
pub enum ResultSink {
    /// Return a GPU-resident handle for the sink input.
    GpuResident {
        /// Node whose output remains GPU-resident.
        input: NodeId,
    },
    /// Materialize the sink input to host Arrow batches with the given schema.
    HostArrow {
        /// Node whose output crosses the device-to-host boundary.
        input: NodeId,
        /// Host Arrow schema expected after materialization.
        schema: SchemaRef,
    },
}

impl ResultSink {
    /// Returns the node consumed by this sink.
    pub fn input(&self) -> NodeId {
        match self {
            Self::GpuResident { input } | Self::HostArrow { input, .. } => *input,
        }
    }

    /// Returns true when executing this sink requires a device-to-host boundary.
    pub fn materializes_host_boundary(&self) -> bool {
        matches!(self, Self::HostArrow { .. })
    }

    pub(super) fn kind_name(&self) -> &'static str {
        match self {
            Self::GpuResident { .. } => "gpu_resident",
            Self::HostArrow { .. } => "host_arrow",
        }
    }
}
