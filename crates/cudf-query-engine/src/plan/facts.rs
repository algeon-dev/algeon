use super::*;

/// Planner/executor facts attached to a node's output.
///
/// These fields describe contracts already established by lowering. They are
/// not optimizer hints, and changing them must not be used to request hidden
/// executor behavior. If behavior needs to change, add an explicit IR field or
/// operator instead.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanFacts {
    /// Schema produced by the node.
    pub output_schema: SchemaRef,
    /// Known ordering of the node output.
    ///
    /// Empty means no ordering contract is known. Sort nodes populate this
    /// directly, and order-preserving operators may carry it forward.
    pub output_ordering: Vec<SortExpr>,
    /// Physical residency of the node output.
    ///
    /// The native engine's normal boundary is GPU memory. Host residency should
    /// be introduced by an explicit source or sink contract, not inferred.
    pub placement: DataPlacement,
    /// Coarse reason this node has its output schema.
    pub schema_behavior: SchemaBehavior,
}

impl PlanFacts {
    pub(crate) fn device(output_schema: SchemaRef, schema_behavior: SchemaBehavior) -> Self {
        Self { output_schema, output_ordering: Vec::new(), placement: DataPlacement::Device, schema_behavior }
    }

    pub(super) fn with_output_ordering(mut self, output_ordering: Vec<SortExpr>) -> Self {
        self.output_ordering = output_ordering;
        self
    }
}

/// Physical residency for a node output.
///
/// Native query execution defaults to GPU-resident data. Host placement is a
/// boundary fact and should correspond to an explicit source/sink or future
/// transfer node, not an implicit materialization point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataPlacement {
    /// Output lives in device/GPU memory.
    Device,
    /// Output lives in host memory.
    Host,
}

impl fmt::Display for DataPlacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Device => "device",
            Self::Host => "host",
        })
    }
}

/// Describes how a node's output schema relates to its inputs.
///
/// This is a diagnostic and validation fact. It is intentionally coarse: the
/// concrete schema is still carried by [`PlanFacts::output_schema`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchemaBehavior {
    /// Schema is established by a source.
    SourceOutput,
    /// Schema is expected to match the input schema.
    PreservesInput,
    /// Schema is produced by projection expressions.
    Projection,
    /// Schema is produced by aggregate group and aggregate expressions.
    Aggregate,
    /// Schema is produced by join type and output projection.
    Join,
    /// Schema is shared by all union inputs.
    Union,
    /// Schema is produced by a graph algorithm result contract.
    GraphAlgorithm,
}

impl fmt::Display for SchemaBehavior {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::SourceOutput => "source_output",
            Self::PreservesInput => "preserves_input",
            Self::Projection => "projection",
            Self::Aggregate => "aggregate",
            Self::Join => "join",
            Self::Union => "union",
            Self::GraphAlgorithm => "graph_algorithm",
        })
    }
}
