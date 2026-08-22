//! Host-only preflight for raw libcugraph graph construction.
//!
//! The owner query reports the temporary workspace selected by the vendored
//! cuGraph sort dispatch. It neither accepts a resource handle nor accesses a
//! CUDA device, so callers can obtain a bound before device admission.

use std::ptr;

use cugraph_sys::{
    cugraph_bfs_side_input_workspace_preflight, cugraph_error_t, cugraph_force_atlas2_side_input_workspace_preflight,
    cugraph_graph_construction_workspace_preflight, cugraph_personalized_pagerank_side_input_workspace_preflight,
};

use crate::{
    error::{Error, ErrorKind, Result, check_host_error},
    types::{DataType, to_bool_t},
};

const OPERATION: &str = "graph_construction_workspace_preflight";
const BFS_SIDE_INPUT_OPERATION: &str = "bfs_side_input_workspace_preflight";
const PPR_SIDE_INPUT_OPERATION: &str = "personalized_pagerank_side_input_workspace_preflight";
const FORCE_ATLAS2_SIDE_INPUT_OPERATION: &str = "force_atlas2_side_input_workspace_preflight";

/// Exact raw single-GPU construction shape for the edgelist-sort workspace
/// preflight.
///
/// `edge_type` is the internal `edge_t` used by raw SG graph construction,
/// not an edge-ID column dtype. The actual ingress dispatches it as the same
/// type as the endpoint vertices, so the constructor preserves that invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct GraphConstructionPreflightInput {
    vertex_type: DataType,
    edge_type: DataType,
    weight_type: Option<DataType>,
    has_edge_ids: bool,
    store_transposed: bool,
    renumber: bool,
    num_edges: usize,
    num_vertices: usize,
}

impl GraphConstructionPreflightInput {
    /// Create a preflight input for raw single-GPU graph construction.
    ///
    /// `weight_type` is `None` for an unweighted graph. Present weights must
    /// match the safe raw graph ingress contract (`Float32` or `Float64`).
    /// The optional edge-ID property uses `edge_type` in that ingress, so it
    /// needs no independent dtype field.
    pub fn new(
        vertex_type: DataType,
        edge_type: DataType,
        weight_type: Option<DataType>,
        has_edge_ids: bool,
        store_transposed: bool,
        renumber: bool,
        num_edges: usize,
        num_vertices: usize,
    ) -> Result<Self> {
        if !vertex_type.is_supported_index_type() {
            return Err(Error::new(ErrorKind::InvalidInput, "GraphConstructionPreflightInput::new")
                .with_message(format!("vertex_type must be Int32 or Int64, got {vertex_type}")));
        }
        if edge_type != vertex_type {
            return Err(Error::new(ErrorKind::InvalidInput, "GraphConstructionPreflightInput::new")
                .with_message("edge_type must match vertex_type for raw single-GPU graph construction")
                .with_context("vertex_type", vertex_type.to_string())
                .with_context("edge_type", edge_type.to_string()));
        }
        if let Some(weight_type) = weight_type
            && !weight_type.is_float()
        {
            return Err(Error::new(ErrorKind::InvalidInput, "GraphConstructionPreflightInput::new")
                .with_message(format!("weight_type must be Float32 or Float64, got {weight_type}")));
        }

        let maximum_edges = match edge_type {
            DataType::Int32 => (i32::MAX - 1) as u128,
            DataType::Int64 => i64::MAX as u128,
            _ => unreachable!("validated raw graph index dtype"),
        };
        if (num_edges as u128) > maximum_edges {
            return Err(Error::new(ErrorKind::InvalidInput, "GraphConstructionPreflightInput::new")
                .with_message(format!("num_edges exceeds the {edge_type} raw graph construction limit"))
                .with_context("num_edges", num_edges.to_string())
                .with_context("edge_type", edge_type.to_string()));
        }
        if (num_vertices as u128) > maximum_edges {
            return Err(Error::new(ErrorKind::InvalidInput, "GraphConstructionPreflightInput::new")
                .with_message(format!("num_vertices exceeds the {vertex_type} raw graph construction limit"))
                .with_context("num_vertices", num_vertices.to_string())
                .with_context("vertex_type", vertex_type.to_string()));
        }

        Ok(Self {
            vertex_type,
            edge_type,
            weight_type,
            has_edge_ids,
            store_transposed,
            renumber,
            num_edges,
            num_vertices,
        })
    }

    /// The raw graph's endpoint dtype.
    pub const fn vertex_type(self) -> DataType {
        self.vertex_type
    }

    /// The raw graph's internal edge-index dtype.
    pub const fn edge_type(self) -> DataType {
        self.edge_type
    }

    /// The edge-weight dtype, or `None` for an unweighted graph.
    pub const fn weight_type(self) -> Option<DataType> {
        self.weight_type
    }

    /// Whether raw construction materializes the edge-ID property.
    pub const fn has_edge_ids(self) -> bool {
        self.has_edge_ids
    }

    /// Whether raw construction stores the graph transposed.
    pub const fn store_transposed(self) -> bool {
        self.store_transposed
    }

    /// Whether construction renumbers external vertex IDs.
    pub const fn renumber(self) -> bool {
        self.renumber
    }

    /// Number of input edges passed to raw construction.
    pub const fn num_edges(self) -> usize {
        self.num_edges
    }

    /// Upper bound on distinct vertices.
    pub const fn num_vertices(self) -> usize {
        self.num_vertices
    }
}

/// Owner-provided phase bounds for raw single-GPU graph construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct GraphConstructionPreflight {
    renumber_workspace_bytes: u64,
    compression_workspace_bytes: u64,
}

impl GraphConstructionPreflight {
    /// Peak renumber workspace beyond the owned input edge columns.
    pub const fn renumber_workspace_bytes(self) -> u64 {
        self.renumber_workspace_bytes
    }

    /// Peak sort-and-compress temporary workspace.
    pub const fn compression_workspace_bytes(self) -> u64 {
        self.compression_workspace_bytes
    }
}

/// Return the owner-provided upper bound for raw edgelist-sort temporary
/// workspace.
///
/// This is a pure host query: it does not require a resource handle, allocate
/// device memory, or access CUDA. An unavailable or incompatible native
/// contract is returned as a structured cuGraph error so callers can remain
/// fail-closed.
pub fn graph_construction_workspace_preflight(
    input: &GraphConstructionPreflightInput,
) -> Result<GraphConstructionPreflight> {
    let mut renumber_workspace_bytes = 0usize;
    let mut compression_workspace_bytes = 0usize;
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    let weight_type = input.weight_type.unwrap_or(DataType::Float32);

    // SAFETY: every scalar originates from a validated safe input; both out
    // pointers are writable for the duration of this host-only C API call.
    let code = unsafe {
        cugraph_graph_construction_workspace_preflight(
            input.vertex_type.into(),
            input.edge_type.into(),
            to_bool_t(input.weight_type.is_some()),
            weight_type.into(),
            to_bool_t(input.has_edge_ids),
            to_bool_t(input.store_transposed),
            to_bool_t(input.renumber),
            input.num_edges,
            input.num_vertices,
            &mut renumber_workspace_bytes,
            &mut compression_workspace_bytes,
            &mut error,
        )
    };
    // SAFETY: the C API owns an error object only on failure; this host-only
    // checker consumes it exactly once without probing CUDA state.
    unsafe { check_host_error(code, error, OPERATION) }?;

    Ok(GraphConstructionPreflight {
        renumber_workspace_bytes: u64::try_from(renumber_workspace_bytes).map_err(|_| {
            Error::new(ErrorKind::Unexpected, OPERATION)
                .with_message("native size_t renumber bound does not fit the public u64 result")
        })?,
        compression_workspace_bytes: u64::try_from(compression_workspace_bytes).map_err(|_| {
            Error::new(ErrorKind::Unexpected, OPERATION)
                .with_message("native size_t compression bound does not fit the public u64 result")
        })?,
    })
}

/// Exact side-input shape consumed by single-GPU cuGraph BFS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct BfsSideInputPreflightInput {
    vertex_type: DataType,
    edge_type: DataType,
    num_vertices: usize,
    num_edges: usize,
    has_sources: bool,
    source_rows: usize,
    has_include_vertices: bool,
    include_vertices_rows: usize,
    has_exclude_vertices: bool,
    exclude_vertices_rows: usize,
    has_target_vertices: bool,
    target_vertices_rows: usize,
    has_include_edge_ids: bool,
    include_edge_ids_rows: usize,
}

impl BfsSideInputPreflightInput {
    /// Construct a validated BFS side-input shape. Predicate flags preserve
    /// the distinction between an absent input and a present empty frame.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vertex_type: DataType,
        edge_type: DataType,
        num_vertices: usize,
        num_edges: usize,
        source_rows: usize,
        include_vertices_rows: Option<usize>,
        exclude_vertices_rows: Option<usize>,
        target_vertices_rows: Option<usize>,
        include_edge_ids_rows: Option<usize>,
    ) -> Result<Self> {
        if !vertex_type.is_supported_index_type() || edge_type != vertex_type {
            return Err(Error::new(ErrorKind::UnsupportedType, BFS_SIDE_INPUT_OPERATION)
                .with_message("BFS vertex_type and edge_type must be the same Int32 or Int64 type"));
        }
        if source_rows == 0 {
            return Err(Error::new(ErrorKind::InvalidInput, BFS_SIDE_INPUT_OPERATION)
                .with_message("BFS source side input must contain at least one row"));
        }
        if include_vertices_rows.is_some() && exclude_vertices_rows.is_some() {
            return Err(Error::new(ErrorKind::InvalidInput, BFS_SIDE_INPUT_OPERATION)
                .with_message("BFS include_vertices and exclude_vertices are mutually exclusive"));
        }
        Ok(Self {
            vertex_type,
            edge_type,
            num_vertices,
            num_edges,
            has_sources: true,
            source_rows,
            has_include_vertices: include_vertices_rows.is_some(),
            include_vertices_rows: include_vertices_rows.unwrap_or(0),
            has_exclude_vertices: exclude_vertices_rows.is_some(),
            exclude_vertices_rows: exclude_vertices_rows.unwrap_or(0),
            has_target_vertices: target_vertices_rows.is_some(),
            target_vertices_rows: target_vertices_rows.unwrap_or(0),
            has_include_edge_ids: include_edge_ids_rows.is_some(),
            include_edge_ids_rows: include_edge_ids_rows.unwrap_or(0),
        })
    }

    pub const fn vertex_type(self) -> DataType {
        self.vertex_type
    }
    pub const fn edge_type(self) -> DataType {
        self.edge_type
    }
    pub const fn num_vertices(self) -> usize {
        self.num_vertices
    }
    pub const fn num_edges(self) -> usize {
        self.num_edges
    }
    pub const fn source_rows(self) -> usize {
        self.source_rows
    }
    pub fn include_vertices_rows(self) -> Option<usize> {
        self.has_include_vertices.then_some(self.include_vertices_rows)
    }
    pub fn exclude_vertices_rows(self) -> Option<usize> {
        self.has_exclude_vertices.then_some(self.exclude_vertices_rows)
    }
    pub fn target_vertices_rows(self) -> Option<usize> {
        self.has_target_vertices.then_some(self.target_vertices_rows)
    }
    pub fn include_edge_ids_rows(self) -> Option<usize> {
        self.has_include_edge_ids.then_some(self.include_edge_ids_rows)
    }
}

/// Owner-reported BFS side-input allocation components and their combined peak.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct BfsSideInputPreflight {
    source_copy_bytes: u64,
    predicate_bitmap_bytes: u64,
    edge_mask_bytes: u64,
    peak_bytes: u64,
}

impl BfsSideInputPreflight {
    pub const fn source_copy_bytes(self) -> u64 {
        self.source_copy_bytes
    }
    pub const fn predicate_bitmap_bytes(self) -> u64 {
        self.predicate_bitmap_bytes
    }
    pub const fn edge_mask_bytes(self) -> u64 {
        self.edge_mask_bytes
    }
    pub const fn peak_bytes(self) -> u64 {
        self.peak_bytes
    }

    /// Additional graph-execution bytes beyond borrowed staged side frames.
    /// The value is the owner-computed co-resident copy/bitmap/mask peak.
    pub const fn additional_peak_bytes(self) -> u64 {
        self.peak_bytes
    }
}

/// Query the owner-provided BFS side-input workspace bound without CUDA/RMM allocation.
pub fn bfs_side_input_workspace_preflight(input: &BfsSideInputPreflightInput) -> Result<BfsSideInputPreflight> {
    let mut source_copy_bytes = 0usize;
    let mut predicate_bitmap_bytes = 0usize;
    let mut edge_mask_bytes = 0usize;
    let mut workspace_bytes = 0usize;
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    // SAFETY: validated scalar input and writable output pointers are passed to
    // the host-only C contract; it does not access a CUDA device.
    let code = unsafe {
        cugraph_bfs_side_input_workspace_preflight(
            input.vertex_type.into(),
            input.edge_type.into(),
            input.num_vertices,
            input.num_edges,
            to_bool_t(input.has_sources),
            input.source_rows,
            to_bool_t(input.has_include_vertices),
            input.include_vertices_rows,
            to_bool_t(input.has_exclude_vertices),
            input.exclude_vertices_rows,
            to_bool_t(input.has_target_vertices),
            input.target_vertices_rows,
            to_bool_t(input.has_include_edge_ids),
            input.include_edge_ids_rows,
            &mut source_copy_bytes,
            &mut predicate_bitmap_bytes,
            &mut edge_mask_bytes,
            &mut workspace_bytes,
            &mut error,
        )
    };
    // SAFETY: the C API owns an error object only on failure; consume it once.
    unsafe { check_host_error(code, error, BFS_SIDE_INPUT_OPERATION) }?;
    Ok(BfsSideInputPreflight {
        source_copy_bytes: u64::try_from(source_copy_bytes).map_err(|_| {
            Error::new(ErrorKind::Unexpected, BFS_SIDE_INPUT_OPERATION)
                .with_message("native size_t source-copy bound does not fit u64")
        })?,
        predicate_bitmap_bytes: u64::try_from(predicate_bitmap_bytes).map_err(|_| {
            Error::new(ErrorKind::Unexpected, BFS_SIDE_INPUT_OPERATION)
                .with_message("native size_t predicate-bitmap bound does not fit u64")
        })?,
        edge_mask_bytes: u64::try_from(edge_mask_bytes).map_err(|_| {
            Error::new(ErrorKind::Unexpected, BFS_SIDE_INPUT_OPERATION)
                .with_message("native size_t edge-mask bound does not fit u64")
        })?,
        peak_bytes: u64::try_from(workspace_bytes).map_err(|_| {
            Error::new(ErrorKind::Unexpected, BFS_SIDE_INPUT_OPERATION)
                .with_message("native size_t workspace bound does not fit u64")
        })?,
    })
}

/// Exact personalization side-input shape copied by single-GPU cuGraph PageRank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct PersonalizedPageRankSideInputPreflightInput {
    vertex_type: DataType,
    weight_type: DataType,
    rows: usize,
}

impl PersonalizedPageRankSideInputPreflightInput {
    /// Construct a validated personalization shape.
    pub fn new(vertex_type: DataType, weight_type: DataType, rows: usize) -> Result<Self> {
        if !vertex_type.is_supported_index_type() {
            return Err(Error::new(ErrorKind::UnsupportedType, PPR_SIDE_INPUT_OPERATION)
                .with_message(format!("personalization vertex type must be Int32 or Int64, got {vertex_type}")));
        }
        if !weight_type.is_float() {
            return Err(Error::new(ErrorKind::UnsupportedType, PPR_SIDE_INPUT_OPERATION)
                .with_message(format!("personalization value type must be Float32 or Float64, got {weight_type}")));
        }
        Ok(Self { vertex_type, weight_type, rows })
    }

    pub const fn vertex_type(self) -> DataType {
        self.vertex_type
    }
    pub const fn weight_type(self) -> DataType {
        self.weight_type
    }
    pub const fn rows(self) -> usize {
        self.rows
    }
}

/// Owner-reported PageRank personalization copy/workspace bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct PersonalizedPageRankSideInputPreflight {
    copy_bytes: u64,
    peak_bytes: u64,
}

impl PersonalizedPageRankSideInputPreflight {
    pub const fn copy_bytes(self) -> u64 {
        self.copy_bytes
    }
    pub const fn peak_bytes(self) -> u64 {
        self.peak_bytes
    }

    /// Additional graph-execution bytes beyond the borrowed personalization frame.
    pub const fn additional_peak_bytes(self) -> u64 {
        self.peak_bytes
    }
}

/// Query the owner-provided PageRank personalization workspace bound without allocation.
pub fn personalized_pagerank_side_input_workspace_preflight(
    input: &PersonalizedPageRankSideInputPreflightInput,
) -> Result<PersonalizedPageRankSideInputPreflight> {
    let mut copy_bytes = 0usize;
    let mut workspace_bytes = 0usize;
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    // SAFETY: validated scalar input and writable outputs for a host-only C API.
    let code = unsafe {
        cugraph_personalized_pagerank_side_input_workspace_preflight(
            input.vertex_type.into(),
            input.weight_type.into(),
            to_bool_t(true),
            input.rows,
            &mut copy_bytes,
            &mut workspace_bytes,
            &mut error,
        )
    };
    // SAFETY: consume the native error object once on failure.
    unsafe { check_host_error(code, error, PPR_SIDE_INPUT_OPERATION) }?;
    Ok(PersonalizedPageRankSideInputPreflight {
        copy_bytes: u64::try_from(copy_bytes).map_err(|_| {
            Error::new(ErrorKind::Unexpected, PPR_SIDE_INPUT_OPERATION)
                .with_message("native size_t personalization bound does not fit u64")
        })?,
        peak_bytes: u64::try_from(workspace_bytes).map_err(|_| {
            Error::new(ErrorKind::Unexpected, PPR_SIDE_INPUT_OPERATION)
                .with_message("native size_t personalization workspace does not fit u64")
        })?,
    })
}

/// Exact optional side-input shape copied and reordered by single-GPU ForceAtlas2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct ForceAtlas2SideInputPreflightInput {
    vertex_type: DataType,
    initial_position_rows: Option<usize>,
    vertex_attribute_rows: Option<usize>,
    has_radius: bool,
    has_mobility: bool,
    has_mass: bool,
}

impl ForceAtlas2SideInputPreflightInput {
    pub fn new(
        vertex_type: DataType,
        initial_position_rows: Option<usize>,
        vertex_attribute_rows: Option<usize>,
        has_radius: bool,
        has_mobility: bool,
        has_mass: bool,
    ) -> Result<Self> {
        if !vertex_type.is_supported_index_type() {
            return Err(Error::new(ErrorKind::UnsupportedType, FORCE_ATLAS2_SIDE_INPUT_OPERATION)
                .with_message(format!("ForceAtlas2 vertex type must be Int32 or Int64, got {vertex_type}")));
        }
        if vertex_attribute_rows.is_none() && (has_radius || has_mobility || has_mass) {
            return Err(Error::new(ErrorKind::InvalidInput, FORCE_ATLAS2_SIDE_INPUT_OPERATION)
                .with_message("ForceAtlas2 attribute flags require a vertex-attribute row bound"));
        }
        if vertex_attribute_rows.is_some() && !(has_radius || has_mobility || has_mass) {
            return Err(Error::new(ErrorKind::InvalidInput, FORCE_ATLAS2_SIDE_INPUT_OPERATION)
                .with_message("ForceAtlas2 vertex attributes require at least one value column"));
        }
        Ok(Self { vertex_type, initial_position_rows, vertex_attribute_rows, has_radius, has_mobility, has_mass })
    }
}

/// Owner-reported ForceAtlas2 side-input copy and reorder peak.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct ForceAtlas2SideInputPreflight {
    copy_bytes: u64,
    peak_bytes: u64,
}

impl ForceAtlas2SideInputPreflight {
    pub const fn copy_bytes(self) -> u64 {
        self.copy_bytes
    }

    pub const fn peak_bytes(self) -> u64 {
        self.peak_bytes
    }
}

/// Query the owner-provided ForceAtlas2 side-input bound without device allocation.
pub fn force_atlas2_side_input_workspace_preflight(
    input: &ForceAtlas2SideInputPreflightInput,
) -> Result<ForceAtlas2SideInputPreflight> {
    let mut copy_bytes = 0usize;
    let mut workspace_bytes = 0usize;
    let mut error: *mut cugraph_error_t = ptr::null_mut();
    // SAFETY: validated scalar input and writable outputs for a host-only C API.
    let code = unsafe {
        cugraph_force_atlas2_side_input_workspace_preflight(
            input.vertex_type.into(),
            to_bool_t(input.initial_position_rows.is_some()),
            input.initial_position_rows.unwrap_or(0),
            to_bool_t(input.has_radius),
            to_bool_t(input.has_mobility),
            to_bool_t(input.has_mass),
            input.vertex_attribute_rows.unwrap_or(0),
            &mut copy_bytes,
            &mut workspace_bytes,
            &mut error,
        )
    };
    // SAFETY: consume the native error object once on failure.
    unsafe { check_host_error(code, error, FORCE_ATLAS2_SIDE_INPUT_OPERATION) }?;
    Ok(ForceAtlas2SideInputPreflight {
        copy_bytes: u64::try_from(copy_bytes).map_err(|_| {
            Error::new(ErrorKind::Unexpected, FORCE_ATLAS2_SIDE_INPUT_OPERATION)
                .with_message("native size_t ForceAtlas2 copy bound does not fit u64")
        })?,
        peak_bytes: u64::try_from(workspace_bytes).map_err(|_| {
            Error::new(ErrorKind::Unexpected, FORCE_ATLAS2_SIDE_INPUT_OPERATION)
                .with_message("native size_t ForceAtlas2 workspace bound does not fit u64")
        })?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_atlas2_side_input_preflight_prices_copy_and_reorder_peak() {
        let input = ForceAtlas2SideInputPreflightInput::new(DataType::Int64, Some(17), Some(17), true, false, true)
            .expect("complete ForceAtlas2 side-input shape is valid");
        let preflight = force_atlas2_side_input_workspace_preflight(&input)
            .expect("native owner must price the ForceAtlas2 side-input shape");

        assert!(preflight.copy_bytes() > 0);
        assert!(preflight.peak_bytes() >= preflight.copy_bytes());
    }

    #[test]
    fn force_atlas2_attribute_flags_require_attribute_rows() {
        let error = ForceAtlas2SideInputPreflightInput::new(DataType::Int32, None, None, true, false, false)
            .expect_err("attribute columns without rows cannot be admitted");

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn input_preserves_the_raw_unweighted_bfs_shape() {
        let input =
            GraphConstructionPreflightInput::new(DataType::Int32, DataType::Int32, None, false, false, true, 47, 11)
                .expect("raw unweighted BFS shape is valid");

        assert_eq!(input.vertex_type(), DataType::Int32);
        assert_eq!(input.edge_type(), DataType::Int32);
        assert_eq!(input.weight_type(), None);
        assert!(!input.has_edge_ids());
        assert!(!input.store_transposed());
        assert!(input.renumber());
        assert_eq!(input.num_edges(), 47);
        assert_eq!(input.num_vertices(), 11);
    }

    #[test]
    fn input_accepts_weighted_transposed_int64_shape() {
        let input = GraphConstructionPreflightInput::new(
            DataType::Int64,
            DataType::Int64,
            Some(DataType::Float64),
            true,
            true,
            true,
            47,
            11,
        )
        .expect("raw weighted Int64 shape is valid");

        assert_eq!(input.weight_type(), Some(DataType::Float64));
        assert!(input.has_edge_ids());
        assert!(input.store_transposed());
    }

    #[test]
    fn input_rejects_a_non_index_vertex_type() {
        let error =
            GraphConstructionPreflightInput::new(DataType::Float32, DataType::Float32, None, false, false, true, 1, 1)
                .expect_err("raw graph endpoints must be integer indices");

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert_eq!(error.operation(), "GraphConstructionPreflightInput::new");
    }

    #[test]
    fn input_rejects_an_edge_type_that_does_not_match_the_raw_ingress() {
        let error =
            GraphConstructionPreflightInput::new(DataType::Int32, DataType::Int64, None, false, false, true, 1, 1)
                .expect_err("raw single-GPU ingress uses the vertex dtype for edge_t");

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(error.message().contains("edge_type must match vertex_type"));
    }

    #[test]
    fn input_rejects_a_non_float_weight_type() {
        let error = GraphConstructionPreflightInput::new(
            DataType::Int32,
            DataType::Int32,
            Some(DataType::Int32),
            false,
            false,
            true,
            1,
            1,
        )
        .expect_err("safe raw graph ingress accepts only floating edge weights");

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(error.message().contains("weight_type must be Float32 or Float64"));
    }

    #[test]
    fn input_rejects_an_int32_edge_count_that_graph_construction_cannot_represent() {
        let error = GraphConstructionPreflightInput::new(
            DataType::Int32,
            DataType::Int32,
            None,
            false,
            false,
            true,
            i32::MAX as usize,
            1,
        )
        .expect_err("the raw C API requires the Int32 edge count to be strictly below INT32_MAX");

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert!(error.message().contains("num_edges exceeds"));
    }

    #[test]
    fn zero_edge_preflight_is_a_host_only_query() {
        let input =
            GraphConstructionPreflightInput::new(DataType::Int32, DataType::Int32, None, false, false, true, 0, 0)
                .expect("empty raw graph shape is valid");

        let preflight = graph_construction_workspace_preflight(&input).expect("host preflight succeeds");
        assert_eq!(preflight.renumber_workspace_bytes(), 0);
        assert_eq!(preflight.compression_workspace_bytes(), size_of::<i32>() as u64);
    }

    #[test]
    fn edge_id_and_plain_construction_use_the_same_owner_preflight() {
        let input = |has_edge_ids| {
            GraphConstructionPreflightInput::new(
                DataType::Int64,
                DataType::Int64,
                Some(DataType::Float64),
                has_edge_ids,
                false,
                true,
                47,
                11,
            )
            .expect("valid graph shape")
        };
        let plain = graph_construction_workspace_preflight(&input(false)).expect("plain preflight");
        let edge_ids = graph_construction_workspace_preflight(&input(true)).expect("edge-ID preflight");
        assert_eq!(edge_ids, plain);
    }

    #[test]
    fn bfs_side_input_preflight_reports_owner_copy_and_bitmap_peak() {
        let input = BfsSideInputPreflightInput::new(
            DataType::Int32,
            DataType::Int32,
            65,
            33,
            2,
            Some(3),
            None,
            Some(1),
            Some(5),
        )
        .expect("valid BFS side-input shape");
        let preflight = bfs_side_input_workspace_preflight(&input).expect("BFS side-input preflight succeeds");
        assert_eq!(preflight.source_copy_bytes(), 8);
        assert_eq!(preflight.predicate_bitmap_bytes(), 24);
        assert_eq!(preflight.edge_mask_bytes(), 8);
        assert_eq!(preflight.peak_bytes(), 60);
    }

    #[test]
    fn personalized_pagerank_preflight_reports_two_owned_copies() {
        let input = PersonalizedPageRankSideInputPreflightInput::new(DataType::Int32, DataType::Float64, 3)
            .expect("valid personalization shape");
        let preflight =
            personalized_pagerank_side_input_workspace_preflight(&input).expect("personalization preflight succeeds");
        assert_eq!(preflight.copy_bytes(), 36);
        assert_eq!(preflight.peak_bytes(), 36);
    }

    #[test]
    fn bfs_side_input_preflight_rejects_conflicting_vertex_predicates() {
        let error =
            BfsSideInputPreflightInput::new(DataType::Int32, DataType::Int32, 1, 1, 1, Some(1), Some(1), None, None)
                .expect_err("include and exclude predicates cannot coexist");
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn side_input_preflight_rejects_unsupported_types() {
        let error = PersonalizedPageRankSideInputPreflightInput::new(DataType::Float32, DataType::Float32, 1)
            .expect_err("personalization vertices must be an index type");
        assert_eq!(error.kind(), ErrorKind::UnsupportedType);
    }
}
