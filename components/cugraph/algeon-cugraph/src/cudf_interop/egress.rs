use cudf::{AggSpec, Exec, GroupByAgg, Runtime, TypeId, core::Column, frame::factorization::Codebook};

use super::errors::map_cudf_error_with;
use crate::{
    centrality::{CentralityResult, EdgeBetweennessCentralityResult, HitsResult},
    community::{
        EcgResult, ExtractEgoResult, KTrussSubgraphResult, LeidenResult, LouvainResult, SpectralClusteringResult,
        TriangleCountResult,
    },
    error::{Error, ErrorKind, Result},
    labeling::LabelingResult,
    layout::LayoutResult,
    similarity::{
        CosineSimilarityResult, JaccardSimilarityResult, OverlapSimilarityResult, SorensenSimilarityResult, VertexPairs,
    },
    structure::{
        CoreNumberResult, KCoreResult,
        degrees::{DegreeValues, DegreesResult, InDegreesResult, OutDegreesResult},
        typed_results::{Distances, IntegerValues, ScoreValues, VertexIds},
    },
    traversal::{bfs::BfsResult, sssp::SsspResult},
    tree::MinimumSpanningTreeResult,
};

#[cfg(test)]
mod tests;

fn series_from_column(column: Column, operation: &'static str) -> Result<cudf::Series> {
    cudf::Series::from_column(column).map_err(|error| {
        map_cudf_error_with(
            error,
            operation,
            "failed to create cudf Series from cugraph result column",
            std::iter::empty(),
        )
    })
}

fn series_from_column_result(column: Result<Column>, operation: &'static str) -> Result<cudf::Series> {
    series_from_column(column?, operation)
}

pub(super) fn dataframe_from_named_series<I, S>(series: I, operation: &'static str) -> Result<cudf::DataFrame>
where
    I: IntoIterator<Item = (S, cudf::Series)>,
    S: Into<String>,
{
    let series = series.into_iter();
    let (series_count, _) = series.size_hint();
    cudf::DataFrame::from_series(series).map_err(|error| {
        map_cudf_error_with(
            error,
            operation,
            "failed to build cudf DataFrame from cugraph result columns",
            [("series_count", series_count.to_string())],
        )
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// to_series; result → cudf Series conversion
// ═══════════════════════════════════════════════════════════════════════════

impl VertexIds<'_> {
    /// Copies this GPU array into an unnamed cudf [`Series`](cudf::Series)
    /// bound to `runtime`.
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
    pub fn to_series(&self, runtime: &Runtime) -> Result<cudf::Series> {
        series_from_column_result(
            match self {
                Self::I32(v) => v.raw_view().to_cudf_column(runtime),
                Self::I64(v) => v.raw_view().to_cudf_column(runtime),
            },
            "VertexIds::to_series",
        )
    }
}

impl ScoreValues<'_> {
    /// Copies this GPU array into an unnamed cudf [`Series`](cudf::Series)
    /// bound to `runtime`.
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
    pub fn to_series(&self, runtime: &Runtime) -> Result<cudf::Series> {
        series_from_column_result(
            match self {
                Self::F32(v) => v.raw_view().to_cudf_column(runtime),
                Self::F64(v) => v.raw_view().to_cudf_column(runtime),
            },
            "ScoreValues::to_series",
        )
    }
}

impl Distances<'_> {
    /// Copies this GPU array into an unnamed cudf [`Series`](cudf::Series)
    /// bound to `runtime`.
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
    pub fn to_series(&self, runtime: &Runtime) -> Result<cudf::Series> {
        series_from_column_result(
            match self {
                Self::I32(v) => v.raw_view().to_cudf_column(runtime),
                Self::I64(v) => v.raw_view().to_cudf_column(runtime),
                Self::F32(v) => v.raw_view().to_cudf_column(runtime),
                Self::F64(v) => v.raw_view().to_cudf_column(runtime),
            },
            "Distances::to_series",
        )
    }
}

impl DegreeValues<'_> {
    /// Copies this GPU array into an unnamed cudf [`Series`](cudf::Series)
    /// bound to `runtime`.
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
    pub fn to_series(&self, runtime: &Runtime) -> Result<cudf::Series> {
        series_from_column_result(
            match self {
                Self::I32(v) => v.raw_view().to_cudf_column(runtime),
                Self::I64(v) => v.raw_view().to_cudf_column(runtime),
            },
            "DegreeValues::to_series",
        )
    }
}

impl IntegerValues<'_> {
    /// Copies this GPU array into an unnamed cudf [`Series`](cudf::Series)
    /// bound to `runtime`.
    ///
    /// Device-to-device copy: unavoidable because the cugraph C API owns
    /// result memory and frees it on drop. Matches Python cudf/cugraph behavior.
    pub fn to_series(&self, runtime: &Runtime) -> Result<cudf::Series> {
        series_from_column_result(
            match self {
                Self::I32(v) => v.raw_view().to_cudf_column(runtime),
                Self::I64(v) => v.raw_view().to_cudf_column(runtime),
            },
            "IntegerValues::to_series",
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TryFrom impls; result structs → cudf::DataFrame
// ═══════════════════════════════════════════════════════════════════════════

pub(crate) fn centrality_result_to_dataframe(result: &CentralityResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "centrality_result_to_dataframe";
    dataframe_from_named_series(
        [("vertex", result.vertex_ids()?.to_series(runtime)?), ("value", result.scores()?.to_series(runtime)?)],
        operation,
    )
}

pub(crate) fn hits_result_to_dataframe(result: &HitsResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "hits_result_to_dataframe";
    dataframe_from_named_series(
        [
            ("vertex", result.vertex_ids()?.to_series(runtime)?),
            ("hub", result.hub_scores()?.to_series(runtime)?),
            ("authority", result.authority_scores()?.to_series(runtime)?),
        ],
        operation,
    )
}

pub(crate) fn edge_betweenness_centrality_result_to_dataframe(
    result: &EdgeBetweennessCentralityResult,
    runtime: &Runtime,
) -> Result<cudf::DataFrame> {
    let operation = "edge_betweenness_centrality_result_to_dataframe";
    let source_series = result.sources()?.to_series(runtime)?;
    let destination_series = result.destinations()?.to_series(runtime)?;
    let edge_id_series = result.edge_ids()?.map(|values| values.to_series(runtime)).transpose()?;
    let score_series = result.scores()?.to_series(runtime)?;

    let mut series = Vec::with_capacity(4);
    series.push(("source", source_series));
    series.push(("destination", destination_series));
    if let Some(edge_id) = edge_id_series {
        series.push(("edge_id", edge_id));
    }
    series.push(("score", score_series));

    dataframe_from_named_series(series, operation)
}

pub(crate) fn louvain_result_to_dataframe(result: &LouvainResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "louvain_result_to_dataframe";
    dataframe_from_named_series(
        [("vertex", result.vertex_ids()?.to_series(runtime)?), ("partition", result.clusters()?.to_series(runtime)?)],
        operation,
    )
}

pub(crate) fn leiden_result_to_dataframe(result: &LeidenResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "leiden_result_to_dataframe";
    dataframe_from_named_series(
        [("vertex", result.vertex_ids()?.to_series(runtime)?), ("partition", result.clusters()?.to_series(runtime)?)],
        operation,
    )
}

pub(crate) fn ecg_result_to_dataframe(result: &EcgResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "ecg_result_to_dataframe";
    dataframe_from_named_series(
        [("vertex", result.vertex_ids()?.to_series(runtime)?), ("partition", result.clusters()?.to_series(runtime)?)],
        operation,
    )
}

pub(crate) fn spectral_clustering_result_to_dataframe(
    result: &SpectralClusteringResult,
    runtime: &Runtime,
) -> Result<cudf::DataFrame> {
    let operation = "spectral_clustering_result_to_dataframe";
    dataframe_from_named_series(
        [("vertex", result.vertex_ids()?.to_series(runtime)?), ("partition", result.clusters()?.to_series(runtime)?)],
        operation,
    )
}

pub(crate) fn bfs_result_to_dataframe(result: &BfsResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "bfs_result_to_dataframe";
    let vertex_series = result.vertex_ids()?.to_series(runtime)?;
    let distance_series = result.distances()?.to_series(runtime)?;

    match result.predecessors()? {
        Some(predecessors) => {
            let predecessor_series = predecessors.to_series(runtime)?;
            dataframe_from_named_series(
                [("vertex", vertex_series), ("distance", distance_series), ("predecessor", predecessor_series)],
                operation,
            )
        },
        None => dataframe_from_named_series([("vertex", vertex_series), ("distance", distance_series)], operation),
    }
}

pub(crate) fn sssp_result_to_dataframe(result: &SsspResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "sssp_result_to_dataframe";
    let vertex_series = result.vertex_ids()?.to_series(runtime)?;
    let distance_series = result.distances()?.to_series(runtime)?;

    match result.predecessors()? {
        Some(predecessors) => {
            let predecessor_series = predecessors.to_series(runtime)?;
            dataframe_from_named_series(
                [("vertex", vertex_series), ("distance", distance_series), ("predecessor", predecessor_series)],
                operation,
            )
        },
        None => dataframe_from_named_series([("vertex", vertex_series), ("distance", distance_series)], operation),
    }
}

pub(crate) fn layout_result_to_dataframe(result: &LayoutResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "layout_result_to_dataframe";
    let vertex_series = result.vertex_ids()?.to_series(runtime)?;
    let x_series = result.x()?.to_series(runtime)?;
    let y_series = result.y()?.to_series(runtime)?;

    dataframe_from_named_series([("vertex", vertex_series), ("x", x_series), ("y", y_series)], operation)
}

pub(crate) fn degrees_result_to_dataframe(result: &DegreesResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "degrees_result_to_dataframe";
    dataframe_from_named_series(
        [
            ("vertex", result.vertex_ids()?.to_series(runtime)?),
            ("in_degree", result.in_degrees()?.to_series(runtime)?),
            ("out_degree", result.out_degrees()?.to_series(runtime)?),
        ],
        operation,
    )
}

pub(crate) fn in_degrees_result_to_dataframe(result: &InDegreesResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "in_degrees_result_to_dataframe";
    dataframe_from_named_series(
        [("vertex", result.vertex_ids()?.to_series(runtime)?), ("in_degree", result.in_degrees()?.to_series(runtime)?)],
        operation,
    )
}

pub(crate) fn out_degrees_result_to_dataframe(result: &OutDegreesResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "out_degrees_result_to_dataframe";
    dataframe_from_named_series(
        [
            ("vertex", result.vertex_ids()?.to_series(runtime)?),
            ("out_degree", result.out_degrees()?.to_series(runtime)?),
        ],
        operation,
    )
}

pub(crate) fn labeling_result_to_dataframe(result: &LabelingResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "labeling_result_to_dataframe";
    let frame = dataframe_from_named_series(
        [("vertex", result.vertex_ids()?.to_series(runtime)?), ("label", result.labels()?.to_series(runtime)?)],
        operation,
    )?;
    canonicalize_component_labels(frame, operation)
}

/// Canonicalize a `("vertex", "label")` component frame by mapping each label
/// to the minimum vertex id of its component.
///
/// Raw cuGraph component labels reflect non-deterministic kernel execution order,
/// and weakly-connected components report labels in the internal renumbered space.
/// Mapping each label to `MIN(vertex)` over its component produces deterministic,
/// run-stable labels expressed in external vertex IDs while preserving row order.
fn canonicalize_component_labels(frame: cudf::DataFrame, operation: &'static str) -> Result<cudf::DataFrame> {
    if frame.num_rows() == 0 {
        return Ok(frame);
    }
    for role in ["vertex", "label"] {
        let dtype = frame.series(role).and_then(|series| series.dtype()).map_err(|error| {
            map_cudf_error_with(error, operation, "failed to inspect component integer dtype", std::iter::empty())
        })?;
        if !matches!(dtype.id(), TypeId::Int32 | TypeId::Int64) {
            return Err(Error::new(ErrorKind::UnsupportedType, operation)
                .with_message(format!("unexpected component {role} dtype: {:?}", dtype.id())));
        }
    }
    let mapping = frame
        .groupby(["label"])
        .on(Exec::RETAINED)
        .agg(&[AggSpec::new("vertex", GroupByAgg::Min).with_alias("canonical")])
        .map_err(|error| {
            map_cudf_error_with(error, operation, "failed to compute component minimum vertex ids", std::iter::empty())
        })?;
    let mapping_keys = mapping.series("label").map_err(|error| {
        map_cudf_error_with(error, operation, "failed to read component label mapping", std::iter::empty())
    })?;
    let codebook = Codebook::from_keys(mapping_keys).map_err(|error| {
        map_cudf_error_with(error, operation, "failed to index component labels", std::iter::empty())
    })?;
    let label = frame.series("label").map_err(|error| {
        map_cudf_error_with(error, operation, "failed to read component label column", std::iter::empty())
    })?;
    let codes = codebook.encode(&label, Exec::RETAINED).map_err(|error| {
        map_cudf_error_with(error, operation, "failed to locate component minimum vertex ids", std::iter::empty())
    })?;
    // Gather requires non-null indices. i32::MAX exceeds the largest valid
    // cuDF row index and preserves absent or null labels as null output rows.
    let indices = codes.on(Exec::RETAINED).fillna(i32::MAX).map_err(|error| {
        map_cudf_error_with(error, operation, "failed to preserve null component labels", std::iter::empty())
    })?;
    let canonical =
        mapping.series("canonical").and_then(|values| values.on(Exec::RETAINED).gather(&indices)).map_err(|error| {
            map_cudf_error_with(error, operation, "failed to gather canonical component labels", std::iter::empty())
        })?;
    let vertex = frame.series("vertex").map_err(|error| {
        map_cudf_error_with(error, operation, "failed to read component vertex column", std::iter::empty())
    })?;
    dataframe_from_named_series([("vertex", vertex), ("label", canonical)], operation)
}

pub(crate) fn core_number_result_to_dataframe(result: &CoreNumberResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "core_number_result_to_dataframe";
    dataframe_from_named_series(
        [
            ("vertex", result.vertex_ids()?.to_series(runtime)?),
            ("core_number", result.core_numbers()?.to_series(runtime)?),
        ],
        operation,
    )
}

pub(crate) fn triangle_count_result_to_dataframe(
    result: &TriangleCountResult,
    runtime: &Runtime,
) -> Result<cudf::DataFrame> {
    let operation = "triangle_count_result_to_dataframe";
    dataframe_from_named_series(
        [
            ("vertex", result.vertex_ids()?.to_series(runtime)?),
            ("triangle_count", result.counts()?.to_series(runtime)?),
        ],
        operation,
    )
}

fn edge_result_to_dataframe(
    sources: VertexIds<'_>,
    destinations: VertexIds<'_>,
    weights: Option<ScoreValues<'_>>,
    edge_ids: Option<IntegerValues<'_>>,
    edge_type_ids: Option<IntegerValues<'_>>,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    let source_series = sources.to_series(runtime)?;
    let destination_series = destinations.to_series(runtime)?;
    let weight_series = weights.map(|values| values.to_series(runtime)).transpose()?;
    let edge_id_series = edge_ids.map(|values| values.to_series(runtime)).transpose()?;
    let edge_type_series = edge_type_ids.map(|values| values.to_series(runtime)).transpose()?;

    let mut series = Vec::with_capacity(5);
    series.push(("source", source_series));
    series.push(("destination", destination_series));
    if let Some(weight) = weight_series {
        series.push(("weight", weight));
    }
    if let Some(edge_id) = edge_id_series {
        series.push(("edge_id", edge_id));
    }
    if let Some(edge_type) = edge_type_series {
        series.push(("edge_type", edge_type));
    }

    dataframe_from_named_series(series, operation)
}

pub(crate) fn k_core_result_to_dataframe(result: &KCoreResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    edge_result_to_dataframe(
        result.sources()?,
        result.destinations()?,
        result.weights()?,
        None,
        None,
        runtime,
        "k_core_result_to_dataframe",
    )
}

pub(crate) fn minimum_spanning_tree_result_to_dataframe(
    result: &MinimumSpanningTreeResult,
    runtime: &Runtime,
) -> Result<cudf::DataFrame> {
    edge_result_to_dataframe(
        result.sources()?,
        result.destinations()?,
        result.weights()?,
        result.edge_ids()?,
        result.edge_type_ids()?,
        runtime,
        "minimum_spanning_tree_result_to_dataframe",
    )
}

pub(crate) fn k_truss_subgraph_result_to_dataframe(
    result: &KTrussSubgraphResult,
    runtime: &Runtime,
) -> Result<cudf::DataFrame> {
    edge_result_to_dataframe(
        result.sources()?,
        result.destinations()?,
        result.weights()?,
        result.edge_ids()?,
        result.edge_type_ids()?,
        runtime,
        "k_truss_subgraph_result_to_dataframe",
    )
}

pub(crate) fn extract_ego_result_to_dataframe(result: &ExtractEgoResult, runtime: &Runtime) -> Result<cudf::DataFrame> {
    edge_result_to_dataframe(
        result.sources()?,
        result.destinations()?,
        result.weights()?,
        result.edge_ids()?,
        result.edge_type_ids()?,
        runtime,
        "extract_ego_result_to_dataframe",
    )
}

pub(crate) fn vertex_pairs_to_dataframe(result: &VertexPairs, runtime: &Runtime) -> Result<cudf::DataFrame> {
    let operation = "vertex_pairs_to_dataframe";
    dataframe_from_named_series(
        [("first", result.first()?.to_series(runtime)?), ("second", result.second()?.to_series(runtime)?)],
        operation,
    )
}

fn similarity_result_to_dataframe(
    first: VertexIds<'_>,
    second: VertexIds<'_>,
    scores: ScoreValues<'_>,
    runtime: &Runtime,
    operation: &'static str,
) -> Result<cudf::DataFrame> {
    dataframe_from_named_series(
        [
            ("first", first.to_series(runtime)?),
            ("second", second.to_series(runtime)?),
            ("similarity", scores.to_series(runtime)?),
        ],
        operation,
    )
}

pub(crate) fn jaccard_similarity_result_to_dataframe(
    result: &JaccardSimilarityResult,
    runtime: &Runtime,
) -> Result<cudf::DataFrame> {
    similarity_result_to_dataframe(
        result.first()?,
        result.second()?,
        result.scores()?,
        runtime,
        "jaccard_similarity_result_to_dataframe",
    )
}

pub(crate) fn sorensen_similarity_result_to_dataframe(
    result: &SorensenSimilarityResult,
    runtime: &Runtime,
) -> Result<cudf::DataFrame> {
    similarity_result_to_dataframe(
        result.first()?,
        result.second()?,
        result.scores()?,
        runtime,
        "sorensen_similarity_result_to_dataframe",
    )
}

pub(crate) fn overlap_similarity_result_to_dataframe(
    result: &OverlapSimilarityResult,
    runtime: &Runtime,
) -> Result<cudf::DataFrame> {
    similarity_result_to_dataframe(
        result.first()?,
        result.second()?,
        result.scores()?,
        runtime,
        "overlap_similarity_result_to_dataframe",
    )
}

pub(crate) fn cosine_similarity_result_to_dataframe(
    result: &CosineSimilarityResult,
    runtime: &Runtime,
) -> Result<cudf::DataFrame> {
    similarity_result_to_dataframe(
        result.first()?,
        result.second()?,
        result.scores()?,
        runtime,
        "cosine_similarity_result_to_dataframe",
    )
}
