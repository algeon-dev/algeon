//! Algorithm wrappers on graphs keyed by transposition state.

use std::num::NonZeroUsize;

use super::{Forward, Graph, Transposed, Transposition};
use crate::{
    array::DeviceArrayView,
    centrality::{
        BetweennessCentralityParams, BetweennessCentralityResult, CentralityResult, EdgeBetweennessCentralityParams,
        EdgeBetweennessCentralityResult, HitsResult,
        betweenness::betweenness_centrality,
        edge_betweenness::edge_betweenness_centrality,
        hits::{HitsInitialHubs, HitsParams, hits},
        pagerank::{
            InitialGuess, PageRankParams, PersonalizationVector, PersonalizedPageRankParams, PrecomputedWeights,
            pagerank, personalized_pagerank,
        },
        vertex::{EigenvectorCentralityParams, KatzCentralityParams, eigenvector_centrality, katz_centrality},
    },
    community::{
        EcgResult, ExtractEgoParams, ExtractEgoResult, KTrussSubgraphParams, KTrussSubgraphResult, LeidenResult,
        LouvainResult, SpectralClusteringResult, TriangleCountResult,
        clustering::{
            SpectralClusteringParams, analyze_clustering_edge_cut, analyze_clustering_modularity,
            analyze_clustering_ratio_cut, spectral_modularity_maximization,
        },
        ecg::{EcgParams, ecg},
        leiden::{LeidenParams, leiden},
        louvain::{LouvainParams, louvain},
        subgraph::{extract_ego, k_truss_subgraph},
        triangle_count::{TriangleCountParams, triangle_count},
    },
    error::Result,
    handle::EnteredContext,
    labeling::{
        ConnectedComponentsParams, LabelingResult,
        connected_components::{strongly_connected_components, weakly_connected_components},
    },
    layout::force_atlas2::{ForceAtlas2Params, ForceAtlas2VertexAttrs, InitialPositions, LayoutResult, force_atlas2},
    random::RngState,
    sampling::{
        NeighborSamplingParams, Node2VecRandomWalkParams, RandomWalkParams, RandomWalkResult, SampleResult,
        biased_random_walks, homogeneous_biased_neighbor_sample, homogeneous_biased_temporal_neighbor_sample,
        homogeneous_uniform_neighbor_sample, homogeneous_uniform_temporal_neighbor_sample, node2vec_random_walks,
        uniform_random_walks,
    },
    similarity::{
        AllPairsSimilarityParams, CosineSimilarityResult, JaccardSimilarityResult, OverlapSimilarityResult,
        SimilarityParams, SorensenSimilarityResult, VertexPairs, all_pairs_cosine_similarity_coefficients,
        all_pairs_jaccard_coefficients, all_pairs_overlap_coefficients, all_pairs_sorensen_coefficients,
        cosine_similarity_coefficients, create_vertex_pairs, jaccard_coefficients, overlap_coefficients,
        sorensen_coefficients,
    },
    structure::{
        CoreNumberParams, CoreNumberResult, DegreesResult, InDegreesResult, KCoreParams, KCoreResult, OutDegreesResult,
        VertexSelection,
        core::{core_number, k_core},
        degrees::{degrees, in_degrees, out_degrees},
    },
    traversal::{
        bfs::{BfsParams, BfsPredicateParams, BfsResult, bfs, bfs_with_predicates},
        sssp::{SsspParams, SsspResult, sssp},
    },
    tree::{MinimumSpanningTreeParams, MinimumSpanningTreeResult, minimum_spanning_tree::minimum_spanning_tree},
    types::CugraphElement,
};

// ═══════════════════════════════════════════════════════════════════════════
// Algorithm methods on Graph<T>
// ═══════════════════════════════════════════════════════════════════════════

impl Graph<Transposed> {
    /// Compute PageRank on this transposed graph.
    ///
    /// Returns a [`CentralityResult`]
    /// containing vertex ids, scores, iteration count, and convergence status.
    ///
    /// Runs PageRank without auxiliary input arrays. For precomputed weights
    /// or initial scores, see [`pagerank_with`](Self::pagerank_with).
    pub fn pagerank(&mut self, handle: &EnteredContext<'_>, params: &PageRankParams) -> Result<CentralityResult> {
        pagerank::<i32, f32>(handle, self, params, None, None)
    }

    /// Compute PageRank with optional precomputed weights and/or initial guess.
    ///
    /// Accepts precomputed out-weight sums and initial score guesses for
    /// iterative refinement.
    pub fn pagerank_with<V: CugraphElement, W: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &PageRankParams,
        precomputed_weights: Option<&PrecomputedWeights<'_, V, W>>,
        initial_guess: Option<&InitialGuess<'_, V, W>>,
    ) -> Result<CentralityResult> {
        pagerank(handle, self, params, precomputed_weights, initial_guess)
    }

    /// Compute personalized PageRank using the allow-nonconvergence native API.
    pub fn personalized_pagerank<V: CugraphElement, W: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &PersonalizedPageRankParams,
        personalization: &PersonalizationVector<'_, V, W>,
    ) -> Result<CentralityResult> {
        personalized_pagerank(handle, self, params, personalization, None, None)
    }

    /// Compute personalized PageRank with optional precomputed weights and/or initial guess.
    pub fn personalized_pagerank_with<V: CugraphElement, W: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &PersonalizedPageRankParams,
        personalization: &PersonalizationVector<'_, V, W>,
        precomputed_weights: Option<&PrecomputedWeights<'_, V, W>>,
        initial_guess: Option<&InitialGuess<'_, V, W>>,
    ) -> Result<CentralityResult> {
        personalized_pagerank(handle, self, params, personalization, precomputed_weights, initial_guess)
    }

    /// Compute eigenvector centrality.
    pub fn eigenvector_centrality(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &EigenvectorCentralityParams,
    ) -> Result<CentralityResult> {
        eigenvector_centrality(handle, self, params)
    }

    /// Compute Katz centrality using a scalar beta value.
    ///
    /// The local cuGraph C API accepts a per-vertex beta array parameter, but
    /// the current implementation does not forward it to the native algorithm.
    /// This safe method therefore exposes only the scalar-beta path.
    pub fn katz_centrality(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &KatzCentralityParams,
    ) -> Result<CentralityResult> {
        katz_centrality(handle, self, params)
    }

    /// Compute HITS hub and authority scores.
    pub fn hits(&mut self, handle: &EnteredContext<'_>, params: &HitsParams) -> Result<HitsResult> {
        hits::<i32, f32>(handle, self, params, None)
    }

    /// Compute HITS hub and authority scores with an optional initial hub guess.
    pub fn hits_with_initial_hubs<V: CugraphElement, W: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &HitsParams,
        initial_hubs: Option<&HitsInitialHubs<'_, V, W>>,
    ) -> Result<HitsResult> {
        hits(handle, self, params, initial_hubs)
    }

    /// Compute both in-degrees and out-degrees for vertices in this graph.
    ///
    /// Pass [`VertexSelection::All`] to compute degrees for every vertex,
    /// [`VertexSelection::Device`] with a device view, or
    /// [`VertexSelection::Host`] with a host slice (uploaded on demand).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`](crate::ErrorKind::UnsupportedType) if `vertices` uses a vertex
    /// dtype that does not match the graph.
    pub fn degrees<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
    ) -> Result<DegreesResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::degrees", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        degrees(handle, self, view.as_ref(), false)
    }

    /// Compute degrees with optional expensive input validation.
    pub fn degrees_checked<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        do_expensive_check: bool,
    ) -> Result<DegreesResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::degrees", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        degrees(handle, self, view.as_ref(), do_expensive_check)
    }

    /// Compute degrees for every vertex in the graph.
    pub fn degrees_all(&mut self, handle: &EnteredContext<'_>) -> Result<DegreesResult> {
        self.degrees(handle, VertexSelection::<i32>::All)
    }

    /// Compute only in-degrees for vertices in this graph.
    ///
    /// Pass [`VertexSelection::All`] to compute degrees for every vertex,
    /// [`VertexSelection::Device`] with a device view, or
    /// [`VertexSelection::Host`] with a host slice (uploaded on demand).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`](crate::ErrorKind::UnsupportedType) if `vertices` uses a vertex
    /// dtype that does not match the graph.
    pub fn in_degrees<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
    ) -> Result<InDegreesResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::in_degrees", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        in_degrees(handle, self, view.as_ref(), false)
    }

    /// Compute in-degrees with optional expensive input validation.
    pub fn in_degrees_checked<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        do_expensive_check: bool,
    ) -> Result<InDegreesResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::in_degrees", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        in_degrees(handle, self, view.as_ref(), do_expensive_check)
    }

    /// Compute in-degrees for every vertex in the graph.
    pub fn in_degrees_all(&mut self, handle: &EnteredContext<'_>) -> Result<InDegreesResult> {
        self.in_degrees(handle, VertexSelection::<i32>::All)
    }
}

impl Graph<Forward> {
    /// Perform breadth-first search from one or more source vertices.
    ///
    /// Returns a [`BfsResult`] containing vertex
    /// ids, distances, and optionally predecessors.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`](crate::ErrorKind::UnsupportedType) if `sources` uses a vertex
    /// dtype that does not match the graph.
    pub fn bfs<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        sources: impl Into<VertexSelection<'a, V>>,
        params: &BfsParams,
    ) -> Result<BfsResult> {
        let sources = sources.into();
        self.validate_vertex_selection(&sources, "Graph::bfs", "sources", false)?;
        let resolved = sources.resolve(handle)?;
        let view = self.resolved_vertex_view(&resolved, "Graph::bfs", "sources")?;
        bfs(handle, self, &view, params)
    }

    /// Perform breadth-first search with optional edge, vertex, and target predicates.
    ///
    /// Predicate inputs must already be concrete device arrays. Higher layers
    /// are responsible for SQL planning, relation-backed edge alignment, and
    /// logical-to-physical vertex-id mapping before calling this API.
    pub fn bfs_with_predicates<'a, V, E>(
        &mut self,
        handle: &EnteredContext<'_>,
        sources: impl Into<VertexSelection<'a, V>>,
        params: &BfsParams,
        predicates: &BfsPredicateParams<'a, V, E>,
    ) -> Result<BfsResult>
    where
        V: CugraphElement + 'a,
        E: CugraphElement + 'a,
    {
        let sources = sources.into();
        self.validate_vertex_selection(&sources, "Graph::bfs_with_predicates", "sources", false)?;
        let resolved = sources.resolve(handle)?;
        let view = self.resolved_vertex_view(&resolved, "Graph::bfs_with_predicates", "sources")?;
        bfs_with_predicates(handle, self, &view, params, predicates)
    }

    /// Perform single-source shortest path from one source vertex.
    ///
    /// Returns an [`SsspResult`] containing
    /// vertex ids, weighted distances, and optionally predecessors.
    pub fn sssp<V>(&mut self, handle: &EnteredContext<'_>, source: V, params: &SsspParams) -> Result<SsspResult>
    where V: CugraphElement + TryInto<usize> + Copy + std::fmt::Debug {
        self.validate_vertex_input_dtype(V::data_type(), "Graph::sssp", "source")?;
        let weight_mode = if self.has_weights { "weighted" } else { "unit_weight" };
        sssp(handle, self, source, params).map_err(|error| error.with_context("weight_mode", weight_mode))
    }

    /// Compute uniform random walks from explicit start vertices.
    pub fn uniform_random_walks<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut RngState,
        start_vertices: &DeviceArrayView<'_, V>,
        params: &RandomWalkParams,
    ) -> Result<RandomWalkResult> {
        uniform_random_walks(handle, rng_state, self, start_vertices, params)
    }

    /// Compute biased random walks from explicit start vertices.
    pub fn biased_random_walks<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut RngState,
        start_vertices: &DeviceArrayView<'_, V>,
        params: &RandomWalkParams,
    ) -> Result<RandomWalkResult> {
        biased_random_walks(handle, rng_state, self, start_vertices, params)
    }

    /// Compute node2vec random walks from explicit start vertices.
    pub fn node2vec_random_walks<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut RngState,
        start_vertices: &DeviceArrayView<'_, V>,
        params: &Node2VecRandomWalkParams,
    ) -> Result<RandomWalkResult> {
        node2vec_random_walks(handle, rng_state, self, start_vertices, params)
    }

    /// Sample homogeneous neighborhoods uniformly from explicit start vertices.
    pub fn homogeneous_uniform_neighbor_sample<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut RngState,
        start_vertices: &DeviceArrayView<'_, V>,
        starting_vertex_label_offsets: Option<&DeviceArrayView<'_, usize>>,
        params: &NeighborSamplingParams,
    ) -> Result<SampleResult> {
        homogeneous_uniform_neighbor_sample(
            handle,
            rng_state,
            self,
            start_vertices,
            starting_vertex_label_offsets,
            params,
        )
    }

    /// Sample homogeneous neighborhoods with biased sampling from explicit start vertices.
    pub fn homogeneous_biased_neighbor_sample<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut RngState,
        start_vertices: &DeviceArrayView<'_, V>,
        starting_vertex_label_offsets: Option<&DeviceArrayView<'_, usize>>,
        params: &NeighborSamplingParams,
    ) -> Result<SampleResult> {
        homogeneous_biased_neighbor_sample(
            handle,
            rng_state,
            self,
            start_vertices,
            starting_vertex_label_offsets,
            params,
        )
    }

    /// Sample homogeneous temporal neighborhoods uniformly from explicit start vertices.
    pub fn homogeneous_uniform_temporal_neighbor_sample<V, Tm>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut RngState,
        temporal_property_name: &str,
        start_vertices: &DeviceArrayView<'_, V>,
        starting_vertex_times: Option<&DeviceArrayView<'_, Tm>>,
        starting_vertex_label_offsets: Option<&DeviceArrayView<'_, usize>>,
        params: &NeighborSamplingParams,
    ) -> Result<SampleResult>
    where
        V: CugraphElement,
        Tm: CugraphElement,
    {
        homogeneous_uniform_temporal_neighbor_sample(
            handle,
            rng_state,
            self,
            temporal_property_name,
            start_vertices,
            starting_vertex_times,
            starting_vertex_label_offsets,
            params,
        )
    }

    /// Sample homogeneous temporal neighborhoods with biased sampling from explicit start vertices.
    pub fn homogeneous_biased_temporal_neighbor_sample<V, Tm>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut RngState,
        temporal_property_name: &str,
        start_vertices: &DeviceArrayView<'_, V>,
        starting_vertex_times: Option<&DeviceArrayView<'_, Tm>>,
        starting_vertex_label_offsets: Option<&DeviceArrayView<'_, usize>>,
        params: &NeighborSamplingParams,
    ) -> Result<SampleResult>
    where
        V: CugraphElement,
        Tm: CugraphElement,
    {
        homogeneous_biased_temporal_neighbor_sample(
            handle,
            rng_state,
            self,
            temporal_property_name,
            start_vertices,
            starting_vertex_times,
            starting_vertex_label_offsets,
            params,
        )
    }

    /// Compute Louvain communities on this forward-stored graph.
    ///
    /// Louvain is an undirected community-detection algorithm. This wrapper
    /// preserves the crate's type-state requirement that the graph must be
    /// stored in forward (non-transposed) layout, but it does not add a new
    /// Rust-side hard check that the graph was constructed with undirected
    /// properties.
    ///
    /// Returns a [`LouvainResult`]
    /// containing vertex ids, per-vertex partitions, and modularity.
    pub fn louvain(&mut self, handle: &EnteredContext<'_>, params: &LouvainParams) -> Result<LouvainResult> {
        louvain(handle, self, params)
    }

    /// Compute Leiden communities on this forward-stored graph.
    ///
    /// Leiden is an undirected community-detection algorithm. This wrapper
    /// requires forward graph storage; libcugraph rejects a directed graph as
    /// [`ErrorKind::InvalidInput`](crate::ErrorKind::InvalidInput). Every
    /// returned community induces a connected subgraph, and the reported
    /// modularity is that of the returned partition.
    pub fn leiden(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut RngState,
        params: &LeidenParams,
    ) -> Result<LeidenResult> {
        leiden(handle, rng_state, self, params)
    }

    /// Compute ECG communities on this forward-stored graph.
    ///
    /// ECG is an undirected community-detection algorithm built from a Louvain
    /// ensemble. This wrapper requires forward graph storage but leaves
    /// graph-property validation to the native library.
    pub fn ecg(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut RngState,
        params: &EcgParams,
    ) -> Result<EcgResult> {
        ecg(handle, rng_state, self, params)
    }

    /// Compute legacy spectral modularity maximization clusters.
    ///
    /// cuGraph requires a symmetric graph for this algorithm. This wrapper
    /// keeps that validation in the native library, matching the rest of the
    /// community-detection surface.
    pub fn spectral_modularity_maximization(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut RngState,
        params: &SpectralClusteringParams,
    ) -> Result<SpectralClusteringResult> {
        spectral_modularity_maximization(handle, rng_state, self, params)
    }

    /// Compute modularity for explicit clustering columns.
    pub fn analyze_clustering_modularity<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        n_clusters: NonZeroUsize,
        vertices: &DeviceArrayView<'_, V>,
        clusters: &DeviceArrayView<'_, V>,
    ) -> Result<f64> {
        analyze_clustering_modularity(handle, self, n_clusters, vertices, clusters)
    }

    /// Compute edge cut for explicit clustering columns.
    pub fn analyze_clustering_edge_cut<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        n_clusters: NonZeroUsize,
        vertices: &DeviceArrayView<'_, V>,
        clusters: &DeviceArrayView<'_, V>,
    ) -> Result<f64> {
        analyze_clustering_edge_cut(handle, self, n_clusters, vertices, clusters)
    }

    /// Compute ratio cut for explicit clustering columns.
    pub fn analyze_clustering_ratio_cut<V: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        n_clusters: NonZeroUsize,
        vertices: &DeviceArrayView<'_, V>,
        clusters: &DeviceArrayView<'_, V>,
    ) -> Result<f64> {
        analyze_clustering_ratio_cut(handle, self, n_clusters, vertices, clusters)
    }
}

impl<T: Transposition> Graph<T> {
    /// Compute vertex betweenness centrality on this graph.
    ///
    /// Pass [`VertexSelection::All`] for exact computation, or provide seed
    /// vertices for approximate computation via [`VertexSelection::Device`] or
    /// [`VertexSelection::Host`].
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`](crate::ErrorKind::UnsupportedType) if `vertex_list` uses a vertex
    /// dtype that does not match the graph.
    pub fn betweenness_centrality<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertex_list: impl Into<VertexSelection<'a, V>>,
        params: &BetweennessCentralityParams,
    ) -> Result<BetweennessCentralityResult> {
        let vertex_list = vertex_list.into();
        self.validate_vertex_selection(&vertex_list, "Graph::betweenness_centrality", "vertex_list", true)?;
        let resolved = vertex_list.resolve(handle)?;
        let view = resolved.view()?;
        betweenness_centrality(handle, self, view.as_ref(), params)
    }

    /// Compute exact betweenness centrality over all vertices.
    pub fn betweenness_centrality_all(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &BetweennessCentralityParams,
    ) -> Result<BetweennessCentralityResult> {
        self.betweenness_centrality(handle, VertexSelection::<i32>::All, params)
    }

    /// Compute edge betweenness centrality on this graph.
    ///
    /// Pass [`VertexSelection::All`] for exact computation, or provide seed
    /// vertices for approximate computation via [`VertexSelection::Device`] or
    /// [`VertexSelection::Host`].
    pub fn edge_betweenness_centrality<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertex_list: impl Into<VertexSelection<'a, V>>,
        params: &EdgeBetweennessCentralityParams,
    ) -> Result<EdgeBetweennessCentralityResult> {
        let vertex_list = vertex_list.into();
        self.validate_vertex_selection(&vertex_list, "Graph::edge_betweenness_centrality", "vertex_list", true)?;
        let resolved = vertex_list.resolve(handle)?;
        let view = resolved.view()?;
        edge_betweenness_centrality(handle, self, view.as_ref(), params)
    }

    /// Compute exact edge betweenness centrality over all vertices.
    pub fn edge_betweenness_centrality_all(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &EdgeBetweennessCentralityParams,
    ) -> Result<EdgeBetweennessCentralityResult> {
        self.edge_betweenness_centrality(handle, VertexSelection::<i32>::All, params)
    }

    /// Compute weakly connected components.
    ///
    /// cuGraph requires the input graph to be symmetric for this algorithm.
    /// This wrapper does not add a Rust-side graph-property hard failure; use
    /// `do_expensive_check` in [`ConnectedComponentsParams`] to request native
    /// validation.
    ///
    /// [`ConnectedComponentsParams`]: crate::labeling::ConnectedComponentsParams
    pub fn weakly_connected_components(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &ConnectedComponentsParams,
    ) -> Result<LabelingResult> {
        weakly_connected_components(handle, self, params)
    }

    /// Compute strongly connected components.
    pub fn strongly_connected_components(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &ConnectedComponentsParams,
    ) -> Result<LabelingResult> {
        strongly_connected_components(handle, self, params)
    }

    /// Compute core numbers for all vertices.
    pub fn core_number(&mut self, handle: &EnteredContext<'_>, params: &CoreNumberParams) -> Result<CoreNumberResult> {
        core_number(handle, self, params)
    }

    /// Extract a k-core subgraph, computing core numbers internally.
    pub fn k_core(&mut self, handle: &EnteredContext<'_>, params: &KCoreParams) -> Result<KCoreResult> {
        k_core(handle, self, params, None)
    }

    /// Extract a k-core subgraph using a precomputed core-number result.
    pub fn k_core_with_core_numbers(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &KCoreParams,
        core_numbers: &CoreNumberResult,
    ) -> Result<KCoreResult> {
        k_core(handle, self, params, Some(core_numbers))
    }

    /// Compute a minimum spanning tree.
    pub fn minimum_spanning_tree(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &MinimumSpanningTreeParams,
    ) -> Result<MinimumSpanningTreeResult> {
        minimum_spanning_tree(handle, self, params)
    }

    /// Create an owned vertex-pair collection for pairwise similarity algorithms.
    ///
    /// `first` and `second` must be explicit vertex selections with matching
    /// lengths and the same vertex dtype as this graph. `VertexSelection::All`
    /// is rejected because pairwise similarity requires aligned input pairs.
    pub fn vertex_pairs<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        first: impl Into<VertexSelection<'a, V>>,
        second: impl Into<VertexSelection<'a, V>>,
    ) -> Result<VertexPairs> {
        let first = first.into();
        let second = second.into();
        self.validate_vertex_selection(&first, "Graph::vertex_pairs", "first", false)?;
        self.validate_vertex_selection(&second, "Graph::vertex_pairs", "second", false)?;
        let first = first.resolve(handle)?;
        let second = second.resolve(handle)?;
        let first_view = self.resolved_vertex_view(&first, "Graph::vertex_pairs", "first")?;
        let second_view = self.resolved_vertex_view(&second, "Graph::vertex_pairs", "second")?;
        create_vertex_pairs(handle, self, &first_view, &second_view)
            .map_err(|error| error.with_operation("Graph::vertex_pairs"))
    }

    /// Compute Jaccard similarity coefficients for explicit vertex pairs.
    ///
    /// cuGraph requires a symmetric graph for this algorithm; native
    /// validation reports detailed graph-property errors.
    pub fn jaccard_coefficients(
        &mut self,
        handle: &EnteredContext<'_>,
        pairs: VertexPairs,
        params: &SimilarityParams,
    ) -> Result<JaccardSimilarityResult> {
        jaccard_coefficients(handle, self, pairs, params)
    }

    /// Compute Sorensen similarity coefficients for explicit vertex pairs.
    ///
    /// cuGraph requires a symmetric graph for this algorithm; native
    /// validation reports detailed graph-property errors.
    pub fn sorensen_coefficients(
        &mut self,
        handle: &EnteredContext<'_>,
        pairs: VertexPairs,
        params: &SimilarityParams,
    ) -> Result<SorensenSimilarityResult> {
        sorensen_coefficients(handle, self, pairs, params)
    }

    /// Compute overlap similarity coefficients for explicit vertex pairs.
    ///
    /// cuGraph requires a symmetric graph for this algorithm; native
    /// validation reports detailed graph-property errors.
    pub fn overlap_coefficients(
        &mut self,
        handle: &EnteredContext<'_>,
        pairs: VertexPairs,
        params: &SimilarityParams,
    ) -> Result<OverlapSimilarityResult> {
        overlap_coefficients(handle, self, pairs, params)
    }

    /// Compute cosine similarity coefficients for explicit vertex pairs.
    ///
    /// cuGraph requires a symmetric graph for this algorithm; native
    /// validation reports detailed graph-property errors.
    pub fn cosine_similarity_coefficients(
        &mut self,
        handle: &EnteredContext<'_>,
        pairs: VertexPairs,
        params: &SimilarityParams,
    ) -> Result<CosineSimilarityResult> {
        cosine_similarity_coefficients(handle, self, pairs, params)
    }

    /// Compute all-pairs Jaccard similarity coefficients.
    ///
    /// Pass [`VertexSelection::All`] to use all vertices or an explicit subset
    /// to seed pair generation. cuGraph requires a symmetric graph.
    pub fn all_pairs_jaccard_coefficients<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        params: &AllPairsSimilarityParams,
    ) -> Result<JaccardSimilarityResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::all_pairs_jaccard_coefficients", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        all_pairs_jaccard_coefficients(handle, self, view.as_ref(), params)
    }

    /// Compute all-pairs Sorensen similarity coefficients.
    ///
    /// Pass [`VertexSelection::All`] to use all vertices or an explicit subset
    /// to seed pair generation. cuGraph requires a symmetric graph.
    pub fn all_pairs_sorensen_coefficients<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        params: &AllPairsSimilarityParams,
    ) -> Result<SorensenSimilarityResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::all_pairs_sorensen_coefficients", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        all_pairs_sorensen_coefficients(handle, self, view.as_ref(), params)
    }

    /// Compute all-pairs overlap similarity coefficients.
    ///
    /// Pass [`VertexSelection::All`] to use all vertices or an explicit subset
    /// to seed pair generation. cuGraph requires a symmetric graph.
    pub fn all_pairs_overlap_coefficients<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        params: &AllPairsSimilarityParams,
    ) -> Result<OverlapSimilarityResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::all_pairs_overlap_coefficients", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        all_pairs_overlap_coefficients(handle, self, view.as_ref(), params)
    }

    /// Compute all-pairs cosine similarity coefficients.
    ///
    /// Pass [`VertexSelection::All`] to use all vertices or an explicit subset
    /// to seed pair generation. cuGraph requires a symmetric graph.
    pub fn all_pairs_cosine_similarity_coefficients<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        params: &AllPairsSimilarityParams,
    ) -> Result<CosineSimilarityResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::all_pairs_cosine_similarity_coefficients", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        all_pairs_cosine_similarity_coefficients(handle, self, view.as_ref(), params)
    }

    /// Compute a Force Atlas 2 layout for this graph.
    ///
    /// Returns a [`LayoutResult`] containing
    /// vertex ids and (x, y) coordinates.
    ///
    /// For advanced features (per-vertex mass/radius, initial positions for
    /// refinement passes), use [`force_atlas2_with`](Self::force_atlas2_with).
    pub fn force_atlas2(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut RngState,
        params: &ForceAtlas2Params,
    ) -> Result<LayoutResult> {
        force_atlas2::<T, i32, f32>(handle, rng_state, self, params, None, None)
    }

    /// Compute a Force Atlas 2 layout with optional per-vertex attributes
    /// and/or initial positions.
    ///
    /// Accepts initial positions for refinement passes or per-vertex
    /// radius, mobility, and mass attributes.
    pub fn force_atlas2_with<V: CugraphElement, W: CugraphElement>(
        &mut self,
        handle: &EnteredContext<'_>,
        rng_state: &mut RngState,
        params: &ForceAtlas2Params,
        vertex_attrs: Option<&ForceAtlas2VertexAttrs<'_, V, W>>,
        initial_positions: Option<&InitialPositions<'_, V, W>>,
    ) -> Result<LayoutResult> {
        force_atlas2(handle, rng_state, self, params, vertex_attrs, initial_positions)
    }

    /// Compute only out-degrees for vertices in this graph.
    ///
    /// Pass [`VertexSelection::All`] to compute degrees for every vertex,
    /// [`VertexSelection::Device`] with a device view, or
    /// [`VertexSelection::Host`] with a host slice (uploaded on demand).
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`](crate::ErrorKind::UnsupportedType) if `vertices` uses a vertex
    /// dtype that does not match the graph.
    pub fn out_degrees<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
    ) -> Result<OutDegreesResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::out_degrees", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        out_degrees(handle, self, view.as_ref(), false)
    }

    /// Compute out-degrees with optional expensive input validation.
    pub fn out_degrees_checked<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        vertices: impl Into<VertexSelection<'a, V>>,
        do_expensive_check: bool,
    ) -> Result<OutDegreesResult> {
        let vertices = vertices.into();
        self.validate_vertex_selection(&vertices, "Graph::out_degrees", "vertices", true)?;
        let resolved = vertices.resolve(handle)?;
        let view = resolved.view()?;
        out_degrees(handle, self, view.as_ref(), do_expensive_check)
    }

    /// Compute out-degrees for every vertex in the graph.
    pub fn out_degrees_all(&mut self, handle: &EnteredContext<'_>) -> Result<OutDegreesResult> {
        self.out_degrees(handle, VertexSelection::<i32>::All)
    }

    /// Compute triangle counts for all vertices or an explicit start-vertex list.
    ///
    /// Pass [`VertexSelection::All`] to count triangles for every vertex,
    /// [`VertexSelection::Device`] with a device view, or
    /// [`VertexSelection::Host`] with a host slice uploaded on demand.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::UnsupportedType`](crate::ErrorKind::UnsupportedType) if `start_vertices` uses a
    /// vertex dtype that does not match the graph.
    pub fn triangle_count<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        start_vertices: impl Into<VertexSelection<'a, V>>,
        params: &TriangleCountParams,
    ) -> Result<TriangleCountResult> {
        let start_vertices = start_vertices.into();
        self.validate_vertex_selection(&start_vertices, "Graph::triangle_count", "start_vertices", true)?;
        let resolved = start_vertices.resolve(handle)?;
        let view = resolved.view()?;
        triangle_count(handle, self, view.as_ref(), params)
    }

    /// Compute triangle counts for every vertex in the graph.
    pub fn triangle_count_all(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &TriangleCountParams,
    ) -> Result<TriangleCountResult> {
        self.triangle_count(handle, VertexSelection::<i32>::All, params)
    }

    /// Extract a k-truss subgraph.
    pub fn k_truss_subgraph(
        &mut self,
        handle: &EnteredContext<'_>,
        params: &KTrussSubgraphParams,
    ) -> Result<KTrussSubgraphResult> {
        k_truss_subgraph(handle, self, params)
    }

    /// Extract ego subgraphs for explicit source vertices.
    ///
    /// `VertexSelection::All` is rejected because the native API requires an
    /// explicit source vertex list.
    pub fn extract_ego<'a, V: CugraphElement + 'a>(
        &mut self,
        handle: &EnteredContext<'_>,
        source_vertices: impl Into<VertexSelection<'a, V>>,
        params: &ExtractEgoParams,
    ) -> Result<ExtractEgoResult> {
        let source_vertices = source_vertices.into();
        self.validate_vertex_selection(&source_vertices, "Graph::extract_ego", "source_vertices", false)?;
        let resolved = source_vertices.resolve(handle)?;
        let view = self.resolved_vertex_view(&resolved, "Graph::extract_ego", "source_vertices")?;
        extract_ego(handle, self, &view, params)
    }
}
