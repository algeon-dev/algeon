mod common;

use std::sync::Arc;

use cudf::{DeviceMemoryManager, QueryMemoryDomain, QueryMemoryDomainOptions, Runtime};
use cudf_sys::test_support;
use cugraph::{
    AllocationFailureSource, AllocationSource, DeviceArray, DeviceArrayView, ErrorKind, Forward, GraphBuilder,
    ResourceHandle, Transposed,
    centrality::{BetweennessCentralityParams, PageRankParams},
    traversal::BfsParams,
};
use cugraph_sys::{
    bool_t, cugraph_centrality_result_free, cugraph_centrality_result_t, cugraph_data_type_id_t,
    cugraph_error_alignment, cugraph_error_allocation_source, cugraph_error_code_t, cugraph_error_cuda_error,
    cugraph_error_domain_id, cugraph_error_failure_source, cugraph_error_free, cugraph_error_generation,
    cugraph_error_member_id, cugraph_error_requested_bytes, cugraph_error_t, cugraph_graph_create_sg_from_csr,
    cugraph_graph_free, cugraph_graph_properties_t, cugraph_graph_t, cugraph_pagerank,
    cugraph_type_erased_device_array_view_create, cugraph_type_erased_device_array_view_free,
    cugraph_type_erased_device_array_view_t,
};

const DOMAIN_LIMIT: usize = 64 * 1024 * 1024;
const DOMAIN_ID: u64 = 82_001;
const MEMBER_ID: u64 = 83_001;

struct AllocationFailureReset;

impl Drop for AllocationFailureReset {
    fn drop(&mut self) {
        unsafe { test_support::cudf_test_query_domain_reset_allocation_failure() };
    }
}

fn assert_usage_baseline(domain: &QueryMemoryDomain, baseline_bytes: u64, baseline_allocations: u64) {
    let usage = domain.usage().expect("query-domain usage");
    assert_eq!(usage.current_bytes, baseline_bytes);
    assert_eq!(usage.current_allocations, baseline_allocations);
}

trait SweepError {
    fn assert_allocation_facts(&self, name: &str, successful_allocations: u64, generation: u64);
}

impl SweepError for cugraph::Error {
    fn assert_allocation_facts(&self, name: &str, successful_allocations: u64, generation: u64) {
        assert_eq!(self.kind(), ErrorKind::OutOfMemory, "{name} allocation {successful_allocations}: {self:?}");
        assert_eq!(self.allocation_source(), AllocationSource::RmmOutOfMemory);
        assert_eq!(self.allocation_failure_source(), AllocationFailureSource::Physical);
        assert_eq!(self.allocation_domain_id(), DOMAIN_ID);
        assert_eq!(self.allocation_member_id(), MEMBER_ID);
        assert_eq!(self.allocation_generation(), generation);
        assert!(self.allocation_requested_bytes() > 0);
        assert!(self.allocation_alignment() > 0);
    }
}

struct RawGraph(*mut cugraph_graph_t);

impl Drop for RawGraph {
    fn drop(&mut self) {
        unsafe { cugraph_graph_free(self.0) };
    }
}

struct RawCentralityResult(*mut cugraph_centrality_result_t);

impl Drop for RawCentralityResult {
    fn drop(&mut self) {
        unsafe { cugraph_centrality_result_free(self.0) };
    }
}

struct RawError(*mut cugraph_error_t);

impl Drop for RawError {
    fn drop(&mut self) {
        unsafe { cugraph_error_free(self.0) };
    }
}

impl SweepError for RawError {
    fn assert_allocation_facts(&self, name: &str, successful_allocations: u64, generation: u64) {
        unsafe {
            assert_eq!(
                cugraph_error_allocation_source(self.0),
                cugraph_sys::cugraph_allocation_source_t::CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY,
                "{name} allocation {successful_allocations}"
            );
            assert_eq!(
                cugraph_error_failure_source(self.0),
                cugraph_sys::cugraph_error_failure_source_t::CUGRAPH_ERROR_FAILURE_SOURCE_PHYSICAL
            );
            assert_eq!(cugraph_error_domain_id(self.0), DOMAIN_ID);
            assert_eq!(cugraph_error_member_id(self.0), MEMBER_ID);
            assert_eq!(cugraph_error_generation(self.0), generation);
            assert!(cugraph_error_requested_bytes(self.0) > 0);
            assert!(cugraph_error_alignment(self.0) > 0);
            assert!(cugraph_error_cuda_error(self.0) >= -1);
        }
    }
}

struct RawArrayView(*mut cugraph_type_erased_device_array_view_t);

impl RawArrayView {
    fn int32(view: &DeviceArrayView<'_, i32>) -> Self {
        let raw = unsafe {
            cugraph_type_erased_device_array_view_create(view.device_ptr(), view.len(), cugraph_data_type_id_t::INT32)
        };
        assert!(!raw.is_null());
        Self(raw)
    }
}

impl Drop for RawArrayView {
    fn drop(&mut self) {
        unsafe { cugraph_type_erased_device_array_view_free(self.0) };
    }
}

fn sweep_operation<T, E: SweepError>(
    name: &str,
    runtime: &Runtime,
    domain: &QueryMemoryDomain,
    mut operation: impl FnMut() -> Result<T, E>,
) {
    runtime.synchronize().expect("baseline synchronize");
    let identity = domain.identity().expect("query-domain identity");
    let baseline = domain.usage().expect("baseline query-domain usage");
    let mut reached_success = false;
    let mut failures = 0;

    for successful_allocations in 0..256 {
        unsafe { test_support::cudf_test_query_domain_fail_allocation_after(DOMAIN_ID, successful_allocations) };
        let reset = AllocationFailureReset;
        let result = operation();
        drop(reset);

        match result {
            Ok(output) => {
                drop(output);
                runtime.synchronize().expect("success synchronize");
                assert_usage_baseline(domain, baseline.current_bytes, baseline.current_allocations);
                reached_success = true;
                eprintln!("{name}: injected_failures={failures}");
                break;
            },
            Err(error) => {
                failures += 1;
                error.assert_allocation_facts(name, successful_allocations, identity.generation);
                runtime.synchronize().expect("failure synchronize");
                assert_usage_baseline(domain, baseline.current_bytes, baseline.current_allocations);
            },
        }
    }

    assert!(reached_success, "{name} exceeded the bounded allocation-path sweep");
    let output = operation().unwrap_or_else(|_| panic!("{name} must succeed immediately after the sweep"));
    drop(output);
    runtime.synchronize().expect("post-sweep synchronize");
    assert_usage_baseline(domain, baseline.current_bytes, baseline.current_allocations);
}

fn build_csr(
    entered: &cugraph::EnteredContext<'_>,
    offsets: &RawArrayView,
    indices: &RawArrayView,
) -> Result<RawGraph, RawError> {
    let properties = cugraph_graph_properties_t { is_symmetric: bool_t::FALSE, is_multigraph: bool_t::FALSE };
    let mut graph = std::ptr::null_mut();
    let mut error = std::ptr::null_mut();
    let code = unsafe {
        cugraph_graph_create_sg_from_csr(
            entered.handle_ptr(),
            &properties,
            offsets.0,
            indices.0,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            bool_t::FALSE,
            bool_t::FALSE,
            bool_t::FALSE,
            bool_t::TRUE,
            &mut graph,
            &mut error,
        )
    };
    if code == cugraph_error_code_t::CUGRAPH_SUCCESS {
        assert!(error.is_null());
        assert!(!graph.is_null());
        Ok(RawGraph(graph))
    } else {
        assert_eq!(code, cugraph_error_code_t::CUGRAPH_ALLOC_ERROR);
        assert!(graph.is_null());
        assert!(!error.is_null());
        Err(RawError(error))
    }
}

fn pagerank_with_transpose(
    entered: &cugraph::EnteredContext<'_>,
    graph: &mut RawGraph,
) -> Result<RawCentralityResult, RawError> {
    let mut result = std::ptr::null_mut();
    let mut error = std::ptr::null_mut();
    let code = unsafe {
        cugraph_pagerank(
            entered.handle_ptr(),
            graph.0,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0.85,
            1.0e-6,
            100,
            bool_t::FALSE,
            &mut result,
            &mut error,
        )
    };
    if code == cugraph_error_code_t::CUGRAPH_SUCCESS {
        assert!(error.is_null());
        assert!(!result.is_null());
        Ok(RawCentralityResult(result))
    } else {
        assert_eq!(code, cugraph_error_code_t::CUGRAPH_ALLOC_ERROR);
        assert!(result.is_null());
        assert!(!error.is_null());
        Err(RawError(error))
    }
}

#[test]
fn pagerank_bfs_betweenness_and_path_decode_failures_are_typed_and_recoverable() {
    let _guard = common::cudf::cuda_test_serial_guard();
    let runtime = Arc::new(Runtime::builder().with_device_memory_manager().build().expect("runtime"));
    let manager = DeviceMemoryManager::for_device(runtime.device()).expect("device manager");
    let domain = manager
        .query_memory_domain(DOMAIN_ID, QueryMemoryDomainOptions::new(DOMAIN_LIMIT).with_ceiling_bytes(DOMAIN_LIMIT))
        .expect("query memory domain");
    let _lease = domain.reserve_stream(runtime.stream(), MEMBER_ID, DOMAIN_LIMIT).expect("stream member");
    let handle = ResourceHandle::for_runtime(Arc::clone(&runtime)).expect("runtime-bound handle");
    let entered = handle.enter("allocation_failure_sweep").expect("entered handle");

    let src = DeviceArray::from_slice(&entered, &[0i32, 1, 2, 2, 3, 4, 5, 5]).expect("sources");
    let dst = DeviceArray::from_slice(&entered, &[1i32, 2, 0, 3, 4, 5, 3, 0]).expect("destinations");
    let src_view = src.view().expect("source view");
    let dst_view = dst.view().expect("destination view");
    let sources = DeviceArray::from_slice(&entered, &[0i32]).expect("BFS sources");
    let destinations = DeviceArray::from_slice(&entered, &[4i32]).expect("path destinations");
    let sources_view = sources.view().expect("BFS source view");
    let destinations_view = destinations.view().expect("path destination view");

    sweep_operation("graph_build_non_renumbered", &runtime, &domain, || {
        GraphBuilder::new(&src_view, &dst_view).do_expensive_check().build::<Forward>(&entered)
    });
    sweep_operation("graph_build_renumbered", &runtime, &domain, || {
        GraphBuilder::new(&src_view, &dst_view).renumber().do_expensive_check().build::<Forward>(&entered)
    });

    let csr_offsets = DeviceArray::from_slice(&entered, &[0i32, 1, 2, 4, 5, 6, 8]).expect("CSR offsets");
    let csr_indices = DeviceArray::from_slice(&entered, &[1i32, 2, 0, 3, 4, 5, 3, 0]).expect("CSR indices");
    let csr_offsets_view = csr_offsets.view().expect("CSR offsets view");
    let csr_indices_view = csr_indices.view().expect("CSR indices view");
    let raw_csr_offsets = RawArrayView::int32(&csr_offsets_view);
    let raw_csr_indices = RawArrayView::int32(&csr_indices_view);
    sweep_operation("graph_build_csr", &runtime, &domain, || build_csr(&entered, &raw_csr_offsets, &raw_csr_indices));

    let mut pagerank_graph =
        GraphBuilder::new(&src_view, &dst_view).build::<Transposed>(&entered).expect("PageRank graph");
    let warmup = pagerank_graph.pagerank(&entered, &PageRankParams::default()).expect("PageRank warmup");
    drop(warmup);
    sweep_operation("pagerank", &runtime, &domain, || pagerank_graph.pagerank(&entered, &PageRankParams::default()));
    sweep_operation("pagerank_with_transpose", &runtime, &domain, || {
        let mut graph = build_csr(&entered, &raw_csr_offsets, &raw_csr_indices)?;
        pagerank_with_transpose(&entered, &mut graph)
    });

    let mut bfs_graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("BFS graph");
    let warmup = bfs_graph.bfs(&entered, &sources_view, &BfsParams::default().with_predecessors()).expect("BFS warmup");
    drop(warmup);
    sweep_operation("bfs", &runtime, &domain, || {
        bfs_graph.bfs(&entered, &sources_view, &BfsParams::default().with_predecessors())
    });

    let mut betweenness_graph =
        GraphBuilder::new(&src_view, &dst_view).renumber().build::<Forward>(&entered).expect("betweenness graph");
    let warmup = betweenness_graph
        .betweenness_centrality_all(&entered, &BetweennessCentralityParams::default())
        .expect("betweenness warmup");
    drop(warmup);
    sweep_operation("betweenness", &runtime, &domain, || {
        betweenness_graph.betweenness_centrality_all(&entered, &BetweennessCentralityParams::default())
    });

    let mut paths_graph = GraphBuilder::new(&src_view, &dst_view).build::<Forward>(&entered).expect("paths graph");
    let bfs_result =
        paths_graph.bfs(&entered, &sources_view, &BfsParams::default().with_predecessors()).expect("paths BFS result");
    let warmup = bfs_result
        .extract_paths(&entered, &mut paths_graph, &sources_view, &destinations_view)
        .expect("path decode warmup");
    drop(warmup);
    sweep_operation("path_decode", &runtime, &domain, || {
        let result = bfs_result.extract_paths(&entered, &mut paths_graph, &sources_view, &destinations_view)?;
        result.paths()?.to_vec_i32(&entered)
    });
}
