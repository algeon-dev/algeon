#[test]
fn profiler_api_symbols_match_the_rust_function_pointer_contract() {
    let start: unsafe extern "C" fn() -> cudf_sys::cudf_status_t = cudf_sys::cudf_cuda_profiler_start;
    let stop: unsafe extern "C" fn() -> cudf_sys::cudf_status_t = cudf_sys::cudf_cuda_profiler_stop;

    std::hint::black_box((start, stop));
}
