use std::sync::Arc;

use cudf::{QueryMemoryDomainOptions, Runtime};
use cuml::{CumlResources, ErrorCode, FailurePhase, MatrixLayout, OwnedFloat32Matrix, PcaModel, PcaParams};

#[test]
fn pca_fit_transform_limit_failure_keeps_query_domain_attribution() {
    assert!(cudf::DeviceId::count().expect("CUDA device count") > 0);

    let runtime = Arc::new(Runtime::builder().with_device_memory_manager().build().expect("runtime"));
    let resources = CumlResources::for_runtime(Arc::clone(&runtime)).expect("cuML resources");

    let rows = 4096;
    let columns = 128;
    let mut input = OwnedFloat32Matrix::new(&resources, rows, columns, MatrixLayout::ColumnMajor)
        .expect("PCA input allocation before the grant is installed");

    const DOMAIN_ID: u64 = 73_901;
    const MEMBER_ID: u64 = 73_902;
    let manager = runtime.device_memory_manager().expect("device memory manager");
    // Model buffers and transformed output fit within this cap; native PCA temporaries do not.
    const GRANT: usize = 3 * 1024 * 1024;
    let domain = manager
        .query_memory_domain(DOMAIN_ID, QueryMemoryDomainOptions::new(GRANT).with_ceiling_bytes(GRANT))
        .expect("query memory domain");
    let _reservation = domain.reserve_stream(runtime.stream(), MEMBER_ID, GRANT).expect("query stream reservation");

    let error = match PcaModel::fit_transform(&resources, &mut input, PcaParams::new(columns)) {
        Err(error) => error,
        Ok(_) => panic!("PCA native allocations must exceed the query grant"),
    };

    assert_eq!(error.code(), ErrorCode::OutOfMemory);
    let detail = error.ffi_detail().expect("allocation failure provenance");
    assert_eq!(detail.failure_source, cuml::CumlErrorFailureSource::ReservationLimit);
    assert_eq!(detail.domain_id, DOMAIN_ID);
    assert_eq!(detail.member_id, MEMBER_ID);
    assert_eq!(cuml::allocation_attribution(&error), cudf::CudfAllocationAttribution::ConfirmedQueryLimit);
    assert_eq!(error.phase(), FailurePhase::NativeCall);
}
