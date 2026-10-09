//! Sweeps allocation failures across every allocation point of a query.
//! Fault injection is scoped to the process and query domain.

use cudf::{
    AggSpec, CudfErrorFailureSource, DataFrame, ErrorKind, Exec, GroupByAgg, MergeOptions, QueryMemoryDomain,
    QueryMemoryDomainOptions, Runtime, SortOptions, ops::ast::AstExpression,
};
use cudf_sys::test_support;

const DOMAIN_LIMIT: usize = 64 * 1024 * 1024;
const MAX_ALLOCATION_PATHS: u64 = 256;

fn arm_failure(domain_id: u64, successful_allocations: u64) {
    unsafe { test_support::cudf_test_query_domain_fail_allocation_after(domain_id, successful_allocations) }
}

fn reset_failure() {
    unsafe { test_support::cudf_test_query_domain_reset_allocation_failure() }
}

fn assert_live_usage_matches(domain: &QueryMemoryDomain, expected: (u64, u64, u64)) {
    let usage = domain.usage().unwrap();
    assert_eq!(
        (usage.current_bytes, usage.current_aligned_bytes, usage.current_allocations),
        expected,
        "failed operation changed live query-domain ownership"
    );
}

fn sweep_operation<T>(
    operation_name: &str,
    runtime: &Runtime,
    domain: &QueryMemoryDomain,
    member_id: u64,
    mut operation: impl FnMut() -> cudf::Result<T>,
    validate: impl Fn(&T),
) {
    let identity = domain.identity().unwrap();
    let initial = domain.usage().unwrap();
    let baseline = (initial.current_bytes, initial.current_aligned_bytes, initial.current_allocations);
    let mut reached_success = false;

    for successful_allocations in 0..MAX_ALLOCATION_PATHS {
        eprintln!("sweeping {operation_name} after {successful_allocations} successful allocations");
        arm_failure(identity.domain_id, successful_allocations);
        match operation() {
            Ok(output) => {
                reset_failure();
                validate(&output);
                drop(output);
                runtime.synchronize().unwrap();
                assert_live_usage_matches(domain, baseline);
                reached_success = true;
                break;
            },
            Err(error) => {
                reset_failure();
                assert_eq!(error.kind(), ErrorKind::ResourceExhausted);
                let detail = error.ffi_detail().expect("allocation failure must retain its native payload");
                assert_eq!(detail.failure_source, CudfErrorFailureSource::Physical);
                assert_eq!(detail.domain_id, identity.domain_id);
                assert_eq!(detail.member_id, member_id);
                assert_eq!(detail.generation, identity.generation);
                assert!(detail.requested_bytes > 0);
                assert!(detail.alignment > 0);
                drop(error);
                runtime.synchronize().unwrap();
                assert_live_usage_matches(domain, baseline);
            },
        }
    }

    assert!(reached_success, "operation exceeded the bounded allocation-path sweep");
    let output = operation().expect("the next operation must succeed on the same backend");
    validate(&output);
    drop(output);
    runtime.synchronize().unwrap();
    assert_live_usage_matches(domain, baseline);
}

fn with_domain(domain_id: u64, member_id: u64, test: impl FnOnce(&Runtime, &QueryMemoryDomain, u64)) {
    let runtime = Runtime::builder().with_device_memory_manager().build().unwrap();
    let manager = runtime.device_memory_manager().unwrap();
    let domain = manager
        .query_memory_domain(domain_id, QueryMemoryDomainOptions::new(DOMAIN_LIMIT).with_ceiling_bytes(DOMAIN_LIMIT))
        .unwrap();
    let _lease = domain.reserve_stream(runtime.stream(), member_id, DOMAIN_LIMIT).unwrap();
    test(&runtime, &domain, member_id);
}

#[test]
fn join_groupby_sort_and_string_allocation_failures_are_typed_and_leave_no_live_output() {
    with_domain(80_005, 81_005, |runtime, domain, member_id| {
        let left = DataFrame::from_series([
            ("key", runtime.series(&[1i32, 1, 2]).unwrap()),
            ("value", runtime.series(&[3i32, 3, 8]).unwrap()),
        ])
        .unwrap();
        let right = DataFrame::from_series([
            ("key", runtime.series(&[1i32, 2]).unwrap()),
            ("value", runtime.series(&[5i32, 6]).unwrap()),
        ])
        .unwrap();
        sweep_operation(
            "mixed_left_state",
            runtime,
            domain,
            member_id,
            || {
                let ctx = cudf::Exec::RETAINED;
                let predicate = AstExpression::less(AstExpression::left_column(0)?, AstExpression::right_column(0)?)?;
                let options = cudf::DataFrameMixedJoinOptions::new()
                    .on(["key"])
                    .left_predicate_columns(["value"])
                    .right_predicate_columns(["value"]);
                let mut state = cudf::DataFrameMixedLeftJoinState::new(&left, predicate, options, ctx)?;
                state.consume(&right)?;
                let finalized = state.seal(false)?;
                finalized.materialize_indices()
            },
            |output| assert_eq!(output.indices().to_vec_strict::<i32>().unwrap(), vec![0, 1]),
        );
    });

    with_domain(80_001, 81_001, |runtime, domain, member_id| {
        let left = DataFrame::from_series([
            ("id", runtime.series(&[1i32, 2, 3]).unwrap()),
            ("score", runtime.series(&[10i32, 20, 30]).unwrap()),
        ])
        .unwrap();
        let right = DataFrame::from_series([
            ("id", runtime.series(&[2i32, 3, 4]).unwrap()),
            ("value", runtime.series(&[200i32, 300, 400]).unwrap()),
        ])
        .unwrap();
        let options = MergeOptions::new().on(["id"]);
        sweep_operation(
            "join",
            runtime,
            domain,
            member_id,
            || left.on(Exec::RETAINED).merge(&right, options.clone()),
            |output| {
                assert_eq!(output.num_rows(), 2);
                assert_eq!(output.column_names(), vec!["id", "score", "value"]);
            },
        );
    });

    with_domain(80_002, 81_002, |runtime, domain, member_id| {
        let frame = DataFrame::from_series([
            ("key", runtime.series(&[1i32, 1, 2, 2]).unwrap()),
            ("value", runtime.series(&[10i64, 20, 30, 40]).unwrap()),
        ])
        .unwrap();
        sweep_operation(
            "groupby",
            runtime,
            domain,
            member_id,
            || frame.groupby(["key"]).on(Exec::RETAINED).agg(&[AggSpec::new("value", GroupByAgg::Sum)]),
            |output| assert_eq!(output.num_rows(), 2),
        );
    });

    with_domain(80_003, 81_003, |runtime, domain, member_id| {
        let frame = DataFrame::from_series([
            ("id", runtime.series(&[2i32, 1, 2, 1]).unwrap()),
            ("score", runtime.series(&[20i32, 10, 40, 30]).unwrap()),
        ])
        .unwrap();
        sweep_operation(
            "sort",
            runtime,
            domain,
            member_id,
            || frame.on(Exec::RETAINED).sort_values(["id", "score"], SortOptions::new()),
            |output| {
                assert_eq!(output.series("id").unwrap().to_vec_strict::<i32>().unwrap(), vec![1, 1, 2, 2]);
            },
        );
    });

    with_domain(80_004, 81_004, |runtime, domain, member_id| {
        let input = runtime.series(&["Hello", "world", "Gpu"]).unwrap();
        sweep_operation(
            "string",
            runtime,
            domain,
            member_id,
            || input.on(Exec::RETAINED).str().upper(),
            |output| {
                assert_eq!(output.to_strings_strict().unwrap(), vec!["HELLO", "WORLD", "GPU"]);
            },
        );
    });
}
