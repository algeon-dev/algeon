use std::{collections::BTreeMap, sync::Arc};

use arrow::datatypes::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use cudf::io::{
    parquet::{
        RemoteCredentialMode, RemoteParquetBackendBuildStatus, RemoteParquetCapability,
        RemoteParquetCapabilityDisposition, RemoteParquetProfile, RemoteParquetSource, RemoteReadMode,
        probe_remote_parquet_datasource, remote_parquet_capability,
    },
    source::{
        LocalParquetFile as CudfLocalParquetFile, ParquetSourceLocation as CudfParquetSourceLocation,
        ParquetVariableWidthEncodingProof as CudfParquetVariableWidthEncodingProof, SourceChunk as CudfSourceChunk,
    },
};
use object_store::aws::AwsCredential;

use super::{local::*, remote::*};
use crate::{
    chunk, error,
    exec::source::{
        ExecutionContext, LocalParquetFile, LocalParquetSource, QueryError, QueryErrorCode, SourceChunk,
        cudf_query_error_without_source,
    },
    expr,
    expr::{BinaryOp, Expr, LiteralValue},
    parquet_stats, plan, rapids_backend, runtime, source,
};

fn remote_direct_test_source() -> LocalParquetSource {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int32, false),
        ArrowField::new("payload", ArrowDataType::Utf8, true),
        ArrowField::new("filter_only", ArrowDataType::Boolean, true),
    ]));
    let resolved = source::ResolvedParquetSource::remote_kvikio(
        "s3://bucket/part.parquet",
        source::CredentialMode::SharedProcessEnv,
    );
    let file = LocalParquetFile::from_resolved_parquet_source(resolved)
        .expect("valid remote direct source")
        .with_stable_id("part")
        .with_file_size_bytes(4096)
        .with_row_groups(vec![1, 3])
        .with_row_group_bytes(vec![512, 768, 1024, 1280])
        .with_projected_row_group_bytes(vec![128, 192, 256, 320], vec![256, 384, 512, 640])
        .with_pure_filter_row_group_bytes(vec![32, 48, 64, 80], vec![64, 96, 128, 160])
        .with_row_group_rows(vec![10, 10, 10, 10]);

    LocalParquetSource::from_files(vec![file], schema, vec!["payload".to_string(), "filter_only".to_string()])
        .with_column_roles(vec!["payload".to_string()], vec!["filter_only".to_string()])
        .with_column_file_indices(vec![1, 2], vec![1], vec![2])
}

const UNREACHABLE_FOOTER_PROFILE_ID: &str = "unreachable-footer";

fn context_with_unreachable_footer_store() -> ExecutionContext {
    ExecutionContext::builder()
        .with_source_storage_profiles([runtime::SourceStorageProfile::s3(
            UNREACHABLE_FOOTER_PROFILE_ID,
            runtime::S3SourceStorageProfile::default_chain()
                .with_region("us-east-1")
                .with_endpoint("http://127.0.0.1:9")
                .with_path_style_access(true)
                .with_unsigned_requests(),
        )])
        .build()
        .expect("execution context builds")
}

fn remote_direct_with_unreachable_footer_store(uri: &str) -> source::ResolvedParquetSource {
    source::ResolvedParquetSource::remote_kvikio(uri, source::CredentialMode::SharedProcessEnv)
        .with_storage_profile_id(UNREACHABLE_FOOTER_PROFILE_ID)
}

fn assert_required_footer_fetch_temporary_failure(error: &QueryError) {
    assert_eq!(error.code(), QueryErrorCode::SourceTemporaryFailure);
    assert_eq!(error.operation(), "query_engine.source_metadata.object_store_footer");
    assert_eq!(error.fact_value("object_store_error_kind"), Some("generic"));
    assert!(error.fact_value("path").is_some_and(|value| value.starts_with("object_store_path:fnv64:")));
}

#[test]
fn local_and_local_cache_contracts_build_only_local_parquet_paths() {
    let context = ExecutionContext::builder().build().expect("execution context builds");
    let cases = [
        source::ResolvedParquetSource::local_kvikio("/tmp/local.parquet"),
        source::ResolvedParquetSource::local_cache_kvikio(
            "s3://bucket/part.parquet",
            "/tmp/algeon-cache/part.parquet",
            source::CredentialMode::ScopedResolved,
            source::LakehouseCacheStatus::Hit,
        ),
    ];

    for resolved in cases {
        let expected_path = resolved.reader_location().to_string();
        let file =
            LocalParquetFile::from_resolved_parquet_source(resolved).expect("valid path source").with_stable_id("part");
        let cudf_file = cudf_local_parquet_file(&context, &file).expect("local cuDF contract builds");

        assert_eq!(cudf_file.reader_path(), expected_path);
        assert!(cudf_file.remote_source().is_none(), "local/cache contract must not become a remote URI");
        assert!(matches!(cudf_file.location(), CudfParquetSourceLocation::LocalPath(_)));
    }
}

#[test]
fn missing_remote_profile_is_structured_and_cannot_fallback() {
    let context = ExecutionContext::builder().build().expect("execution context builds");
    let resolved = source::ResolvedParquetSource::remote_kvikio(
        "s3://bucket/private/part.parquet?signature=secret",
        source::CredentialMode::ScopedResolved,
    )
    .with_storage_profile_id("missing-profile");
    let cudf_file =
        CudfLocalParquetFile::object_store_uri(resolved.reader_location(), "part", []).expect("valid object-store URI");

    let error =
        attach_cudf_remote_parquet_contract(&context, cudf_file, &resolved, "query_engine.test.remote_contract")
            .unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("resource"), Some("remote_parquet_source"));
    assert_eq!(error.fact_value("reason"), Some("remote_storage_profile_unavailable"));
    assert_eq!(error.fact_value("remote_read_mode"), Some("require_kvikio"));
    assert_eq!(error.fact_value("fallback_allowed"), Some("false"));
    assert!(!error.to_string().contains("s3://bucket"));
    assert!(!error.to_string().contains("signature=secret"));
}

#[test]
fn unavailable_remote_backend_is_permanent_structured_and_cannot_fallback() {
    let resolved = source::ResolvedParquetSource::remote_kvikio(
        "s3://bucket/private/part.parquet?signature=secret",
        source::CredentialMode::SharedProcessEnv,
    );
    let capability = RemoteParquetCapability {
        disposition: RemoteParquetCapabilityDisposition::BackendBuildDisabled,
        uri_scheme: "s3".to_string(),
        read_mode: RemoteReadMode::RequireKvikio,
        credential_mode: RemoteCredentialMode::SharedProcessEnv,
        backend_build_status: RemoteParquetBackendBuildStatus::KvikioRemoteIoDisabled,
    };

    let error =
        validate_remote_kvikio_capability(&resolved, &capability, "query_engine.test.remote_contract").unwrap_err();

    assert_eq!(error.code(), QueryErrorCode::CudfUnsupported);
    assert_eq!(error.status(), error::QueryErrorStatus::Permanent);
    assert_eq!(error.fact_value("reason"), Some("remote_kvikio_backend_unavailable"));
    assert_eq!(error.fact_value("remote_capability_disposition"), Some("backend_build_disabled"));
    assert_eq!(error.fact_value("remote_backend_build_status"), Some("kvikio_remote_io_disabled"));
    assert_eq!(error.fact_value("fallback_allowed"), Some("false"));
    assert!(!error.to_string().contains("s3://bucket"));
    assert!(!error.to_string().contains("signature=secret"));
}

#[test]
fn remote_parquet_profile_carries_only_non_secret_storage_hints() {
    let storage = runtime::SourceStorageProfile::s3(
        "rest-s3",
        runtime::S3SourceStorageProfile::default_chain()
            .with_region("us-west-2")
            .with_endpoint("http://localhost:9000")
            .with_path_style_access(true)
            .with_static_credentials("access", "secret", Some("token".to_string())),
    );

    let profile = cudf_remote_parquet_profile_from_storage(Some("rest-s3"), Some(&storage), None);

    assert_eq!(profile.profile_id(), Some("rest-s3"));
    assert_eq!(profile.region(), Some("us-west-2"));
    assert_eq!(profile.endpoint(), Some("http://localhost:9000"));
    assert_eq!(profile.path_style_access(), Some(true));
    assert_eq!(profile.credential_mode(), RemoteCredentialMode::ScopedProfile);
}

#[test]
fn remote_parquet_profile_passes_resolved_default_chain_credentials_as_scoped_static() {
    let storage = runtime::SourceStorageProfile::s3(
        "glue-default-chain",
        runtime::S3SourceStorageProfile::default_chain().with_region("us-east-1"),
    );
    let credential = AwsCredential {
        key_id: "resolved-access-key".to_string(),
        secret_key: "resolved-secret-key".to_string(),
        token: Some("resolved-session-token".to_string()),
    };

    let profile =
        cudf_remote_parquet_profile_from_storage(Some("glue-default-chain"), Some(&storage), Some(&credential));

    assert_eq!(profile.profile_id(), Some("glue-default-chain"));
    assert_eq!(profile.region(), Some("us-east-1"));
    assert_eq!(profile.credential_mode(), RemoteCredentialMode::ScopedProfile);
    let rendered = format!("{profile:?}");
    assert!(rendered.contains("Static"), "resolved default-chain credential must become a scoped static credential");
    assert!(!rendered.contains("resolved-access-key"));
    assert!(!rendered.contains("resolved-secret-key"));
    assert!(!rendered.contains("resolved-session-token"));
}

#[test]
fn remote_parquet_profile_treats_region_override_as_scoped_hint() {
    let storage = runtime::SourceStorageProfile::s3(
        "glue-region",
        runtime::S3SourceStorageProfile::default_chain().with_region("us-west-2"),
    );

    let profile = cudf_remote_parquet_profile_from_storage(Some("glue-region"), Some(&storage), None);

    assert_eq!(profile.profile_id(), Some("glue-region"));
    assert_eq!(profile.region(), Some("us-west-2"));
    assert_eq!(profile.credential_mode(), RemoteCredentialMode::ScopedProfile);
}

#[test]
fn typed_remote_parquet_contract_hardwires_fail_closed_kvikio_mode() {
    let context = ExecutionContext::builder().build().expect("execution context builds");
    let cudf_file = CudfLocalParquetFile::object_store_uri("s3://bucket/part.parquet", "part", [])
        .expect("valid object store file");
    let resolved = source::ResolvedParquetSource::remote_kvikio(
        "s3://bucket/part.parquet",
        source::CredentialMode::SharedProcessEnv,
    );
    let file = attach_cudf_remote_parquet_contract(&context, cudf_file, &resolved, "query_engine.test")
        .expect("remote contract attaches");
    let remote_source = file.remote_source().expect("remote source is preserved");
    let capability = remote_parquet_capability(remote_source);

    assert_eq!(remote_source.uri(), "s3://bucket/part.parquet");
    assert_eq!(remote_source.read_mode(), RemoteReadMode::RequireKvikio);
    assert_eq!(capability.disposition, RemoteParquetCapabilityDisposition::KvikioRemoteDatasource);
    assert_eq!(resolved.read_strategy(), source::ParquetReadStrategy::RemoteDirect);
}

#[test]
fn datasource_probe_cudf_failures_preserve_facts_without_source_text() {
    let source = RemoteParquetSource::s3("s3://bucket/part.parquet").expect("valid remote source").with_profile(
        RemoteParquetProfile::shared_process_env().with_credential_mode(RemoteCredentialMode::ScopedProfile),
    );
    let cudf_error = probe_remote_parquet_datasource(&source, Some(4096))
        .expect_err("scoped profiles are rejected before datasource open");

    let error = cudf_query_error_without_source(
        "query_engine.test.probe",
        "failed to probe resolved remote Parquet datasource capability",
        &cudf_error,
    )
    .with_fact("resource", "remote_parquet_datasource")
    .with_fact("uri_scheme", "s3");

    assert_eq!(error.fact_value("cudf_operation"), Some("io::parquet::probe_remote_parquet_datasource"));
    assert_eq!(error.fact_value("resource"), Some("remote_parquet_datasource"));
    assert_eq!(error.fact_value("uri_scheme"), Some("s3"));
    assert!(error.source_error().is_none());
    assert!(!error.to_string().contains(" source="));
    assert!(!format!("{error:?}").contains("source: Some"));
}

#[test]
fn remote_direct_cudf_contract_preserves_projection_row_groups_and_reader_mode() {
    let context = ExecutionContext::builder().build().expect("execution context builds");
    let source = remote_direct_test_source();
    let (cudf_source, cudf_chunk) = cudf_local_parquet_source_file_selection(&context, &source, &source.files[0], 0, 7)
        .expect("source contract lowers through algeon conversion path");

    assert_eq!(cudf_source.read_columns(), ["payload", "filter_only"]);
    assert_eq!(cudf_source.output_columns().len(), 1);
    assert_eq!(cudf_source.required_columns().len(), 1);
    assert_eq!(cudf_source.files()[0].row_groups(), Some([1, 3].as_slice()));
    assert_eq!(cudf_source.files()[0].remote_source().unwrap().uri(), "s3://bucket/part.parquet");
    assert_eq!(cudf_source.files()[0].remote_source().unwrap().read_mode(), RemoteReadMode::RequireKvikio);
    assert_eq!(cudf_chunk.remote_source().unwrap().uri(), "s3://bucket/part.parquet");
    assert!(!cudf_chunk.reads_all_row_groups());
    assert_eq!(cudf_chunk.row_group_indexes(), [1, 3]);
    assert_eq!(cudf_chunk.read_columns(), ["payload", "filter_only"]);
    assert_eq!(cudf_chunk.known_row_group_bytes(), Some(768 + 1280));
    assert_eq!(cudf_chunk.known_projected_compressed_bytes(), Some(192 + 320));
    assert_eq!(cudf_chunk.known_pure_filter_compressed_bytes(), Some(48 + 80));
}

#[test]
fn local_parquet_descriptor_propagates_variable_width_payload_evidence() {
    let context = ExecutionContext::builder().build().expect("execution context builds");
    let mut source = remote_direct_test_source();
    source.files[0] = source.files[0].clone().with_variable_width_payload_evidence(BTreeMap::from([(
        1,
        parquet_stats::ParquetVariableWidthPayloadEvidence::new(
            1,
            vec![1, 3],
            2048,
            parquet_stats::ParquetVariableWidthEncodingProof::PlainAndDictionary,
        ),
    )]));

    let (cudf_source, _) = cudf_local_parquet_source_file_selection(&context, &source, &source.files[0], 0, 7)
        .expect("source contract lowers through algeon conversion path");
    let evidence = cudf_source.files()[0]
        .variable_width_payload_evidence()
        .get(&1)
        .expect("variable-width payload evidence propagates to cudf");

    assert_eq!(evidence.physical_column_index, 1);
    assert_eq!(evidence.selected_row_groups, [1, 3]);
    assert_eq!(evidence.payload_upper_bound_bytes, 2048);
    assert_eq!(evidence.encoding_proof, CudfParquetVariableWidthEncodingProof::PlainAndDictionary);
}

#[test]
fn remote_multi_chunk_batch_preserves_uri_row_groups_projection_and_reader_mode() {
    let context = ExecutionContext::builder().build().expect("execution context builds");
    let source = remote_direct_test_source();
    let plan = chunk::plan_local_parquet_source_chunks(
        plan::NodeId::from_index_for_internal_use(1),
        &source,
        chunk::SourceChunkPlanningPolicy::new(None, 1),
    )
    .expect("selected row groups split into a multi-chunk plan");
    assert_eq!(plan.chunks.len(), 2);

    let (cudf_source, cudf_chunks) = cudf_local_parquet_source_chunk_batch(&context, &source, &plan.chunks)
        .expect("multi-chunk remote source lowers through the typed batch builder");

    assert_eq!(cudf_source.files().len(), 1);
    assert_eq!(cudf_chunks.len(), 2);
    assert_eq!(cudf_source.read_columns(), ["payload", "filter_only"]);
    assert_eq!(cudf_chunks[0].row_group_indexes(), [1]);
    assert_eq!(cudf_chunks[1].row_group_indexes(), [3]);
    assert!(cudf_chunks.iter().all(|chunk| chunk.file_index() == 0));
    for file in cudf_source.files() {
        let remote = file.remote_source().expect("batch file retains its typed remote source");
        assert_eq!(remote.uri(), "s3://bucket/part.parquet");
        assert_eq!(remote.read_mode(), RemoteReadMode::RequireKvikio);
    }
}

fn local_test_source(files: Vec<LocalParquetFile>) -> LocalParquetSource {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int32, false),
        ArrowField::new("payload", ArrowDataType::Utf8, true),
    ]));
    LocalParquetSource::from_files(files, schema, vec!["id".to_string(), "payload".to_string()])
}

fn local_test_file(path: &str, stable_id: &str, row_group_count: usize) -> LocalParquetFile {
    LocalParquetFile::from_resolved_parquet_source(source::ResolvedParquetSource::local_kvikio(path))
        .expect("valid local source")
        .with_stable_id(stable_id)
        .with_file_size_bytes(4096)
        .with_row_group_bytes(vec![512; row_group_count])
        .with_row_group_rows(vec![10; row_group_count])
}

#[test]
fn local_chunk_batch_lists_each_source_file_once_and_keeps_chunk_row_groups() {
    let context = ExecutionContext::builder().build().expect("execution context builds");
    // The same path appears as two source files with distinct identities; the
    // batch must keep them apart instead of merging by path.
    let source = local_test_source(vec![
        local_test_file("/tmp/lineitem-a.parquet", "a", 3),
        local_test_file("/tmp/lineitem-b.parquet", "b", 2),
        local_test_file("/tmp/lineitem-a.parquet", "a-again", 1),
    ]);
    let plan = chunk::plan_local_parquet_source_chunks(
        plan::NodeId::from_index_for_internal_use(1),
        &source,
        chunk::SourceChunkPlanningPolicy::new(None, 1),
    )
    .expect("one row group per chunk splits every file");
    assert_eq!(plan.chunks.iter().map(|chunk| chunk.file_index).collect::<Vec<_>>(), [0, 0, 0, 1, 1, 2]);

    let (cudf_source, cudf_chunks) = cudf_local_parquet_source_chunk_batch(&context, &source, &plan.chunks)
        .expect("same-file chunks lower through one cudf file entry");

    assert_eq!(
        cudf_source.files().iter().map(CudfLocalParquetFile::stable_id).collect::<Vec<_>>(),
        ["a", "b", "a-again"]
    );
    assert_eq!(cudf_chunks.iter().map(CudfSourceChunk::file_index).collect::<Vec<_>>(), [0, 0, 0, 1, 1, 2]);
    assert_eq!(
        cudf_chunks.iter().map(|chunk| chunk.row_group_indexes().to_vec()).collect::<Vec<_>>(),
        [vec![0], vec![1], vec![2], vec![0], vec![1], vec![0]]
    );
    assert_eq!(cudf_chunks.iter().map(CudfSourceChunk::chunk_index).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5]);
    assert_eq!(
        cudf_chunks.iter().map(CudfSourceChunk::stable_id).collect::<Vec<_>>(),
        ["a", "a", "a", "b", "b", "a-again"]
    );
}

#[test]
fn remote_direct_contract_keeps_filter_only_predicate_attachable() {
    let context = ExecutionContext::builder().build().expect("execution context builds");
    let source = remote_direct_test_source().with_reader_predicate(expr::Expr::binary(
        expr::Expr::column("filter_only"),
        expr::BinaryOp::Eq,
        expr::Expr::literal(expr::LiteralValue::Boolean(true)),
    ));
    let predicate = source.reader_predicate.as_ref().expect("reader predicate set");
    assert!(source.reader_predicate_is_supported(predicate, rapids_backend::rapids_analysis_context()));

    let (cudf_source, _cudf_chunk) =
        cudf_local_parquet_source_file_selection(&context, &source, &source.files[0], 0, 7)
            .expect("source contract lowers through algeon conversion path");

    assert_eq!(cudf_source.read_columns(), ["payload", "filter_only"]);
    assert!(cudf_source.reader_predicate().is_some());
    assert_eq!(cudf_source.files()[0].remote_source().unwrap().uri(), "s3://bucket/part.parquet");
    assert_eq!(cudf_source.files()[0].remote_source().unwrap().read_mode(), RemoteReadMode::RequireKvikio);
}

#[test]
fn remote_direct_chunk_batch_propagates_required_footer_fetch_failure_for_unverified_decimal_type() {
    let context = context_with_unreachable_footer_store();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new("discount", ArrowDataType::Decimal128(15, 2), true)]));
    let resolved = remote_direct_with_unreachable_footer_store("s3://bucket/lineitem.parquet");
    let file = LocalParquetFile::from_resolved_parquet_source(resolved)
        .expect("valid remote direct source")
        .with_stable_id("lineitem")
        .with_file_size_bytes(4096)
        .with_row_groups(vec![0])
        .with_row_group_bytes(vec![4096])
        .with_row_group_rows(vec![10])
        .with_reader_decimal_types(vec![source::ParquetReaderDecimalType::new(
            "discount",
            source::ParquetReaderDecimalTypeId::Decimal64,
            -2,
        )]);
    let predicate = Expr::binary(
        Expr::column("discount"),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
    );
    let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
        .with_column_file_indices(vec![0], vec![0], vec![])
        .with_reader_predicate(predicate);
    let predicate = source.reader_predicate.as_ref().expect("reader predicate set");
    assert!(source.reader_predicate_is_supported(predicate, rapids_backend::rapids_analysis_context()));
    assert!(source.reader_predicate_requires_source_contract(predicate));
    let file = &source.files[0];
    let chunk = SourceChunk {
        source_node_id: plan::NodeId::from_index_for_internal_use(1),
        chunk_index: 7,
        file_index: 0,
        location: file.location.clone(),
        path: file.path.clone(),
        resolved_source: file.resolved_source.clone(),
        stable_id: file.stable_id.clone(),
        row_group_indexes: vec![0],
        reads_all_row_groups: false,
        known_file_bytes: file.file_size_bytes,
        known_row_group_bytes: Some(4096),
        known_projected_compressed_bytes: Some(4096),
        known_projected_uncompressed_bytes: Some(4096),
        known_pure_filter_compressed_bytes: Some(0),
        known_pure_filter_uncompressed_bytes: Some(0),
        estimated_chunk_bytes: Some(4096),
        estimated_source_read_bytes: Some(4096),
        exact_source_read_preflight_bytes: Some(4096),
        source_file_row_groups: file.row_groups.clone(),
        source_file_row_group_bytes: file.row_group_bytes.clone(),
        source_file_projected_compressed_bytes: file.projected_row_group_compressed_bytes.clone(),
        source_file_projected_uncompressed_bytes: file.projected_row_group_uncompressed_bytes.clone(),
        source_file_pure_filter_compressed_bytes: file.pure_filter_row_group_compressed_bytes.clone(),
        source_file_pure_filter_uncompressed_bytes: file.pure_filter_row_group_uncompressed_bytes.clone(),
        source_file_row_group_rows: file.row_group_rows.clone(),
        reader_decimal_types: file.reader_decimal_types.clone(),
        reader_decimal_types_footer_verified: false,
        read_columns: vec!["discount".to_string()],
        partition_values: vec![],
        zero_column_count: false,
        known_row_count: Some(10),
        scan_limit_rows: None,
        size_estimate_based: false,
        source_batching_reason: Some("batched".to_string()),
        source_chunk_planning_mode: runtime::SourceChunkPlanningMode::Conservative,
        source_chunk_planning_reason: "test".to_string(),
    };

    let error = cudf_local_parquet_source_chunk_batch(&context, &source, &[chunk])
        .expect_err("chunk batches must verify preattached remote decimal reader metadata against the footer");

    assert_required_footer_fetch_temporary_failure(&error);
    assert!(!error.to_string().contains("s3://bucket"));
}

#[test]
fn remote_direct_chunk_batch_reuses_planner_verified_decimal_reader_type() {
    let context = ExecutionContext::builder().build().expect("execution context builds");
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new("discount", ArrowDataType::Decimal128(15, 2), true)]));
    let reader_decimal_types =
        vec![source::ParquetReaderDecimalType::new("discount", source::ParquetReaderDecimalTypeId::Decimal64, -2)];
    let resolved = source::ResolvedParquetSource::remote_kvikio(
        "s3://bucket/lineitem.parquet",
        source::CredentialMode::SharedProcessEnv,
    );
    let file = LocalParquetFile::from_resolved_parquet_source(resolved)
        .expect("valid remote direct source")
        .with_stable_id("lineitem")
        .with_file_size_bytes(4096)
        .with_row_groups(vec![0])
        .with_row_group_bytes(vec![4096])
        .with_row_group_rows(vec![10])
        .with_reader_decimal_types(reader_decimal_types.clone());
    let predicate = Expr::binary(
        Expr::column("discount"),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
    );
    let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
        .with_column_file_indices(vec![0], vec![0], vec![])
        .with_reader_predicate(predicate);
    let file = &source.files[0];
    let chunk = SourceChunk {
        source_node_id: plan::NodeId::from_index_for_internal_use(1),
        chunk_index: 7,
        file_index: 0,
        location: file.location.clone(),
        path: file.path.clone(),
        resolved_source: file.resolved_source.clone(),
        stable_id: file.stable_id.clone(),
        row_group_indexes: vec![0],
        reads_all_row_groups: false,
        known_file_bytes: file.file_size_bytes,
        known_row_group_bytes: Some(4096),
        known_projected_compressed_bytes: Some(4096),
        known_projected_uncompressed_bytes: Some(4096),
        known_pure_filter_compressed_bytes: Some(0),
        known_pure_filter_uncompressed_bytes: Some(0),
        estimated_chunk_bytes: Some(4096),
        estimated_source_read_bytes: Some(4096),
        exact_source_read_preflight_bytes: Some(4096),
        source_file_row_groups: file.row_groups.clone(),
        source_file_row_group_bytes: file.row_group_bytes.clone(),
        source_file_projected_compressed_bytes: file.projected_row_group_compressed_bytes.clone(),
        source_file_projected_uncompressed_bytes: file.projected_row_group_uncompressed_bytes.clone(),
        source_file_pure_filter_compressed_bytes: file.pure_filter_row_group_compressed_bytes.clone(),
        source_file_pure_filter_uncompressed_bytes: file.pure_filter_row_group_uncompressed_bytes.clone(),
        source_file_row_group_rows: file.row_group_rows.clone(),
        reader_decimal_types,
        reader_decimal_types_footer_verified: true,
        read_columns: vec!["discount".to_string()],
        partition_values: vec![],
        zero_column_count: false,
        known_row_count: Some(10),
        scan_limit_rows: None,
        size_estimate_based: false,
        source_batching_reason: Some("batched".to_string()),
        source_chunk_planning_mode: runtime::SourceChunkPlanningMode::Conservative,
        source_chunk_planning_reason: "test".to_string(),
    };

    let (cudf_source, _chunks) = cudf_local_parquet_source_chunk_batch(&context, &source, &[chunk])
        .expect("planner-verified chunk reader decimal metadata should not refetch object-store footers");

    assert_eq!(cudf_source.files()[0].reader_decimal_types()[0].data_type().id(), cudf::TypeId::Decimal64);
    assert!(cudf_source.reader_predicate().is_some());
}

#[test]
fn chunk_batch_rejects_chunk_identity_mismatch_before_footer_verification() {
    let context = ExecutionContext::builder().build().expect("execution context builds");
    let source = remote_direct_test_source();
    let file = &source.files[0];
    let mut chunk = SourceChunk {
        source_node_id: plan::NodeId::from_index_for_internal_use(1),
        chunk_index: 7,
        file_index: 0,
        location: file.location.clone(),
        path: file.path.clone(),
        resolved_source: file.resolved_source.clone(),
        stable_id: file.stable_id.clone(),
        row_group_indexes: vec![1],
        reads_all_row_groups: false,
        known_file_bytes: file.file_size_bytes,
        known_row_group_bytes: Some(768),
        known_projected_compressed_bytes: Some(192),
        known_projected_uncompressed_bytes: Some(384),
        known_pure_filter_compressed_bytes: Some(48),
        known_pure_filter_uncompressed_bytes: Some(96),
        estimated_chunk_bytes: Some(384),
        estimated_source_read_bytes: Some(816),
        exact_source_read_preflight_bytes: Some(384),
        source_file_row_groups: file.row_groups.clone(),
        source_file_row_group_bytes: file.row_group_bytes.clone(),
        source_file_projected_compressed_bytes: file.projected_row_group_compressed_bytes.clone(),
        source_file_projected_uncompressed_bytes: file.projected_row_group_uncompressed_bytes.clone(),
        source_file_pure_filter_compressed_bytes: file.pure_filter_row_group_compressed_bytes.clone(),
        source_file_pure_filter_uncompressed_bytes: file.pure_filter_row_group_uncompressed_bytes.clone(),
        source_file_row_group_rows: file.row_group_rows.clone(),
        reader_decimal_types: Vec::new(),
        reader_decimal_types_footer_verified: false,
        read_columns: source.read_columns.clone(),
        partition_values: vec![],
        zero_column_count: false,
        known_row_count: Some(10),
        scan_limit_rows: None,
        size_estimate_based: false,
        source_batching_reason: Some("batched".to_string()),
        source_chunk_planning_mode: runtime::SourceChunkPlanningMode::Conservative,
        source_chunk_planning_reason: "test".to_string(),
    };
    chunk.location = source::ParquetSourceLocation::object_store_uri("s3://bucket/other.parquet").unwrap();

    let error = cudf_local_parquet_source_chunk_batch(&context, &source, &[chunk])
        .expect_err("chunk must not verify one file while decoding another object");
    let rendered = format!("{error}");

    assert_eq!(error.code(), QueryErrorCode::InvalidPlan);
    assert_eq!(error.fact_value("reason"), Some("source_chunk_identity_mismatch"));
    assert_eq!(error.fact_value("field"), Some("location"));
    assert!(error.fact_value("path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")));
    assert!(error.fact_value("source_file_path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")));
    assert!(!rendered.contains("s3://bucket"));
}

#[test]
fn remote_direct_contract_propagates_required_footer_fetch_failure_without_footer_type_metadata() {
    let context = context_with_unreachable_footer_store();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new("discount", ArrowDataType::Decimal128(15, 2), true)]));
    let resolved = remote_direct_with_unreachable_footer_store("s3://bucket/lineitem.parquet");
    let file = LocalParquetFile::from_resolved_parquet_source(resolved)
        .expect("valid remote direct source")
        .with_stable_id("lineitem")
        .with_file_size_bytes(4096)
        .with_row_groups(vec![0])
        .with_row_group_bytes(vec![4096])
        .with_row_group_rows(vec![10]);
    let predicate = Expr::binary(
        Expr::column("discount"),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
    );
    let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
        .with_column_file_indices(vec![0], vec![0], vec![])
        .with_reader_predicate(predicate);

    let error = cudf_local_parquet_source_file_selection(&context, &source, &source.files[0], 0, 7)
        .expect_err("remote decimal reader predicates need proven footer dtype metadata");

    assert_required_footer_fetch_temporary_failure(&error);
    assert!(!error.to_string().contains("s3://bucket"));
}

#[test]
fn remote_direct_contract_propagates_required_footer_fetch_failure_for_unverified_decimal_type() {
    let context = context_with_unreachable_footer_store();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new("discount", ArrowDataType::Decimal128(15, 2), true)]));
    let resolved = remote_direct_with_unreachable_footer_store("s3://bucket/lineitem.parquet");
    let file = LocalParquetFile::from_resolved_parquet_source(resolved)
        .expect("valid remote direct source")
        .with_stable_id("lineitem")
        .with_file_size_bytes(4096)
        .with_row_groups(vec![0])
        .with_row_group_bytes(vec![4096])
        .with_row_group_rows(vec![10])
        .with_reader_decimal_types(vec![source::ParquetReaderDecimalType::new(
            "discount",
            source::ParquetReaderDecimalTypeId::Decimal32,
            -2,
        )]);
    let predicate = Expr::binary(
        Expr::column("discount"),
        BinaryOp::GtEq,
        Expr::literal(LiteralValue::Decimal128 { raw: 5, precision: 15, scale: 2 }),
    );
    let source = LocalParquetSource::from_files(vec![file], schema, vec!["discount".to_string()])
        .with_column_file_indices(vec![0], vec![0], vec![])
        .with_reader_predicate(predicate);

    let error = cudf_local_parquet_source_file_selection(&context, &source, &source.files[0], 0, 7)
        .expect_err("direct remote decimal reader predicates must verify preattached types against the footer");

    assert_required_footer_fetch_temporary_failure(&error);
    assert!(!error.to_string().contains("s3://bucket"));
}

#[test]
fn remote_direct_cudf_contract_errors_redact_object_store_locations() {
    let context = ExecutionContext::builder().build().expect("execution context builds");
    let sensitive_uri = "s3://secret-bucket/private/key.parquet?X-Amz-Signature=abc";
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new("payload", ArrowDataType::Utf8, true)]));
    let resolved =
        source::ResolvedParquetSource::remote_kvikio(sensitive_uri, source::CredentialMode::SharedProcessEnv);
    let file = LocalParquetFile::from_resolved_parquet_source(resolved)
        .expect("valid remote direct source")
        .with_stable_id("secret-bucket/private/key.parquet?X-Amz-Signature=abc")
        .with_row_group_bytes(vec![10, 20])
        .with_row_group_rows(vec![1]);
    let source = LocalParquetSource::from_files(vec![file], schema, vec!["payload".to_string()]);

    let error = cudf_local_parquet_source_file_selection(&context, &source, &source.files[0], 0, 0)
        .expect_err("mismatched row-group metadata should fail before read");
    let rendered = format!("{error}");
    let debug = format!("{error:?}");

    assert!(error.fact_value("path").is_some_and(|value| value.starts_with("object_store:s3:fnv64:")));
    assert!(error.fact_value("stable_id").is_some_and(|value| value.starts_with("object_store:s3:stable_id_fnv64:")),);
    for sensitive in ["secret-bucket", "private/key", "X-Amz-Signature", "abc"] {
        assert!(!rendered.contains(sensitive), "{sensitive} leaked in display error: {rendered}");
        assert!(!debug.contains(sensitive), "{sensitive} leaked in debug error: {debug}");
    }
    assert!(error.source_error().is_none());
}
