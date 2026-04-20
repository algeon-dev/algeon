use cudf::{Error, ErrorKind, ErrorStatus};
use rapids_interop::{InteropError, InteropErrorKind};

#[test]
fn new_defaults_to_permanent() {
    let err = Error::new(ErrorKind::TypeError, "bad type");
    assert_eq!(err.kind(), ErrorKind::TypeError);
    assert_eq!(err.message(), "bad type");
    assert!(err.is_permanent());
    assert!(!err.is_temporary());
    assert!(!err.is_persistent());
    assert_eq!(err.operation(), "");
    assert!(err.context().is_empty());
}

#[test]
fn builder_methods_chain() {
    let err = Error::new(ErrorKind::CudaError, "out of memory")
        .with_operation("reduce")
        .with_context("dtype", "float64")
        .with_context("rows", "1000000")
        .set_temporary();

    assert_eq!(err.kind(), ErrorKind::CudaError);
    assert!(err.is_temporary());
    assert_eq!(err.operation(), "reduce");
    assert_eq!(err.context().len(), 2);
    assert_eq!(err.context()[0], ("dtype", "float64".to_string()));
    assert_eq!(err.context()[1], ("rows", "1000000".to_string()));
}

#[test]
fn status_accessor_reflects_status() {
    let err = Error::new(ErrorKind::TypeError, "bad type");
    assert_eq!(err.status(), ErrorStatus::Permanent);

    let tmp = Error::new(ErrorKind::CudaError, "oom").set_temporary();
    assert_eq!(tmp.status(), ErrorStatus::Temporary);

    let persistent = Error::new(ErrorKind::FatalCudaError, "device lost").set_persistent();
    assert_eq!(persistent.status(), ErrorStatus::Persistent);
}

#[test]
fn set_source_attaches_cause() {
    let cause = std::io::Error::new(std::io::ErrorKind::NotFound, "file gone");
    let err = Error::new(ErrorKind::IoError, "read failed").set_source(cause);

    let src = std::error::Error::source(&err);
    assert!(src.is_some());
}

#[test]
fn set_source_accepts_boxed_std_error_and_preserves_chain() {
    let cause: Box<dyn std::error::Error + Send + Sync + 'static> =
        Box::new(std::io::Error::other("boxed child cause"));
    let err = Error::new(ErrorKind::IoError, "read failed").set_source(cause);

    let src = std::error::Error::source(&err).expect("expected boxed source");
    assert_eq!(src.to_string(), "boxed child cause");
}

#[test]
fn display_minimal() {
    let err = Error::new(ErrorKind::TypeError, "cannot cast");
    let s = err.to_string();
    assert!(s.contains("TypeError"));
    assert!(s.contains("permanent"));
    assert!(s.contains("cannot cast"));
    assert!(!s.contains(" at "));
    assert!(!s.contains("context:"));
}

#[test]
fn display_full() {
    let err = Error::new(ErrorKind::CudaError, "out of memory")
        .with_operation("reduce")
        .with_context("dtype", "float64")
        .set_temporary();

    let s = err.to_string();
    assert!(s.contains("CudaError (temporary)"));
    assert!(s.contains("at reduce"));
    assert!(s.contains("dtype: float64"));
    assert!(s.contains("out of memory"));
}

#[test]
fn debug_is_multiline() {
    let err = Error::new(ErrorKind::CudaError, "oom").with_operation("sort").with_context("rows", "1M");

    let s = format!("{:?}", err);
    assert!(s.contains("operation: sort"));
    assert!(s.contains("rows: 1M"));
    assert!(s.contains("message: oom"));
}

#[test]
fn default_status_logic_errors_are_permanent() {
    let type_err = Error::new(ErrorKind::TypeError, "bad type");
    assert_eq!(type_err.status(), ErrorStatus::Permanent);

    let invalid_arg = Error::new(ErrorKind::InvalidArgument, "bad argument");
    assert_eq!(invalid_arg.status(), ErrorStatus::Permanent);

    let not_found = Error::new(ErrorKind::NotFound, "missing column");
    assert_eq!(not_found.status(), ErrorStatus::Permanent);

    let not_supported = Error::new(ErrorKind::NotSupported, "not supported");
    assert_eq!(not_supported.status(), ErrorStatus::Permanent);

    let invalid_op = Error::new(ErrorKind::InvalidOperation, "bad op");
    assert_eq!(invalid_op.status(), ErrorStatus::Permanent);

    let index_err = Error::new(ErrorKind::IndexError, "bad index");
    assert_eq!(index_err.status(), ErrorStatus::Permanent);
}

#[test]
fn set_persistent_works() {
    let err = Error::new(ErrorKind::Unknown, "repeated failure").set_persistent();
    assert!(err.is_persistent());
    assert!(!err.is_permanent());
    assert!(!err.is_temporary());
}

#[test]
fn display_with_source() {
    let cause = std::io::Error::new(std::io::ErrorKind::NotFound, "no such file");
    let err = Error::new(ErrorKind::IoError, "read failed").with_operation("read_csv").set_source(cause);

    let s = err.to_string();
    assert!(s.contains("source:"));
    assert!(s.contains("no such file"));
}

#[test]
fn with_operation_preserves_prior_failure_site_in_called_context() {
    let err = Error::new(ErrorKind::InvalidOperation, "inner failure")
        .with_operation("inner::op")
        .with_operation("outer::op");

    assert_eq!(err.operation(), "outer::op");
    assert_eq!(err.context(), [("called", "inner::op".to_string())]);
}

#[test]
fn with_operation_stacks_multiple_called_context_entries() {
    let err = Error::new(ErrorKind::InvalidOperation, "inner failure")
        .with_operation("inner::op")
        .with_operation("middle::op")
        .with_operation("outer::op");

    assert_eq!(err.operation(), "outer::op");
    assert_eq!(err.context(), [("called", "inner::op".to_string()), ("called", "middle::op".to_string()),]);
}

#[test]
fn interop_error_converts_with_kind_context_status_and_source() {
    let interop = InteropError::new(InteropErrorKind::UnsupportedType, "dictionary export is not supported")
        .with_context("operation", "interop::PreparedColumnExport::try_prepare_device_column_export")
        .with_context("type_id", "Dictionary32");

    let err = Error::from(interop);

    assert_eq!(err.kind(), ErrorKind::TypeError);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "dictionary export is not supported");
    assert_eq!(err.operation(), "interop::PreparedColumnExport::try_prepare_device_column_export");
    assert_eq!(err.context(), [("type_id", "Dictionary32".to_string())]);
    assert_eq!(
        std::error::Error::source(&err).map(std::string::ToString::to_string).as_deref(),
        Some("dictionary export is not supported")
    );
}

#[test]
fn interop_error_nested_source_chain_is_preserved_when_mapping_into_cudf_error() {
    let interop = InteropError::new(InteropErrorKind::InvalidView, "borrowed view rejected")
        .with_context("column_id", "7")
        .set_source(std::io::Error::other("child cause"));

    let err = Error::from(interop);
    let source = std::error::Error::source(&err).expect("expected interop source");

    assert_eq!(source.to_string(), "borrowed view rejected");
    assert_eq!(source.source().map(std::string::ToString::to_string).as_deref(), Some("child cause"));
}

#[test]
fn interop_not_found_maps_to_not_found() {
    let err = Error::from(
        InteropError::new(InteropErrorKind::NotFound, "prepared export referenced a missing buffer binding")
            .with_context("buffer_id", "10"),
    );

    assert_eq!(err.kind(), ErrorKind::NotFound);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "prepared export referenced a missing buffer binding");
    assert_eq!(err.context(), [("buffer_id", "10".to_string())]);
}

#[test]
fn interop_readiness_unavailable_without_cudf_source_maps_to_permanent_invalid_operation() {
    let err = Error::from(
        InteropError::new(
            InteropErrorKind::ReadinessUnavailable,
            "CUDA stream handle must carry explicit device identity",
        )
        .with_context("field", "cuda_stream.device"),
    );

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.context(), [("field", "cuda_stream.device".to_string())]);
}

#[test]
fn interop_unsupported_layout_maps_to_not_supported() {
    let err = Error::from(
        InteropError::new(InteropErrorKind::UnsupportedLayout, "struct columns are not supported for this export path")
            .with_context("layout", "struct"),
    );

    assert_eq!(err.kind(), ErrorKind::NotSupported);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "struct columns are not supported for this export path");
    assert_eq!(err.context(), [("layout", "struct".to_string())]);
}

#[test]
fn interop_error_promotes_nested_operation_context_into_operation_and_called() {
    let interop = InteropError::new(InteropErrorKind::InvalidView, "borrowed view rejected")
        .with_context("operation", "interop::ActiveReadLease::complete")
        .with_context("operation", "runtime::completion::wait")
        .with_context("column_id", "7");

    let err = Error::from(interop);

    assert_eq!(err.kind(), ErrorKind::InvalidOperation);
    assert_eq!(err.operation(), "interop::ActiveReadLease::complete");
    assert_eq!(err.context(), [("called", "runtime::completion::wait".to_string()), ("column_id", "7".to_string())]);
}

#[test]
fn interop_error_with_cudf_source_preserves_cudf_kind_and_status() {
    let cudf_error =
        Error::new(ErrorKind::CudaError, "CUDA runtime busy").with_operation("runtime::synchronize").set_temporary();
    let interop = InteropError::new(InteropErrorKind::ExecutionBindingUnavailable, "CUDA runtime busy")
        .with_context("operation", "interop::ActiveReadLease::complete")
        .set_source(cudf_error);

    let err = Error::from(interop);

    assert_eq!(err.kind(), ErrorKind::CudaError);
    assert_eq!(err.status(), ErrorStatus::Temporary);
    assert_eq!(err.operation(), "interop::ActiveReadLease::complete");
    assert_eq!(err.context(), [("called", "runtime::synchronize".to_string())]);
    assert_eq!(
        std::error::Error::source(&err).map(std::string::ToString::to_string).as_deref(),
        Some("CUDA runtime busy")
    );
}

#[test]
fn question_mark_propagates_interop_error_into_cudf_result() {
    fn helper() -> cudf::Result<()> {
        Err::<(), _>(
            InteropError::new(InteropErrorKind::NullMaskUnsupported, "byte-granularity validity is not supported")
                .with_context("column", "validity"),
        )?;
        Ok(())
    }

    let err = helper().unwrap_err();
    assert_eq!(err.kind(), ErrorKind::NullError);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "byte-granularity validity is not supported");
    assert_eq!(err.context(), [("column", "validity".to_string())]);
}
