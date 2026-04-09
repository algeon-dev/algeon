use cudf::{
    Error, ErrorKind, ErrorStatus,
    interop::{InteropError, InteropErrorKind},
};

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
fn interop_error_converts_with_kind_context_status_and_source() {
    let interop = InteropError::new(InteropErrorKind::UnsupportedType, "dictionary export is not supported")
        .with_context("operation", "interop::PreparedColumnExport::try_prepare_device_column_export")
        .with_context("type_id", "Dictionary32");

    let err = Error::from(interop);

    assert_eq!(err.kind(), ErrorKind::TypeError);
    assert_eq!(err.status(), ErrorStatus::Permanent);
    assert_eq!(err.message(), "dictionary export is not supported");
    assert_eq!(err.context().len(), 2);
    assert_eq!(
        err.context()[0],
        ("operation", "interop::PreparedColumnExport::try_prepare_device_column_export".to_string())
    );
    assert_eq!(err.context()[1], ("type_id", "Dictionary32".to_string()));
    assert_eq!(
        std::error::Error::source(&err).map(std::string::ToString::to_string).as_deref(),
        Some("dictionary export is not supported")
    );
}

#[test]
fn interop_readiness_unavailable_maps_to_temporary_cuda_error() {
    let err = Error::from(
        InteropError::new(
            InteropErrorKind::ReadinessUnavailable,
            "CUDA stream handle must carry explicit device identity",
        )
        .with_context("field", "cuda_stream.device"),
    );

    assert_eq!(err.kind(), ErrorKind::CudaError);
    assert_eq!(err.status(), ErrorStatus::Temporary);
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
