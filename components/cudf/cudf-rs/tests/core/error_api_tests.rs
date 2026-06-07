use cudf::{Error, ErrorKind, ErrorStatus};

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
