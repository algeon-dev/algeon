use cudf::ErrorKind;

fn main() {
    let _ = ErrorKind::InvalidOperation;
    let _ = ErrorKind::TypeError;
    let _ = ErrorKind::NotSupported;
    let _ = ErrorKind::Unknown;
}
