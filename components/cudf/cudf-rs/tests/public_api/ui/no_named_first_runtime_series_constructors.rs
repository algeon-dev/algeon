use cudf::{Runtime, Series};

fn main() {
    let rt = Runtime::new().unwrap();
    let _ = rt.series("value", &[1i32, 2, 3]).unwrap();
    let _ = rt.nulls("value", cudf::TypeId::Int32, 3).unwrap();

    let scalar = rt.scalar(7i32).unwrap();
    let _ = rt.series_from_scalar("value", &scalar, 2).unwrap();
    let _ = Series::from_scalar("value", &scalar, 2).unwrap();
}
