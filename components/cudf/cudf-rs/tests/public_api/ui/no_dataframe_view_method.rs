use cudf::DataFrame;

fn main() {
    let f: fn(&DataFrame) -> _ = DataFrame::view;
    let _ = f;
}
