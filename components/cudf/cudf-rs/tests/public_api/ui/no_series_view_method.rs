use cudf::Series;

fn main() {
    let f: fn(&Series) -> _ = Series::view;
    let _ = f;
}
