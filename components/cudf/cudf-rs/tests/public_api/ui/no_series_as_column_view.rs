use cudf::Series;

fn main() {
    let f: fn(&Series) -> _ = Series::as_column_view;
    let _ = f;
}
