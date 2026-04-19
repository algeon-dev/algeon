use cudf::testing::ColumnView;

fn main() {
    let f: fn(&ColumnView<'_>) -> _ = ColumnView::factorize_parts;
    let _ = f;
}
