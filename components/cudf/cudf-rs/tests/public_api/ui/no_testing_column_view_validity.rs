use cudf::testing::ColumnView;

fn main() {
    let f: fn(&ColumnView<'_>) -> _ = ColumnView::validity;
    let _ = f;
}
