use cudf::core::Column;

fn main() {
    let f: fn(&Column) -> _ = Column::view;
    let _ = f;
}
