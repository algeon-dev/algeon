use cudf::core::Column;

fn main() {
    let column = Column::from_strs(&["alpha", "beta"]).unwrap();
    let view = column.view().unwrap();
    let _ = view.to_strings().unwrap();
}
