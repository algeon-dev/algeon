use cudf::core::{ColumnView, TableView};

fn main() {
    let _ = ::std::mem::size_of::<ColumnView<'static>>();
    let _ = ::std::mem::size_of::<TableView<'static>>();
}
