use cudf::core::Table;

fn main() {
    let f: fn(&Table) -> _ = Table::view;
    let _ = f;
}
