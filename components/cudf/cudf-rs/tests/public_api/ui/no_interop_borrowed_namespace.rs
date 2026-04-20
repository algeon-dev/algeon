use cudf::interop::borrowed::{BorrowedColumnView, BorrowedTableView};

fn main() {
    let _: Option<BorrowedColumnView<'static>> = None;
    let _: Option<BorrowedTableView<'static>> = None;
}
