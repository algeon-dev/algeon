use cudf::interop::{
    BorrowedColumnView, CompletionAuthority, OwnedColumnTransfer, PreparedColumnExport, ReadStartAuthority,
    TryBorrowColumnView, TryPrepareDeviceColumnExport,
};

fn main() {
    let _ = std::any::TypeId::of::<BorrowedColumnView<'static>>();
    let _ = std::any::TypeId::of::<PreparedColumnExport<'static>>();
    let _ = std::any::TypeId::of::<OwnedColumnTransfer>();
    let _ = std::any::TypeId::of::<CompletionAuthority<'static>>();
    let _ = std::any::TypeId::of::<ReadStartAuthority<'static>>();
    let _ = std::any::TypeId::of::<fn() -> Option<&'static dyn TryBorrowColumnView>>();
    let _ = std::any::TypeId::of::<fn() -> Option<&'static dyn TryPrepareDeviceColumnExport>>();
}
