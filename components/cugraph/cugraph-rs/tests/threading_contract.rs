use cugraph::handle::ResourceHandle;
use static_assertions::assert_not_impl_any;

#[test]
fn resource_handle_is_not_send_or_sync() {
    assert_not_impl_any!(ResourceHandle: Send, Sync);
}
