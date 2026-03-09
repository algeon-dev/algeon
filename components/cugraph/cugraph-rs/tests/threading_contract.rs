use static_assertions::assert_not_impl_any;

use cugraph::handle::ResourceHandle;

#[test]
fn resource_handle_is_not_send_or_sync() {
    assert_not_impl_any!(ResourceHandle: Send, Sync);
}
