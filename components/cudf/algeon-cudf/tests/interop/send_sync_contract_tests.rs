use std::thread;

use cudf::{
    Runtime,
    core::column::DeviceSlice,
    interop::{
        arrow_device::{
            DeviceArrowColumnExport, DeviceArrowColumnOwnedAbiParts, DeviceArrowColumnRawParts, DeviceArrowTableExport,
            DeviceArrowTableOwnedAbiParts, DeviceArrowTableRawParts, ForeignDeviceArrowColumn, ForeignDeviceArrowTable,
        },
        cuda_stream_handle,
    },
    ops::join::{HashJoin, HashJoinOptions, JoinIndices, SingleSideJoinIndices},
    runtime::CudaStream,
};
use rapids_interop::{
    CompletionAuthority, PreparedColumnExport, PreparedTableExport, ProducerReadiness, ReadStartAuthority,
};
use static_assertions::assert_impl_all;

use crate::common;

assert_impl_all!(DeviceSlice<'static>: Send, Sync);
assert_impl_all!(ProducerReadiness<'static>: Send, Sync);
assert_impl_all!(PreparedColumnExport<'static>: Send, Sync);
assert_impl_all!(PreparedTableExport<'static>: Send, Sync);
assert_impl_all!(DeviceArrowColumnExport: Send, Sync);
assert_impl_all!(DeviceArrowColumnRawParts: Send, Sync);
assert_impl_all!(DeviceArrowColumnOwnedAbiParts: Send, Sync);
assert_impl_all!(ForeignDeviceArrowColumn: Send, Sync);
assert_impl_all!(DeviceArrowTableExport: Send, Sync);
assert_impl_all!(DeviceArrowTableRawParts: Send, Sync);
assert_impl_all!(DeviceArrowTableOwnedAbiParts: Send, Sync);
assert_impl_all!(ForeignDeviceArrowTable: Send, Sync);
assert_impl_all!(HashJoin: Send, Sync);
assert_impl_all!(HashJoinOptions: Send, Sync);
assert_impl_all!(JoinIndices: Send, Sync);
assert_impl_all!(SingleSideJoinIndices: Send, Sync);

#[test]
fn prepared_column_export_can_activate_and_complete_on_worker_thread() {
    let _observation_lock = common::lock_event_observation();
    let producer_runtime = Runtime::builder().build().unwrap();
    let series = producer_runtime.series(&[1i32, 2, 3]).unwrap();
    let export = series.try_prepare_device_column_export().unwrap();

    let consumer_stream = CudaStream::new().unwrap();
    let _consumer_runtime =
        Runtime::builder().with_device(producer_runtime.device()).with_stream(consumer_stream.clone()).build().unwrap();

    thread::scope(|scope| {
        scope
            .spawn(move || {
                let lease = export
                    .try_activate_read_lease(
                        ReadStartAuthority::ConsumerStream { stream: cuda_stream_handle(&consumer_stream).unwrap() },
                        CompletionAuthority::SameDeviceStream { stream: cuda_stream_handle(&consumer_stream).unwrap() },
                    )
                    .unwrap();

                let mut completion = lease.complete().unwrap();
                completion.synchronize().unwrap();
                assert!(completion.is_ready().unwrap());
            })
            .join()
            .unwrap();
    });
}
