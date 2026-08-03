use nexus_query_engine::runtime::{DeviceLedger, DeviceResourceProfile};

fn main() {
    let profile = DeviceResourceProfile::resolve(0, 1, 0, 0, 1, 1, 1, 2, 2).unwrap();
    let _ = DeviceLedger::new(profile);
}
