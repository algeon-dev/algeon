use std::sync::Mutex;

use cudf::{Column, DataFrame};

/// Guards test-only mutations of the process-global current device resource.
///
/// Some allocation-accounting tests temporarily swap the active RMM resource
/// via cudf-sys test hooks. Those hooks mutate global process state, so tests
/// across different modules must share one lock when installing a stats
/// resource.
#[allow(dead_code)]
pub static CURRENT_DEVICE_RESOURCE_LOCK: Mutex<()> = Mutex::new(());

#[allow(dead_code)]
pub fn sample_df() -> DataFrame {
    let ids = Column::from_slice(&[3i32, 1, 4, 1, 5]).unwrap();
    let scores = Column::from_slice(&[30.0f64, 10.0, 40.0, 15.0, 50.0]).unwrap();
    DataFrame::new(vec!["id".to_string(), "score".to_string()], vec![ids, scores]).unwrap()
}
