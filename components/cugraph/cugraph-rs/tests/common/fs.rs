#![allow(dead_code)]

use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_TEMP_FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

pub fn unique_temp_dir(prefix: &str) -> PathBuf {
    let id = NEXT_TEMP_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("cugraph_{prefix}_{}_{}", std::process::id(), id))
}
