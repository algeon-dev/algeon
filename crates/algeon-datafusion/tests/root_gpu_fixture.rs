use std::{
    sync::{Arc, OnceLock},
    thread,
    time::Duration,
};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

static SERIAL: OnceLock<Arc<Semaphore>> = OnceLock::new();

fn semaphore() -> Arc<Semaphore> {
    Arc::clone(SERIAL.get_or_init(|| Arc::new(Semaphore::new(1))))
}

pub(crate) fn lock() -> OwnedSemaphorePermit {
    loop {
        match semaphore().try_acquire_owned() {
            Ok(permit) => return permit,
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
}
