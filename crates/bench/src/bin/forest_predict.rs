use std::error::Error;

use bench::forest_predict::{CPU_THREADS, run};
use tokio::runtime::Builder;

fn main() -> Result<(), Box<dyn Error>> {
    Builder::new_multi_thread().worker_threads(CPU_THREADS).enable_all().build()?.block_on(run())
}
