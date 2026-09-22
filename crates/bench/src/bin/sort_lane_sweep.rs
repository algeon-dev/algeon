//! Sort-lane sweep: width comparison and mixed-load service check for
//! lane-parallel top-k. See `bench::sort_lane_sweep`. GPU runs must be
//! wrapped in `flock /tmp/cudf-gpu.lock`.

use bench::sort_lane_sweep::run;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    match run().await {
        Ok(()) => Ok(()),
        Err(error) => {
            eprintln!("sort_lane_sweep: {error}");
            Err(error)
        },
    }
}
