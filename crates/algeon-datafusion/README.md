# Algeon for DataFusion

`algeon-datafusion` runs supported [Apache DataFusion](https://datafusion.apache.org/) plans
on NVIDIA GPUs through RAPIDS. A physical optimizer rule replaces whole relational plan
candidates with cuDF execution and leaves the rest of the plan on the DataFusion CPU path.
SQL table functions expose cuGraph, cuVS and cuML algorithms, so one query can select an
algorithm's input, run it and join the result. Setup, supported SQL, configuration and
design are documented at [algeon.dev](https://algeon.dev). Algeon is experimental; APIs
and query behavior may change between releases.

## Dependencies

```toml
[dependencies]
algeon-datafusion = "0.1.1"
datafusion = "55.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
futures = "0.3"
```

| Feature | Adds | Native libraries |
| --- | --- | --- |
| none | Relational execution on cuDF | cuDF, RMM |
| `cuvs` | Vector search and clustering table functions | cuVS |
| `cugraph` | Graph table functions | cuVS, cuGraph |
| `cuml` | Machine learning and forest inference table functions | cuVS, cuML |
| `server` | Embedded Arrow Flight SQL server | none |
| `nvml` | Live device diagnostics through NVML | none |

No feature is on by default. Algeon needs Linux, an NVIDIA GPU and a CUDA toolkit, and
Cargo does not build the RAPIDS libraries. Build them from the Algeon source tag
that matches the crate version in the application's `Cargo.lock` (`v0.1.1` for `0.1.1`) and
point `ALGEON_NATIVE_DIR` and `LD_LIBRARY_PATH` at the installed prefixes.
[Building for an application](https://algeon.dev/build/rust-dependency) gives the build
order per feature and the loader setup; [Prerequisites](https://algeon.dev/build/prerequisites)
covers the driver, toolkit and host compiler.

## Example

```rust,no_run
use algeon_datafusion::{
    AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, AttemptOptions, GpuMemoryOwnership,
};
use datafusion::{execution::SessionStateBuilder, prelude::SessionContext};
use futures::TryStreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Device 0 with process-wide allocator ownership; no memory is allocated here.
    let device = AlgeonGpuDeviceConfig::new(
        AlgeonGpuDeviceProfile::new(0),
        GpuMemoryOwnership::WholeDeviceExclusive,
    );
    let backend = AlgeonGpuBackend::builder().devices([device]).build()?;
    let state = backend.install_on(SessionStateBuilder::new_with_default_features())?.build();
    let ctx = SessionContext::new_with_state(state);
    let attempt = backend.begin_attempt(AttemptOptions::default())?;
    // Taken before planning so the terminal report is observable even if planning fails.
    let completion = attempt.completion();
    let batches = attempt.sql(&ctx, "SELECT 1 AS value").await?.try_collect::<Vec<_>>().await?;
    let report = completion.await;
    println!("{} batch(es), outcome {:?}", batches.len(), report.outcome);
    backend.close_and_wait_async(std::time::Duration::from_secs(10)).await?;
    Ok(())
}
```

[DataFusion session](https://algeon.dev/integration/datafusion-session) walks through the
same steps, adds the algorithm SQL families and explains execution modes.
