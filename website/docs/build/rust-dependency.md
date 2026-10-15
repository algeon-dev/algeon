---
sidebar_position: 6
sidebar_label: Application Build
description: Build the native libraries an external Rust application needs from the matching Algeon release, then configure its build and runtime loader environment.
---

# Building for an Application

An application takes `algeon-datafusion` from the registry and builds the
native RAPIDS libraries and Algeon shims separately, from the Algeon source
tag that matches the crate version. This page covers that native side: which
install prefixes each Cargo feature needs, how to build them from the matching
tag, and how the application finds them at build time and at run time. The
application code, from the backend to execution modes, is in
[DataFusion Session](../integration/datafusion-session.mdx).

The commands describe the `0.1.0` release with native libraries from the
`v0.1.0` tag and require that tag and the release packages to be available.

## 1. Select the release and configure the native build

Install the [prerequisites](./prerequisites.md), including a compatible NVIDIA
driver, CUDA toolkit, native host compiler, CMake, Ninja and Rust 1.95 or newer.
A registry application that enables `cugraph` also compiles a C++20 shim against
the installed headers. See
[Registry application build environment](./prerequisites.md#registry-application-build-environment).

Clone the matching native source release into a directory separate from the
application:

```bash
git clone --branch v0.1.0 --recurse-submodules \
  https://github.com/algeon-dev/algeon.git algeon-native-v0.1.0
cd algeon-native-v0.1.0
git submodule status --recursive
```

For an existing checkout, select the tag and update its pinned submodules
before building:

```bash
cd /absolute/path/to/algeon-native-v0.1.0
git checkout v0.1.0
git submodule sync --recursive
git submodule update --init --recursive
```

Run the [build shell block](./from-source.md#2-configure-the-build-shell) from
that checkout. It sets the absolute `ALGEON_ROOT`, CUDA paths and host compiler
paths, and clears native overrides from other checkouts. Choose one
`CMAKE_CUDA_ARCHITECTURES` value for all native layers before building. `NATIVE`
targets the build host's GPU. For deployment to another GPU model, use an
[explicit architecture list](./from-source.md#choosing-cuda-architectures).

## 2. Build the required native libraries

The adapter has no default features. Every configuration needs cuDF and its
shared RMM install. Build the required layers in the order shown:

| Application features | Required directories under `ALGEON_NATIVE_DIR` | Native build order |
| --- | --- | --- |
| none | `cudf-install` | cuDF/RMM |
| `cuvs` | `cudf-install`, `cuvs-install` | cuDF/RMM → cuVS |
| `cugraph` | `cudf-install`, `cuvs-install`, `cugraph-install` | cuDF/RMM → cuVS → cuGraph |
| `cuml` | `cudf-install`, `cuvs-install`, `cuml-install` | cuDF/RMM → cuVS → cuML |
| `cugraph,cuvs,cuml` | all four install directories | cuDF/RMM → cuVS → cuGraph → cuML |

From the native checkout, build cuDF/RMM:

```bash
bash scripts/build/build_libcudf.sh
```

For any of `cuvs`, `cugraph` or `cuml`, build cuVS next:

```bash
bash scripts/build/build_libcuvs.sh
```

For `cugraph`, then build cuGraph:

```bash
bash scripts/build/build_libcugraph.sh
```

For `cuml`, build cuML after cuVS:

```bash
bash scripts/build/build_libcuml.sh
```

The scripts install libraries, headers and build metadata under
`$ALGEON_ROOT/target/native`. The cuDF, cuGraph and cuML installations carry
Algeon crate-version markers. Their Rust crates reject missing or incompatible
markers. Copy complete prefixes when staging artifacts, including `include/`,
CMake exports and `share/` metadata. All layers use the cuDF prefix's shared RMM.
The [source-build reference](./from-source.md#3-build-the-native-stack) lists
installation checks for each component.

`server` adds the embedded Flight SQL API and `nvml` adds optional live device
diagnostics. Neither adds a RAPIDS install directory. Algorithm features select
their own native directories as listed above. Select application features
explicitly. `--all-features` on Algeon's implementation crates can include
developer-only `test-hooks`.

### NVML diagnostics

`nvml` is off by default for the adapter, the `algeon-server` package and the
shipped server image. Selecting it links `nvml-wrapper`, which loads the
NVIDIA driver's `libnvidia-ml.so.1` at run time. It adds these values to
`device_telemetry_snapshots()` and the backend startup records:

| Value | Without `nvml` | `nvml` built, NVML unavailable |
| --- | --- | --- |
| `gpu_utilization_pct`, `memory_utilization_pct` | `None` | `None` |
| NVML framebuffer total, reserved, used and free bytes | `None` | `None` |
| Startup `model_name`, `uuid`, `physical_total_bytes` | `None` | `None` |
| `nvml_status` | `FeatureDisabled` | `Unavailable` |

A missing value stays `None`, or `null` in JSON. It is never reported as zero.
`cuda_status` and `cuda_free_bytes` come from CUDA and are sampled whether or
not `nvml` is selected. NVML initialization or sampling failures leave query
execution unchanged. Query allocation and pool statistics, admission and the
device ledger use CUDA and RMM data only; see
[Memory and query lifetime](../design/admission.mdx#device-observations).

## 3. Create an external application

Save the native root as an absolute path, then create the application outside
that checkout. The following commands copy the same complete example used by
Algeon's release consumer check:

```bash
export ALGEON_NATIVE_DIR="$ALGEON_ROOT/target/native"
export ALGEON_APP_ROOT="$(mktemp -d /tmp/algeon-registry-app.XXXXXX)"
mkdir -p "$ALGEON_APP_ROOT/src"
cp "$ALGEON_ROOT/scripts/ci/fixtures/registry-consumer/Cargo.toml" \
  "$ALGEON_APP_ROOT/Cargo.toml"
cp "$ALGEON_ROOT/scripts/ci/fixtures/registry-consumer/src/main.rs" \
  "$ALGEON_APP_ROOT/src/main.rs"
cd "$ALGEON_APP_ROOT"
```

The standalone manifest below uses registry dependencies and forwards optional
features to the adapter. Its `datafusion = "55.1"` dependency accepts compatible
55.x updates; `algeon-datafusion = "=0.1.0"` pins the Algeon release. Cargo
creates an application lockfile on the first build. Keep that lockfile for
subsequent `--locked` builds.

{/* registry-consumer:Cargo.toml */}
```toml
[package]
name = "registry-consumer"
version = "0.0.0"
edition = "2024"
rust-version = "1.95"
publish = false

[workspace]

[features]
default = []
cuvs = ["algeon-datafusion/cuvs"]
cugraph = ["algeon-datafusion/cugraph"]
cuml = ["algeon-datafusion/cuml"]
server = ["algeon-datafusion/server"]
nvml = ["algeon-datafusion/nvml"]

[dependencies]
algeon-datafusion = "=0.1.0"
datafusion = "55.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
futures = "0.3"
```

The copied `src/main.rs` is the release consumer check's application. It
writes a small Parquet input in a temporary directory, installs one backend
into a caller-owned session, executes a relational aggregate with
`NativeRequired` and checks the result values and the attempt's completion
report. Optional features add small cuVS, cuGraph and cuML queries, and the
application closes the backend and removes its input directory afterwards,
including on query errors. [DataFusion Session](../integration/datafusion-session.mdx)
explains the same steps one at a time.

<details>
<summary>The complete <code>src/main.rs</code></summary>

{/* registry-consumer:src/main.rs */}
```rust
use std::{
    env,
    error::Error,
    fs::{self, File},
    io,
    path::Path,
    process,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "cugraph")]
use algeon_datafusion::cugraph_sql::CugraphSqlConfig;
#[cfg(any(feature = "cuvs", feature = "cugraph", feature = "cuml"))]
use algeon_datafusion::session::AlgeonSessionStateBuilderExt;
use algeon_datafusion::{
    attempt::{AttemptDisposition, AttemptOptions},
    backend::{AlgeonGpuBackend, AlgeonGpuDeviceConfig, AlgeonGpuDeviceProfile, GpuMemoryOwnership},
    planner::{AlgeonGpuExecutionMode, FinalPlanDisposition, classify_final_plan},
};
#[cfg(any(feature = "cuvs", feature = "cuml"))]
use datafusion::arrow::array::Float32Array;
use datafusion::{
    arrow::{
        array::{Array, ArrayRef, Int64Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    execution::SessionStateBuilder,
    parquet::arrow::ArrowWriter,
    prelude::{ParquetReadOptions, SessionConfig, SessionContext},
};
use futures::TryStreamExt;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

#[tokio::main]
async fn main() -> Result<()> {
    let device = AlgeonGpuDeviceConfig::new(AlgeonGpuDeviceProfile::new(0), GpuMemoryOwnership::WholeDeviceExclusive);
    let backend = AlgeonGpuBackend::builder()
        .with_execution_mode(AlgeonGpuExecutionMode::NativeRequired)
        .devices([device])
        .build()?;

    let result = run(&backend).await;
    let close = backend.close_and_wait_async(Duration::from_secs(30)).await;
    result?;
    let close = close?;
    require(close.gpu_resource_quiescent && close.observation_quiescent, "backend cleanup did not complete")?;
    println!("registry consumer passed");
    Ok(())
}

async fn run(backend: &AlgeonGpuBackend) -> Result<()> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let directory = env::temp_dir().join(format!("algeon-registry-consumer-{}-{nonce}", process::id()));
    fs::create_dir(&directory)?;
    let result = run_session(backend, &directory).await;
    let cleanup = fs::remove_dir_all(&directory);
    result?;
    cleanup?;
    Ok(())
}

async fn run_session(backend: &AlgeonGpuBackend, directory: &Path) -> Result<()> {
    let builder = backend.install_on(
        SessionStateBuilder::new_with_default_features().with_config(SessionConfig::new().with_target_partitions(1)),
    )?;
    #[cfg(feature = "cuvs")]
    let builder = builder.try_with_cuvs_sql()?;
    #[cfg(feature = "cugraph")]
    let builder = builder.try_with_cugraph_sql(CugraphSqlConfig::default())?;
    #[cfg(feature = "cuml")]
    let builder = builder.try_with_cuml_sql()?;
    let ctx = SessionContext::new_with_state(builder.build());

    let orders = directory.join("orders.parquet");
    write_parquet(
        &orders,
        vec![Field::new("amount", DataType::Int64, false)],
        vec![Arc::new(Int64Array::from(vec![10, 20, 30, 40]))],
    )?;
    register_parquet(&ctx, "orders", &orders).await?;
    check_relational(backend, &ctx).await?;

    #[cfg(any(feature = "cuvs", feature = "cuml"))]
    {
        let vectors = directory.join("vectors.parquet");
        let x = (0..8).map(|i| i as f32 - 3.0).collect::<Vec<_>>();
        write_parquet(
            &vectors,
            vec![
                Field::new("id", DataType::Int64, false),
                Field::new("d0", DataType::Float32, false),
                Field::new("d1", DataType::Float32, false),
                Field::new("d2", DataType::Float32, false),
            ],
            vec![
                Arc::new(Int64Array::from_iter_values(0..8)),
                Arc::new(Float32Array::from(x.clone())),
                Arc::new(Float32Array::from(x.iter().map(|v| 2.0 * v).collect::<Vec<_>>())),
                Arc::new(Float32Array::from(x.iter().map(|v| -v).collect::<Vec<_>>())),
            ],
        )?;
        register_parquet(&ctx, "vectors", &vectors).await?;
    }

    #[cfg(feature = "cuvs")]
    check_cuvs(backend, &ctx).await?;
    #[cfg(feature = "cugraph")]
    check_cugraph(backend, &ctx, directory).await?;
    #[cfg(feature = "cuml")]
    check_cuml(backend, &ctx).await?;
    #[cfg(feature = "server")]
    check_server(backend, &orders).await?;
    check_telemetry(backend)?;
    Ok(())
}

async fn execute_native(backend: &AlgeonGpuBackend, ctx: &SessionContext, sql: &str) -> Result<Vec<RecordBatch>> {
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    require(classify_final_plan(plan.as_ref()) == FinalPlanDisposition::Native, "query plan contains CPU execution")?;

    let attempt = backend.begin_attempt(AttemptOptions::default())?;
    let completion = attempt.completion();
    let result = async { attempt.sql(ctx, sql).await?.try_collect::<Vec<_>>().await }.await;
    let report = completion.await;
    let batches = result?;
    require(
        report.disposition == AttemptDisposition::Native && report.admission.is_some(),
        "query did not execute with native GPU admission",
    )?;
    println!("native query passed: {sql}");
    Ok(batches)
}

async fn check_relational(backend: &AlgeonGpuBackend, ctx: &SessionContext) -> Result<()> {
    let batches = execute_native(backend, ctx, "SELECT SUM(amount) AS total FROM orders WHERE amount >= 20").await?;
    let totals = int64_values(&batches, 0)?;
    require(totals == vec![90], "relational SUM result differs from 90")
}

#[cfg(feature = "cuvs")]
async fn check_cuvs(backend: &AlgeonGpuBackend, ctx: &SessionContext) -> Result<()> {
    let batches = execute_native(
        backend,
        ctx,
        "SELECT query_id, neighbor_id, distance FROM cuvs_brute_force_knn(\
         dataset => (SELECT id, d0, d1, d2 FROM vectors), \
         queries => (SELECT id, d0, d1, d2 FROM vectors WHERE id = 7), \
         k => 1, metric => 'l2_expanded')",
    )
    .await?;
    require(
        int64_values(&batches, 0)? == vec![7] && int64_values(&batches, 1)? == vec![7],
        "cuVS nearest neighbor differs from id 7",
    )?;
    let distances = float32_values(&batches, 2)?;
    require(distances.len() == 1 && distances[0].abs() < 1e-5, "cuVS self distance differs from zero")
}

#[cfg(feature = "cugraph")]
async fn check_cugraph(backend: &AlgeonGpuBackend, ctx: &SessionContext, directory: &Path) -> Result<()> {
    let edges = directory.join("edges.parquet");
    write_parquet(
        &edges,
        vec![Field::new("src", DataType::Int64, false), Field::new("dst", DataType::Int64, false)],
        vec![Arc::new(Int64Array::from(vec![0, 1, 2])), Arc::new(Int64Array::from(vec![1, 2, 3]))],
    )?;
    register_parquet(ctx, "edges", &edges).await?;
    let batches = execute_native(
        backend,
        ctx,
        "SELECT vertex, distance FROM cugraph_bfs(edges => (SELECT src, dst FROM edges), source_vertex => 0)",
    )
    .await?;
    let mut rows = int64_values(&batches, 0)?.into_iter().zip(int64_values(&batches, 1)?).collect::<Vec<_>>();
    rows.sort_unstable();
    require(rows == vec![(0, 0), (1, 1), (2, 2), (3, 3)], "cuGraph BFS distances differ from the path graph")
}

#[cfg(feature = "cuml")]
async fn check_cuml(backend: &AlgeonGpuBackend, ctx: &SessionContext) -> Result<()> {
    let batches = execute_native(
        backend,
        ctx,
        "SELECT id, pc_0, pc_1 FROM cuml_tsvd(input => (SELECT id, d0, d1, d2 FROM vectors), n_components => 2)",
    )
    .await?;
    let ids = int64_values(&batches, 0)?;
    let scores = float32_values(&batches, 1)?;
    let second = float32_values(&batches, 2)?;
    let mut seen = ids.clone();
    seen.sort_unstable();
    require(seen == (0..8).collect::<Vec<_>>(), "cuML TSVD did not retain all row ids")?;
    for ((id, score), residual) in ids.into_iter().zip(scores).zip(second) {
        let expected = (id as f32 - 3.0).abs() * 6.0_f32.sqrt();
        require(
            (score.abs() - expected).abs() < 1e-2 && residual.abs() < 1e-2,
            "cuML TSVD differs from the rank-one matrix oracle",
        )?;
    }
    Ok(())
}

#[cfg(feature = "server")]
async fn check_server(backend: &AlgeonGpuBackend, orders: &Path) -> Result<()> {
    use std::iter;

    use algeon_datafusion::server::{ServerConfig, build_session_context_with_backend};

    let config = ServerConfig::from_vars(iter::empty::<(String, String)>())?;
    let ctx = build_session_context_with_backend(&config, backend).await?;
    register_parquet(&ctx, "orders", orders).await?;
    check_relational(backend, &ctx).await?;
    println!("server session initialization passed");
    Ok(())
}

fn check_telemetry(backend: &AlgeonGpuBackend) -> Result<()> {
    use algeon_datafusion::observability::DeviceTelemetryStatus;

    let snapshots = backend.device_telemetry_snapshots()?;
    require(snapshots.len() == 1 && snapshots[0].device_ordinal == 0, "telemetry did not sample CUDA device 0")?;
    let snapshot = &snapshots[0];
    require(snapshot.cuda_status == DeviceTelemetryStatus::Available, "CUDA telemetry is unavailable")?;
    if cfg!(feature = "nvml") {
        require(snapshot.nvml_status != DeviceTelemetryStatus::FeatureDisabled, "NVML feature was not forwarded")?;
    } else {
        require(snapshot.nvml_status == DeviceTelemetryStatus::FeatureDisabled, "NVML is enabled without the feature")?;
        require(snapshot.nvml_used_bytes.is_none(), "NVML memory was reported without the feature")?;
    }
    // NVML availability is optional, independently of CUDA execution support.
    println!("nvml initialization status: {}", snapshot.nvml_status.as_str());
    Ok(())
}

fn write_parquet(path: &Path, fields: Vec<Field>, columns: Vec<ArrayRef>) -> Result<()> {
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
    let mut writer = ArrowWriter::try_new(File::create(path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

async fn register_parquet(ctx: &SessionContext, name: &str, path: &Path) -> Result<()> {
    let path = path.to_str().ok_or_else(|| io::Error::other("Parquet path is not UTF-8"))?;
    ctx.register_parquet(name, path, ParquetReadOptions::default()).await?;
    Ok(())
}

fn int64_values(batches: &[RecordBatch], column: usize) -> Result<Vec<i64>> {
    let mut values = Vec::new();
    for batch in batches {
        let array = batch
            .column(column)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| io::Error::other("expected non-null Int64 result"))?;
        require(array.null_count() == 0, "query returned a null Int64 result")?;
        values.extend(array.values().iter().copied());
    }
    Ok(values)
}

#[cfg(any(feature = "cuvs", feature = "cuml"))]
fn float32_values(batches: &[RecordBatch], column: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    for batch in batches {
        let array = batch
            .column(column)
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| io::Error::other("expected non-null Float32 result"))?;
        require(array.null_count() == 0, "query returned a null Float32 result")?;
        values.extend(array.values().iter().copied());
    }
    Ok(values)
}

fn require(condition: bool, message: &str) -> Result<()> {
    if condition { Ok(()) } else { Err(io::Error::other(message).into()) }
}
```

</details>

The manifest and source are the canonical
[release consumer fixture](https://github.com/algeon-dev/algeon/tree/main/scripts/ci/fixtures/registry-consumer).
No path dependency or repository `.cargo/config.toml` is required by the
application.

## 4. Build and run the application

Keep `ALGEON_NATIVE_DIR`, the CUDA selectors and the compiler settings from the
native build shell in the application shell. `ALGEON_NATIVE_DIR` supplies build
discovery; `LD_LIBRARY_PATH` supplies runtime discovery. A dependency's
build-script rpath arguments do not propagate to the final application binary.
The application may supply its own RUNPATH as a deployment choice. The commands
here use an explicit loader environment.

For the relational configuration, export the cuDF/RMM and CUDA directories,
build, and directly execute the binary from another working directory:

```bash
export LD_LIBRARY_PATH="$ALGEON_NATIVE_DIR/cudf-install/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo build --release
export ALGEON_APP_BINARY="$ALGEON_APP_ROOT/target/release/registry-consumer"
cd /tmp
"$ALGEON_APP_BINARY"
```

A passing run prints `registry consumer passed` after the relational result is
checked against `90` and backend shutdown completes.

`CUDA_HOME/lib64` is the loader directory for the toolkit layout used by the
build shell. Substitute the installed CUDA runtime directory when the toolkit
uses another layout. The driver runtime must also be discoverable by the Linux
loader.

For optional features, return to the application directory and add the matching
library directories before executing it. For example, after building all four
native layers, build and run the complete algorithm configuration:

```bash
cd "$ALGEON_APP_ROOT"
export LD_LIBRARY_PATH="$ALGEON_NATIVE_DIR/cugraph-install/lib:$ALGEON_NATIVE_DIR/cuml-install/lib:$ALGEON_NATIVE_DIR/cudf-install/lib:$ALGEON_NATIVE_DIR/cuvs-install/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cargo build --release --locked --features cugraph,cuvs,cuml
cd /tmp
"$ALGEON_APP_BINARY"
```

For a smaller feature selection, include `cuvs-install/lib` for `cuvs`,
`cugraph` or `cuml`, then include `cugraph-install/lib` for `cugraph` and
`cuml-install/lib` for `cuml`. cuDF/RMM and CUDA remain required. Use the same
absolute paths in a service manager or deployment environment.

Inspect required library resolution with the GNU loader on x86-64 Linux:

```bash
/lib64/ld-linux-x86-64.so.2 --list "$ALGEON_APP_BINARY"
```

The command should succeed and resolve Algeon and RAPIDS libraries to the
selected install prefixes. ELF `DT_AUXILIARY` filters are
[optional](https://www.sourceware.org/binutils/docs/ld/Options.html#index-g_t_002df).
`ldd` can list a missing auxiliary filter even when required libraries resolve.
Direct binary execution confirms the query's result assertions on the available
GPU.

Return to [DataFusion Session](../integration/datafusion-session.mdx) for the
application code: the backend, SQL function families, execution modes and
attempt observation.
