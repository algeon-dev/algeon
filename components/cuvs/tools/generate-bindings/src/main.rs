use std::{env, path::PathBuf, process::Command};

use anyhow::{Context, Result, bail, ensure};

const PINNED_CUVS_REVISION: &str = "2bd7cd71d31e39eb9ee00c2a250085a1afb84977";

fn main() -> Result<()> {
    let output = env::args_os().nth(1).map(PathBuf::from).context("usage: cuvs-nexus-bindings-generator OUTPUT")?;
    ensure!(env::args_os().nth(2).is_none(), "usage: cuvs-nexus-bindings-generator OUTPUT");

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|tools| tools.parent())
        .context("bindings generator must remain under tools/generate-bindings")?;
    let sys_dir = workspace_root.join("cuvs-nexus-sys");
    let source_dir = env::var_os("CUVS_SOURCE_DIR").map(PathBuf::from).unwrap_or_else(|| workspace_root.join("cuvs"));
    let include_dir = source_dir.join("c/include");
    let header = include_dir.join("cuvs/core/all.h");
    ensure!(
        header.is_file(),
        "pinned cuVS headers are missing at '{}'; run `git submodule update --init --recursive cuvs`",
        header.display()
    );

    let revision = Command::new("git")
        .args(["-C"])
        .arg(&source_dir)
        .args(["rev-parse", "HEAD"])
        .output()
        .with_context(|| format!("failed to inspect cuVS source revision under {}", source_dir.display()))?;
    if !revision.status.success() {
        bail!("git rev-parse failed for {}", source_dir.display());
    }
    let revision = String::from_utf8(revision.stdout).context("cuVS source revision was not UTF-8")?;
    ensure!(
        revision.trim().eq_ignore_ascii_case(PINNED_CUVS_REVISION),
        "cuVS source revision mismatch: expected {PINNED_CUVS_REVISION}, found {}",
        revision.trim()
    );

    let dlpack_include = env::var_os("DLPACK_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| source_dir.join("cpp/build/_deps/dlpack-src/include"));
    ensure!(
        dlpack_include.join("dlpack/dlpack.h").is_file(),
        "DLPack headers are missing under '{}'; build the canonical cuVS prefix or set DLPACK_INCLUDE_DIR",
        dlpack_include.display()
    );

    let mut builder = bindgen::Builder::default()
        .header(sys_dir.join("cuvs_c_wrapper.h").to_string_lossy())
        .must_use_type("cuvsError_t")
        .allowlist_function("cuvs.*")
        .allowlist_type("(cuvs|DL).*")
        .rustified_enum("(cuvs|DL).*")
        .blocklist_item("cuda.*")
        .blocklist_item("CUstream_st")
        .raw_line("use crate::{cudaDataType_t, cudaStream_t};")
        .rustfmt_configuration_file(Some(manifest_dir.join("rustfmt.toml")))
        .clang_arg(format!("-I{}", sys_dir.join("bindgen-stubs").display()))
        .clang_arg(format!("-I{}", include_dir.display()))
        .clang_arg(format!("-I{}", dlpack_include.display()));

    if let Some(cuda_include) = env::var_os("CUDA_HOME").map(PathBuf::from).map(|path| path.join("include"))
        && cuda_include.is_dir()
    {
        builder = builder.clang_arg(format!("-I{}", cuda_include.display()));
    }

    builder
        .generate()
        .context("bindgen failed to generate cuVS bindings from the pinned headers")?
        .write_to_file(&output)
        .with_context(|| format!("failed to write generated bindings to {}", output.display()))?;
    Ok(())
}
