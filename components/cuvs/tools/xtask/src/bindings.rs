use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{Context, Result, bail, ensure};

use crate::source;

fn generate(workspace_root: &Path, output: &Path) -> Result<String> {
    let revision = source::verify(workspace_root)?;
    let manifest_dir = workspace_root.join("tools/xtask");
    let sys_dir = workspace_root.join("cuvs-nexus-sys");
    let source_dir = workspace_root.join("cuvs");
    let include_dir = source_dir.join("c/include");
    let header = include_dir.join("cuvs/core/all.h");
    ensure!(header.is_file(), "pinned cuVS headers are missing at {}", header.display());

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
        .write_to_file(output)
        .with_context(|| format!("failed to write generated bindings to {}", output.display()))?;
    Ok(revision)
}

fn print_diff(checked_in: &Path, generated: &Path) -> Result<()> {
    let output =
        Command::new("diff").arg("-u").arg(checked_in).arg(generated).output().context("failed to run diff")?;
    let diff = String::from_utf8_lossy(&output.stdout);
    for line in diff.lines().take(200) {
        eprintln!("{line}");
    }
    Ok(())
}

pub(crate) fn run(workspace_root: &Path, action: &str) -> Result<()> {
    ensure!(matches!(action, "generate" | "check"), "usage: cargo xtask bindings <generate|check>");
    let temporary = tempfile::Builder::new().prefix("cuvs-nexus-bindings-").tempdir()?;
    let generated = temporary.path().join("bindings.rs");
    let revision = generate(workspace_root, &generated)?;
    ensure!(generated.metadata()?.len() > 0, "bindings generator produced an empty file");
    let checked_in = workspace_root.join("cuvs-nexus-sys/src/bindings.rs");

    if action == "generate" {
        fs::copy(&generated, &checked_in)
            .with_context(|| format!("failed to update checked-in bindings at {}", checked_in.display()))?;
        println!("updated {} from pinned cuVS {}", checked_in.display(), revision);
        return Ok(());
    }
    if fs::read(&generated)? != fs::read(&checked_in)? {
        eprintln!("error: checked-in bindings are stale: {}", checked_in.display());
        print_diff(&checked_in, &generated)?;
        bail!("run `cargo xtask bindings generate`");
    }
    println!("checked-in bindings match pinned cuVS headers: {}", checked_in.display());
    Ok(())
}
