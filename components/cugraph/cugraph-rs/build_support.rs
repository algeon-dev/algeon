use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompilerFamily {
    GnuLike,
    Msvc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShimCompileLayout {
    pub(crate) source_file: PathBuf,
    pub(crate) local_include_dir: PathBuf,
    pub(crate) native_include_dirs: Vec<PathBuf>,
}

pub(crate) fn resolve_shim_compile_layout(
    manifest_dir: &Path,
    native_include_dirs: Vec<PathBuf>,
) -> Result<ShimCompileLayout, String> {
    if native_include_dirs.is_empty() {
        return Err("cugraph-rs/build.rs did not receive DEP_CUGRAPH_NATIVE_INCLUDE_DIRS from cugraph-sys".to_string());
    }

    Ok(ShimCompileLayout {
        source_file: manifest_dir.join("shim").join("runtime_resource_handle.cpp"),
        local_include_dir: manifest_dir.join("shim"),
        native_include_dirs,
    })
}

pub(crate) fn system_include_args(family: CompilerFamily, dir: &Path) -> Vec<OsString> {
    match family {
        CompilerFamily::GnuLike => vec![OsString::from("-isystem"), dir.as_os_str().to_owned()],
        CompilerFamily::Msvc => vec![OsString::from(format!("/external:I{}", dir.display()))],
    }
}

pub(crate) fn external_warning_control_args(family: CompilerFamily) -> Vec<OsString> {
    match family {
        CompilerFamily::GnuLike => Vec::new(),
        CompilerFamily::Msvc => vec![OsString::from("/external:W0")],
    }
}
