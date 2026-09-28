//! Planning-time model file resolution for `cuml_forest_predict`.

#[cfg(feature = "cuml")]
use std::os::fd::OwnedFd;
use std::{path::PathBuf, sync::Arc};

#[cfg(feature = "cuml")]
use cuml::{ForestModelFormat, ForestOutput, ForestOutputKind, ForestPrecision};
use query_engine::plan::{MlForestModelFormat, MlForestModelSnapshot};
#[cfg(feature = "cuml")]
use query_engine::{
    native_error::cuml_query_error,
    plan::{MlForestModelFacts, MlForestOutputContract, MlForestPrecision},
};

#[cfg(feature = "cuml")]
use super::error::invalid_option;
use super::{
    CumlModelStoreConfig, CumlOperation, CumlValidatedOptions,
    error::{feature_disabled, invalid_call},
};
use crate::error::{Error, ErrorCode, Result};

/// Output the SQL `output` option requests from a forest model.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ForestOutputRequest {
    Scores,
    ClassIndex,
}

/// Validated `model`, `model_format`, and `output` options of one call.
#[derive(Debug)]
#[cfg_attr(not(feature = "cuml"), expect(dead_code, reason = "only the cuml model loader reads the reference"))]
pub(crate) struct ForestModelReference {
    pub(crate) path: String,
    pub(crate) format: MlForestModelFormat,
    pub(crate) output: ForestOutputRequest,
}

/// Models root that `cuml_forest_predict` reads model files from.
///
/// Every file reachable beneath the root, including hard links and bind
/// mounts placed there, is trusted as a model source, so the root must not be
/// writable by untrusted users. Symlinks must be relative and stay beneath
/// the root. Each planning pass (`GetFlightInfo`, `DoGet`, and prepared
/// statement creation) reads and parses the named file synchronously, so
/// planning latency grows with model size.
#[derive(Debug)]
pub(crate) struct CumlModelStore {
    /// Canonical absolute directory at install time.
    root: PathBuf,
    max_model_bytes: u64,
    /// `O_PATH` descriptor of `root` opened at install. Model paths resolve
    /// beneath this directory inode, so renaming the root or swapping a
    /// symlink among its ancestors after install cannot move the anchor.
    #[cfg(feature = "cuml")]
    root_fd: OwnedFd,
}

/// Installations compare by configuration; the descriptor identity is not
/// part of it.
impl PartialEq for CumlModelStore {
    fn eq(&self, other: &Self) -> bool {
        (&self.root, self.max_model_bytes) == (&other.root, other.max_model_bytes)
    }
}

impl Eq for CumlModelStore {}

impl CumlModelStore {
    pub(crate) fn try_new(config: CumlModelStoreConfig) -> Result<Self> {
        let invalid = |message: &str| {
            Error::new(ErrorCode::InvalidRequest, "algeon_datafusion.cuml_sql.install", message)
                .with_log_safe_fact("reason", "invalid_model_store")
        };
        if config.max_model_bytes == 0 {
            return Err(invalid("cuML max_model_bytes must be positive"));
        }
        let root = config
            .root
            .canonicalize()
            .map_err(|error| invalid("cuML models root cannot be resolved").with_source(error))?;
        if !root.is_dir() {
            return Err(invalid("cuML models root is not a directory"));
        }
        Ok(Self {
            #[cfg(feature = "cuml")]
            root_fd: open::open_root(&root)
                .map_err(|error| invalid("cuML models root cannot be opened").with_source(error))?,
            root,
            max_model_bytes: config.max_model_bytes,
        })
    }
}

/// Binds the model a validated call names, or `None` for operations without
/// one. The feature and root checks run before any file access.
pub(super) fn bind_operation_model(
    operation: CumlOperation,
    store: Option<&CumlModelStore>,
    options: &CumlValidatedOptions,
) -> Result<Option<Arc<MlForestModelSnapshot>>> {
    let Some(reference) = options.forest_model_reference() else {
        return Ok(None);
    };
    if !cfg!(feature = "cuml") {
        return Err(feature_disabled(operation));
    }
    let Some(store) = store else {
        return Err(invalid_call(
            operation,
            "model_root_unconfigured",
            "cuml_forest_predict requires a configured models root",
        ));
    };
    load_snapshot(operation, store, &reference).map(|snapshot| Some(Arc::new(snapshot)))
}

#[cfg(not(feature = "cuml"))]
fn load_snapshot(
    _operation: CumlOperation,
    _store: &CumlModelStore,
    _reference: &ForestModelReference,
) -> Result<MlForestModelSnapshot> {
    unreachable!("bind_operation_model rejects forest models without the cuml feature")
}

#[cfg(feature = "cuml")]
fn load_snapshot(
    operation: CumlOperation,
    store: &CumlModelStore,
    reference: &ForestModelReference,
) -> Result<MlForestModelSnapshot> {
    const INSPECT: &str = "algeon_datafusion.cuml_sql.inspect_forest_model";
    let bytes = open::read_model(operation, store, &reference.path)?;
    let format = match reference.format {
        MlForestModelFormat::XgboostJson => ForestModelFormat::XgboostJson,
        MlForestModelFormat::XgboostUbjson => ForestModelFormat::XgboostUbjson,
        MlForestModelFormat::XgboostLegacy => ForestModelFormat::XgboostLegacy,
        MlForestModelFormat::LightGbm => ForestModelFormat::LightGbm,
        MlForestModelFormat::TreeliteCheckpoint => ForestModelFormat::TreeliteCheckpoint,
    };
    let (info, checkpoint) = cuml::inspect_forest_model(&bytes, format).map_err(|error| {
        Error::from_native(
            INSPECT,
            "nvForest rejected the model file",
            cuml_query_error(INSPECT, "failed to inspect forest model", error),
        )
        .with_log_safe_fact("reason", "model_rejected")
        .with_log_safe_fact("model_format", reference.format.as_str())
    })?;
    let request = match reference.output {
        ForestOutputRequest::Scores => ForestOutput::Scores,
        ForestOutputRequest::ClassIndex => ForestOutput::ClassIndex,
    };
    let output = match info.output_kind(request) {
        Ok(ForestOutputKind::Score) => MlForestOutputContract::Score,
        Ok(ForestOutputKind::ClassIndex { classes }) => MlForestOutputContract::ClassIndex { classes },
        Ok(ForestOutputKind::Vector(width)) => MlForestOutputContract::Vector(width),
        Err(_) => {
            return Err(invalid_option(
                operation,
                "class_index_requires_multiple_outputs",
                Some("output"),
                "output => 'class_index' requires a model with at least two outputs",
            ));
        },
    };
    let facts = MlForestModelFacts {
        num_features: info.num_features,
        num_trees: info.num_trees,
        native_output_stride: info.native_output_stride,
        precision: match info.precision {
            ForestPrecision::Single => MlForestPrecision::Single,
            ForestPrecision::Double => MlForestPrecision::Double,
        },
    };
    MlForestModelSnapshot::new(&bytes, checkpoint, reference.format, facts, output).map_err(|error| {
        Error::from_native(INSPECT, "forest model facts are inconsistent with the requested output", error)
    })
}

#[cfg(feature = "cuml")]
mod open {
    use std::{
        ffi::CString,
        fs::{File, OpenOptions},
        io::{self, Read},
        mem,
        os::{
            fd::{AsRawFd, FromRawFd, OwnedFd},
            unix::fs::{MetadataExt, OpenOptionsExt},
        },
        path::{Component, Path},
    };

    use super::CumlModelStore;
    use crate::{
        cuml_sql::{CumlOperation, error::invalid_call},
        error::{Error, ErrorCode, Result},
    };

    const OPEN: &str = "algeon_datafusion.cuml_sql.model_store";
    /// `openat2` attempts before a lookup that keeps racing concurrent
    /// renames or mounts is rejected.
    const MAX_RESOLVE_ATTEMPTS: usize = 8;

    /// Opens the models root as an `O_PATH | O_DIRECTORY | O_CLOEXEC`
    /// descriptor; `std` adds `O_CLOEXEC`, and `O_PATH` ignores the read mode.
    pub(super) fn open_root(root: &Path) -> io::Result<OwnedFd> {
        OpenOptions::new().read(true).custom_flags(libc::O_PATH | libc::O_DIRECTORY).open(root).map(OwnedFd::from)
    }

    /// Reads a model file named relative to the models root.
    ///
    /// The path resolves with `openat2(RESOLVE_BENEATH)` from the root
    /// descriptor opened at install, so the kernel rejects `..` escapes, absolute symlinks, and
    /// relative symlinks leaving the root with `EXDEV` before it looks up
    /// anything outside the root. Relative symlinks that stay beneath the root
    /// resolve normally. The resolved file is held as an `O_PATH` descriptor,
    /// which opens no device and never blocks on a FIFO; only a regular file
    /// within `max_model_bytes` is then reopened for reading.
    pub(super) fn read_model(operation: CumlOperation, store: &CumlModelStore, relative: &str) -> Result<Vec<u8>> {
        let rejected = |rejection: &'static str, message: &str| -> Error {
            invalid_call(operation, "model_path_rejected", message)
                .with_log_safe_fact("rejection", rejection)
                .with_fact("model", relative)
        };
        if relative.is_empty() || relative.contains('\0') {
            return Err(rejected("invalid_path", "model path must be a non-empty path without NUL bytes"));
        }
        for component in Path::new(relative).components() {
            match component {
                Component::Normal(_) | Component::CurDir => {},
                Component::ParentDir => {
                    return Err(rejected("parent_component", "model path must not contain '..'"));
                },
                Component::RootDir | Component::Prefix(_) => {
                    return Err(rejected("absolute_path", "model path must be relative to the models root"));
                },
            }
        }
        let path = open_beneath(&store.root_fd, relative.as_bytes()).map_err(|error| match error.raw_os_error() {
            Some(libc::EXDEV) => rejected("outside_root", "model path resolves outside the models root"),
            Some(libc::ENOENT | libc::ENOTDIR) => {
                rejected("not_found", "model file does not exist under the models root")
            },
            Some(libc::EACCES) => rejected("permission_denied", "model file is not readable"),
            Some(libc::ELOOP) => rejected("symlink_loop", "model path has too many levels of symbolic links"),
            Some(libc::ENAMETOOLONG) => rejected("name_too_long", "model path or one of its components is too long"),
            Some(libc::EAGAIN | libc::EINTR) => rejected(
                "resolution_retry_exhausted",
                "model path lookup kept racing concurrent renames or mounts under the models root",
            ),
            // Seccomp filters report a blocked syscall as EPERM.
            Some(libc::ENOSYS | libc::EPERM) => Error::new(
                ErrorCode::Internal,
                OPEN,
                "openat2 is unavailable (ENOSYS/EPERM); model files require Linux 5.6+",
            )
            .with_log_safe_fact("reason", "openat2_unavailable")
            .with_source(error),
            _ => io_error("failed to resolve the model path", error),
        })?;
        let path = File::from(path);
        let metadata = path.metadata().map_err(|error| io_error("failed to stat the model file", error))?;
        if !metadata.is_file() {
            return Err(rejected("not_regular_file", "model path is not a regular file"));
        }
        if metadata.len() > store.max_model_bytes {
            return Err(too_large(operation, store.max_model_bytes));
        }
        // Reopening the O_PATH descriptor's magic link yields the same inode
        // without resolving the path again.
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_NOCTTY)
            .open(format!("/proc/self/fd/{}", path.as_raw_fd()))
            .map_err(|error| match error.raw_os_error() {
                Some(libc::EACCES) => rejected("permission_denied", "model file is not readable"),
                _ => io_error("failed to open the model file", error),
            })?;
        let opened = file.metadata().map_err(|error| io_error("failed to stat the model file", error))?;
        if (opened.dev(), opened.ino()) != (metadata.dev(), metadata.ino()) {
            return Err(Error::new(ErrorCode::Internal, OPEN, "reopened model file is a different inode"));
        }
        read_bounded(operation, file, store.max_model_bytes)
    }

    /// `openat2(2)` with `O_PATH | O_CLOEXEC` and
    /// `RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS`, relative to `directory`.
    fn open_beneath(directory: &OwnedFd, path: &[u8]) -> io::Result<OwnedFd> {
        let path = CString::new(path).map_err(io::Error::other)?;
        // `open_how` is non-exhaustive; the kernel requires unused bytes to be zero.
        // SAFETY: `open_how` is plain integers, so the all-zero value is valid.
        let mut how: libc::open_how = unsafe { mem::zeroed() };
        how.flags = (libc::O_PATH | libc::O_CLOEXEC) as u64;
        how.resolve = libc::RESOLVE_BENEATH | libc::RESOLVE_NO_MAGICLINKS;
        let mut error = io::Error::from_raw_os_error(libc::EAGAIN);
        for _ in 0..MAX_RESOLVE_ATTEMPTS {
            // SAFETY: `path` is NUL-terminated and `how` outlives the call with
            // its exact size passed.
            let fd = unsafe {
                libc::syscall(
                    libc::SYS_openat2,
                    directory.as_raw_fd(),
                    path.as_ptr(),
                    &raw const how,
                    mem::size_of::<libc::open_how>(),
                )
            };
            if fd >= 0 {
                // SAFETY: a successful openat2 returns a new descriptor this
                // function owns; descriptors fit in `c_int`.
                return Ok(unsafe { OwnedFd::from_raw_fd(fd as libc::c_int) });
            }
            error = io::Error::last_os_error();
            // RESOLVE_BENEATH fails with EAGAIN when a concurrent rename or
            // mount could have raced the lookup; the retry resolves afresh.
            if !matches!(error.raw_os_error(), Some(libc::EAGAIN | libc::EINTR)) {
                return Err(error);
            }
        }
        Err(error)
    }

    fn read_bounded(operation: CumlOperation, file: File, max_model_bytes: u64) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        file.take(max_model_bytes.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|error| io_error("failed to read the model file", error))?;
        // The file can grow between fstat and read.
        if bytes.len() as u64 > max_model_bytes {
            return Err(too_large(operation, max_model_bytes));
        }
        Ok(bytes)
    }

    fn too_large(operation: CumlOperation, max_model_bytes: u64) -> Error {
        invalid_call(operation, "model_too_large", "model file exceeds the configured max_model_bytes")
            .with_log_safe_fact("max_model_bytes", max_model_bytes)
    }

    fn io_error(message: &'static str, error: io::Error) -> Error {
        Error::new(ErrorCode::Internal, OPEN, message).with_source(error)
    }
}
