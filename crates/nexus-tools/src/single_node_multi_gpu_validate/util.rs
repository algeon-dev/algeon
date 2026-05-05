//! SHA-256 helpers and the atomic single-artifact writer.

use std::{
    fs::{self, File},
    io::{Read as _, Write as _},
    path::{Path, PathBuf},
};

use sha2::{Digest as _, Sha256};

use super::ARTIFACT_FILE_NAME;
use crate::ToolResult;

pub(super) fn sha256_file(path: &Path) -> ToolResult<String> {
    let mut file = File::open(path).map_err(|_| format!("cannot read {} for SHA-256", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let bytes_read = file.read(&mut buffer).map_err(|_| format!("cannot hash {}", path.display()))?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    Ok(hex_digest(hasher.finalize().as_ref()))
}

pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_digest(digest.as_ref())
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

pub(super) fn write_new_artifact(out_dir: &Path, contents: &[u8]) -> ToolResult<PathBuf> {
    if out_dir.exists() {
        return Err(format!(
            "--out-dir {} already exists; refuse to mix or overwrite run artifacts",
            out_dir.display()
        )
        .into());
    }
    let parent = output_parent(out_dir);
    if !parent.is_dir() {
        return Err(format!("--out-dir parent {} does not exist", parent.display()).into());
    }
    fs::create_dir(out_dir).map_err(|_| format!("cannot create --out-dir {}", out_dir.display()))?;
    let temporary = out_dir.join(format!(".{ARTIFACT_FILE_NAME}.{}.tmp", std::process::id()));
    let artifact = out_dir.join(ARTIFACT_FILE_NAME);
    let write_result = (|| -> ToolResult<()> {
        let mut file = File::options().write(true).create_new(true).open(&temporary)?;
        file.write_all(contents)?;
        file.sync_all()?;
        fs::rename(&temporary, &artifact)?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        let _ = fs::remove_dir(out_dir);
        return Err(error);
    }
    Ok(artifact)
}

pub(super) fn output_parent(out_dir: &Path) -> &Path {
    out_dir.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."))
}
