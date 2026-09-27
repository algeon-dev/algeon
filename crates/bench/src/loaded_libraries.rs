use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    os::unix::fs::MetadataExt,
    path::{Path, PathBuf},
};

use datafusion::common::{DataFusionError, Result as DataFusionResult};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Serialize)]
struct LoadedLibrary {
    path: PathBuf,
    mapped_path: PathBuf,
    device: u64,
    mapped_device: String,
    inode: u64,
    size_bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
struct LoadedBinary {
    path: PathBuf,
    device: u64,
    inode: u64,
    size_bytes: u64,
    sha256: String,
}

fn sha256_file(path: &Path) -> DataFusionResult<String> {
    let mut file = fs::File::open(path).map_err(|error| DataFusionError::External(Box::new(error)))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 256 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| DataFusionError::External(Box::new(error)))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest.finalize().iter().map(|byte| format!("{byte:02x}")).collect())
}

pub(crate) fn write_loaded_libraries(dir: &Path) -> DataFusionResult<()> {
    let executable = Path::new("/proc/self/exe");
    let binary_path = fs::read_link(executable).map_err(|error| DataFusionError::External(Box::new(error)))?;
    let binary_metadata = fs::metadata(executable).map_err(|error| DataFusionError::External(Box::new(error)))?;
    let binary = LoadedBinary {
        path: binary_path,
        device: binary_metadata.dev(),
        inode: binary_metadata.ino(),
        size_bytes: binary_metadata.len(),
        sha256: sha256_file(executable)?,
    };
    let maps = fs::read_to_string("/proc/self/maps").map_err(|error| DataFusionError::External(Box::new(error)))?;
    let mut mapped = BTreeMap::<PathBuf, (String, u64)>::new();
    for line in maps.lines() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 6 || !fields[1].contains('x') || !fields[5].starts_with('/') {
            continue;
        }
        let mapped_path = PathBuf::from(fields[5..].join(" "));
        if !mapped_path.file_name().is_some_and(|name| name.to_string_lossy().contains(".so")) {
            continue;
        }
        let inode = fields[4].parse::<u64>().map_err(|error| DataFusionError::External(Box::new(error)))?;
        let identity = (fields[3].to_string(), inode);
        if let Some(previous) = mapped.insert(mapped_path.clone(), identity.clone())
            && previous != identity
        {
            return Err(DataFusionError::Execution(format!(
                "loaded library changed mapping identity during benchmark: {}",
                mapped_path.display()
            )));
        }
    }
    let mut libraries = Vec::with_capacity(mapped.len());
    for (mapped_path, (mapped_device, inode)) in mapped {
        let path = mapped_path.canonicalize().map_err(|error| DataFusionError::External(Box::new(error)))?;
        let metadata = fs::metadata(&path).map_err(|error| DataFusionError::External(Box::new(error)))?;
        if metadata.ino() != inode {
            return Err(DataFusionError::Execution(format!(
                "loaded library path no longer resolves to mapped inode: {}",
                mapped_path.display()
            )));
        }
        let sha256 = sha256_file(&path)?;
        libraries.push(LoadedLibrary {
            path,
            mapped_path,
            device: metadata.dev(),
            mapped_device,
            inode,
            size_bytes: metadata.len(),
            sha256,
        });
    }
    for required in ["libcudf_rust.so", "libcudf.so"] {
        if !libraries
            .iter()
            .any(|library| library.path.file_name().is_some_and(|name| name.to_string_lossy().starts_with(required)))
        {
            return Err(DataFusionError::Execution(format!(
                "benchmark did not map required native library {required}"
            )));
        }
    }
    let record = serde_json::json!({"source": "/proc/self/maps", "binary": binary, "libraries": libraries});
    let bytes = serde_json::to_vec_pretty(&record).map_err(|error| DataFusionError::External(Box::new(error)))?;
    fs::write(dir.join("loaded-native-libraries.json"), bytes)
        .map_err(|error| DataFusionError::External(Box::new(error)))
}
