//! Bounded, std-only ELF dependency inspection for runtime-identity evidence.

use std::{
    fs,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

/// Linkage classification for an optional shared runtime dependency of the
/// native build (e.g. `librapids_logger.so`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum OptionalRuntimeDepLinkage {
    /// A shared candidate exists and at least one RAPIDS consumer links it.
    Shared(PathBuf),
    /// No shared candidate exists; the dependency is statically absorbed
    /// into its consumers (or unused).
    Absorbed,
    /// A shared candidate exists but no RAPIDS consumer links it: a stale
    /// artifact from an earlier shared build.
    StaleSharedArtifact(PathBuf),
}

/// Classify an optional shared runtime dependency by inspecting `DT_NEEDED` on
/// the RAPIDS shared libraries that would link it. Presence alone is not
/// evidence because CMake installation does not remove old shared artifacts.
pub(crate) fn classify_optional_runtime_dep(
    candidate: Option<PathBuf>,
    consumers: &[PathBuf],
    file_name: &str,
) -> Result<OptionalRuntimeDepLinkage, String> {
    let Some(candidate) = candidate else { return Ok(OptionalRuntimeDepLinkage::Absorbed) };
    let mut saw_consumer = false;
    for consumer in consumers {
        if !consumer.is_file() {
            continue;
        }
        saw_consumer = true;
        if elf_dt_needed(consumer)?.iter().any(|dependency| dependency == file_name) {
            return Ok(OptionalRuntimeDepLinkage::Shared(candidate));
        }
    }
    if !saw_consumer {
        return Ok(OptionalRuntimeDepLinkage::Shared(candidate));
    }
    Ok(OptionalRuntimeDepLinkage::StaleSharedArtifact(candidate))
}

// std-only ELF64 reader for DT_NEEDED entries. Only the header, program
// headers, dynamic section, and dynamic string table are read: libcudf.so is
// hundreds of MB, so whole-file reads are not acceptable.
const ELF64_EHDR_SIZE: usize = 64;
const ELF64_PHDR_SIZE: usize = 56;
const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const DT_NULL: i64 = 0;
const DT_NEEDED: i64 = 1;
const DT_STRTAB: i64 = 5;
const DT_STRSZ: i64 = 10;
/// Sanity caps so a corrupt header cannot turn into a huge read.
const MAX_PROGRAM_HEADERS: usize = 4096;
const MAX_DYNAMIC_SECTION_BYTES: u64 = 1 << 24;
const MAX_STRTAB_BYTES: u64 = 1 << 24;

fn u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
}

fn u64_le(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("slice of eight bytes"))
}

fn i64_le(bytes: &[u8], offset: usize) -> i64 {
    i64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("slice of eight bytes"))
}

/// Read the `DT_NEEDED` shared-library dependencies of an ELF64 binary.
/// Returns an empty list for binaries without a dynamic section. Only
/// little-endian ELF64 executables and shared libraries are supported.
pub(crate) fn elf_dt_needed(path: &Path) -> Result<Vec<String>, String> {
    let context = |error: String| format!("{}: {error}", path.display());
    let file = fs::File::open(path).map_err(|error| context(format!("failed to open: {error}")))?;
    let mut ehdr = [0u8; ELF64_EHDR_SIZE];
    file.read_exact_at(&mut ehdr, 0).map_err(|error| context(format!("failed to read ELF header: {error}")))?;
    if ehdr[..4] != [0x7f, b'E', b'L', b'F'] {
        return Err(context("not an ELF file".to_owned()));
    }
    if ehdr[4] != 2 {
        return Err(context("not a 64-bit ELF file".to_owned()));
    }
    if ehdr[5] != 1 {
        return Err(context("not a little-endian ELF file".to_owned()));
    }
    let elf_type = u16_le(&ehdr, 16);
    if elf_type != 2 && elf_type != 3 {
        return Err(context(format!("unsupported ELF type {elf_type} (expected executable or shared library)")));
    }
    let phoff = u64_le(&ehdr, 32);
    let phentsize = u16_le(&ehdr, 54) as usize;
    let phnum = u16_le(&ehdr, 56) as usize;
    if phnum > MAX_PROGRAM_HEADERS {
        return Err(context(format!("unreasonable program header count {phnum}")));
    }
    if phnum == 0 {
        return Ok(Vec::new());
    }
    if phentsize < ELF64_PHDR_SIZE {
        return Err(context(format!("program header entry size {phentsize} is smaller than the ELF64 size")));
    }

    let mut phdrs = vec![0u8; phnum * phentsize];
    file.read_exact_at(&mut phdrs, phoff)
        .map_err(|error| context(format!("failed to read program headers: {error}")))?;
    let mut loads: Vec<(u64, u64, u64)> = Vec::new();
    let mut dynamic: Option<(u64, u64)> = None;
    for index in 0..phnum {
        let phdr = &phdrs[index * phentsize..][..ELF64_PHDR_SIZE];
        match u32_le(phdr, 0) {
            PT_LOAD => loads.push((u64_le(phdr, 8), u64_le(phdr, 16), u64_le(phdr, 32))),
            PT_DYNAMIC => dynamic = Some((u64_le(phdr, 8), u64_le(phdr, 32))),
            _ => {},
        }
    }
    let Some((dynamic_offset, dynamic_size)) = dynamic else { return Ok(Vec::new()) };
    if dynamic_size > MAX_DYNAMIC_SECTION_BYTES {
        return Err(context(format!("unreasonable dynamic section size {dynamic_size}")));
    }
    let mut dynamic_bytes = vec![0u8; dynamic_size as usize];
    file.read_exact_at(&mut dynamic_bytes, dynamic_offset)
        .map_err(|error| context(format!("failed to read dynamic section: {error}")))?;

    let mut strtab_vaddr: Option<u64> = None;
    let mut strtab_size = 0u64;
    let mut needed_offsets = Vec::new();
    for entry in dynamic_bytes.as_chunks::<16>().0 {
        let tag = i64_le(entry, 0);
        let value = u64_le(entry, 8);
        match tag {
            DT_NULL => break,
            DT_NEEDED => needed_offsets.push(value),
            DT_STRTAB => strtab_vaddr = Some(value),
            DT_STRSZ => strtab_size = value,
            _ => {},
        }
    }
    if needed_offsets.is_empty() {
        return Ok(Vec::new());
    }
    let strtab_vaddr = strtab_vaddr.ok_or_else(|| context("DT_NEEDED entries without a DT_STRTAB entry".to_owned()))?;
    if strtab_size == 0 || strtab_size > MAX_STRTAB_BYTES {
        return Err(context(format!("unreasonable dynamic string table size {strtab_size}")));
    }
    let strtab_offset = loads
        .iter()
        .find_map(|(segment_offset, segment_vaddr, segment_filesz)| {
            let delta = strtab_vaddr.checked_sub(*segment_vaddr)?;
            let end = delta.checked_add(strtab_size)?;
            (end <= *segment_filesz).then(|| segment_offset + delta)
        })
        .ok_or_else(|| context("dynamic string table is not covered by any PT_LOAD segment".to_owned()))?;
    let mut strtab = vec![0u8; strtab_size as usize];
    file.read_exact_at(&mut strtab, strtab_offset)
        .map_err(|error| context(format!("failed to read dynamic string table: {error}")))?;

    needed_offsets
        .iter()
        .map(|offset| {
            let start = usize::try_from(*offset)
                .map_err(|_| context(format!("DT_NEEDED string offset {offset} is not addressable")))?;
            if start >= strtab.len() {
                return Err(context(format!("DT_NEEDED string offset {start} is outside the string table")));
            }
            let end = strtab[start..]
                .iter()
                .position(|byte| *byte == 0)
                .map(|position| start + position)
                .ok_or_else(|| context(format!("unterminated DT_NEEDED string at offset {start}")))?;
            String::from_utf8(strtab[start..end].to_vec())
                .map_err(|error| context(format!("DT_NEEDED string at offset {start} is not UTF-8: {error}")))
        })
        .collect()
}
