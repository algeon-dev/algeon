use std::path::PathBuf;

#[allow(dead_code)]
#[path = "../build_support.rs"]
mod build_support;

pub fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn include_dir() -> PathBuf {
    let submodule_include = manifest_dir().join("../cugraph/cpp/include");
    build_support::resolve_include_dir(std::env::var("CUGRAPH_INCLUDE_DIR").ok().as_deref(), &submodule_include)
        .unwrap_or_else(|err| panic!("{err}"))
}

pub fn strip_c_comments(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let bytes = source.as_bytes();
    let mut i = 0;
    let mut in_block_comment = false;

    while i < bytes.len() {
        if in_block_comment {
            if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                in_block_comment = false;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }

        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            in_block_comment = true;
            i += 2;
            continue;
        }

        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        out.push(bytes[i] as char);
        i += 1;
    }

    out
}
