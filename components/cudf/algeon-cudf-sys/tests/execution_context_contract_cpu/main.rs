#[path = "../support/source_contract_support.rs"]
mod source_contract_support;

mod io;
mod required_options;

const IO_PARQUET_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/io_parquet.cpp"));

fn find_marker_ignoring_ascii_whitespace(source: &str, marker: &str) -> Option<usize> {
    let marker = marker.bytes().filter(|byte| !byte.is_ascii_whitespace()).collect::<Vec<_>>();
    if marker.is_empty() {
        return Some(0);
    }

    let source = source.as_bytes();
    (0..source.len()).find(|&start| {
        let mut source_index = start;
        for &expected in &marker {
            while source_index < source.len() && source[source_index].is_ascii_whitespace() {
                source_index += 1;
            }
            if source.get(source_index) != Some(&expected) {
                return false;
            }
            source_index += 1;
        }
        true
    })
}

fn function_region<'a>(source: &'a str, start_marker: &str, end_marker: &str) -> &'a str {
    let start = find_marker_ignoring_ascii_whitespace(source, start_marker).expect("start marker must exist");
    let tail = &source[start..];
    let end = find_marker_ignoring_ascii_whitespace(tail, end_marker)
        .unwrap_or_else(|| panic!("end marker must exist after {start_marker}: {end_marker}"));
    &tail[..end]
}
