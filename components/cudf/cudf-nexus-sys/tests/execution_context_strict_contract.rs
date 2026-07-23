mod source_contract_support;

use source_contract_support::contains_normalized;

const EXECUTION_CONTEXT_CPP: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/execution_context.cpp"));
const EXECUTION_CONTEXT_INTERNAL_HPP: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/execution_context_internal.hpp"));
const ARROW_DEVICE_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/arrow_device.cpp"));
const ARROW_HOST_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/arrow_host.h"));
const ARROW_HOST_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/arrow_host.cpp"));
const BINARY_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/binary.h"));
const BINARY_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/binary.cpp"));
const CONCATENATE_H: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/concatenate.h"));
const CONCATENATE_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/concatenate.cpp"));
const COLUMN_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/column.h"));
const COLUMN_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/column.cpp"));
const COPYING_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/copying.h"));
const COPYING_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/copying.cpp"));
const DATETIME_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/datetime.h"));
const DATETIME_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/datetime.cpp"));
const DEVICE_BUFFER_H: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/device_buffer.h"));
const DEVICE_BUFFER_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/device_buffer.cpp"));
const DICTIONARY_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/dictionary.h"));
const DICTIONARY_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/dictionary.cpp"));
const FILTER_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/filter.h"));
const FILTER_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/filter.cpp"));
const GROUPBY_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/groupby.h"));
const GROUPBY_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/groupby.cpp"));
const HASHING_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/hashing.h"));
const HASHING_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/hashing.cpp"));
const IO_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/io.h"));
const IO_CSV_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/io_csv.cpp"));
const IO_JSON_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/io_json.cpp"));
const IO_PARQUET_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/io_parquet.cpp"));
const JOIN_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/join.h"));
const JOIN_COMMON_HPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/join_common.hpp"));
const JOIN_HASH_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/join_hash.cpp"));
const JOIN_CONDITIONAL_CPP: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/join_conditional.cpp"));
const JOIN_FILTER_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/join_filter.cpp"));
const JOIN_MIXED_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/join_mixed.cpp"));
const PARTITIONING_H: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/partitioning.h"));
const PARTITIONING_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/partitioning.cpp"));
const PREPARED_IMPORT_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/prepared_import.cpp"));
const REPLACE_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/replace.h"));
const REPLACE_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/replace.cpp"));
const REDUCTION_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/reduction.h"));
const REDUCTION_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/reduction.cpp"));
const RESHAPE_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/reshape.h"));
const RESHAPE_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/reshape.cpp"));
const SEARCH_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/search.h"));
const SEARCH_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/search.cpp"));
const SCALAR_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/scalar.h"));
const SCALAR_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/scalar.cpp"));
const SORTING_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/sorting.h"));
const SORTING_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/sorting.cpp"));
const VENDORED_SORTING_HPP: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../cudf/cpp/include/cudf/sorting.hpp"));
const VENDORED_SORTED_ORDER_RADIX_CU: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../cudf/cpp/src/sort/sorted_order_radix.cu"));
const STREAM_COMPACTION_H: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/stream_compaction.h"));
const STREAM_COMPACTION_CPP: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/stream_compaction.cpp"));
const STRINGS_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/strings.h"));
const STRINGS_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/strings.cpp"));
const RESERVATION_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/memory/reservation.cpp"));
const RESERVATION_ACCOUNTING_HPP: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/reservation_accounting.hpp"));
const TABLE_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/table.h"));
const TABLE_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/table.cpp"));
const UNARY_H: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/include/cudf_rust/unary.h"));
const UNARY_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/unary.cpp"));
const VALIDATION_CU: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/validation.cu"));
// All raw-ABI assertions below are against the one checked-in bindgen output,
// rather than the removed handwritten per-domain FFI files.
const BINDINGS_RS: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/bindings.rs"));

use BINDINGS_RS as ARROW_HOST_FFI_RS;
use BINDINGS_RS as BINARY_FFI_RS;
use BINDINGS_RS as COLUMN_FFI_RS;
use BINDINGS_RS as CONCATENATE_FFI_RS;
use BINDINGS_RS as COPYING_FFI_RS;
use BINDINGS_RS as DATETIME_FFI_RS;
use BINDINGS_RS as DEVICE_BUFFER_FFI_RS;
use BINDINGS_RS as DICTIONARY_FFI_RS;
use BINDINGS_RS as FILTER_FFI_RS;
use BINDINGS_RS as GROUPBY_FFI_RS;
use BINDINGS_RS as HASHING_FFI_RS;
use BINDINGS_RS as IO_CSV_FFI_RS;
use BINDINGS_RS as IO_JSON_FFI_RS;
use BINDINGS_RS as IO_PARQUET_FFI_RS;
use BINDINGS_RS as JOIN_FFI_RS;
use BINDINGS_RS as PARTITIONING_FFI_RS;
use BINDINGS_RS as REDUCTION_FFI_RS;
use BINDINGS_RS as REPLACE_FFI_RS;
use BINDINGS_RS as RESHAPE_FFI_RS;
use BINDINGS_RS as SCALAR_FFI_RS;
use BINDINGS_RS as SEARCH_FFI_RS;
use BINDINGS_RS as SORTING_FFI_RS;
use BINDINGS_RS as STREAM_COMPACTION_FFI_RS;
use BINDINGS_RS as STRINGS_FFI_RS;
use BINDINGS_RS as TABLE_FFI_RS;
use BINDINGS_RS as UNARY_FFI_RS;
use BINDINGS_RS as VALIDATION_FFI_RS;

#[path = "execution_context_strict_contract/interop.rs"]
mod interop;
#[path = "execution_context_strict_contract/io.rs"]
mod io;
#[path = "execution_context_strict_contract/operation.rs"]
mod operation;
#[path = "execution_context_strict_contract/resolver.rs"]
mod resolver;
#[path = "execution_context_strict_contract/scalar.rs"]
mod scalar;

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

fn function_region_by_function_name<'a>(source: &'a str, function: &str, end_marker: &str) -> &'a str {
    for (start, _) in source.match_indices(function) {
        let prefix = &source[start.saturating_sub(96)..start];
        if !prefix.contains("cudf_status_t") {
            continue;
        }
        let tail = &source[start..];
        let end = find_marker_ignoring_ascii_whitespace(tail, end_marker).expect("end marker must exist");
        return &tail[..end];
    }
    panic!("{function} definition marker must exist");
}
