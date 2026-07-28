#ifndef CUDF_RUST_TEST_SUPPORT_INTERNAL_HPP
#define CUDF_RUST_TEST_SUPPORT_INTERNAL_HPP

#include "cudf_rust/error.h"

#include <cstddef>
#include <cstdint>

namespace cudf::rust::test_support {

#ifdef CUDF_RUST_ENABLE_TEST_HOOKS

void record_exec_context(std::uintptr_t stream_raw, void* mr_raw);
void record_synchronized_stream(std::uintptr_t stream_raw);

bool take_column_view_metadata_read_failure();
bool take_fixed_width_host_import_after_data_copy();
bool take_fixed_width_host_import_after_null_mask_copy();
bool take_strings_host_import_after_offsets_copy();
bool take_strings_host_import_after_chars_copy();
bool take_strings_host_import_after_null_mask_copy();
bool take_strings_host_import_before_make_strings_column();
bool take_strings_host_import_after_make_strings_column();

bool take_column_handle_release_failure();
bool take_table_release_columns_handle_array_alloc_failure();
bool take_table_view_negative_num_columns_failure();
bool take_table_view_negative_num_rows_failure();
bool take_groupby_negative_total_columns_failure();
cudf_status_t take_provenance_destroy_failure();

void record_apply_boolean_mask();
void record_row_predicate_filter();
void record_row_predicate_compute_mask();
void record_row_predicate_final_filter();

void record_event_synchronize(std::uintptr_t event_handle_raw);
bool event_synchronize_should_fail();
void wait_for_event_synchronize_release();
void record_stream_wait(std::uintptr_t stream_raw, std::uintptr_t event_raw);
void record_raw_event_record(std::uintptr_t stream_raw, std::uintptr_t event_raw);

void pause_query_domain_route_if_armed(std::uint64_t domain_id);

int32_t take_io_metadata_fault();
cudf_status_t apply_io_metadata_fault(int32_t fault, char*** out_names, int32_t* out_num_names);

#else

inline void record_exec_context(std::uintptr_t, void*) {}
inline void record_synchronized_stream(std::uintptr_t) {}

inline bool take_column_view_metadata_read_failure() { return false; }
inline bool take_fixed_width_host_import_after_data_copy() { return false; }
inline bool take_fixed_width_host_import_after_null_mask_copy() { return false; }
inline bool take_strings_host_import_after_offsets_copy() { return false; }
inline bool take_strings_host_import_after_chars_copy() { return false; }
inline bool take_strings_host_import_after_null_mask_copy() { return false; }
inline bool take_strings_host_import_before_make_strings_column() { return false; }
inline bool take_strings_host_import_after_make_strings_column() { return false; }

inline bool take_column_handle_release_failure() { return false; }
inline bool take_table_release_columns_handle_array_alloc_failure() { return false; }
inline bool take_table_view_negative_num_columns_failure() { return false; }
inline bool take_table_view_negative_num_rows_failure() { return false; }
inline bool take_groupby_negative_total_columns_failure() { return false; }
inline cudf_status_t take_provenance_destroy_failure() { return CUDF_SUCCESS; }

inline void record_apply_boolean_mask() {}
inline void record_row_predicate_filter() {}
inline void record_row_predicate_compute_mask() {}
inline void record_row_predicate_final_filter() {}

inline void record_event_synchronize(std::uintptr_t) {}
inline bool event_synchronize_should_fail() { return false; }
inline void wait_for_event_synchronize_release() {}
inline void record_stream_wait(std::uintptr_t, std::uintptr_t) {}
inline void record_raw_event_record(std::uintptr_t, std::uintptr_t) {}

inline void pause_query_domain_route_if_armed(std::uint64_t) {}

inline int32_t take_io_metadata_fault() { return 0; }
inline cudf_status_t apply_io_metadata_fault(int32_t, char***, int32_t*) { return CUDF_SUCCESS; }

#endif

}  // namespace cudf::rust::test_support

#endif  // CUDF_RUST_TEST_SUPPORT_INTERNAL_HPP
