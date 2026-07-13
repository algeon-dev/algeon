// Build-time ABI guardrails for enums shared with Rust.
// Rust declares FFI enums with `#[repr(i32)]`, so the C/C++ shim must keep
// them 32-bit and preserve the discriminants expected by Rust.

#include "arrow_device.h"
#include "ast.h"
#include "binary.h"
#include "copying.h"
#include "datetime.h"
#include "device.h"
#include "execution_context.h"
#include "filter.h"
#include "groupby.h"
#include "join.h"
#include "memory_resource.h"
#include "reduction.h"
#include "sorting.h"
#include "stream_compaction.h"
#include "strings.h"
#include "types.h"
#include "unary.h"

#include <cudf/ast/ast_operator.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/side_type.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

static_assert(sizeof(cudf_status_t) == sizeof(int32_t), "cudf_status_t must be 32-bit");
static_assert(sizeof(cudf_ast_operator_t) == sizeof(int32_t), "cudf_ast_operator_t must be 32-bit");
static_assert(sizeof(cudf_ast_table_reference_t) == sizeof(int32_t),
              "cudf_ast_table_reference_t must be 32-bit");
static_assert(sizeof(cudf_type_id_t) == sizeof(int32_t), "cudf_type_id_t must be 32-bit");
static_assert(sizeof(cudf_order_t) == sizeof(int32_t), "cudf_order_t must be 32-bit");
static_assert(sizeof(cudf_null_order_t) == sizeof(int32_t), "cudf_null_order_t must be 32-bit");
static_assert(sizeof(cudf_aggregation_kind_t) == sizeof(int32_t),
              "cudf_aggregation_kind_t must be 32-bit");
static_assert(sizeof(cudf_null_policy_t) == sizeof(int32_t), "cudf_null_policy_t must be 32-bit");
static_assert(sizeof(cudf_null_equality_t) == sizeof(int32_t),
              "cudf_null_equality_t must be 32-bit");
static_assert(sizeof(cudf_nullable_join_t) == sizeof(int32_t),
              "cudf_nullable_join_t must be 32-bit");
static_assert(sizeof(cudf_join_kind_t) == sizeof(int32_t), "cudf_join_kind_t must be 32-bit");
static_assert(sizeof(cudf_unary_operator_t) == sizeof(int32_t),
              "cudf_unary_operator_t must be 32-bit");
static_assert(sizeof(cudf_binary_operator_t) == sizeof(int32_t),
              "cudf_binary_operator_t must be 32-bit");
static_assert(sizeof(cudf_datetime_component_t) == sizeof(int32_t),
              "cudf_datetime_component_t must be 32-bit");
static_assert(sizeof(cudf_reduce_aggregation_kind_t) == sizeof(int32_t),
              "cudf_reduce_aggregation_kind_t must be 32-bit");
static_assert(sizeof(cudf_scan_type_t) == sizeof(int32_t), "cudf_scan_type_t must be 32-bit");
static_assert(sizeof(cudf_out_of_bounds_policy_t) == sizeof(int32_t),
              "cudf_out_of_bounds_policy_t must be 32-bit");
static_assert(sizeof(cudf_duplicate_keep_option_t) == sizeof(int32_t),
              "cudf_duplicate_keep_option_t must be 32-bit");
static_assert(sizeof(cudf_stream_compaction_null_equality_t) == sizeof(int32_t),
              "cudf_stream_compaction_null_equality_t must be 32-bit");
static_assert(sizeof(cudf_stream_compaction_nan_equality_t) == sizeof(int32_t),
              "cudf_stream_compaction_nan_equality_t must be 32-bit");
static_assert(sizeof(cudf_strip_side_t) == sizeof(int32_t), "cudf_strip_side_t must be 32-bit");
static_assert(sizeof(cudf_device_attribute_t) == sizeof(int32_t),
              "cudf_device_attribute_t must be 32-bit");
static_assert(sizeof(cudf_arrow_device_export_materialization_t) == sizeof(int32_t),
              "cudf_arrow_device_export_materialization_t must be 32-bit");
static_assert(sizeof(cudf_arrow_device_import_materialization_t) == sizeof(int32_t),
              "cudf_arrow_device_import_materialization_t must be 32-bit");
static_assert(CUDF_ARROW_DEVICE_EXPORT_PRODUCER_RETAINED_VIEW == 0);
static_assert(CUDF_ARROW_DEVICE_EXPORT_BRIDGE_OWNED_DEVICE_COPY == 1);
static_assert(CUDF_ARROW_DEVICE_IMPORT_FOREIGN_RETAINED_VIEW == 0);
static_assert(CUDF_ARROW_DEVICE_IMPORT_BRIDGE_OWNED_DEVICE_COPY == 1);
static_assert(std::is_standard_layout_v<cudf_exec_options_t>,
              "cudf_exec_options_t must be standard layout");
static_assert(offsetof(cudf_exec_options_t, struct_size) == 0,
              "cudf_exec_options_t::struct_size must be the first field");
static_assert(sizeof(static_cast<cudf_exec_options_t*>(nullptr)->device) == sizeof(int32_t),
              "cudf_exec_options_t::device must be 32-bit");
static_assert(offsetof(cudf_exec_options_t, device) > offsetof(cudf_exec_options_t, struct_size),
              "cudf_exec_options_t::device must follow struct_size");
static_assert(sizeof(static_cast<cudf_exec_options_t*>(nullptr)->flags) == sizeof(uint32_t),
              "cudf_exec_options_t::flags must be 32-bit");
static_assert(offsetof(cudf_exec_options_t, flags) > offsetof(cudf_exec_options_t, device),
              "cudf_exec_options_t::flags must follow device");
static_assert(offsetof(cudf_exec_options_t, stream) > offsetof(cudf_exec_options_t, flags),
              "cudf_exec_options_t::stream must follow flags");
static_assert(offsetof(cudf_exec_options_t, mr) > offsetof(cudf_exec_options_t, stream),
              "cudf_exec_options_t::mr must follow stream");
static_assert(sizeof(static_cast<cudf_exec_options_t*>(nullptr)->reserved) == sizeof(uint64_t) * 4,
              "cudf_exec_options_t::reserved must expose four 64-bit slots");
static_assert(std::is_standard_layout_v<cudf_memory_resource_usage_t>,
              "cudf_memory_resource_usage_t must be standard layout");
static_assert(offsetof(cudf_memory_resource_usage_t, struct_size) == 0,
              "cudf_memory_resource_usage_t::struct_size must be the first field");
static_assert(sizeof(static_cast<cudf_memory_resource_usage_t*>(nullptr)->reserved) ==
                sizeof(uint64_t) * CUDF_MEMORY_RESOURCE_USAGE_RESERVED_SLOTS,
              "cudf_memory_resource_usage_t::reserved must expose the advertised slots");
static_assert(sizeof(cudf_memory_manager_base_kind_t) == sizeof(int32_t),
              "cudf_memory_manager_base_kind_t must be 32-bit");
static_assert(sizeof(cudf_memory_failure_action_t) == sizeof(int32_t),
              "cudf_memory_failure_action_t must be 32-bit");
static_assert(CUDF_MEMORY_FAILURE_ACTION_DECLINE == 0);
static_assert(CUDF_MEMORY_FAILURE_ACTION_RETRY == 1);
static_assert(CUDF_MEMORY_FAILURE_ACTION_TRIM_AND_RETRY == 2);
static_assert(sizeof(cudf_memory_failure_source_t) == sizeof(int32_t),
              "cudf_memory_failure_source_t must be 32-bit");
static_assert(CUDF_MEMORY_FAILURE_SOURCE_PHYSICAL == 0);
static_assert(CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT == 1);
static_assert(sizeof(cudf_memory_failure_domain_t) == sizeof(int32_t),
              "cudf_memory_failure_domain_t must be 32-bit");
static_assert(CUDF_MEMORY_FAILURE_DOMAIN_NONE == 0);
static_assert(CUDF_MEMORY_FAILURE_DOMAIN_STREAM_RESERVATION == 1);
static_assert(CUDF_MEMORY_FAILURE_DOMAIN_RESERVATION_ARENA == 2);
static_assert(CUDF_MEMORY_FAILURE_DOMAIN_QUERY_MEMORY_DOMAIN == 3);
static_assert(std::is_standard_layout_v<cudf_memory_pool_usage_t>,
              "cudf_memory_pool_usage_t must be standard layout");
static_assert(offsetof(cudf_memory_pool_usage_t, struct_size) == 0,
              "cudf_memory_pool_usage_t::struct_size must be the first field");
static_assert(sizeof(static_cast<cudf_memory_pool_usage_t*>(nullptr)->reserved) ==
                sizeof(uint64_t) * CUDF_MEMORY_POOL_USAGE_RESERVED_SLOTS,
              "cudf_memory_pool_usage_t::reserved must expose the advertised slots");
static_assert(std::is_standard_layout_v<cudf_memory_failure_info_t>,
              "cudf_memory_failure_info_t must be standard layout");
static_assert(offsetof(cudf_memory_failure_info_t, struct_size) == 0,
              "cudf_memory_failure_info_t::struct_size must be the first field");
static_assert(sizeof(static_cast<cudf_memory_failure_info_t*>(nullptr)->device_ordinal) ==
                sizeof(int32_t),
              "cudf_memory_failure_info_t::device_ordinal must be 32-bit");
static_assert(sizeof(static_cast<cudf_memory_failure_info_t*>(nullptr)->flags) ==
                sizeof(uint32_t),
              "cudf_memory_failure_info_t::flags must be 32-bit");
static_assert(sizeof(static_cast<cudf_memory_failure_info_t*>(nullptr)->failure_source) ==
                sizeof(uint32_t),
              "cudf_memory_failure_info_t::failure_source must be 32-bit");
static_assert(sizeof(static_cast<cudf_memory_failure_info_t*>(nullptr)
                       ->reserved_failure_info_padding) == sizeof(uint32_t),
              "cudf_memory_failure_info_t::reserved_failure_info_padding must be 32-bit");
static_assert(sizeof(static_cast<cudf_memory_failure_info_t*>(nullptr)->failure_source) +
                  sizeof(static_cast<cudf_memory_failure_info_t*>(nullptr)
                           ->reserved_failure_info_padding) +
                  sizeof(static_cast<cudf_memory_failure_info_t*>(nullptr)->reserved) ==
                sizeof(uint64_t) * CUDF_MEMORY_FAILURE_INFO_RESERVED_SLOTS,
              "cudf_memory_failure_info_t failure source must fit in the advertised reserved slots");
static_assert(std::is_standard_layout_v<cudf_memory_manager_options_t>,
              "cudf_memory_manager_options_t must be standard layout");
static_assert(offsetof(cudf_memory_manager_options_t, struct_size) == 0,
              "cudf_memory_manager_options_t::struct_size must be the first field");
static_assert(sizeof(static_cast<cudf_memory_manager_options_t*>(nullptr)->base_kind) ==
                sizeof(int32_t),
              "cudf_memory_manager_options_t::base_kind must be 32-bit");
static_assert(sizeof(static_cast<cudf_memory_manager_options_t*>(nullptr)->flags) ==
                sizeof(uint32_t),
              "cudf_memory_manager_options_t::flags must be 32-bit");
static_assert(sizeof(static_cast<cudf_memory_manager_options_t*>(nullptr)->reserved) ==
                sizeof(uint64_t) * CUDF_MEMORY_MANAGER_OPTIONS_RESERVED_SLOTS,
              "cudf_memory_manager_options_t::reserved must expose the advertised slots");
static_assert(std::is_standard_layout_v<cudf_row_predicate_node_t>,
              "cudf_row_predicate_node_t must be standard layout");
static_assert(offsetof(cudf_row_predicate_node_t, kind) == 0,
              "cudf_row_predicate_node_t::kind must be the first field");
static_assert(sizeof(static_cast<cudf_row_predicate_node_t*>(nullptr)->lhs_index) ==
                sizeof(int32_t),
              "cudf_row_predicate_node_t::lhs_index must be 32-bit");
static_assert(sizeof(static_cast<cudf_row_predicate_node_t*>(nullptr)->rhs_index) ==
                sizeof(int32_t),
              "cudf_row_predicate_node_t::rhs_index must be 32-bit");
static_assert(sizeof(static_cast<cudf_row_predicate_node_t*>(nullptr)->column_index) ==
                sizeof(int32_t),
              "cudf_row_predicate_node_t::column_index must be 32-bit");
static_assert(sizeof(static_cast<cudf_row_predicate_node_t*>(nullptr)->binary_op) ==
                sizeof(int32_t),
              "cudf_row_predicate_node_t::binary_op must be 32-bit");
static_assert(sizeof(static_cast<cudf_row_predicate_node_t*>(nullptr)->unary_op) == sizeof(int32_t),
              "cudf_row_predicate_node_t::unary_op must be 32-bit");
static_assert(sizeof(static_cast<cudf_row_predicate_node_t*>(nullptr)->literal) ==
                sizeof(cudf_scalar_handle),
              "cudf_row_predicate_node_t::literal must match cudf_scalar_handle width");
static_assert(offsetof(cudf_row_predicate_node_t, lhs_index) >
                offsetof(cudf_row_predicate_node_t, kind),
              "cudf_row_predicate_node_t::lhs_index must follow kind");
static_assert(offsetof(cudf_row_predicate_node_t, rhs_index) >
                offsetof(cudf_row_predicate_node_t, lhs_index),
              "cudf_row_predicate_node_t::rhs_index must follow lhs_index");
static_assert(offsetof(cudf_row_predicate_node_t, column_index) >
                offsetof(cudf_row_predicate_node_t, rhs_index),
              "cudf_row_predicate_node_t::column_index must follow rhs_index");
static_assert(offsetof(cudf_row_predicate_node_t, binary_op) >
                offsetof(cudf_row_predicate_node_t, column_index),
              "cudf_row_predicate_node_t::binary_op must follow column_index");
static_assert(offsetof(cudf_row_predicate_node_t, unary_op) >
                offsetof(cudf_row_predicate_node_t, binary_op),
              "cudf_row_predicate_node_t::unary_op must follow binary_op");
static_assert(offsetof(cudf_row_predicate_node_t, literal) >
                offsetof(cudf_row_predicate_node_t, unary_op),
              "cudf_row_predicate_node_t::literal must follow unary_op");

static_assert(CUDF_SUCCESS == 0);
static_assert(CUDF_ERROR_INVALID_OPERATION == 1);
static_assert(CUDF_ERROR_CUDA == 2);
static_assert(CUDF_ERROR_OUT_OF_MEMORY == 3);
static_assert(CUDF_ERROR_IO == 4);
static_assert(CUDF_ERROR_TYPE == 5);
static_assert(CUDF_ERROR_NULL == 6);
static_assert(CUDF_ERROR_INDEX == 7);
static_assert(CUDF_ERROR_UNKNOWN == 8);
static_assert(CUDF_ERROR_FATAL_CUDA == 9);
static_assert(CUDF_ERROR_INVALID_ARGUMENT == 10);
static_assert(CUDF_ERROR_UNSUPPORTED == 11);
static_assert(CUDF_MEMORY_MANAGER_BASE_CUDA_ASYNC == 0);
static_assert(CUDF_MEMORY_MANAGER_BASE_POOL == 1);

static_assert(CUDF_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT == 1);
static_assert(CUDF_DEVICE_ATTRIBUTE_MANAGED_MEMORY == 2);
static_assert(CUDF_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS == 3);

static_assert(CUDF_AST_TABLE_REFERENCE_LEFT == static_cast<int>(cudf::ast::table_reference::LEFT));
static_assert(CUDF_AST_TABLE_REFERENCE_RIGHT ==
              static_cast<int>(cudf::ast::table_reference::RIGHT));
static_assert(CUDF_AST_TABLE_REFERENCE_OUTPUT ==
              static_cast<int>(cudf::ast::table_reference::OUTPUT));

static_assert(CUDF_AST_ADD == static_cast<int>(cudf::ast::ast_operator::ADD));
static_assert(CUDF_AST_SUB == static_cast<int>(cudf::ast::ast_operator::SUB));
static_assert(CUDF_AST_MUL == static_cast<int>(cudf::ast::ast_operator::MUL));
static_assert(CUDF_AST_DIV == static_cast<int>(cudf::ast::ast_operator::DIV));
static_assert(CUDF_AST_TRUE_DIV == static_cast<int>(cudf::ast::ast_operator::TRUE_DIV));
static_assert(CUDF_AST_FLOOR_DIV == static_cast<int>(cudf::ast::ast_operator::FLOOR_DIV));
static_assert(CUDF_AST_MOD == static_cast<int>(cudf::ast::ast_operator::MOD));
static_assert(CUDF_AST_PYMOD == static_cast<int>(cudf::ast::ast_operator::PYMOD));
static_assert(CUDF_AST_POW == static_cast<int>(cudf::ast::ast_operator::POW));
static_assert(CUDF_AST_EQUAL == static_cast<int>(cudf::ast::ast_operator::EQUAL));
static_assert(CUDF_AST_NULL_EQUAL == static_cast<int>(cudf::ast::ast_operator::NULL_EQUAL));
static_assert(CUDF_AST_NOT_EQUAL == static_cast<int>(cudf::ast::ast_operator::NOT_EQUAL));
static_assert(CUDF_AST_LESS == static_cast<int>(cudf::ast::ast_operator::LESS));
static_assert(CUDF_AST_GREATER == static_cast<int>(cudf::ast::ast_operator::GREATER));
static_assert(CUDF_AST_LESS_EQUAL == static_cast<int>(cudf::ast::ast_operator::LESS_EQUAL));
static_assert(CUDF_AST_GREATER_EQUAL == static_cast<int>(cudf::ast::ast_operator::GREATER_EQUAL));
static_assert(CUDF_AST_LOGICAL_AND == static_cast<int>(cudf::ast::ast_operator::LOGICAL_AND));
static_assert(CUDF_AST_LOGICAL_OR == static_cast<int>(cudf::ast::ast_operator::LOGICAL_OR));
static_assert(CUDF_AST_IS_NULL == static_cast<int>(cudf::ast::ast_operator::IS_NULL));
static_assert(CUDF_AST_NOT == static_cast<int>(cudf::ast::ast_operator::NOT));
static_assert(CUDF_AST_CAST_TO_FLOAT64 ==
              static_cast<int>(cudf::ast::ast_operator::CAST_TO_FLOAT64));

static_assert(CUDF_TYPE_EMPTY == static_cast<int>(cudf::type_id::EMPTY));
static_assert(CUDF_TYPE_INT8 == static_cast<int>(cudf::type_id::INT8));
static_assert(CUDF_TYPE_INT16 == static_cast<int>(cudf::type_id::INT16));
static_assert(CUDF_TYPE_INT32 == static_cast<int>(cudf::type_id::INT32));
static_assert(CUDF_TYPE_INT64 == static_cast<int>(cudf::type_id::INT64));
static_assert(CUDF_TYPE_UINT8 == static_cast<int>(cudf::type_id::UINT8));
static_assert(CUDF_TYPE_UINT16 == static_cast<int>(cudf::type_id::UINT16));
static_assert(CUDF_TYPE_UINT32 == static_cast<int>(cudf::type_id::UINT32));
static_assert(CUDF_TYPE_UINT64 == static_cast<int>(cudf::type_id::UINT64));
static_assert(CUDF_TYPE_FLOAT32 == static_cast<int>(cudf::type_id::FLOAT32));
static_assert(CUDF_TYPE_FLOAT64 == static_cast<int>(cudf::type_id::FLOAT64));
static_assert(CUDF_TYPE_BOOL8 == static_cast<int>(cudf::type_id::BOOL8));
static_assert(CUDF_TYPE_TIMESTAMP_DAYS == static_cast<int>(cudf::type_id::TIMESTAMP_DAYS));
static_assert(CUDF_TYPE_TIMESTAMP_SECONDS == static_cast<int>(cudf::type_id::TIMESTAMP_SECONDS));
static_assert(CUDF_TYPE_TIMESTAMP_MILLISECONDS ==
              static_cast<int>(cudf::type_id::TIMESTAMP_MILLISECONDS));
static_assert(CUDF_TYPE_TIMESTAMP_MICROSECONDS ==
              static_cast<int>(cudf::type_id::TIMESTAMP_MICROSECONDS));
static_assert(CUDF_TYPE_TIMESTAMP_NANOSECONDS ==
              static_cast<int>(cudf::type_id::TIMESTAMP_NANOSECONDS));
static_assert(CUDF_TYPE_DURATION_DAYS == static_cast<int>(cudf::type_id::DURATION_DAYS));
static_assert(CUDF_TYPE_DURATION_SECONDS == static_cast<int>(cudf::type_id::DURATION_SECONDS));
static_assert(CUDF_TYPE_DURATION_MILLISECONDS ==
              static_cast<int>(cudf::type_id::DURATION_MILLISECONDS));
static_assert(CUDF_TYPE_DURATION_MICROSECONDS ==
              static_cast<int>(cudf::type_id::DURATION_MICROSECONDS));
static_assert(CUDF_TYPE_DURATION_NANOSECONDS ==
              static_cast<int>(cudf::type_id::DURATION_NANOSECONDS));
static_assert(CUDF_TYPE_DICTIONARY32 == static_cast<int>(cudf::type_id::DICTIONARY32));
static_assert(CUDF_TYPE_STRING == static_cast<int>(cudf::type_id::STRING));
static_assert(CUDF_TYPE_LIST == static_cast<int>(cudf::type_id::LIST));
static_assert(CUDF_TYPE_DECIMAL32 == static_cast<int>(cudf::type_id::DECIMAL32));
static_assert(CUDF_TYPE_DECIMAL64 == static_cast<int>(cudf::type_id::DECIMAL64));
static_assert(CUDF_TYPE_DECIMAL128 == static_cast<int>(cudf::type_id::DECIMAL128));
static_assert(CUDF_TYPE_STRUCT == static_cast<int>(cudf::type_id::STRUCT));
// Exhaustiveness pin: if libcudf adds a type id, this fails the build so the
// new value gets mirrored in types.h, cudf-sys/src/types.rs, and
// TypeId::from_raw before any column metadata read can observe it.
static_assert(static_cast<int>(cudf::type_id::NUM_TYPE_IDS) == CUDF_TYPE_STRUCT + 1,
              "libcudf added a cudf::type_id value that is not mirrored in the FFI enums");

static_assert(CUDF_ORDER_ASCENDING == static_cast<int>(cudf::order::ASCENDING));
static_assert(CUDF_ORDER_DESCENDING == static_cast<int>(cudf::order::DESCENDING));

static_assert(CUDF_NULL_ORDER_AFTER == static_cast<int>(cudf::null_order::AFTER));
static_assert(CUDF_NULL_ORDER_BEFORE == static_cast<int>(cudf::null_order::BEFORE));

static_assert(CUDF_NULL_POLICY_EXCLUDE == static_cast<int>(cudf::null_policy::EXCLUDE));
static_assert(CUDF_NULL_POLICY_INCLUDE == static_cast<int>(cudf::null_policy::INCLUDE));

static_assert(CUDF_NULL_EQUALITY_EQUAL == static_cast<int>(cudf::null_equality::EQUAL));
static_assert(CUDF_NULL_EQUALITY_UNEQUAL == static_cast<int>(cudf::null_equality::UNEQUAL));

static_assert(CUDF_NULLABLE_JOIN_YES == static_cast<int>(cudf::nullable_join::YES));
static_assert(CUDF_NULLABLE_JOIN_NO == static_cast<int>(cudf::nullable_join::NO));

static_assert(CUDF_JOIN_KIND_INNER_JOIN == static_cast<int>(cudf::join_kind::INNER_JOIN));
static_assert(CUDF_JOIN_KIND_LEFT_JOIN == static_cast<int>(cudf::join_kind::LEFT_JOIN));
static_assert(CUDF_JOIN_KIND_FULL_JOIN == static_cast<int>(cudf::join_kind::FULL_JOIN));
static_assert(CUDF_JOIN_KIND_LEFT_SEMI_JOIN == static_cast<int>(cudf::join_kind::LEFT_SEMI_JOIN));
static_assert(CUDF_JOIN_KIND_LEFT_ANTI_JOIN == static_cast<int>(cudf::join_kind::LEFT_ANTI_JOIN));

static_assert(CUDF_SCAN_INCLUSIVE == static_cast<int>(cudf::scan_type::INCLUSIVE));
static_assert(CUDF_SCAN_EXCLUSIVE == static_cast<int>(cudf::scan_type::EXCLUSIVE));

static_assert(CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY ==
              static_cast<int>(cudf::out_of_bounds_policy::NULLIFY));
static_assert(CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK ==
              static_cast<int>(cudf::out_of_bounds_policy::DONT_CHECK));

static_assert(CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY ==
              static_cast<int>(cudf::duplicate_keep_option::KEEP_ANY));
static_assert(CUDF_DUPLICATE_KEEP_OPTION_KEEP_FIRST ==
              static_cast<int>(cudf::duplicate_keep_option::KEEP_FIRST));
static_assert(CUDF_DUPLICATE_KEEP_OPTION_KEEP_LAST ==
              static_cast<int>(cudf::duplicate_keep_option::KEEP_LAST));
static_assert(CUDF_DUPLICATE_KEEP_OPTION_KEEP_NONE ==
              static_cast<int>(cudf::duplicate_keep_option::KEEP_NONE));

static_assert(CUDF_STREAM_COMPACTION_NULL_EQUALITY_EQUAL ==
              static_cast<int>(cudf::null_equality::EQUAL));
static_assert(CUDF_STREAM_COMPACTION_NULL_EQUALITY_UNEQUAL ==
              static_cast<int>(cudf::null_equality::UNEQUAL));

static_assert(CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL ==
              static_cast<int>(cudf::nan_equality::ALL_EQUAL));
static_assert(CUDF_STREAM_COMPACTION_NAN_EQUALITY_UNEQUAL ==
              static_cast<int>(cudf::nan_equality::UNEQUAL));

static_assert(CUDF_UNARY_SIN == static_cast<int>(cudf::unary_operator::SIN));
static_assert(CUDF_UNARY_COS == static_cast<int>(cudf::unary_operator::COS));
static_assert(CUDF_UNARY_TAN == static_cast<int>(cudf::unary_operator::TAN));
static_assert(CUDF_UNARY_ARCSIN == static_cast<int>(cudf::unary_operator::ARCSIN));
static_assert(CUDF_UNARY_ARCCOS == static_cast<int>(cudf::unary_operator::ARCCOS));
static_assert(CUDF_UNARY_ARCTAN == static_cast<int>(cudf::unary_operator::ARCTAN));
static_assert(CUDF_UNARY_SINH == static_cast<int>(cudf::unary_operator::SINH));
static_assert(CUDF_UNARY_COSH == static_cast<int>(cudf::unary_operator::COSH));
static_assert(CUDF_UNARY_TANH == static_cast<int>(cudf::unary_operator::TANH));
static_assert(CUDF_UNARY_ARCSINH == static_cast<int>(cudf::unary_operator::ARCSINH));
static_assert(CUDF_UNARY_ARCCOSH == static_cast<int>(cudf::unary_operator::ARCCOSH));
static_assert(CUDF_UNARY_ARCTANH == static_cast<int>(cudf::unary_operator::ARCTANH));
static_assert(CUDF_UNARY_EXP == static_cast<int>(cudf::unary_operator::EXP));
static_assert(CUDF_UNARY_LOG == static_cast<int>(cudf::unary_operator::LOG));
static_assert(CUDF_UNARY_SQRT == static_cast<int>(cudf::unary_operator::SQRT));
static_assert(CUDF_UNARY_CBRT == static_cast<int>(cudf::unary_operator::CBRT));
static_assert(CUDF_UNARY_CEIL == static_cast<int>(cudf::unary_operator::CEIL));
static_assert(CUDF_UNARY_FLOOR == static_cast<int>(cudf::unary_operator::FLOOR));
static_assert(CUDF_UNARY_ABS == static_cast<int>(cudf::unary_operator::ABS));
static_assert(CUDF_UNARY_RINT == static_cast<int>(cudf::unary_operator::RINT));
static_assert(CUDF_UNARY_BIT_COUNT == static_cast<int>(cudf::unary_operator::BIT_COUNT));
static_assert(CUDF_UNARY_BIT_INVERT == static_cast<int>(cudf::unary_operator::BIT_INVERT));
static_assert(CUDF_UNARY_NOT == static_cast<int>(cudf::unary_operator::NOT));
static_assert(CUDF_UNARY_NEGATE == static_cast<int>(cudf::unary_operator::NEGATE));

static_assert(CUDF_DATETIME_COMPONENT_YEAR == 0);
static_assert(CUDF_DATETIME_COMPONENT_MONTH == 1);
static_assert(CUDF_DATETIME_COMPONENT_DAY == 2);

static_assert(CUDF_BINARY_ADD == static_cast<int>(cudf::binary_operator::ADD));
static_assert(CUDF_BINARY_SUB == static_cast<int>(cudf::binary_operator::SUB));
static_assert(CUDF_BINARY_MUL == static_cast<int>(cudf::binary_operator::MUL));
static_assert(CUDF_BINARY_DIV == static_cast<int>(cudf::binary_operator::DIV));
static_assert(CUDF_BINARY_TRUE_DIV == static_cast<int>(cudf::binary_operator::TRUE_DIV));
static_assert(CUDF_BINARY_FLOOR_DIV == static_cast<int>(cudf::binary_operator::FLOOR_DIV));
static_assert(CUDF_BINARY_MOD == static_cast<int>(cudf::binary_operator::MOD));
static_assert(CUDF_BINARY_PMOD == static_cast<int>(cudf::binary_operator::PMOD));
static_assert(CUDF_BINARY_PYMOD == static_cast<int>(cudf::binary_operator::PYMOD));
static_assert(CUDF_BINARY_POW == static_cast<int>(cudf::binary_operator::POW));
static_assert(CUDF_BINARY_INT_POW == static_cast<int>(cudf::binary_operator::INT_POW));
static_assert(CUDF_BINARY_LOG_BASE == static_cast<int>(cudf::binary_operator::LOG_BASE));
static_assert(CUDF_BINARY_ATAN2 == static_cast<int>(cudf::binary_operator::ATAN2));
static_assert(CUDF_BINARY_SHIFT_LEFT == static_cast<int>(cudf::binary_operator::SHIFT_LEFT));
static_assert(CUDF_BINARY_SHIFT_RIGHT == static_cast<int>(cudf::binary_operator::SHIFT_RIGHT));
static_assert(CUDF_BINARY_SHIFT_RIGHT_UNSIGNED ==
              static_cast<int>(cudf::binary_operator::SHIFT_RIGHT_UNSIGNED));
static_assert(CUDF_BINARY_BITWISE_AND == static_cast<int>(cudf::binary_operator::BITWISE_AND));
static_assert(CUDF_BINARY_BITWISE_OR == static_cast<int>(cudf::binary_operator::BITWISE_OR));
static_assert(CUDF_BINARY_BITWISE_XOR == static_cast<int>(cudf::binary_operator::BITWISE_XOR));
static_assert(CUDF_BINARY_LOGICAL_AND == static_cast<int>(cudf::binary_operator::LOGICAL_AND));
static_assert(CUDF_BINARY_LOGICAL_OR == static_cast<int>(cudf::binary_operator::LOGICAL_OR));
static_assert(CUDF_BINARY_EQUAL == static_cast<int>(cudf::binary_operator::EQUAL));
static_assert(CUDF_BINARY_NOT_EQUAL == static_cast<int>(cudf::binary_operator::NOT_EQUAL));
static_assert(CUDF_BINARY_LESS == static_cast<int>(cudf::binary_operator::LESS));
static_assert(CUDF_BINARY_GREATER == static_cast<int>(cudf::binary_operator::GREATER));
static_assert(CUDF_BINARY_LESS_EQUAL == static_cast<int>(cudf::binary_operator::LESS_EQUAL));
static_assert(CUDF_BINARY_GREATER_EQUAL == static_cast<int>(cudf::binary_operator::GREATER_EQUAL));
static_assert(CUDF_BINARY_NULL_EQUALS == static_cast<int>(cudf::binary_operator::NULL_EQUALS));
static_assert(CUDF_BINARY_NULL_NOT_EQUALS ==
              static_cast<int>(cudf::binary_operator::NULL_NOT_EQUALS));
static_assert(CUDF_BINARY_NULL_MAX == static_cast<int>(cudf::binary_operator::NULL_MAX));
static_assert(CUDF_BINARY_NULL_MIN == static_cast<int>(cudf::binary_operator::NULL_MIN));
static_assert(CUDF_BINARY_GENERIC_BINARY == static_cast<int>(cudf::binary_operator::GENERIC_BINARY));
static_assert(CUDF_BINARY_NULL_LOGICAL_AND ==
              static_cast<int>(cudf::binary_operator::NULL_LOGICAL_AND));
static_assert(CUDF_BINARY_NULL_LOGICAL_OR ==
              static_cast<int>(cudf::binary_operator::NULL_LOGICAL_OR));

static_assert(CUDF_STRIP_LEFT == static_cast<int>(cudf::strings::side_type::LEFT));
static_assert(CUDF_STRIP_RIGHT == static_cast<int>(cudf::strings::side_type::RIGHT));
static_assert(CUDF_STRIP_BOTH == static_cast<int>(cudf::strings::side_type::BOTH));
