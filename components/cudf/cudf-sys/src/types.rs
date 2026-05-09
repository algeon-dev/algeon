use std::os::raw::c_void;

// ---------------------------------------------------------------------------
// Error codes
// ---------------------------------------------------------------------------

pub const CUDF_SUCCESS: i32 = 0;
pub const CUDF_ERROR_INVALID_OPERATION: i32 = 1;
pub const CUDF_ERROR_CUDA: i32 = 2;
pub const CUDF_ERROR_OUT_OF_MEMORY: i32 = 3;
pub const CUDF_ERROR_IO: i32 = 4;
pub const CUDF_ERROR_TYPE: i32 = 5;
pub const CUDF_ERROR_NULL: i32 = 6;
pub const CUDF_ERROR_INDEX: i32 = 7;
pub const CUDF_ERROR_UNKNOWN: i32 = 8;
pub const CUDF_ERROR_FATAL_CUDA: i32 = 9;
pub const CUDF_ERROR_INVALID_ARGUMENT: i32 = 10;
pub const CUDF_ERROR_UNSUPPORTED: i32 = 11;
pub const CUDF_TEST_IO_METADATA_FAULT_NONE: i32 = 0;
pub const CUDF_TEST_IO_METADATA_FAULT_NEGATIVE_NAME_COUNT: i32 = 1;
pub const CUDF_TEST_IO_METADATA_FAULT_NULL_NAME_ARRAY: i32 = 2;
pub const CUDF_TEST_IO_METADATA_FAULT_NULL_NAME_ENTRY: i32 = 3;
pub const CUDF_TEST_IO_METADATA_FAULT_INVALID_UTF8_NAME: i32 = 4;
pub const CUDF_TEST_IO_METADATA_FAULT_ZERO_NAME_COUNT: i32 = 5;

pub type cudf_status_t = i32;

// ---------------------------------------------------------------------------
// AST enums (mirror cudf::ast)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_ast_table_reference_t {
    CUDF_AST_TABLE_REFERENCE_LEFT = 0,
    CUDF_AST_TABLE_REFERENCE_RIGHT = 1,
    CUDF_AST_TABLE_REFERENCE_OUTPUT = 2,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_ast_operator_t {
    CUDF_AST_ADD = 0,
    CUDF_AST_SUB = 1,
    CUDF_AST_MUL = 2,
    CUDF_AST_DIV = 3,
    CUDF_AST_TRUE_DIV = 4,
    CUDF_AST_FLOOR_DIV = 5,
    CUDF_AST_MOD = 6,
    CUDF_AST_PYMOD = 7,
    CUDF_AST_POW = 8,
    CUDF_AST_EQUAL = 9,
    CUDF_AST_NULL_EQUAL = 10,
    CUDF_AST_NOT_EQUAL = 11,
    CUDF_AST_LESS = 12,
    CUDF_AST_GREATER = 13,
    CUDF_AST_LESS_EQUAL = 14,
    CUDF_AST_GREATER_EQUAL = 15,
    CUDF_AST_BITWISE_AND = 16,
    CUDF_AST_BITWISE_OR = 17,
    CUDF_AST_BITWISE_XOR = 18,
    CUDF_AST_LOGICAL_AND = 19,
    CUDF_AST_NULL_LOGICAL_AND = 20,
    CUDF_AST_LOGICAL_OR = 21,
    CUDF_AST_NULL_LOGICAL_OR = 22,
    CUDF_AST_IDENTITY = 23,
    CUDF_AST_IS_NULL = 24,
    CUDF_AST_SIN = 25,
    CUDF_AST_COS = 26,
    CUDF_AST_TAN = 27,
    CUDF_AST_ARCSIN = 28,
    CUDF_AST_ARCCOS = 29,
    CUDF_AST_ARCTAN = 30,
    CUDF_AST_SINH = 31,
    CUDF_AST_COSH = 32,
    CUDF_AST_TANH = 33,
    CUDF_AST_ARCSINH = 34,
    CUDF_AST_ARCCOSH = 35,
    CUDF_AST_ARCTANH = 36,
    CUDF_AST_EXP = 37,
    CUDF_AST_LOG = 38,
    CUDF_AST_SQRT = 39,
    CUDF_AST_CBRT = 40,
    CUDF_AST_CEIL = 41,
    CUDF_AST_FLOOR = 42,
    CUDF_AST_ABS = 43,
    CUDF_AST_RINT = 44,
    CUDF_AST_BIT_INVERT = 45,
    CUDF_AST_NOT = 46,
    CUDF_AST_CAST_TO_INT64 = 47,
    CUDF_AST_CAST_TO_UINT64 = 48,
    CUDF_AST_CAST_TO_FLOAT64 = 49,
}

// ---------------------------------------------------------------------------
// Type IDs (mirrors cudf::type_id)
//
// FFI enums use `#[repr(i32)]` to make the Rust-side ABI explicitly 32-bit.
// The C/C++ shim must preserve that contract and validates enum sizes and
// discriminants in `native/src/abi_checks.cpp`. Update those checks whenever
// changing an FFI enum.
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_type_id_t {
    CUDF_TYPE_EMPTY = 0,
    CUDF_TYPE_INT8,
    CUDF_TYPE_INT16,
    CUDF_TYPE_INT32,
    CUDF_TYPE_INT64,
    CUDF_TYPE_UINT8,
    CUDF_TYPE_UINT16,
    CUDF_TYPE_UINT32,
    CUDF_TYPE_UINT64,
    CUDF_TYPE_FLOAT32,
    CUDF_TYPE_FLOAT64,
    CUDF_TYPE_BOOL8,
    CUDF_TYPE_TIMESTAMP_DAYS,
    CUDF_TYPE_TIMESTAMP_SECONDS,
    CUDF_TYPE_TIMESTAMP_MILLISECONDS,
    CUDF_TYPE_TIMESTAMP_MICROSECONDS,
    CUDF_TYPE_TIMESTAMP_NANOSECONDS,
    CUDF_TYPE_DURATION_DAYS,
    CUDF_TYPE_DURATION_SECONDS,
    CUDF_TYPE_DURATION_MILLISECONDS,
    CUDF_TYPE_DURATION_MICROSECONDS,
    CUDF_TYPE_DURATION_NANOSECONDS,
    CUDF_TYPE_DICTIONARY32,
    CUDF_TYPE_STRING,
    CUDF_TYPE_LIST,
    CUDF_TYPE_DECIMAL32,
    CUDF_TYPE_DECIMAL64,
    CUDF_TYPE_DECIMAL128,
    CUDF_TYPE_STRUCT,
}

// ---------------------------------------------------------------------------
// Sorting enums (mirrors cudf::order, cudf::null_order)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_order_t {
    CUDF_ORDER_ASCENDING = 0,
    CUDF_ORDER_DESCENDING = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_null_order_t {
    CUDF_NULL_ORDER_AFTER = 0,
    CUDF_NULL_ORDER_BEFORE = 1,
}

// ---------------------------------------------------------------------------
// Groupby enums (mirrors cudf::aggregation::Kind, cudf::null_policy)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_aggregation_kind_t {
    CUDF_AGG_SUM = 0,
    CUDF_AGG_MIN = 1,
    CUDF_AGG_MAX = 2,
    CUDF_AGG_COUNT_VALID = 3,
    CUDF_AGG_COUNT_ALL = 4,
    CUDF_AGG_MEAN = 5,
    CUDF_AGG_MEDIAN = 6,
    CUDF_AGG_NUNIQUE = 7,
    CUDF_AGG_STD = 8,
    CUDF_AGG_VARIANCE = 9,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_null_policy_t {
    CUDF_NULL_POLICY_EXCLUDE = 0,
    CUDF_NULL_POLICY_INCLUDE = 1,
}

// ---------------------------------------------------------------------------
// Join enums (mirrors cudf::null_equality)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_null_equality_t {
    CUDF_NULL_EQUALITY_EQUAL = 0,
    CUDF_NULL_EQUALITY_UNEQUAL = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_nullable_join_t {
    CUDF_NULLABLE_JOIN_YES = 0,
    CUDF_NULLABLE_JOIN_NO = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_join_kind_t {
    CUDF_JOIN_KIND_INNER_JOIN = 0,
    CUDF_JOIN_KIND_LEFT_JOIN = 1,
    CUDF_JOIN_KIND_FULL_JOIN = 2,
    CUDF_JOIN_KIND_LEFT_SEMI_JOIN = 3,
    CUDF_JOIN_KIND_LEFT_ANTI_JOIN = 4,
}

// ---------------------------------------------------------------------------
// Unary enums (mirrors cudf::unary_operator)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_unary_operator_t {
    CUDF_UNARY_SIN = 0,
    CUDF_UNARY_COS = 1,
    CUDF_UNARY_TAN = 2,
    CUDF_UNARY_ARCSIN = 3,
    CUDF_UNARY_ARCCOS = 4,
    CUDF_UNARY_ARCTAN = 5,
    CUDF_UNARY_SINH = 6,
    CUDF_UNARY_COSH = 7,
    CUDF_UNARY_TANH = 8,
    CUDF_UNARY_ARCSINH = 9,
    CUDF_UNARY_ARCCOSH = 10,
    CUDF_UNARY_ARCTANH = 11,
    CUDF_UNARY_EXP = 12,
    CUDF_UNARY_LOG = 13,
    CUDF_UNARY_SQRT = 14,
    CUDF_UNARY_CBRT = 15,
    CUDF_UNARY_CEIL = 16,
    CUDF_UNARY_FLOOR = 17,
    CUDF_UNARY_ABS = 18,
    CUDF_UNARY_RINT = 19,
    CUDF_UNARY_BIT_COUNT = 20,
    CUDF_UNARY_BIT_INVERT = 21,
    CUDF_UNARY_NOT = 22,
    CUDF_UNARY_NEGATE = 23,
}

// ---------------------------------------------------------------------------
// Binary enums (mirrors cudf::binary_operator)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_binary_operator_t {
    CUDF_BINARY_ADD = 0,
    CUDF_BINARY_SUB = 1,
    CUDF_BINARY_MUL = 2,
    CUDF_BINARY_DIV = 3,
    CUDF_BINARY_TRUE_DIV = 4,
    CUDF_BINARY_FLOOR_DIV = 5,
    CUDF_BINARY_MOD = 6,
    CUDF_BINARY_PMOD = 7,
    CUDF_BINARY_PYMOD = 8,
    CUDF_BINARY_POW = 9,
    CUDF_BINARY_INT_POW = 10,
    CUDF_BINARY_LOG_BASE = 11,
    CUDF_BINARY_ATAN2 = 12,
    CUDF_BINARY_SHIFT_LEFT = 13,
    CUDF_BINARY_SHIFT_RIGHT = 14,
    CUDF_BINARY_SHIFT_RIGHT_UNSIGNED = 15,
    CUDF_BINARY_BITWISE_AND = 16,
    CUDF_BINARY_BITWISE_OR = 17,
    CUDF_BINARY_BITWISE_XOR = 18,
    CUDF_BINARY_LOGICAL_AND = 19,
    CUDF_BINARY_LOGICAL_OR = 20,
    CUDF_BINARY_EQUAL = 21,
    CUDF_BINARY_NOT_EQUAL = 22,
    CUDF_BINARY_LESS = 23,
    CUDF_BINARY_GREATER = 24,
    CUDF_BINARY_LESS_EQUAL = 25,
    CUDF_BINARY_GREATER_EQUAL = 26,
    CUDF_BINARY_NULL_EQUALS = 27,
    CUDF_BINARY_NULL_NOT_EQUALS = 28,
    CUDF_BINARY_NULL_MAX = 29,
    CUDF_BINARY_NULL_MIN = 30,
}

// ---------------------------------------------------------------------------
// Datetime enums
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_datetime_component_t {
    CUDF_DATETIME_COMPONENT_YEAR = 0,
    CUDF_DATETIME_COMPONENT_MONTH = 1,
    CUDF_DATETIME_COMPONENT_DAY = 2,
}

// ---------------------------------------------------------------------------
// Row-predicate filtering nodes
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_row_predicate_node_kind_t {
    CUDF_ROW_PREDICATE_NODE_COLUMN_REF = 0,
    CUDF_ROW_PREDICATE_NODE_LITERAL = 1,
    CUDF_ROW_PREDICATE_NODE_BINARY_OP = 2,
    CUDF_ROW_PREDICATE_NODE_UNARY_OP = 3,
    CUDF_ROW_PREDICATE_NODE_IS_NULL = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_row_predicate_node_t {
    pub kind: cudf_row_predicate_node_kind_t,
    pub lhs_index: i32,
    pub rhs_index: i32,
    pub column_index: i32,
    pub binary_op: cudf_binary_operator_t,
    pub unary_op: cudf_unary_operator_t,
    pub literal: *mut c_void,
}

// ---------------------------------------------------------------------------
// Reduction enums (mirrors cudf::scan_type, reduction aggregation kind)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_reduce_aggregation_kind_t {
    CUDF_REDUCE_AGG_SUM = 0,
    CUDF_REDUCE_AGG_MIN = 1,
    CUDF_REDUCE_AGG_MAX = 2,
    CUDF_REDUCE_AGG_PRODUCT = 3,
    CUDF_REDUCE_AGG_MEAN = 4,
    CUDF_REDUCE_AGG_MEDIAN = 5,
    CUDF_REDUCE_AGG_ANY = 6,
    CUDF_REDUCE_AGG_ALL = 7,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_scan_type_t {
    CUDF_SCAN_INCLUSIVE = 0,
    CUDF_SCAN_EXCLUSIVE = 1,
}

// ---------------------------------------------------------------------------
// Copying enums (mirrors cudf::out_of_bounds_policy)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_out_of_bounds_policy_t {
    CUDF_OUT_OF_BOUNDS_POLICY_NULLIFY = 0,
    CUDF_OUT_OF_BOUNDS_POLICY_DONT_CHECK = 1,
}

// ---------------------------------------------------------------------------
// Stream compaction enums
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_duplicate_keep_option_t {
    CUDF_DUPLICATE_KEEP_OPTION_KEEP_ANY = 0,
    CUDF_DUPLICATE_KEEP_OPTION_KEEP_FIRST = 1,
    CUDF_DUPLICATE_KEEP_OPTION_KEEP_LAST = 2,
    CUDF_DUPLICATE_KEEP_OPTION_KEEP_NONE = 3,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_stream_compaction_null_equality_t {
    CUDF_STREAM_COMPACTION_NULL_EQUALITY_EQUAL = 0,
    CUDF_STREAM_COMPACTION_NULL_EQUALITY_UNEQUAL = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_stream_compaction_nan_equality_t {
    CUDF_STREAM_COMPACTION_NAN_EQUALITY_ALL_EQUAL = 0,
    CUDF_STREAM_COMPACTION_NAN_EQUALITY_UNEQUAL = 1,
}

// ---------------------------------------------------------------------------
// String enums (mirrors cudf::strings::side_type)
// ---------------------------------------------------------------------------

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_strip_side_t {
    CUDF_STRIP_LEFT = 0,
    CUDF_STRIP_RIGHT = 1,
    CUDF_STRIP_BOTH = 2,
}

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

pub type cudf_data_type_handle = *mut c_void;
pub type cudf_column_handle = *mut c_void;
pub type cudf_column_view_handle = *mut c_void;
pub type cudf_table_handle = *mut c_void;
pub type cudf_table_view_handle = *mut c_void;
pub type cudf_scalar_handle = *mut c_void;
pub type cudf_ast_expression_handle = *mut c_void;
pub type cudf_distinct_hash_join_handle = *mut c_void;
pub type cudf_filtered_join_handle = *mut c_void;
pub type cudf_filtered_join_probe_handle = *mut c_void;
pub type cudf_hash_join_handle = *mut c_void;
pub type cudf_conditional_full_join_size_data_handle = *mut c_void;
pub type cudf_mixed_join_size_data_handle = *mut c_void;
pub type cudf_device_buffer_handle = *mut c_void;
pub type cudf_stream_handle = *mut c_void;
pub type cudf_event_handle = *mut c_void;
pub type cudf_memory_resource_handle = *mut c_void;
pub type cudf_arrow_schema_handle = *mut c_void;
pub type cudf_arrow_device_array_handle = *mut c_void;

pub const CUDF_MEMORY_RESOURCE_USAGE_RESERVED_SLOTS: usize = 8;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct cudf_memory_resource_usage_t {
    pub struct_size: usize,
    pub allocation_limit_bytes: u64,
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub total_bytes: u64,
    pub current_aligned_bytes: u64,
    pub peak_aligned_bytes: u64,
    pub total_allocations: u64,
    pub current_allocations: u64,
    pub peak_allocations: u64,
    pub failed_allocation_count: u64,
    pub largest_failed_allocation_bytes: u64,
    pub reserved: [u64; CUDF_MEMORY_RESOURCE_USAGE_RESERVED_SLOTS],
}

impl cudf_memory_resource_usage_t {
    pub const fn defaults() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            allocation_limit_bytes: 0,
            current_bytes: 0,
            peak_bytes: 0,
            total_bytes: 0,
            current_aligned_bytes: 0,
            peak_aligned_bytes: 0,
            total_allocations: 0,
            current_allocations: 0,
            peak_allocations: 0,
            failed_allocation_count: 0,
            largest_failed_allocation_bytes: 0,
            reserved: [0; CUDF_MEMORY_RESOURCE_USAGE_RESERVED_SLOTS],
        }
    }
}

impl Default for cudf_memory_resource_usage_t {
    fn default() -> Self {
        Self::defaults()
    }
}

pub const CUDF_ARROW_DEVICE_TYPE_CPU: i32 = 1;
pub const CUDF_ARROW_DEVICE_TYPE_CUDA: i32 = 2;
pub const CUDF_ARROW_DEVICE_TYPE_CUDA_HOST: i32 = 3;
pub const CUDF_ARROW_DEVICE_TYPE_CUDA_MANAGED: i32 = 13;

// ---------------------------------------------------------------------------
// Execution options ABI
// ---------------------------------------------------------------------------

pub const CUDF_EXEC_OPTIONS_RESERVED_SLOTS: usize = 4;

/// Bit 0 of `flags`: when set, `device` was explicitly provided.
pub const CUDF_EXEC_FLAG_DEVICE_EXPLICIT: u32 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_exec_options_t {
    pub struct_size: usize,
    pub device: i32,
    pub flags: u32,
    pub stream: cudf_stream_handle,
    pub mr: cudf_memory_resource_handle,
    pub reserved: [u64; CUDF_EXEC_OPTIONS_RESERVED_SLOTS],
}

impl cudf_exec_options_t {
    pub const fn defaults() -> Self {
        Self {
            struct_size: std::mem::size_of::<Self>(),
            device: -1,
            flags: 0,
            stream: std::ptr::null_mut(),
            mr: std::ptr::null_mut(),
            reserved: [0; CUDF_EXEC_OPTIONS_RESERVED_SLOTS],
        }
    }
}

impl Default for cudf_exec_options_t {
    fn default() -> Self {
        Self::defaults()
    }
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_interop_placement_kind_t {
    CUDF_INTEROP_PLACEMENT_HOST = 0,
    CUDF_INTEROP_PLACEMENT_CUDA_DEVICE = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_interop_validity_kind_t {
    CUDF_INTEROP_VALIDITY_NONE = 0,
    CUDF_INTEROP_VALIDITY_MASK = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_interop_layout_kind_t {
    CUDF_INTEROP_LAYOUT_FIXED_WIDTH = 0,
    CUDF_INTEROP_LAYOUT_BINARY_LIKE = 1,
    CUDF_INTEROP_LAYOUT_LIST = 2,
    CUDF_INTEROP_LAYOUT_STRUCT = 3,
    CUDF_INTEROP_LAYOUT_DICTIONARY = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_interop_buffer_binding_t {
    pub buffer_id: u32,
    pub data: *const c_void,
    pub byte_len: usize,
    pub placement_kind: cudf_interop_placement_kind_t,
    pub device: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_interop_semantic_column_t {
    pub column_id: u32,
    pub type_id: cudf_type_id_t,
    pub scale: i32,
    pub len: i32,
    pub offset: i32,
    pub validity_kind: cudf_interop_validity_kind_t,
    pub validity_buffer_id: u32,
    pub null_count: i32,
    pub layout_kind: cudf_interop_layout_kind_t,
    pub values_buffer_id: u32,
    pub offsets_buffer_id: u32,
    pub data_buffer_id: u32,
    pub indices_buffer_id: u32,
    pub child_column_id: u32,
    pub child_column_ids: *const u32,
    pub num_child_column_ids: i32,
    pub keys_column_id: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct cudf_interop_semantic_table_t {
    pub num_rows: i32,
    pub column_ids: *const u32,
    pub num_columns: i32,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_arrow_device_export_materialization_t {
    CUDF_ARROW_DEVICE_EXPORT_PRODUCER_RETAINED_VIEW = 0,
    CUDF_ARROW_DEVICE_EXPORT_BRIDGE_OWNED_DEVICE_COPY = 1,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum cudf_arrow_device_import_materialization_t {
    CUDF_ARROW_DEVICE_IMPORT_FOREIGN_RETAINED_VIEW = 0,
    CUDF_ARROW_DEVICE_IMPORT_BRIDGE_OWNED_DEVICE_COPY = 1,
}
