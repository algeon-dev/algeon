#include "binary.h"
#include "copying.h"
#include "groupby.h"
#include "join.h"
#include "reduction.h"
#include "sorting.h"
#include "stream_compaction.h"
#include "types.h"
#include "unary.h"

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/reduction.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>

#include <cstdint>
#include <type_traits>

static_assert(sizeof(cudf_status_t) == sizeof(int32_t), "cudf_status_t must be 32-bit");
static_assert(sizeof(cudf_type_id_t) == sizeof(int32_t), "cudf_type_id_t must be 32-bit");
static_assert(sizeof(cudf_order_t) == sizeof(int32_t), "cudf_order_t must be 32-bit");
static_assert(sizeof(cudf_null_order_t) == sizeof(int32_t), "cudf_null_order_t must be 32-bit");
static_assert(sizeof(cudf_aggregation_kind_t) == sizeof(int32_t),
              "cudf_aggregation_kind_t must be 32-bit");
static_assert(sizeof(cudf_null_policy_t) == sizeof(int32_t),
              "cudf_null_policy_t must be 32-bit");
static_assert(sizeof(cudf_null_equality_t) == sizeof(int32_t),
              "cudf_null_equality_t must be 32-bit");
static_assert(sizeof(cudf_unary_operator_t) == sizeof(int32_t),
              "cudf_unary_operator_t must be 32-bit");
static_assert(sizeof(cudf_binary_operator_t) == sizeof(int32_t),
              "cudf_binary_operator_t must be 32-bit");
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
static_assert(CUDF_TYPE_STRUCT == static_cast<int>(cudf::type_id::STRUCT));

static_assert(CUDF_ORDER_ASCENDING == static_cast<int>(cudf::order::ASCENDING));
static_assert(CUDF_ORDER_DESCENDING == static_cast<int>(cudf::order::DESCENDING));

static_assert(CUDF_NULL_ORDER_AFTER == static_cast<int>(cudf::null_order::AFTER));
static_assert(CUDF_NULL_ORDER_BEFORE == static_cast<int>(cudf::null_order::BEFORE));

static_assert(CUDF_NULL_POLICY_EXCLUDE == static_cast<int>(cudf::null_policy::EXCLUDE));
static_assert(CUDF_NULL_POLICY_INCLUDE == static_cast<int>(cudf::null_policy::INCLUDE));

static_assert(CUDF_NULL_EQUALITY_EQUAL == static_cast<int>(cudf::null_equality::EQUAL));
static_assert(CUDF_NULL_EQUALITY_UNEQUAL == static_cast<int>(cudf::null_equality::UNEQUAL));

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
static_assert(CUDF_UNARY_NEGATE == static_cast<int>(cudf::unary_operator::NEGATE));

static_assert(CUDF_BINARY_ADD == static_cast<int>(cudf::binary_operator::ADD));
static_assert(CUDF_BINARY_SUB == static_cast<int>(cudf::binary_operator::SUB));
static_assert(CUDF_BINARY_MUL == static_cast<int>(cudf::binary_operator::MUL));
static_assert(CUDF_BINARY_DIV == static_cast<int>(cudf::binary_operator::DIV));
static_assert(CUDF_BINARY_NULL_MIN == static_cast<int>(cudf::binary_operator::NULL_MIN));
