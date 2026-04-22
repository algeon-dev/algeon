#pragma once

#include "io.h"

cudf_test_io_metadata_fault_t cudf_test_io_take_next_metadata_fault();
cudf_status_t cudf_test_io_apply_metadata_fault(cudf_test_io_metadata_fault_t fault,
                                                char*** out_names,
                                                int32_t* out_num_names);
