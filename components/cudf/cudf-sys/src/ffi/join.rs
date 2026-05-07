use crate::{
    cudf_ast_expression_handle, cudf_column_handle, cudf_column_view_handle,
    cudf_conditional_full_join_size_data_handle, cudf_distinct_hash_join_handle, cudf_exec_options_t,
    cudf_filtered_join_handle, cudf_filtered_join_probe_handle, cudf_hash_join_handle, cudf_join_kind_t,
    cudf_mixed_join_size_data_handle, cudf_null_equality_t, cudf_nullable_join_t, cudf_status_t, cudf_table_handle,
    cudf_table_view_handle,
};

extern "C" {
    // -- Join ----------------------------------------------------------------

    pub fn cudf_inner_join(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Inner join with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive both returned tables.
    pub fn cudf_inner_join_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_left_join(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Left join with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive both returned tables.
    pub fn cudf_left_join_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_full_join(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Full join with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive both returned tables.
    pub fn cudf_full_join_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        left_payload: cudf_table_view_handle,
        right_payload: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left: *mut cudf_table_handle,
        out_right: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_cross_join(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    /// Cross join with explicit execution context.
    ///
    /// When `options->mr` is non-null, it must outlive the returned table.
    pub fn cudf_cross_join_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_projected_cross_join(
        left_selected: cudf_table_view_handle,
        right_selected: cudf_table_view_handle,
        left_num_rows: usize,
        right_num_rows: usize,
        out: *mut cudf_table_handle,
    ) -> cudf_status_t;

    pub fn cudf_projected_cross_join_ex(
        left_selected: cudf_table_view_handle,
        right_selected: cudf_table_view_handle,
        left_num_rows: usize,
        right_num_rows: usize,
        out: *mut cudf_table_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_projected_cross_join_device_allocated_size_ex(
        left_selected: cudf_table_view_handle,
        right_selected: cudf_table_view_handle,
        left_num_rows: usize,
        right_num_rows: usize,
        out: *mut usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_inner_join_indices(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_inner_join_indices_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_left_join_indices(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_left_join_indices_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_full_join_indices(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_full_join_indices_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_left_semi_join_indices(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_left_semi_join_indices_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_left_anti_join_indices(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_left_anti_join_indices_ex(
        left_keys: cudf_table_view_handle,
        right_keys: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_inner_join_indices(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Conditional inner join indices with explicit execution context.
    ///
    /// Passing null `options`, or leaving `options->stream` / `options->mr`
    /// null, uses cuDF's default stream and current device memory resource.
    /// When `options->mr` is non-null, it must outlive both returned index
    /// columns.
    pub fn cudf_conditional_inner_join_indices_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_inner_join_indices_with_output_size(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        output_size: usize,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_conditional_inner_join_indices_with_output_size_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        output_size: usize,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_inner_join_size(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_conditional_inner_join_size_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out: *mut usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_join_indices(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Conditional left join indices with explicit execution context.
    ///
    /// Passing null `options`, or leaving `options->stream` / `options->mr`
    /// null, uses cuDF's default stream and current device memory resource.
    /// When `options->mr` is non-null, it must outlive both returned index
    /// columns.
    pub fn cudf_conditional_left_join_indices_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_join_indices_with_output_size(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        output_size: usize,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_join_indices_with_output_size_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        output_size: usize,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_join_size(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_join_size_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out: *mut usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_full_join_indices(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Conditional full join indices with explicit execution context.
    ///
    /// Passing null `options`, or leaving `options->stream` / `options->mr`
    /// null, uses cuDF's default stream and current device memory resource.
    /// When `options->mr` is non-null, it must outlive both returned index
    /// columns.
    pub fn cudf_conditional_full_join_indices_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_full_join_begin_size_data(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out: *mut cudf_conditional_full_join_size_data_handle,
    ) -> cudf_status_t;

    pub fn cudf_conditional_full_join_begin_size_data_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out: *mut cudf_conditional_full_join_size_data_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_full_join_size_data_destroy(handle: cudf_conditional_full_join_size_data_handle);

    pub fn cudf_conditional_full_join_size_data_output_size(
        handle: cudf_conditional_full_join_size_data_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_conditional_full_join_size_data_device_allocated_size(
        handle: cudf_conditional_full_join_size_data_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_conditional_full_join_size_data_gather_map_size(
        handle: cudf_conditional_full_join_size_data_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_conditional_full_join_size_data_materialize_indices(
        size_data: cudf_conditional_full_join_size_data_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_conditional_full_join_size_data_materialize_indices_ex(
        size_data: cudf_conditional_full_join_size_data_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_semi_join_indices(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Conditional left semi join indices with explicit execution context.
    ///
    /// Passing null `options`, or leaving `options->stream` / `options->mr`
    /// null, uses cuDF's default stream and current device memory resource.
    /// When `options->mr` is non-null, it must outlive the returned index
    /// column.
    pub fn cudf_conditional_left_semi_join_indices_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_semi_join_indices_with_output_size(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        output_size: usize,
        out_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_semi_join_indices_with_output_size_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        output_size: usize,
        out_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_semi_join_size(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_semi_join_size_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out: *mut usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_anti_join_indices(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Conditional left anti join indices with explicit execution context.
    ///
    /// Passing null `options`, or leaving `options->stream` / `options->mr`
    /// null, uses cuDF's default stream and current device memory resource.
    /// When `options->mr` is non-null, it must outlive the returned index
    /// column.
    pub fn cudf_conditional_left_anti_join_indices_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_anti_join_indices_with_output_size(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        output_size: usize,
        out_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_anti_join_indices_with_output_size_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        output_size: usize,
        out_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_anti_join_size(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_conditional_left_anti_join_size_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        out: *mut usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_filter_join_indices(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        left_indices: cudf_column_view_handle,
        right_indices: cudf_column_view_handle,
        predicate: cudf_ast_expression_handle,
        join_kind: cudf_join_kind_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Filter already-computed join-index pairs with explicit execution
    /// context. Only inner, left, and full join kinds are supported.
    pub fn cudf_filter_join_indices_ex(
        left: cudf_table_view_handle,
        right: cudf_table_view_handle,
        left_indices: cudf_column_view_handle,
        right_indices: cudf_column_view_handle,
        predicate: cudf_ast_expression_handle,
        join_kind: cudf_join_kind_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_mixed_inner_join_indices(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Mixed inner join indices with explicit execution context.
    ///
    /// Passing null `options`, or leaving `options->stream` / `options->mr`
    /// null, uses cuDF's default stream and current device memory resource.
    /// When `options->mr` is non-null, it must outlive both returned index
    /// columns.
    pub fn cudf_mixed_inner_join_indices_ex(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_mixed_left_join_indices(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Mixed left join indices with explicit execution context.
    ///
    /// Passing null `options`, or leaving `options->stream` / `options->mr`
    /// null, uses cuDF's default stream and current device memory resource.
    /// When `options->mr` is non-null, it must outlive both returned index
    /// columns.
    pub fn cudf_mixed_left_join_indices_ex(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_mixed_full_join_indices(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Mixed full join indices with explicit execution context.
    ///
    /// Passing null `options`, or leaving `options->stream` / `options->mr`
    /// null, uses cuDF's default stream and current device memory resource.
    /// When `options->mr` is non-null, it must outlive both returned index
    /// columns.
    pub fn cudf_mixed_full_join_indices_ex(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_mixed_left_semi_join_indices(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Mixed left semi join indices with explicit execution context.
    ///
    /// Passing null `options`, or leaving `options->stream` / `options->mr`
    /// null, uses cuDF's default stream and current device memory resource.
    /// When `options->mr` is non-null, it must outlive the returned index
    /// column.
    pub fn cudf_mixed_left_semi_join_indices_ex(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_mixed_left_anti_join_indices(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    /// Mixed left anti join indices with explicit execution context.
    ///
    /// Passing null `options`, or leaving `options->stream` / `options->mr`
    /// null, uses cuDF's default stream and current device memory resource.
    /// When `options->mr` is non-null, it must outlive the returned index
    /// column.
    pub fn cudf_mixed_left_anti_join_indices_ex(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_mixed_inner_join_begin_size_data(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out: *mut cudf_mixed_join_size_data_handle,
    ) -> cudf_status_t;

    pub fn cudf_mixed_inner_join_begin_size_data_ex(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out: *mut cudf_mixed_join_size_data_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_mixed_left_join_begin_size_data(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out: *mut cudf_mixed_join_size_data_handle,
    ) -> cudf_status_t;

    pub fn cudf_mixed_left_join_begin_size_data_ex(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out: *mut cudf_mixed_join_size_data_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_mixed_full_join_begin_size_data(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out: *mut cudf_mixed_join_size_data_handle,
    ) -> cudf_status_t;

    pub fn cudf_mixed_full_join_begin_size_data_ex(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out: *mut cudf_mixed_join_size_data_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_mixed_left_semi_join_begin_size_data(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out: *mut cudf_mixed_join_size_data_handle,
    ) -> cudf_status_t;

    pub fn cudf_mixed_left_semi_join_begin_size_data_ex(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out: *mut cudf_mixed_join_size_data_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_mixed_left_anti_join_begin_size_data(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out: *mut cudf_mixed_join_size_data_handle,
    ) -> cudf_status_t;

    pub fn cudf_mixed_left_anti_join_begin_size_data_ex(
        left_equality: cudf_table_view_handle,
        right_equality: cudf_table_view_handle,
        left_conditional: cudf_table_view_handle,
        right_conditional: cudf_table_view_handle,
        predicate: cudf_ast_expression_handle,
        null_equality: cudf_null_equality_t,
        out: *mut cudf_mixed_join_size_data_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_mixed_join_size_data_destroy(handle: cudf_mixed_join_size_data_handle);

    pub fn cudf_mixed_join_size_data_output_size(
        handle: cudf_mixed_join_size_data_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_mixed_join_size_data_device_allocated_size(
        handle: cudf_mixed_join_size_data_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_mixed_join_size_data_gather_map_size(
        handle: cudf_mixed_join_size_data_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_mixed_join_size_data_materialize_indices(
        size_data: cudf_mixed_join_size_data_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_mixed_join_size_data_materialize_indices_ex(
        size_data: cudf_mixed_join_size_data_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_mixed_join_size_data_materialize_single_indices(
        size_data: cudf_mixed_join_size_data_handle,
        out_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_mixed_join_size_data_materialize_single_indices_ex(
        size_data: cudf_mixed_join_size_data_handle,
        out_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_distinct_hash_join_create(
        build: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out: *mut cudf_distinct_hash_join_handle,
    ) -> cudf_status_t;

    /// Create a distinct hash-join build artifact with explicit execution
    /// context. When `options->mr` is non-null, it must outlive the returned
    /// join handle.
    pub fn cudf_distinct_hash_join_create_ex(
        build: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        out: *mut cudf_distinct_hash_join_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_distinct_hash_join_destroy(handle: cudf_distinct_hash_join_handle);

    pub fn cudf_distinct_hash_join_left_join(
        join: cudf_distinct_hash_join_handle,
        probe: cudf_table_view_handle,
        out: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_distinct_hash_join_left_join_ex(
        join: cudf_distinct_hash_join_handle,
        probe: cudf_table_view_handle,
        out: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_create(
        build: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        load_factor: f64,
        out: *mut cudf_filtered_join_handle,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_create_ex(
        build: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        load_factor: f64,
        out: *mut cudf_filtered_join_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_pre_build_reservation_size(
        build: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        load_factor: f64,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_pre_build_reservation_size_ex(
        build: cudf_table_view_handle,
        null_equality: cudf_null_equality_t,
        load_factor: f64,
        out: *mut usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_destroy(handle: cudf_filtered_join_handle);

    pub fn cudf_filtered_join_device_allocated_size(
        handle: cudf_filtered_join_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_begin_left_semi_probe(
        join: cudf_filtered_join_handle,
        probe: cudf_table_view_handle,
        out: *mut cudf_filtered_join_probe_handle,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_begin_left_semi_probe_ex(
        join: cudf_filtered_join_handle,
        probe: cudf_table_view_handle,
        out: *mut cudf_filtered_join_probe_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_begin_left_anti_probe(
        join: cudf_filtered_join_handle,
        probe: cudf_table_view_handle,
        out: *mut cudf_filtered_join_probe_handle,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_begin_left_anti_probe_ex(
        join: cudf_filtered_join_handle,
        probe: cudf_table_view_handle,
        out: *mut cudf_filtered_join_probe_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_probe_destroy(handle: cudf_filtered_join_probe_handle);

    pub fn cudf_filtered_join_probe_output_size(
        handle: cudf_filtered_join_probe_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_probe_device_allocated_size(
        handle: cudf_filtered_join_probe_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_probe_materialize_indices(
        handle: cudf_filtered_join_probe_handle,
        out_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_probe_materialize_indices_ex(
        handle: cudf_filtered_join_probe_handle,
        out_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_left_semi_join_size(
        join: cudf_filtered_join_handle,
        probe: cudf_table_view_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_left_semi_join_size_ex(
        join: cudf_filtered_join_handle,
        probe: cudf_table_view_handle,
        out: *mut usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_left_anti_join_size(
        join: cudf_filtered_join_handle,
        probe: cudf_table_view_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_left_anti_join_size_ex(
        join: cudf_filtered_join_handle,
        probe: cudf_table_view_handle,
        out: *mut usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_left_semi_join_with_output_size(
        join: cudf_filtered_join_handle,
        probe: cudf_table_view_handle,
        output_size: usize,
        out_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_left_semi_join_with_output_size_ex(
        join: cudf_filtered_join_handle,
        probe: cudf_table_view_handle,
        output_size: usize,
        out_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_left_anti_join_with_output_size(
        join: cudf_filtered_join_handle,
        probe: cudf_table_view_handle,
        output_size: usize,
        out_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_filtered_join_left_anti_join_with_output_size_ex(
        join: cudf_filtered_join_handle,
        probe: cudf_table_view_handle,
        output_size: usize,
        out_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_create(
        build: cudf_table_view_handle,
        nullable_join: cudf_nullable_join_t,
        null_equality: cudf_null_equality_t,
        load_factor: f64,
        out: *mut cudf_hash_join_handle,
    ) -> cudf_status_t;

    /// Create a hash-join build artifact with explicit execution context. When
    /// `options->mr` is non-null, it must outlive the returned join handle.
    pub fn cudf_hash_join_create_ex(
        build: cudf_table_view_handle,
        nullable_join: cudf_nullable_join_t,
        null_equality: cudf_null_equality_t,
        load_factor: f64,
        out: *mut cudf_hash_join_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_pre_build_reservation_size(
        build: cudf_table_view_handle,
        load_factor: f64,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_pre_build_reservation_size_ex(
        build: cudf_table_view_handle,
        load_factor: f64,
        out: *mut usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_destroy(handle: cudf_hash_join_handle);

    pub fn cudf_hash_join_device_allocated_size(handle: cudf_hash_join_handle, out: *mut usize) -> cudf_status_t;

    pub fn cudf_hash_join_inner_join(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_inner_join_ex(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_inner_join_with_output_size(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        output_size: usize,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_inner_join_with_output_size_ex(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        output_size: usize,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_inner_join_size(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_inner_join_size_ex(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        out: *mut usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_left_join(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_left_join_ex(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_left_join_with_output_size(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        output_size: usize,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_left_join_with_output_size_ex(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        output_size: usize,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_left_join_size(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_left_join_size_ex(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        out: *mut usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_full_join(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_full_join_ex(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_full_join_with_output_size(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        output_size: usize,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_full_join_with_output_size_ex(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        output_size: usize,
        out_left_indices: *mut cudf_column_handle,
        out_right_indices: *mut cudf_column_handle,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_full_join_size(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        out: *mut usize,
    ) -> cudf_status_t;

    pub fn cudf_hash_join_full_join_size_ex(
        join: cudf_hash_join_handle,
        probe: cudf_table_view_handle,
        out: *mut usize,
        options: *const cudf_exec_options_t,
    ) -> cudf_status_t;
}
