use crate::{
    cudf_ast_expression_handle, cudf_ast_operator_t, cudf_ast_table_reference_t, cudf_scalar_handle, cudf_status_t,
};

unsafe extern "C" {
    pub fn cudf_ast_column_reference_create(
        column_index: i32,
        table_source: cudf_ast_table_reference_t,
        out: *mut cudf_ast_expression_handle,
    ) -> cudf_status_t;

    pub fn cudf_ast_literal_create(scalar: cudf_scalar_handle, out: *mut cudf_ast_expression_handle) -> cudf_status_t;

    pub fn cudf_ast_operation_unary_create(
        op: cudf_ast_operator_t,
        input: cudf_ast_expression_handle,
        out: *mut cudf_ast_expression_handle,
    ) -> cudf_status_t;

    pub fn cudf_ast_operation_binary_create(
        op: cudf_ast_operator_t,
        left: cudf_ast_expression_handle,
        right: cudf_ast_expression_handle,
        out: *mut cudf_ast_expression_handle,
    ) -> cudf_status_t;

    pub fn cudf_ast_expression_destroy(expr: cudf_ast_expression_handle);
}
