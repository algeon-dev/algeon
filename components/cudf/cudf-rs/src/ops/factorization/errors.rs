use crate::core::{
    error::{self, Error, FACT_TYPE_ID},
    types::TypeId,
};

pub(super) fn foreign_factorization_row_code_dtype_error(operation: &'static str, actual_type_id: TypeId) -> Error {
    error::foreign_invariant_violation(
        operation,
        "factorization row-code dtype must be Int32",
        "row_codes",
        "unexpected_dtype",
        "ensure the factorization backend emits Int32 row codes",
    )
    .with_fact(FACT_TYPE_ID, actual_type_id)
}
