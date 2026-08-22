//! Reader-predicate validation and semantics-preserving rewrites.

mod rewrite;
mod validate;

pub(super) use rewrite::rewrite_fixed_prefix_substring_reader_predicate;
pub(super) use validate::{
    reader_predicate_expr_is_supported, reader_predicate_references_decimal_column, reader_predicate_root_is_supported,
    reader_predicate_source_contract_shape_is_supported,
};
