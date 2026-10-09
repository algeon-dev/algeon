use crate::NamedCase;

mod cast_consistency_tests;
mod high_level_semantic_baseline_tests;
mod shared_row_selection_contract_tests;

pub(crate) const TABLES: &[&[NamedCase]] = &[
    cast_consistency_tests::CASES,
    high_level_semantic_baseline_tests::CASES,
    shared_row_selection_contract_tests::CASES,
];
