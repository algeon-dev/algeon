use crate::NamedCase;

mod column_ops_tests;
mod column_string_tests;
mod column_tests;
mod device_buffer_tests;
mod error_path_tests;
mod matrix_tests;
mod no_panic_contract_tests;
mod scalar_tests;
mod table_release_tests;
mod table_tests;

pub(crate) const TABLES: &[&[NamedCase]] = &[
    table_release_tests::CASES,
    column_tests::CASES,
    matrix_tests::CASES,
    scalar_tests::CASES,
    no_panic_contract_tests::CASES,
    column_ops_tests::CASES,
    device_buffer_tests::CASES,
    column_string_tests::CASES,
    table_tests::CASES,
    error_path_tests::CASES,
];
