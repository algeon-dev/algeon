use crate::NamedCase;

mod expr_tests;
mod factorize_tests;
mod partitioning_tests;
mod replace_tests;
mod search_tests;

pub(crate) const TABLES: &[&[NamedCase]] =
    &[partitioning_tests::CASES, search_tests::CASES, expr_tests::CASES, replace_tests::CASES, factorize_tests::CASES];
