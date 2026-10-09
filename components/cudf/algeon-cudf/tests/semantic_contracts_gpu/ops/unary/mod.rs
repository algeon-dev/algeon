use crate::NamedCase;

mod cast_spec_tests;
mod unary_tests;

pub(crate) const TABLES: &[&[NamedCase]] = &[unary_tests::CASES, cast_spec_tests::CASES];
