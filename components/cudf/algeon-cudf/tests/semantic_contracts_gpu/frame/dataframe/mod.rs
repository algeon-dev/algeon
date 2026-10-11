use crate::NamedCase;

mod arc_sharing_tests;
mod arrow_tests;
mod basics_tests;
mod column_ops_tests;
mod concat_tests;
mod construction_contract_tests;
mod gather_tests;
mod high_level_d3_tests;
mod high_level_entrypoint_contract_tests;
mod info_tests;
mod metadata_ops_tests;
mod series_bridge_tests;
mod slice_tests;

pub(crate) const TABLES: &[&[NamedCase]] = &[
    gather_tests::CASES,
    series_bridge_tests::CASES,
    slice_tests::CASES,
    arc_sharing_tests::CASES,
    basics_tests::CASES,
    concat_tests::CASES,
    arrow_tests::CASES,
    metadata_ops_tests::CASES,
    info_tests::CASES,
    column_ops_tests::CASES,
    high_level_d3_tests::CASES,
    high_level_entrypoint_contract_tests::CASES,
    construction_contract_tests::CASES,
];
