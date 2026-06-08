mod source_contract_support;

const PREPARED_IMPORT_CPP: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/prepared_import.cpp"));

#[test]
fn prepared_import_uses_2606_owning_lists_factory_signature() {
    assert!(
        source_contract_support::contains_normalized(
            PREPARED_IMPORT_CPP,
            "return cudf::make_lists_column(
              size, std::move(offsets_col), std::move(child), null_count, std::move(null_mask));"
        ),
        "expected prepared_import.cpp to use the cuDF 26.06 five-argument owning lists factory"
    );
    assert!(
        !source_contract_support::contains_normalized(
            PREPARED_IMPORT_CPP,
            "return cudf::make_lists_column(\n          size, std::move(offsets_col), std::move(child), null_count, std::move(null_mask), exec.stream, exec.mr);"
        ),
        "prepared_import.cpp must not assume an owning lists factory overload with stream/MR arguments"
    );
}

#[test]
fn prepared_import_uses_2606_owning_dictionary_factory_signature() {
    assert!(
        source_contract_support::contains_normalized(
            PREPARED_IMPORT_CPP,
            "return cudf::make_dictionary_column(std::move(keys), std::move(indices), std::move(null_mask), null_count);"
        ),
        "expected prepared_import.cpp to use the cuDF 26.06 four-argument owning dictionary factory with null-mask ownership"
    );
    assert!(
        !source_contract_support::contains_normalized(
            PREPARED_IMPORT_CPP,
            "return cudf::make_dictionary_column(\n          std::move(keys), std::move(indices), std::move(null_mask), null_count, exec.stream, exec.mr);"
        ),
        "prepared_import.cpp must not assume an owning dictionary factory overload that combines null-mask ownership with stream/MR arguments"
    );
}
