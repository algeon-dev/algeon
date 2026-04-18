const PREPARED_IMPORT_CPP: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/native/src/prepared_import.cpp"));

#[test]
fn prepared_import_uses_2604_lists_factory_signature() {
    assert!(
        PREPARED_IMPORT_CPP.contains("return cudf::make_lists_column(\n          size, std::move(offsets_col), std::move(child), null_count, std::move(null_mask));"),
        "expected prepared_import.cpp to use the cuDF 26.04 five-argument lists factory"
    );
    assert!(
        !PREPARED_IMPORT_CPP.contains(
            "return cudf::make_lists_column(\n          size, std::move(offsets_col), std::move(child), null_count, std::move(null_mask), exec.stream, exec.mr);"
        ),
        "prepared_import.cpp still uses the pre-26.04 lists factory overload"
    );
}

#[test]
fn prepared_import_uses_2604_dictionary_factory_signature() {
    assert!(
        PREPARED_IMPORT_CPP.contains(
            "return cudf::make_dictionary_column(std::move(keys), std::move(indices), std::move(null_mask), null_count);"
        ),
        "expected prepared_import.cpp to use the cuDF 26.04 four-argument dictionary factory"
    );
    assert!(
        !PREPARED_IMPORT_CPP.contains(
            "return cudf::make_dictionary_column(\n          std::move(keys), std::move(indices), std::move(null_mask), null_count, exec.stream, exec.mr);"
        ),
        "prepared_import.cpp still uses the pre-26.04 dictionary factory overload"
    );
}
