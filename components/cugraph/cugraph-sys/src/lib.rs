//! Raw FFI bindings to the cuGraph C API.
//!
//! This crate provides auto-generated bindings via `bindgen`. Users should
//! prefer the safe `cugraph` crate instead of using these bindings directly.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_values() {
        assert_eq!(cugraph_error_code_t::CUGRAPH_SUCCESS.0, 0);
        assert_eq!(cugraph_error_code_t::CUGRAPH_UNKNOWN_ERROR.0, 1);
        assert_eq!(cugraph_error_code_t::CUGRAPH_INVALID_HANDLE.0, 2);
        assert_eq!(cugraph_error_code_t::CUGRAPH_ALLOC_ERROR.0, 3);
        assert_eq!(cugraph_error_code_t::CUGRAPH_INVALID_INPUT.0, 4);
        assert_eq!(cugraph_error_code_t::CUGRAPH_NOT_IMPLEMENTED.0, 5);
        assert_eq!(cugraph_error_code_t::CUGRAPH_UNSUPPORTED_TYPE_COMBINATION.0, 6);
    }

    #[test]
    fn test_bool_values() {
        assert_eq!(bool_t::FALSE.0, 0);
        assert_eq!(bool_t::TRUE.0, 1);
    }

    #[test]
    fn test_data_type_values() {
        assert_eq!(cugraph_data_type_id_t::INT32.0, 2);
        assert_eq!(cugraph_data_type_id_t::INT64.0, 3);
        assert_eq!(cugraph_data_type_id_t::FLOAT32.0, 8);
        assert_eq!(cugraph_data_type_id_t::FLOAT64.0, 9);
    }

    #[test]
    fn test_unknown_error_code_is_representable() {
        let unknown = cugraph_error_code_t(999);
        assert_ne!(unknown, cugraph_error_code_t::CUGRAPH_SUCCESS);
        assert_eq!(unknown.0, 999);
    }
}
