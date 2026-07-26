//! Raw FFI bindings to the cuGraph C API.
//!
//! This crate provides auto-generated bindings via `bindgen`. Users should
//! prefer the safe `cugraph` crate instead of using these bindings directly.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_values() {
        assert_eq!(cugraph_error_code_t::CUGRAPH_SUCCESS as u32, 0);
        assert_eq!(cugraph_error_code_t::CUGRAPH_UNKNOWN_ERROR as u32, 1);
        assert_eq!(cugraph_error_code_t::CUGRAPH_INVALID_HANDLE as u32, 2);
        assert_eq!(cugraph_error_code_t::CUGRAPH_ALLOC_ERROR as u32, 3);
        assert_eq!(cugraph_error_code_t::CUGRAPH_INVALID_INPUT as u32, 4);
        assert_eq!(cugraph_error_code_t::CUGRAPH_NOT_IMPLEMENTED as u32, 5);
        assert_eq!(
            cugraph_error_code_t::CUGRAPH_UNSUPPORTED_TYPE_COMBINATION as u32,
            6
        );
    }

    #[test]
    fn test_bool_values() {
        assert_eq!(bool_t::FALSE as u32, 0);
        assert_eq!(bool_t::TRUE as u32, 1);
    }

    #[test]
    fn test_data_type_values() {
        assert_eq!(cugraph_data_type_id_t::INT32 as u32, 2);
        assert_eq!(cugraph_data_type_id_t::INT64 as u32, 3);
        assert_eq!(cugraph_data_type_id_t::FLOAT32 as u32, 8);
        assert_eq!(cugraph_data_type_id_t::FLOAT64 as u32, 9);
    }
}
