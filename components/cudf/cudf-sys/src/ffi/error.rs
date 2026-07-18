use std::os::raw::c_char;

unsafe extern "C" {
    // -- Error ---------------------------------------------------------------

    pub fn cudf_last_error_message() -> *const c_char;
    pub fn cudf_clear_error();
}
