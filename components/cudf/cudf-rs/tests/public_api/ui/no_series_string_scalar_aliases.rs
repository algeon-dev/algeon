use cudf::Series;

fn main() {
    let _ = Series::eq_scalar_str;
    let _ = Series::ne_scalar_str;
    let _ = Series::gt_scalar_str;
    let _ = Series::lt_scalar_str;
    let _ = Series::ge_scalar_str;
    let _ = Series::le_scalar_str;
}
