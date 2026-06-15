use cudf::{DropNaOptions, InfoDetail, JsonReadOptions, MeltOptions, MultiParquetReadOptions};

fn main() {
    let _ = std::mem::size_of::<DropNaOptions>();
    let _ = std::mem::size_of::<InfoDetail>();
    let _ = std::mem::size_of::<JsonReadOptions>();
    let _ = std::mem::size_of::<MeltOptions>();
    let _ = std::mem::size_of::<MultiParquetReadOptions>();
}
