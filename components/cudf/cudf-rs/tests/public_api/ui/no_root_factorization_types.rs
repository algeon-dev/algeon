use cudf::{Codebook, DictionaryEncodingOptions, DictionaryKeyOrder, EncodedSeries, Factorized};

fn main() {
    let _ = std::mem::size_of::<Codebook>();
    let _ = std::mem::size_of::<DictionaryEncodingOptions>();
    let _ = std::mem::size_of::<DictionaryKeyOrder>();
    let _ = std::mem::size_of::<EncodedSeries>();
    let _ = std::mem::size_of::<Factorized>();
}
