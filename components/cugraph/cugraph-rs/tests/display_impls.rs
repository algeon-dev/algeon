use cugraph::types::{DataType, GraphProperties};

#[test]
fn datatype_display_all_variants() {
    assert_eq!(DataType::Int8.to_string(), "Int8");
    assert_eq!(DataType::Int16.to_string(), "Int16");
    assert_eq!(DataType::Int32.to_string(), "Int32");
    assert_eq!(DataType::Int64.to_string(), "Int64");
    assert_eq!(DataType::Uint8.to_string(), "Uint8");
    assert_eq!(DataType::Uint16.to_string(), "Uint16");
    assert_eq!(DataType::Uint32.to_string(), "Uint32");
    assert_eq!(DataType::Uint64.to_string(), "Uint64");
    assert_eq!(DataType::Float32.to_string(), "Float32");
    assert_eq!(DataType::Float64.to_string(), "Float64");
    assert_eq!(DataType::SizeT.to_string(), "SizeT");
    assert_eq!(DataType::Bool.to_string(), "Bool");
}

#[test]
fn graph_properties_display() {
    let directed_simple = GraphProperties { is_symmetric: false, is_multigraph: false };
    assert_eq!(directed_simple.to_string(), "directed simple graph");

    let symmetric_simple = GraphProperties { is_symmetric: true, is_multigraph: false };
    assert_eq!(symmetric_simple.to_string(), "symmetric simple graph");

    let directed_multi = GraphProperties { is_symmetric: false, is_multigraph: true };
    assert_eq!(directed_multi.to_string(), "directed multigraph");

    let symmetric_multi = GraphProperties { is_symmetric: true, is_multigraph: true };
    assert_eq!(symmetric_multi.to_string(), "symmetric multigraph");
}
