//! Fundamental types shared across the cuGraph Rust API.

use cugraph_sys::{bool_t, cugraph_data_type_id_t, cugraph_graph_properties_t};

/// Element data type for type-erased arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Float32,
    Float64,
    SizeT,
    Bool,
}

impl DataType {
    /// Size of one element in bytes.
    pub fn size_of(&self) -> usize {
        match self {
            DataType::Int8 | DataType::Uint8 | DataType::Bool => 1,
            DataType::Int16 | DataType::Uint16 => 2,
            DataType::Int32 | DataType::Uint32 | DataType::Float32 => 4,
            DataType::Int64 | DataType::Uint64 | DataType::Float64 | DataType::SizeT => 8,
        }
    }
}

impl From<DataType> for cugraph_data_type_id_t {
    fn from(dt: DataType) -> Self {
        match dt {
            DataType::Int8 => cugraph_data_type_id_t::INT8,
            DataType::Int16 => cugraph_data_type_id_t::INT16,
            DataType::Int32 => cugraph_data_type_id_t::INT32,
            DataType::Int64 => cugraph_data_type_id_t::INT64,
            DataType::Uint8 => cugraph_data_type_id_t::UINT8,
            DataType::Uint16 => cugraph_data_type_id_t::UINT16,
            DataType::Uint32 => cugraph_data_type_id_t::UINT32,
            DataType::Uint64 => cugraph_data_type_id_t::UINT64,
            DataType::Float32 => cugraph_data_type_id_t::FLOAT32,
            DataType::Float64 => cugraph_data_type_id_t::FLOAT64,
            DataType::SizeT => cugraph_data_type_id_t::SIZE_T,
            DataType::Bool => cugraph_data_type_id_t::BOOL,
        }
    }
}

impl TryFrom<cugraph_data_type_id_t> for DataType {
    type Error = ();

    fn try_from(dt: cugraph_data_type_id_t) -> std::result::Result<Self, Self::Error> {
        match dt {
            cugraph_data_type_id_t::INT8 => Ok(DataType::Int8),
            cugraph_data_type_id_t::INT16 => Ok(DataType::Int16),
            cugraph_data_type_id_t::INT32 => Ok(DataType::Int32),
            cugraph_data_type_id_t::INT64 => Ok(DataType::Int64),
            cugraph_data_type_id_t::UINT8 => Ok(DataType::Uint8),
            cugraph_data_type_id_t::UINT16 => Ok(DataType::Uint16),
            cugraph_data_type_id_t::UINT32 => Ok(DataType::Uint32),
            cugraph_data_type_id_t::UINT64 => Ok(DataType::Uint64),
            cugraph_data_type_id_t::FLOAT32 => Ok(DataType::Float32),
            cugraph_data_type_id_t::FLOAT64 => Ok(DataType::Float64),
            cugraph_data_type_id_t::SIZE_T => Ok(DataType::SizeT),
            cugraph_data_type_id_t::BOOL => Ok(DataType::Bool),
            _ => Err(()),
        }
    }
}

/// Properties that describe a graph's structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphProperties {
    pub is_symmetric: bool,
    pub is_multigraph: bool,
}

impl From<GraphProperties> for cugraph_graph_properties_t {
    fn from(props: GraphProperties) -> Self {
        cugraph_graph_properties_t {
            is_symmetric: to_bool_t(props.is_symmetric),
            is_multigraph: to_bool_t(props.is_multigraph),
        }
    }
}

/// Convert a Rust `bool` to the C `bool_t` enum.
pub(crate) fn to_bool_t(b: bool) -> bool_t {
    if b {
        bool_t::TRUE
    } else {
        bool_t::FALSE
    }
}

/// Convert the C `bool_t` enum to a Rust `bool`, validating that the value is
/// either TRUE or FALSE.
pub(crate) fn try_from_bool_t(b: bool_t) -> Option<bool> {
    if b == bool_t::TRUE {
        Some(true)
    } else if b == bool_t::FALSE {
        Some(false)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_roundtrip() {
        let variants = [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::Uint8,
            DataType::Uint16,
            DataType::Uint32,
            DataType::Uint64,
            DataType::Float32,
            DataType::Float64,
            DataType::SizeT,
            DataType::Bool,
        ];
        for dt in &variants {
            let c_dt: cugraph_data_type_id_t = (*dt).into();
            let back = DataType::try_from(c_dt).expect("known cugraph_data_type_id_t");
            assert_eq!(*dt, back, "roundtrip failed for {:?}", dt);
        }
    }

    #[test]
    fn test_data_type_size_of() {
        assert_eq!(DataType::Int8.size_of(), 1);
        assert_eq!(DataType::Int16.size_of(), 2);
        assert_eq!(DataType::Int32.size_of(), 4);
        assert_eq!(DataType::Int64.size_of(), 8);
        assert_eq!(DataType::Float32.size_of(), 4);
        assert_eq!(DataType::Float64.size_of(), 8);
        assert_eq!(DataType::SizeT.size_of(), 8);
        assert_eq!(DataType::Bool.size_of(), 1);
    }

    #[test]
    fn test_bool_conversion() {
        assert_eq!(to_bool_t(true), bool_t::TRUE);
        assert_eq!(to_bool_t(false), bool_t::FALSE);
        assert_eq!(try_from_bool_t(bool_t::TRUE), Some(true));
        assert_eq!(try_from_bool_t(bool_t::FALSE), Some(false));
        assert_eq!(try_from_bool_t(bool_t(2)), None);
    }

    #[test]
    fn test_graph_properties_to_c() {
        let props = GraphProperties {
            is_symmetric: true,
            is_multigraph: false,
        };
        let c_props: cugraph_graph_properties_t = props.into();
        assert_eq!(c_props.is_symmetric, bool_t::TRUE);
        assert_eq!(c_props.is_multigraph, bool_t::FALSE);
    }
}
