use cudf::{TypeId as CudfTypeId, interop::contract::TypeId as InteropTypeId};

use crate::types::DataType;

macro_rules! map_type_id_to_cugraph {
    ($type_id:expr, $type:path) => {
        match $type_id {
            <$type>::Int8 => Some(DataType::Int8),
            <$type>::Int16 => Some(DataType::Int16),
            <$type>::Int32 => Some(DataType::Int32),
            <$type>::Int64 => Some(DataType::Int64),
            <$type>::UInt8 => Some(DataType::Uint8),
            <$type>::UInt16 => Some(DataType::Uint16),
            <$type>::UInt32 => Some(DataType::Uint32),
            <$type>::UInt64 => Some(DataType::Uint64),
            <$type>::Float32 => Some(DataType::Float32),
            <$type>::Float64 => Some(DataType::Float64),
            <$type>::Bool8 => Some(DataType::Bool),
            _ => None,
        }
    };
}

/// Convert a cudf [`cudf::TypeId`] to a cugraph [`DataType`].
///
/// Returns `None` for cudf types that have no direct cugraph equivalent
/// (e.g. timestamps, strings, structs).
pub fn cudf_type_to_cugraph(type_id: CudfTypeId) -> Option<DataType> {
    map_type_id_to_cugraph!(type_id, CudfTypeId)
}

pub(super) fn borrowed_type_to_cugraph(type_id: InteropTypeId) -> Option<DataType> {
    map_type_id_to_cugraph!(type_id, InteropTypeId)
}

/// Convert a cugraph [`DataType`] to a cudf [`cudf::TypeId`].
///
/// Returns `None` for cugraph types that have no direct cudf equivalent
/// (i.e. [`DataType::SizeT`], which is an internal cugraph sizing type).
pub fn cugraph_type_to_cudf(dtype: DataType) -> Option<CudfTypeId> {
    match dtype {
        DataType::Int8 => Some(CudfTypeId::Int8),
        DataType::Int16 => Some(CudfTypeId::Int16),
        DataType::Int32 => Some(CudfTypeId::Int32),
        DataType::Int64 => Some(CudfTypeId::Int64),
        DataType::Uint8 => Some(CudfTypeId::UInt8),
        DataType::Uint16 => Some(CudfTypeId::UInt16),
        DataType::Uint32 => Some(CudfTypeId::UInt32),
        DataType::Uint64 => Some(CudfTypeId::UInt64),
        DataType::Float32 => Some(CudfTypeId::Float32),
        DataType::Float64 => Some(CudfTypeId::Float64),
        DataType::Bool => Some(CudfTypeId::Bool8),
        DataType::SizeT => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAPPED_DATA_TYPES: &[DataType] = &[
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
        DataType::Bool,
    ];

    const MAPPED_TYPE_IDS: &[CudfTypeId] = &[
        CudfTypeId::Int8,
        CudfTypeId::Int16,
        CudfTypeId::Int32,
        CudfTypeId::Int64,
        CudfTypeId::UInt8,
        CudfTypeId::UInt16,
        CudfTypeId::UInt32,
        CudfTypeId::UInt64,
        CudfTypeId::Float32,
        CudfTypeId::Float64,
        CudfTypeId::Bool8,
    ];

    #[test]
    fn cugraph_to_cudf_roundtrip() {
        for &dt in MAPPED_DATA_TYPES {
            let cudf_tid =
                cugraph_type_to_cudf(dt).unwrap_or_else(|| panic!("cugraph_type_to_cudf returned None for {:?}", dt));
            let back = cudf_type_to_cugraph(cudf_tid)
                .unwrap_or_else(|| panic!("cudf_type_to_cugraph returned None for {:?}", cudf_tid));
            assert_eq!(dt, back, "round-trip failed: {:?} -> {:?} -> {:?}", dt, cudf_tid, back);
        }
    }

    #[test]
    fn cudf_to_cugraph_roundtrip() {
        for &tid in MAPPED_TYPE_IDS {
            let cugraph_dt =
                cudf_type_to_cugraph(tid).unwrap_or_else(|| panic!("cudf_type_to_cugraph returned None for {:?}", tid));
            let back = cugraph_type_to_cudf(cugraph_dt)
                .unwrap_or_else(|| panic!("cugraph_type_to_cudf returned None for {:?}", cugraph_dt));
            assert_eq!(tid, back, "round-trip failed: {:?} -> {:?} -> {:?}", tid, cugraph_dt, back);
        }
    }

    #[test]
    fn borrowed_type_to_cugraph_matches_cudf_mapping() {
        let borrowed = [
            (InteropTypeId::Int8, CudfTypeId::Int8),
            (InteropTypeId::Int16, CudfTypeId::Int16),
            (InteropTypeId::Int32, CudfTypeId::Int32),
            (InteropTypeId::Int64, CudfTypeId::Int64),
            (InteropTypeId::UInt8, CudfTypeId::UInt8),
            (InteropTypeId::UInt16, CudfTypeId::UInt16),
            (InteropTypeId::UInt32, CudfTypeId::UInt32),
            (InteropTypeId::UInt64, CudfTypeId::UInt64),
            (InteropTypeId::Float32, CudfTypeId::Float32),
            (InteropTypeId::Float64, CudfTypeId::Float64),
            (InteropTypeId::Bool8, CudfTypeId::Bool8),
        ];

        for (borrowed_tid, cudf_tid) in borrowed {
            assert_eq!(
                borrowed_type_to_cugraph(borrowed_tid),
                cudf_type_to_cugraph(cudf_tid),
                "borrowed and cudf mappings diverged for {:?} / {:?}",
                borrowed_tid,
                cudf_tid
            );
        }
    }

    #[test]
    fn size_t_has_no_cudf_equivalent() {
        assert_eq!(cugraph_type_to_cudf(DataType::SizeT), None);
    }

    #[test]
    fn unsupported_cudf_types_return_none() {
        let unsupported = [
            CudfTypeId::Empty,
            CudfTypeId::TimestampDays,
            CudfTypeId::TimestampSeconds,
            CudfTypeId::TimestampMilliseconds,
            CudfTypeId::TimestampMicroseconds,
            CudfTypeId::TimestampNanoseconds,
            CudfTypeId::DurationDays,
            CudfTypeId::DurationSeconds,
            CudfTypeId::DurationMilliseconds,
            CudfTypeId::DurationMicroseconds,
            CudfTypeId::DurationNanoseconds,
            CudfTypeId::Dictionary32,
            CudfTypeId::String,
            CudfTypeId::List,
            CudfTypeId::Decimal32,
            CudfTypeId::Decimal64,
            CudfTypeId::Decimal128,
            CudfTypeId::Struct,
        ];
        for tid in unsupported {
            assert_eq!(cudf_type_to_cugraph(tid), None, "expected None for unsupported cudf TypeId {:?}", tid);
        }
    }

    #[test]
    fn unsupported_borrowed_types_return_none() {
        let unsupported = [
            InteropTypeId::Empty,
            InteropTypeId::TimestampDays,
            InteropTypeId::TimestampSeconds,
            InteropTypeId::TimestampMilliseconds,
            InteropTypeId::TimestampMicroseconds,
            InteropTypeId::TimestampNanoseconds,
            InteropTypeId::DurationDays,
            InteropTypeId::DurationSeconds,
            InteropTypeId::DurationMilliseconds,
            InteropTypeId::DurationMicroseconds,
            InteropTypeId::DurationNanoseconds,
            InteropTypeId::Dictionary32,
            InteropTypeId::String,
            InteropTypeId::List,
            InteropTypeId::Decimal32,
            InteropTypeId::Decimal64,
            InteropTypeId::Decimal128,
            InteropTypeId::Struct,
        ];

        for tid in unsupported {
            assert_eq!(borrowed_type_to_cugraph(tid), None, "expected None for unsupported interop TypeId {:?}", tid);
        }
    }
}
