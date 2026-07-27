//! Runtime-typed result value enums for algorithm outputs.
//!
//! These enums wrap the two possible element widths for vertex IDs and
//! score/weight values returned by cuGraph algorithms, following the same
//! pattern as [`DegreeValues`](super::DegreeValues).

use crate::{
    array::{DeviceArrayView, RawDeviceArrayView},
    error::{Error, ErrorKind, Result},
    handle::ResourceHandle,
    types::DataType,
};

// ═══════════════════════════════════════════════════════════════════════════
// VertexIds — typed enum for runtime-width vertex id arrays
// ═══════════════════════════════════════════════════════════════════════════

/// Runtime-typed vertex IDs from algorithm results.
///
/// The cuGraph C API may return vertex id arrays as either `i32` or `i64`
/// depending on the graph's vertex type. This enum wraps the two
/// possibilities so callers can handle both without raw byte manipulation.
///
/// For convenience, [`to_vec_i64`](VertexIds::to_vec_i64) widens `i32`
/// values to `i64` automatically.
#[derive(Debug)]
pub enum VertexIds<'a> {
    /// Vertex IDs stored as 32-bit integers.
    I32(DeviceArrayView<'a, i32>),
    /// Vertex IDs stored as 64-bit integers.
    I64(DeviceArrayView<'a, i64>),
}

impl<'a> VertexIds<'a> {
    /// Construct from a type-erased raw view after checking the dtype.
    ///
    /// Returns an error if the dtype is neither `Int32` nor `Int64`.
    pub(crate) fn from_raw(raw: RawDeviceArrayView<'a>) -> Result<Self> {
        match raw.dtype()? {
            DataType::Int32 => Ok(Self::I32(DeviceArrayView::from_raw_unchecked(raw))),
            DataType::Int64 => Ok(Self::I64(DeviceArrayView::from_raw_unchecked(raw))),
            other => Err(Error::new(ErrorKind::InvalidState, "VertexIds::from_raw")
                .with_message(format!("unexpected vertex id dtype: {other}"))),
        }
    }

    /// Number of elements in the vertex id array.
    pub fn len(&self) -> usize {
        match self {
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
        }
    }

    /// Whether the vertex id array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy to host as `Vec<i64>`, widening `i32` values if needed.
    pub fn to_vec_i64(&self, handle: &ResourceHandle) -> Result<Vec<i64>> {
        match self {
            Self::I32(v) => {
                let vals = v.to_vec(handle)?;
                Ok(vals.into_iter().map(|x| x as i64).collect())
            },
            Self::I64(v) => v.to_vec(handle),
        }
    }

    /// Copy to host as `Vec<i32>`.
    ///
    /// Returns an error if the underlying values are `i64`, since narrowing
    /// could lose data.
    pub fn to_vec_i32(&self, handle: &ResourceHandle) -> Result<Vec<i32>> {
        match self {
            Self::I32(v) => v.to_vec(handle),
            Self::I64(_) => Err(Error::new(ErrorKind::InvalidState, "VertexIds::to_vec_i32")
                .with_message("vertex ids are i64; use to_vec_i64() to avoid data loss")),
        }
    }

    /// Get the inner view as `&DeviceArrayView<i32>`, or `None` if `i64`.
    pub fn as_i32(&self) -> Option<&DeviceArrayView<'a, i32>> {
        match self {
            Self::I32(v) => Some(v),
            Self::I64(_) => None,
        }
    }

    /// Get the inner view as `&DeviceArrayView<i64>`, or `None` if `i32`.
    pub fn as_i64(&self) -> Option<&DeviceArrayView<'a, i64>> {
        match self {
            Self::I32(_) => None,
            Self::I64(v) => Some(v),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ScoreValues — typed enum for runtime-width float score arrays
// ═══════════════════════════════════════════════════════════════════════════

/// Runtime-typed score/weight values from algorithm results.
///
/// The cuGraph C API may return score arrays as either `f32` or `f64`
/// depending on the graph's weight type. This enum wraps the two
/// possibilities so callers can handle both without raw byte manipulation.
///
/// For convenience, [`to_vec_f64`](ScoreValues::to_vec_f64) widens `f32`
/// values to `f64` automatically.
#[derive(Debug)]
pub enum ScoreValues<'a> {
    /// Score values stored as 32-bit floats.
    F32(DeviceArrayView<'a, f32>),
    /// Score values stored as 64-bit floats.
    F64(DeviceArrayView<'a, f64>),
}

impl<'a> ScoreValues<'a> {
    /// Construct from a type-erased raw view after checking the dtype.
    ///
    /// Returns an error if the dtype is neither `Float32` nor `Float64`.
    pub(crate) fn from_raw(raw: RawDeviceArrayView<'a>) -> Result<Self> {
        match raw.dtype()? {
            DataType::Float32 => Ok(Self::F32(DeviceArrayView::from_raw_unchecked(raw))),
            DataType::Float64 => Ok(Self::F64(DeviceArrayView::from_raw_unchecked(raw))),
            other => Err(Error::new(ErrorKind::InvalidState, "ScoreValues::from_raw")
                .with_message(format!("unexpected score dtype: {other}"))),
        }
    }

    /// Number of elements in the score array.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    /// Whether the score array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy to host as `Vec<f64>`, widening `f32` values if needed.
    pub fn to_vec_f64(&self, handle: &ResourceHandle) -> Result<Vec<f64>> {
        match self {
            Self::F32(v) => {
                let vals = v.to_vec(handle)?;
                Ok(vals.into_iter().map(|x| x as f64).collect())
            },
            Self::F64(v) => v.to_vec(handle),
        }
    }

    /// Copy to host as `Vec<f32>`.
    ///
    /// Returns an error if the underlying values are `f64`, since narrowing
    /// could lose data.
    pub fn to_vec_f32(&self, handle: &ResourceHandle) -> Result<Vec<f32>> {
        match self {
            Self::F32(v) => v.to_vec(handle),
            Self::F64(_) => Err(Error::new(ErrorKind::InvalidState, "ScoreValues::to_vec_f32")
                .with_message("score values are f64; use to_vec_f64() to avoid data loss")),
        }
    }

    /// Get the inner view as `&DeviceArrayView<f32>`, or `None` if `f64`.
    pub fn as_f32(&self) -> Option<&DeviceArrayView<'a, f32>> {
        match self {
            Self::F32(v) => Some(v),
            Self::F64(_) => None,
        }
    }

    /// Get the inner view as `&DeviceArrayView<f64>`, or `None` if `f32`.
    pub fn as_f64(&self) -> Option<&DeviceArrayView<'a, f64>> {
        match self {
            Self::F32(_) => None,
            Self::F64(v) => Some(v),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Distances — typed enum for algorithm distance arrays (integer or float)
// ═══════════════════════════════════════════════════════════════════════════

/// Runtime-typed distance values from traversal algorithm results.
///
/// BFS returns integer hop-count distances (`I32`/`I64`), while SSSP returns
/// floating-point weighted distances (`F32`/`F64`). Both share the same
/// `cugraph_paths_result_t` C API result type, so this enum unifies them.
///
/// For convenience:
/// - [`to_vec_i64`](Distances::to_vec_i64) widens `i32` values (errors on float variants)
/// - [`to_vec_f64`](Distances::to_vec_f64) widens all variants to `f64`
#[derive(Debug)]
pub enum Distances<'a> {
    /// Integer distances stored as 32-bit integers (BFS hop counts).
    I32(DeviceArrayView<'a, i32>),
    /// Integer distances stored as 64-bit integers (BFS hop counts).
    I64(DeviceArrayView<'a, i64>),
    /// Float distances stored as 32-bit floats (SSSP weighted distances).
    F32(DeviceArrayView<'a, f32>),
    /// Float distances stored as 64-bit floats (SSSP weighted distances).
    F64(DeviceArrayView<'a, f64>),
}

impl<'a> Distances<'a> {
    /// Construct from a type-erased raw view after checking the dtype.
    ///
    /// Accepts `Int32`, `Int64`, `Float32`, and `Float64`.
    pub(crate) fn from_raw(raw: RawDeviceArrayView<'a>) -> Result<Self> {
        match raw.dtype()? {
            DataType::Int32 => Ok(Self::I32(DeviceArrayView::from_raw_unchecked(raw))),
            DataType::Int64 => Ok(Self::I64(DeviceArrayView::from_raw_unchecked(raw))),
            DataType::Float32 => Ok(Self::F32(DeviceArrayView::from_raw_unchecked(raw))),
            DataType::Float64 => Ok(Self::F64(DeviceArrayView::from_raw_unchecked(raw))),
            other => Err(Error::new(ErrorKind::InvalidState, "Distances::from_raw")
                .with_message(format!("unexpected distance dtype: {other}"))),
        }
    }

    /// Number of elements in the distance array.
    pub fn len(&self) -> usize {
        match self {
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }

    /// Whether the distance array is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy to host as `Vec<i64>`, widening `i32` values if needed.
    ///
    /// Returns an error for float variants — use [`to_vec_f64`](Distances::to_vec_f64)
    /// instead.
    pub fn to_vec_i64(&self, handle: &ResourceHandle) -> Result<Vec<i64>> {
        match self {
            Self::I32(v) => {
                let vals = v.to_vec(handle)?;
                Ok(vals.into_iter().map(|x| x as i64).collect())
            },
            Self::I64(v) => v.to_vec(handle),
            Self::F32(_) | Self::F64(_) => Err(Error::new(ErrorKind::InvalidState, "Distances::to_vec_i64")
                .with_message("distances are floating-point; use to_vec_f64()")),
        }
    }

    /// Copy to host as `Vec<i32>`.
    ///
    /// Returns an error if the underlying values are `i64` (narrowing) or
    /// floating-point.
    pub fn to_vec_i32(&self, handle: &ResourceHandle) -> Result<Vec<i32>> {
        match self {
            Self::I32(v) => v.to_vec(handle),
            Self::I64(_) => Err(Error::new(ErrorKind::InvalidState, "Distances::to_vec_i32")
                .with_message("distances are i64; use to_vec_i64() to avoid data loss")),
            Self::F32(_) | Self::F64(_) => Err(Error::new(ErrorKind::InvalidState, "Distances::to_vec_i32")
                .with_message("distances are floating-point; use to_vec_f64()")),
        }
    }

    /// Copy to host as `Vec<f64>`, widening all variants.
    ///
    /// Integer distances are converted to `f64` losslessly (for i32) or with
    /// possible precision loss for very large i64 values.
    pub fn to_vec_f64(&self, handle: &ResourceHandle) -> Result<Vec<f64>> {
        match self {
            Self::I32(v) => {
                let vals = v.to_vec(handle)?;
                Ok(vals.into_iter().map(|x| x as f64).collect())
            },
            Self::I64(v) => {
                let vals = v.to_vec(handle)?;
                Ok(vals.into_iter().map(|x| x as f64).collect())
            },
            Self::F32(v) => {
                let vals = v.to_vec(handle)?;
                Ok(vals.into_iter().map(|x| x as f64).collect())
            },
            Self::F64(v) => v.to_vec(handle),
        }
    }

    /// Copy to host as `Vec<f32>`.
    ///
    /// Returns an error if the underlying values are `i64`, `f64`, or `i32`
    /// with values outside `f32` range. Only succeeds for `F32` variant.
    pub fn to_vec_f32(&self, handle: &ResourceHandle) -> Result<Vec<f32>> {
        match self {
            Self::F32(v) => v.to_vec(handle),
            _ => Err(Error::new(ErrorKind::InvalidState, "Distances::to_vec_f32")
                .with_message("distances are not f32; use to_vec_f64() for a lossless conversion")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        array::{DeviceArray, RawDeviceArrayView},
        error::ErrorKind,
        handle::ResourceHandle,
        types::DataType,
    };

    #[test]
    fn vertex_ids_i64_to_vec_i32_is_invalid_state() {
        let handle = ResourceHandle::new().expect("handle");
        let ids = DeviceArray::from_slice(&handle, &[1i64, 2, 3]).expect("ids");
        let values = VertexIds::I64(ids.view().expect("view"));

        let err = values.to_vec_i32(&handle).expect_err("i64 vertex ids should reject i32 copy");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "VertexIds::to_vec_i32");
    }

    #[test]
    fn score_values_f64_to_vec_f32_is_invalid_state() {
        let handle = ResourceHandle::new().expect("handle");
        let scores = DeviceArray::from_slice(&handle, &[1.0f64, 2.0, 3.0]).expect("scores");
        let values = ScoreValues::F64(scores.view().expect("view"));

        let err = values.to_vec_f32(&handle).expect_err("f64 score values should reject f32 copy");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "ScoreValues::to_vec_f32");
    }

    #[test]
    fn distances_float_to_vec_i64_is_invalid_state() {
        let handle = ResourceHandle::new().expect("handle");
        let distances = DeviceArray::from_slice(&handle, &[1.5f32, 2.5, 3.5]).expect("distances");
        let values = Distances::F32(distances.view().expect("view"));

        let err = values.to_vec_i64(&handle).expect_err("float distances should reject integer copy");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "Distances::to_vec_i64");
    }

    #[test]
    fn distances_i64_to_vec_i32_is_invalid_state() {
        let handle = ResourceHandle::new().expect("handle");
        let distances = DeviceArray::from_slice(&handle, &[10i64, 20, 30]).expect("distances");
        let values = Distances::I64(distances.view().expect("view"));

        let err = values.to_vec_i32(&handle).expect_err("i64 distances should reject i32 copy");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "Distances::to_vec_i32");
    }

    #[test]
    fn distances_non_f32_to_vec_f32_is_invalid_state() {
        let handle = ResourceHandle::new().expect("handle");
        let distances = DeviceArray::from_slice(&handle, &[10i32, 20, 30]).expect("distances");
        let values = Distances::I32(distances.view().expect("view"));

        let err = values.to_vec_f32(&handle).expect_err("non-f32 distances should reject f32 copy");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "Distances::to_vec_f32");
    }

    #[test]
    fn vertex_ids_from_raw_unexpected_dtype_is_invalid_state() {
        let handle = ResourceHandle::new().expect("handle");
        let ids = DeviceArray::from_slice(&handle, &[1i16, 2, 3]).expect("ids");
        let view = ids.view().expect("view");
        let raw = unsafe {
            RawDeviceArrayView::try_from_raw(view.device_ptr(), view.len(), DataType::Int16, "test_raw")
                .expect("raw view")
        };

        let err = VertexIds::from_raw(raw).expect_err("unexpected vertex dtype should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "VertexIds::from_raw");
    }

    #[test]
    fn score_values_from_raw_unexpected_dtype_is_invalid_state() {
        let handle = ResourceHandle::new().expect("handle");
        let scores = DeviceArray::from_slice(&handle, &[1i32, 2, 3]).expect("scores");
        let view = scores.view().expect("view");
        let raw = unsafe {
            RawDeviceArrayView::try_from_raw(view.device_ptr(), view.len(), DataType::Int32, "test_raw")
                .expect("raw view")
        };

        let err = ScoreValues::from_raw(raw).expect_err("unexpected score dtype should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "ScoreValues::from_raw");
    }

    #[test]
    fn distances_from_raw_unexpected_dtype_is_invalid_state() {
        let handle = ResourceHandle::new().expect("handle");
        let distances = DeviceArray::from_slice(&handle, &[1i16, 2, 3]).expect("distances");
        let view = distances.view().expect("view");
        let raw = unsafe {
            RawDeviceArrayView::try_from_raw(view.device_ptr(), view.len(), DataType::Int16, "test_raw")
                .expect("raw view")
        };

        let err = Distances::from_raw(raw).expect_err("unexpected distance dtype should be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidState);
        assert_eq!(err.operation(), "Distances::from_raw");
    }
}
