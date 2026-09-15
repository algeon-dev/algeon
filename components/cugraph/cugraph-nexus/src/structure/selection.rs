use crate::{
    array::{DeviceArray, DeviceArrayView},
    error::Result,
    handle::EnteredContext,
    types::CugraphElement,
};

/// Unified public vertex-selection input used across graph algorithms.
///
/// This type covers three distinct semantics:
///
/// - [`All`](Self::All) for algorithms that operate over every graph vertex
/// - [`Device`](Self::Device) for an explicit device-resident subset
/// - [`Host`](Self::Host) for a host-resident subset uploaded at the API boundary
///
/// # Examples
///
/// ```no_run
/// use cugraph::{
///     ResourceHandle, cudf_interop::device_array_view_from_slice,
///     structure::VertexSelection,
/// };
///
/// let runtime = std::sync::Arc::new(cudf::Runtime::new().unwrap());
/// let handle =
///     ResourceHandle::for_runtime(std::sync::Arc::clone(&runtime)).unwrap();
///
/// let device_vertices = runtime.series(&[0i32, 1]).unwrap();
/// let column = device_vertices.as_column();
/// let slice = column.device_slice::<i32>().unwrap();
/// let device_view = device_array_view_from_slice::<i32>(&slice).unwrap();
/// let selection = VertexSelection::from(&device_view);
///
/// let host_selection: VertexSelection<'_, i32> =
///     VertexSelection::from(&[0i32, 1][..]);
/// let all: VertexSelection<'_, i32> = VertexSelection::All;
/// # let _ = (selection, host_selection, all);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub enum VertexSelection<'a, V: CugraphElement = i32> {
    /// Select all graph vertices.
    #[default]
    All,
    /// Select an explicit subset that already lives on the GPU.
    Device(&'a DeviceArrayView<'a, V>),
    /// Select an explicit subset from host memory, uploaded on demand.
    Host(&'a [V]),
}

impl<'a, V: CugraphElement> From<&'a DeviceArrayView<'a, V>> for VertexSelection<'a, V> {
    fn from(view: &'a DeviceArrayView<'a, V>) -> Self {
        Self::Device(view)
    }
}

impl<'a, V: CugraphElement> From<&'a [V]> for VertexSelection<'a, V> {
    fn from(slice: &'a [V]) -> Self {
        Self::Host(slice)
    }
}

impl<'a, V: CugraphElement> VertexSelection<'a, V> {
    pub(crate) fn resolve(self, handle: &EnteredContext<'_>) -> Result<ResolvedVertexSelection<'a, V>> {
        match self {
            Self::All => Ok(ResolvedVertexSelection::All),
            Self::Device(view) => Ok(ResolvedVertexSelection::Borrowed(view)),
            Self::Host(slice) => {
                let array = DeviceArray::from_slice(handle, slice)?;
                Ok(ResolvedVertexSelection::Owned(array))
            },
        }
    }
}

pub(crate) enum ResolvedVertexSelection<'a, V: CugraphElement> {
    All,
    Borrowed(&'a DeviceArrayView<'a, V>),
    Owned(DeviceArray<V>),
}

impl<V: CugraphElement> ResolvedVertexSelection<'_, V> {
    pub(crate) fn view(&self) -> Result<Option<DeviceArrayView<'_, V>>> {
        match self {
            Self::All => Ok(None),
            Self::Borrowed(view) => Ok(Some(DeviceArrayView::from_raw_unchecked(view.raw_view()))),
            Self::Owned(array) => array.view().map(Some),
        }
    }
}
