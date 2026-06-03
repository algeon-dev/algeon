//! Private retained execution-domain substrate for owned GPU objects.

use std::{
    num::NonZeroU64,
    os::raw::c_void,
    sync::{Arc, OnceLock},
};

use crate::{
    error::{Error, ErrorKind, Result},
    memory::DeviceMemoryResource,
    stream::CudaStream,
};

static AMBIENT_DEFAULT_DOMAIN: OnceLock<Arc<ExecutionDomain>> = OnceLock::new();
pub(crate) struct ExecutionDomain {
    kind: DomainKind,
}

pub(crate) enum DomainKind {
    AmbientDefault,
    Explicit { _stream: Arc<CudaStream>, _mr: Arc<DeviceMemoryResource>, identity: ExecutionIdentity },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ExecutionIdentity(NonZeroU64);

impl ExecutionIdentity {
    fn new(value: u64) -> Result<Self> {
        NonZeroU64::new(value).map(Self).ok_or_else(|| {
            Error::new(ErrorKind::InvalidOperation, "explicit execution domain identity must be non-zero")
                .with_operation("execution_domain::ExecutionIdentity::new")
                .with_context("identity", value.to_string())
        })
    }
    fn get(self) -> u64 {
        self.0.get()
    }
}

impl ExecutionDomain {
    pub(crate) fn ambient_default() -> Arc<Self> {
        Arc::clone(AMBIENT_DEFAULT_DOMAIN.get_or_init(|| Arc::new(Self { kind: DomainKind::AmbientDefault })))
    }

    pub(crate) fn debug_explicit(identity: u64) -> Result<Arc<Self>> {
        let stream = Arc::new(CudaStream::new()?);
        let mr = Arc::new(DeviceMemoryResource::cuda()?);
        Ok(Arc::new(Self {
            kind: DomainKind::Explicit { _stream: stream, _mr: mr, identity: ExecutionIdentity::new(identity)? },
        }))
    }

    pub(crate) fn debug_kind(&self) -> &'static str {
        match self.kind {
            DomainKind::AmbientDefault => "ambient_default",
            DomainKind::Explicit { .. } => "explicit",
        }
    }

    pub(crate) fn debug_id(&self) -> u64 {
        match self.kind {
            DomainKind::AmbientDefault => 0,
            DomainKind::Explicit { identity, .. } => identity.get(),
        }
    }

    pub(crate) fn debug_label(&self) -> String {
        match self.kind {
            DomainKind::AmbientDefault => "ambient_default".to_string(),
            DomainKind::Explicit { identity, .. } => format!("explicit:{}", identity.get()),
        }
    }

    pub(crate) fn debug_stream_value(&self) -> usize {
        match &self.kind {
            DomainKind::AmbientDefault => 0,
            DomainKind::Explicit { _stream, .. } => _stream.debug_handle_value().unwrap_or(0),
        }
    }

    pub(crate) fn debug_mr_value(&self) -> usize {
        match &self.kind {
            DomainKind::AmbientDefault => 0,
            DomainKind::Explicit { _mr, .. } => _mr.__debug_raw_handle_value(),
        }
    }

    pub(crate) fn synchronize_for_host_visibility(&self, operation: &'static str) -> Result<()> {
        match &self.kind {
            DomainKind::AmbientDefault => Ok(()),
            DomainKind::Explicit { _stream, .. } => _stream
                .synchronize()
                .map_err(|e| e.with_operation(operation).with_context("sync_stream", self.debug_label())),
        }
    }

    pub(crate) fn explicit_exec_options_raw(&self) -> Option<cudf_sys::cudf_exec_options_t> {
        match &self.kind {
            DomainKind::AmbientDefault => None,
            DomainKind::Explicit { _stream, _mr, .. } => Some(cudf_sys::cudf_exec_options_t {
                struct_size: std::mem::size_of::<cudf_sys::cudf_exec_options_t>(),
                stream: _stream.as_raw() as *mut c_void,
                mr: _mr.as_raw() as *mut c_void,
                flags: 0,
                reserved: [0; cudf_sys::CUDF_EXEC_OPTIONS_RESERVED_SLOTS],
            }),
        }
    }

    pub(crate) fn resolve_with_borrowed_override(
        input: &Arc<Self>,
        stream: Option<&CudaStream>,
        mr: Option<&DeviceMemoryResource>,
        operation: &'static str,
    ) -> Result<Arc<Self>> {
        match &input.kind {
            DomainKind::AmbientDefault => {
                if stream.is_some() || mr.is_some() {
                    return Err(Error::new(
                        ErrorKind::InvalidOperation,
                        "borrowed execution-context overrides cannot promote ambient-default objects",
                    )
                    .with_operation(operation)
                    .with_context("operation_domain", input.debug_label())
                    .with_context("borrowed_override_stream", stream.is_some().to_string())
                    .with_context("borrowed_override_memory_resource", mr.is_some().to_string()));
                }

                Ok(Arc::clone(input))
            },
            DomainKind::Explicit { _stream, _mr, identity } => Ok(Arc::new(Self {
                kind: DomainKind::Explicit {
                    _stream: stream.map(|stream| Arc::new(stream.clone())).unwrap_or_else(|| Arc::clone(_stream)),
                    _mr: mr.map(|mr| Arc::new(mr.clone())).unwrap_or_else(|| Arc::clone(_mr)),
                    identity: *identity,
                },
            })),
        }
    }

    pub(crate) fn same_identity(left: &Arc<Self>, right: &Arc<Self>) -> bool {
        left.debug_id() == right.debug_id() && left.debug_kind() == right.debug_kind()
    }

    pub(crate) fn compatible_domain(left: &Arc<Self>, right: &Arc<Self>, operation: &'static str) -> Result<Arc<Self>> {
        if Self::same_identity(left, right) {
            Ok(Arc::clone(left))
        } else {
            Err(Error::new(ErrorKind::InvalidOperation, "mixed execution domains require one shared retained lineage")
                .with_operation(operation)
                .with_context("left_domain", left.debug_label())
                .with_context("right_domain", right.debug_label()))
        }
    }
}
