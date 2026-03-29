//! Private retained execution-domain substrate for owned GPU objects.

use std::{
    num::NonZeroU64,
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
