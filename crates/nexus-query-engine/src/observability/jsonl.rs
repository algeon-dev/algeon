//! JSONL export for sealed runtime observation snapshots.

use std::{error::Error as StdError, fmt, io, io::Write};

use super::{RuntimeObservationRecord, RuntimeTopologyRecord, SealedRuntimeObservationSnapshot};

/// Machine-readable export failure category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RuntimeObservationExportErrorKind {
    /// A typed record could not be encoded. Retrying unchanged data will not help.
    InvalidRecord,
    /// The destination rejected a write.
    DestinationUnavailable,
}

/// Retry status derived from the export failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeObservationExportErrorStatus {
    Permanent,
    Temporary,
}

#[derive(Debug)]
enum RuntimeObservationExportErrorSource {
    Encode(serde_json::Error),
    Write(io::Error),
}

impl fmt::Display for RuntimeObservationExportErrorSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode(error) => error.fmt(formatter),
            Self::Write(error) => error.fmt(formatter),
        }
    }
}

impl StdError for RuntimeObservationExportErrorSource {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Encode(error) => Some(error),
            Self::Write(error) => Some(error),
        }
    }
}

/// Typed JSONL export error. Export failure never mutates the sealed snapshot.
#[derive(Debug)]
pub struct RuntimeObservationExportError {
    kind: RuntimeObservationExportErrorKind,
    status: RuntimeObservationExportErrorStatus,
    record_index: usize,
    source: RuntimeObservationExportErrorSource,
}

impl RuntimeObservationExportError {
    /// Returns the caller-action category.
    pub const fn kind(&self) -> RuntimeObservationExportErrorKind {
        self.kind
    }

    /// Returns whether retrying with a healthy destination can help.
    pub const fn status(&self) -> RuntimeObservationExportErrorStatus {
        self.status
    }

    /// Zero-based JSONL record index that failed.
    pub const fn record_index(&self) -> usize {
        self.record_index
    }

    /// Returns whether the failure is classified as temporary.
    pub const fn is_retryable(&self) -> bool {
        matches!(self.status, RuntimeObservationExportErrorStatus::Temporary)
    }

    fn encode(record_index: usize, source: serde_json::Error) -> Self {
        if let Some(kind) = source.io_error_kind() {
            let status = match kind {
                io::ErrorKind::Interrupted | io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock => {
                    RuntimeObservationExportErrorStatus::Temporary
                },
                _ => RuntimeObservationExportErrorStatus::Permanent,
            };
            return Self {
                kind: RuntimeObservationExportErrorKind::DestinationUnavailable,
                status,
                record_index,
                source: RuntimeObservationExportErrorSource::Encode(source),
            };
        }
        Self {
            kind: RuntimeObservationExportErrorKind::InvalidRecord,
            status: RuntimeObservationExportErrorStatus::Permanent,
            record_index,
            source: RuntimeObservationExportErrorSource::Encode(source),
        }
    }

    fn write(record_index: usize, source: io::Error) -> Self {
        let status = match source.kind() {
            io::ErrorKind::Interrupted | io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock => {
                RuntimeObservationExportErrorStatus::Temporary
            },
            _ => RuntimeObservationExportErrorStatus::Permanent,
        };
        Self {
            kind: RuntimeObservationExportErrorKind::DestinationUnavailable,
            status,
            record_index,
            source: RuntimeObservationExportErrorSource::Write(source),
        }
    }
}

impl fmt::Display for RuntimeObservationExportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "runtime observation JSONL export failed: kind={:?}, status={:?}, record_index={}: {}",
            self.kind, self.status, self.record_index, self.source
        )
    }
}

impl StdError for RuntimeObservationExportError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        Some(&self.source)
    }
}

/// Writes a sealed snapshot as ordered, independently parseable JSONL records.
pub fn write_runtime_observation_jsonl<W: Write>(
    snapshot: &SealedRuntimeObservationSnapshot,
    writer: &mut W,
) -> Result<(), RuntimeObservationExportError> {
    let data = snapshot.data();
    let mut record_index = 0;
    write_record(writer, record_index, &RuntimeObservationRecord::Manifest(data.manifest.clone()))?;
    record_index += 1;

    for retained in &data.topology {
        let record = match &retained.record {
            RuntimeTopologyRecord::PlanNode(row) => {
                RuntimeObservationRecord::PlanNode { sequence: retained.sequence, row: row.clone() }
            },
            RuntimeTopologyRecord::PlanEdge(row) => {
                RuntimeObservationRecord::PlanEdge { sequence: retained.sequence, row: row.clone() }
            },
            RuntimeTopologyRecord::TaskDefinition(row) => {
                RuntimeObservationRecord::TaskDefinition { sequence: retained.sequence, row: row.clone() }
            },
            RuntimeTopologyRecord::CompiledEdge(row) => {
                RuntimeObservationRecord::CompiledEdge { sequence: retained.sequence, row: row.clone() }
            },
            RuntimeTopologyRecord::BarrierDefinition(row) => {
                RuntimeObservationRecord::BarrierDefinition { sequence: retained.sequence, row: row.clone() }
            },
            RuntimeTopologyRecord::SinkDefinition(row) => {
                RuntimeObservationRecord::SinkDefinition { sequence: retained.sequence, row: row.clone() }
            },
            RuntimeTopologyRecord::ReservationGate(row) => {
                RuntimeObservationRecord::ReservationGate { sequence: retained.sequence, row: row.clone() }
            },
            RuntimeTopologyRecord::JobDefinition(row) => {
                RuntimeObservationRecord::JobDefinition { sequence: retained.sequence, row: row.clone() }
            },
            RuntimeTopologyRecord::WorkerDefinition(row) => {
                RuntimeObservationRecord::WorkerDefinition { sequence: retained.sequence, row: row.clone() }
            },
            RuntimeTopologyRecord::Resource(row) => {
                RuntimeObservationRecord::Resource { sequence: retained.sequence, row: row.clone() }
            },
        };
        write_record(writer, record_index, &record)?;
        record_index += 1;
    }
    for event in &data.events {
        write_record(writer, record_index, &RuntimeObservationRecord::TimelineEvent(event.clone()))?;
        record_index += 1;
    }
    write_record(writer, record_index, &RuntimeObservationRecord::Footer(data.footer.clone()))
}

fn write_record<W: Write>(
    writer: &mut W,
    record_index: usize,
    record: &RuntimeObservationRecord,
) -> Result<(), RuntimeObservationExportError> {
    // Serialize directly into the destination. Besides avoiding one complete
    // record-sized allocation, this lets bounded destinations stop serde as
    // soon as their hard byte limit is reached.
    serde_json::to_writer(&mut *writer, record)
        .map_err(|error| RuntimeObservationExportError::encode(record_index, error))?;
    writer.write_all(b"\n").map_err(|error| RuntimeObservationExportError::write(record_index, error))
}
