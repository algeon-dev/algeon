use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

use query_planning::util::StableFnv1a64;
use serde::{Deserialize, Serialize, de};

pub const COMPOSED_SCHEDULE_PROJECTION_SCHEMA: &str = "query_engine.composed_schedule_projection";

static NEXT_ATTEMPT_ID: AtomicU64 = AtomicU64::new(1);

fn attempt_id_seed() -> u64 {
    static SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SEED.get_or_init(|| {
        let mut random = [0_u8; std::mem::size_of::<u64>()];
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0);
        random.copy_from_slice(&(nanos ^ ((std::process::id() as u64) << 32) ^ 0x9e37_79b9_7f4a_7c15).to_le_bytes());
        u64::from_le_bytes(random)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Opaque execution-attempt identity minted by the runtime.
/// Display and serialization use `attempt-` followed by 32 hexadecimal digits.
/// The identity is independent of caller query and trace correlations.
pub struct AttemptId([u8; 16]);

impl AttemptId {
    /// Returns the opaque 16-byte identity without its display prefix.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Mints a process-local execution-attempt correlation identity independent
    /// of caller-supplied query and trace identifiers.
    pub fn mint() -> Self {
        let counter = NEXT_ATTEMPT_ID.fetch_add(1, Ordering::Relaxed).to_be_bytes();
        let seed = attempt_id_seed().to_be_bytes();
        let mut bytes = [0; 16];
        bytes[..8].copy_from_slice(&counter);
        bytes[8..].copy_from_slice(&seed);
        Self(bytes)
    }
}

impl fmt::Display for AttemptId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "attempt-")?;
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl Serialize for AttemptId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for AttemptId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        let value = String::deserialize(deserializer)?;
        let hex = value.strip_prefix("attempt-").ok_or_else(|| de::Error::custom("attempt id prefix"))?;
        if hex.len() != 32 {
            return Err(de::Error::custom("attempt id length"));
        }
        let mut bytes = [0; 16];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).map_err(de::Error::custom)?;
        }
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
/// One producer occurrence in a composed execution memory schedule.
pub struct AttemptComposedScheduleOccurrence {
    /// Producer occurrence identity within the composed schedule.
    pub occurrence_id: String,
    /// Fingerprint of the producer's isolated execution scope.
    pub execution_scope_fingerprint: String,
    /// Peak capacity required by this producer's isolated scope, in bytes.
    pub isolated_peak_bytes: u64,
    /// Producer output bytes that remain owned after its scope completes.
    pub retained_output_bytes: u64,
    /// Peak scheduled capacity for this occurrence, including previously retained outputs.
    pub stage_peak_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
/// Immutable composed memory schedule bound to an admitted attempt.
///
/// The projection records producer order, retained outputs and the complete
/// capacity requirement. It cannot grant capacity or change the selected device.
pub struct AttemptComposedScheduleProjection {
    /// Composed-schedule schema label checked by the content fingerprint validator.
    pub schema: String,
    /// Attempt that owns this immutable composed schedule.
    pub attempt_id: AttemptId,
    /// CUDA device selected by the admission grant.
    pub device_ordinal: usize,
    /// Immutable ledger grant identity to which the schedule is bound.
    pub grant_id: u64,
    /// Immutable admitted query capacity in bytes.
    pub grant_bytes: u64,
    /// Capacity required for the entire attempt outside individual scope peaks, in bytes.
    pub attempt_wide_fixed_bytes: u64,
    /// Producer occurrences in their committed execution order.
    pub producer_occurrences: Vec<AttemptComposedScheduleOccurrence>,
    /// Fingerprint of the final consumer's execution scope.
    pub consumer_scope_fingerprint: String,
    /// Consumer capacity beyond retained producer inputs, in bytes.
    pub consumer_additional_peak_bytes: u64,
    /// Consumer stage peak including retained producer outputs, in bytes.
    pub consumer_stage_peak_bytes: u64,
    /// Largest stage capacity required by the complete composed schedule, in bytes.
    pub complete_peak_bytes: u64,
    /// Deterministic content fingerprint of memory quantities and scope order.
    pub fingerprint: String,
}

impl AttemptComposedScheduleProjection {
    #[cfg(test)]
    pub(crate) fn for_test(device_ordinal: usize, grant_id: u64, grant_bytes: u64) -> Self {
        let mut projection = Self {
            schema: COMPOSED_SCHEDULE_PROJECTION_SCHEMA.to_owned(),
            attempt_id: AttemptId::mint(),
            device_ordinal,
            grant_id,
            grant_bytes,
            attempt_wide_fixed_bytes: 0,
            producer_occurrences: Vec::new(),
            consumer_scope_fingerprint: "consumer".to_owned(),
            consumer_additional_peak_bytes: 1,
            consumer_stage_peak_bytes: 1,
            complete_peak_bytes: 1,
            fingerprint: String::new(),
        };
        projection.fingerprint = Self::fingerprint_for(
            projection.attempt_wide_fixed_bytes,
            &projection.producer_occurrences,
            &projection.consumer_scope_fingerprint,
            projection.consumer_additional_peak_bytes,
            projection.consumer_stage_peak_bytes,
            projection.complete_peak_bytes,
        );
        projection
    }

    /// Checks the schema label and deterministic memory-schedule fingerprint.
    /// Attempt identity, device ordinal and grant binding are excluded from the
    /// fingerprint; this check does not validate the admission grant.
    pub fn fingerprint_matches_contents(&self) -> bool {
        self.schema == COMPOSED_SCHEDULE_PROJECTION_SCHEMA
            && self.fingerprint
                == Self::fingerprint_for(
                    self.attempt_wide_fixed_bytes,
                    &self.producer_occurrences,
                    &self.consumer_scope_fingerprint,
                    self.consumer_additional_peak_bytes,
                    self.consumer_stage_peak_bytes,
                    self.complete_peak_bytes,
                )
    }

    pub(crate) fn fingerprint_for(
        attempt_wide_fixed_bytes: u64,
        producer_occurrences: &[AttemptComposedScheduleOccurrence],
        consumer_scope_fingerprint: &str,
        consumer_additional_peak_bytes: u64,
        consumer_stage_peak_bytes: u64,
        complete_peak_bytes: u64,
    ) -> String {
        let mut hasher = StableFnv1a64::new();
        hasher.update_field(COMPOSED_SCHEDULE_PROJECTION_SCHEMA);
        hasher.update_field(&attempt_wide_fixed_bytes.to_string());
        hasher.update_field(&producer_occurrences.len().to_string());
        for occurrence in producer_occurrences {
            hasher.update_field(&occurrence.occurrence_id);
            hasher.update_field(&occurrence.execution_scope_fingerprint);
            hasher.update_field(&occurrence.isolated_peak_bytes.to_string());
            hasher.update_field(&occurrence.retained_output_bytes.to_string());
            hasher.update_field(&occurrence.stage_peak_bytes.to_string());
        }
        hasher.update_field(consumer_scope_fingerprint);
        hasher.update_field(&consumer_additional_peak_bytes.to_string());
        hasher.update_field(&consumer_stage_peak_bytes.to_string());
        hasher.update_field(&complete_peak_bytes.to_string());
        format!("composed-schedule-fnv1a64:{:016x}", hasher.finish())
    }
}
