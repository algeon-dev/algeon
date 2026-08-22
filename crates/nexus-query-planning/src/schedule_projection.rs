use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

use serde::{Deserialize, Serialize};

use crate::util::StableFnv1a64;

pub const COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1: &str = "nexus_query_engine.composed_schedule_projection.v1";

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
pub struct AttemptId([u8; 16]);

impl AttemptId {
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

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
        let hex = value.strip_prefix("attempt-").ok_or_else(|| serde::de::Error::custom("attempt id prefix"))?;
        if hex.len() != 32 {
            return Err(serde::de::Error::custom("attempt id length"));
        }
        let mut bytes = [0; 16];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).map_err(serde::de::Error::custom)?;
        }
        Ok(Self(bytes))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AttemptComposedScheduleOccurrence {
    pub occurrence_id: String,
    pub execution_scope_fingerprint: String,
    pub isolated_peak_bytes: u64,
    pub retained_output_bytes: u64,
    pub stage_peak_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AttemptComposedScheduleProjection {
    pub schema: String,
    pub attempt_id: AttemptId,
    pub device_ordinal: usize,
    pub grant_id: u64,
    pub grant_bytes: u64,
    pub attempt_wide_fixed_bytes: u64,
    pub producer_occurrences: Vec<AttemptComposedScheduleOccurrence>,
    pub consumer_scope_fingerprint: String,
    pub consumer_additional_peak_bytes: u64,
    pub consumer_stage_peak_bytes: u64,
    pub complete_peak_bytes: u64,
    pub fingerprint: String,
}

impl AttemptComposedScheduleProjection {
    #[cfg(any(test, feature = "test-hooks"))]
    #[doc(hidden)]
    pub fn for_test(device_ordinal: usize, grant_id: u64, grant_bytes: u64) -> Self {
        let mut projection = Self {
            schema: COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1.to_owned(),
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

    pub fn fingerprint_matches_contents(&self) -> bool {
        self.schema == COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1
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

    #[doc(hidden)]
    pub fn fingerprint_for(
        attempt_wide_fixed_bytes: u64,
        producer_occurrences: &[AttemptComposedScheduleOccurrence],
        consumer_scope_fingerprint: &str,
        consumer_additional_peak_bytes: u64,
        consumer_stage_peak_bytes: u64,
        complete_peak_bytes: u64,
    ) -> String {
        let mut hasher = StableFnv1a64::new();
        hasher.update_field(COMPOSED_SCHEDULE_PROJECTION_SCHEMA_V1);
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
