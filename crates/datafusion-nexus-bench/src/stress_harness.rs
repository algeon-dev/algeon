//! Shared validation helpers for explicit stress binaries.

/// Validate that a barrier-released workload cannot shed requests merely
/// because its admission queue is smaller than the synchronized request wave.
///
/// This intentionally uses the full wave size rather than subtracting the
/// configured active-query limit: memory, placement, or per-device caps can
/// make the immediately active subset smaller than that limit.
pub fn validate_synchronized_admission_queue(
    queue_env: &str,
    max_admission_queue_len: usize,
    synchronized_request_count: usize,
) -> Result<(), String> {
    if max_admission_queue_len < synchronized_request_count {
        return Err(format!(
            "{queue_env} must be at least the synchronized request count ({synchronized_request_count}), got {max_admission_queue_len}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synchronized_queue_accepts_exact_wave_capacity() {
        validate_synchronized_admission_queue("QUEUE_ENV", 96, 96).unwrap();
    }

    #[test]
    fn synchronized_queue_rejects_capacity_below_wave() {
        let error = validate_synchronized_admission_queue("QUEUE_ENV", 64, 96).unwrap_err();

        assert!(error.contains("QUEUE_ENV"), "{error}");
        assert!(error.contains("96"), "{error}");
        assert!(error.contains("64"), "{error}");
    }
}
