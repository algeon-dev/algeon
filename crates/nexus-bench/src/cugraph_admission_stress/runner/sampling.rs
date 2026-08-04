use super::*;

impl SampleRow {
    pub(super) fn to_tsv(&self) -> String {
        [
            self.epoch_ms.to_string(),
            self.governor_capacity_bytes.to_string(),
            self.governor_outstanding_bytes.to_string(),
            self.governor_available_bytes.to_string(),
            self.active_grants.to_string(),
            self.active_queries.to_string(),
            self.queued_queries.to_string(),
            self.pool_reserved_current_bytes.to_string(),
            self.pool_reserved_high_bytes.to_string(),
            self.pool_used_current_bytes.to_string(),
            self.pool_used_high_bytes.to_string(),
            self.allocation_failure_count.to_string(),
            self.trim_retry_count.to_string(),
        ]
        .join("\t")
    }
}

impl DeviceSampleRow {
    pub(super) fn to_tsv(&self) -> String {
        [
            self.epoch_ms.to_string(),
            self.device_ordinal.to_string(),
            self.facts.governor_capacity_bytes.to_string(),
            self.facts.governor_outstanding_bytes.to_string(),
            self.facts.governor_available_bytes.to_string(),
            self.facts.active_grants.to_string(),
            self.active_queries.to_string(),
            self.queued_queries.to_string(),
            self.facts.gpu_execution_worker_count.to_string(),
            self.facts.gpu_execution_active_jobs.to_string(),
            self.facts.gpu_execution_queued_jobs.to_string(),
            self.facts.gpu_execution_active_query_jobs.to_string(),
            self.facts.pool_reserved_current_bytes.to_string(),
            self.facts.pool_reserved_high_bytes.to_string(),
            self.facts.pool_used_current_bytes.to_string(),
            self.facts.pool_used_high_bytes.to_string(),
            self.facts.allocation_failure_count.to_string(),
            self.facts.trim_retry_count.to_string(),
        ]
        .join("\t")
    }
}

pub(super) fn device_sample_rows(
    epoch_ms: u128,
    device_ordinals: &[usize],
    facts_by_device: &BTreeMap<usize, DeviceSampleFacts>,
    active_queries: usize,
    queued_queries: usize,
) -> Vec<DeviceSampleRow> {
    device_ordinals
        .iter()
        .map(|device_ordinal| DeviceSampleRow {
            epoch_ms,
            device_ordinal: *device_ordinal,
            active_queries,
            queued_queries,
            facts: facts_by_device.get(device_ordinal).copied().unwrap_or_default(),
        })
        .collect()
}

pub(super) fn collect_sample(
    backend: &NexusGpuBackend,
    device_ordinals: &[usize],
    managers: &BTreeMap<usize, cudf::DeviceMemoryManager>,
) -> (SampleRow, Vec<DeviceSampleRow>) {
    let epoch_ms = epoch_ms();
    let mut facts_by_device = device_ordinals
        .iter()
        .copied()
        .map(|ordinal| (ordinal, DeviceSampleFacts::default()))
        .collect::<BTreeMap<_, _>>();

    if let Ok(reports) = backend.device_ledger_reports() {
        for report in reports {
            if let Some(facts) = facts_by_device.get_mut(&report.device_ordinal) {
                facts.governor_capacity_bytes = report.managed_capacity_bytes;
                facts.governor_outstanding_bytes = report.live_grants_bytes;
                facts.governor_available_bytes = report.available_bytes;
                facts.active_grants = report.grants.len();
            }
        }
    }

    let (active_queries, queued_queries) = backend
        .admission_snapshot()
        .map(|snapshot| (snapshot.active_queries, snapshot.queued_queries))
        .unwrap_or_default();

    let mut allocation_failure_count = 0u64;
    let mut trim_retry_count = 0u64;
    for report in backend.device_memory_failure_reports().unwrap() {
        allocation_failure_count = allocation_failure_count.saturating_add(report.allocation_failure_count);
        trim_retry_count = trim_retry_count.saturating_add(report.trim_retry_count);
        if let Some(facts) = facts_by_device.get_mut(&report.device_ordinal) {
            facts.allocation_failure_count = report.allocation_failure_count;
            facts.trim_retry_count = report.trim_retry_count;
        }
    }

    for report in backend.gpu_execution_pool_reports().unwrap() {
        if let Some(facts) = facts_by_device.get_mut(&report.device_ordinal) {
            facts.gpu_execution_worker_count = report.worker_count;
            facts.gpu_execution_active_jobs = report.active_jobs;
            facts.gpu_execution_queued_jobs = report.queued_jobs;
            facts.gpu_execution_active_query_jobs = report.active_query_jobs;
        }
    }

    for (device_ordinal, manager) in managers {
        if let Some(facts) = facts_by_device.get_mut(device_ordinal)
            && let Ok(usage) = manager.pool_usage()
        {
            facts.pool_reserved_current_bytes = usage.reserved_current_bytes;
            facts.pool_reserved_high_bytes = usage.reserved_high_bytes;
            facts.pool_used_current_bytes = usage.used_current_bytes;
            facts.pool_used_high_bytes = usage.used_high_bytes;
        }
    }

    let primary_facts =
        device_ordinals.first().and_then(|ordinal| facts_by_device.get(ordinal)).copied().unwrap_or_default();
    let aggregate = SampleRow {
        epoch_ms,
        governor_capacity_bytes: primary_facts.governor_capacity_bytes,
        governor_outstanding_bytes: primary_facts.governor_outstanding_bytes,
        governor_available_bytes: primary_facts.governor_available_bytes,
        active_grants: primary_facts.active_grants,
        active_queries,
        queued_queries,
        pool_reserved_current_bytes: primary_facts.pool_reserved_current_bytes,
        pool_reserved_high_bytes: primary_facts.pool_reserved_high_bytes,
        pool_used_current_bytes: primary_facts.pool_used_current_bytes,
        pool_used_high_bytes: primary_facts.pool_used_high_bytes,
        allocation_failure_count,
        trim_retry_count,
    };
    let by_device = device_sample_rows(epoch_ms, device_ordinals, &facts_by_device, active_queries, queued_queries);
    (aggregate, by_device)
}

pub(super) fn spawn_sampler(
    backend: &NexusGpuBackend,
    config: &HarnessConfig,
    stop_flag: Arc<AtomicBool>,
    samples: Arc<Mutex<SampleCollection>>,
) -> std::thread::JoinHandle<()> {
    let backend = backend.clone();
    let interval = Duration::from_millis(config.sample_interval_ms);
    let device_ordinals = config.device_ordinals.clone();
    std::thread::spawn(move || {
        let managers = device_ordinals
            .iter()
            .filter_map(|ordinal| {
                let device = cudf::DeviceId::new(*ordinal).ok()?;
                cudf::DeviceMemoryManager::install_for_device(device).ok().map(|manager| (*ordinal, manager))
            })
            .collect::<BTreeMap<_, _>>();
        while !stop_flag.load(Ordering::SeqCst) {
            let (aggregate, by_device) = collect_sample(&backend, &device_ordinals, &managers);
            let mut samples = samples.lock().unwrap();
            samples.aggregate.push(aggregate);
            samples.by_device.extend(by_device);
            drop(samples);
            std::thread::sleep(interval);
        }
    })
}
