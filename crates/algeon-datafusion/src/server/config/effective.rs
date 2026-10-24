//! The effective configuration projected back onto the TOML schema, for
//! `algeon_server --check-config` and the startup log.

use std::{fmt::Write as _, sync::OnceLock};

use query_runtime::runtime::S3SourceStorageCredential;

use super::{
    AlgeonGpuDeviceProfile, ConfigSources, ConfigValueSource, Duration, ServerConfig, ServerFile,
    ServerIcebergCatalogConfig, ServerIcebergCatalogKind, ServerIcebergRestAuth, ServerTlsFileConfig,
    env::scalar_leaves,
    file::{
        ServerAdmissionDeviceProfileFileConfig, ServerAdmissionFileConfig, ServerCugraphFileConfig,
        ServerDataFileConfig, ServerIcebergFileConfig, ServerModelsFileConfig, ServerNativeFileConfig,
        ServerObservabilityFileConfig, ServerS3FileConfig,
    },
    table_lookup,
};
use crate::server::backend::GpuDeviceMemoryLimit;

/// `config` in the file schema. A leaf left `None` is unset and takes an
/// automatic value. TLS paths come from `tls`, because the tonic config keeps
/// no paths. Iceberg catalogs stay out of the table so their secrets cannot
/// reach a rendering; [`render_iceberg_catalogs`] prints them. The struct
/// literals are exhaustive so a new file field must be projected here before
/// the crate compiles.
fn project(config: &ServerConfig, tls: Option<&ServerTlsFileConfig>) -> ServerFile {
    let native = config.native_optimizer_config.native_execution_config();
    let chunking = native.source_chunking();
    let observability = &config.observability;
    ServerFile {
        bind: Some(config.bind_addr.to_string()),
        statement_ticket_ttl_secs: Some(config.statement_ticket_ttl.as_secs()),
        http2_keepalive_interval_secs: Some(config.http2_keepalive.interval.as_secs()),
        http2_keepalive_timeout_secs: Some(config.http2_keepalive.timeout.as_secs()),
        init_sql: config.init_sql.clone(),
        admission: ServerAdmissionFileConfig {
            max_queued_attempts: Some(config.queue_policy.max_queued_attempts()),
            max_queue_wait_ms: Some(saturating_millis(config.queue_policy.max_queue_wait())),
            device_profiles: config.gpu_device_profiles.iter().map(project_profile).collect(),
        },
        native: ServerNativeFileConfig {
            execution_mode: Some(config.native_optimizer_config.execution_mode().as_str().to_string()),
            max_source_chunk_bytes: chunking.explicit_max_source_chunk_bytes().map(|bytes| bytes.get()),
            max_row_groups_per_chunk: Some(chunking.max_row_groups_per_chunk.get()),
            max_retry_attempts: Some(chunking.max_retry_attempts.get()),
            max_source_reads_in_flight: Some(native.max_source_reads_in_flight()),
            source_chunk_planning_mode: Some(native.source_chunk_planning_mode().as_str().to_string()),
            statistical_aggregate_execution_mode: Some(
                native.statistical_aggregate_execution_mode().as_str().to_string(),
            ),
        },
        observability: ServerObservabilityFileConfig {
            per_query_max_records: Some(observability.per_query_max_records),
            per_query_max_bytes: Some(observability.per_query_max_bytes),
            active_logical_capture_max_bytes: observability.active_logical_capture_max_bytes,
            global_sealed_max_count: Some(observability.global_sealed_max_count),
            global_sealed_max_bytes: Some(observability.global_sealed_max_bytes),
            sealed_ttl_secs: Some(observability.sealed_ttl.as_secs()),
            shutdown_grace_secs: Some(observability.shutdown_grace.as_secs()),
        },
        cugraph: match &config.cugraph {
            Some(cugraph) => ServerCugraphFileConfig {
                enabled: Some(true),
                construction_policy: Some(cugraph.default_graph_options.construction_policy.as_str().to_string()),
                max_input_staging_bytes: Some(cugraph.host_boundary_config.max_input_staging_bytes()),
                max_output_staging_bytes: Some(cugraph.host_boundary_config.max_output_staging_bytes()),
            },
            None => ServerCugraphFileConfig {
                enabled: Some(false),
                construction_policy: None,
                max_input_staging_bytes: None,
                max_output_staging_bytes: None,
            },
        },
        models: match &config.models {
            Some(models) => ServerModelsFileConfig {
                root: Some(models.root.clone()),
                max_model_bytes: Some(models.max_model_bytes),
            },
            None => ServerModelsFileConfig { root: None, max_model_bytes: None },
        },
        data: ServerDataFileConfig { root: config.data_root.clone() },
        s3: native.s3_storage().map(|storage| ServerS3FileConfig {
            endpoint: storage.endpoint.clone(),
            region: storage.region.clone(),
            path_style: storage.virtual_hosted_style_request.map(|virtual_hosted| !virtual_hosted),
            unsigned: Some(storage.credential == S3SourceStorageCredential::Unsigned),
        }),
        iceberg: ServerIcebergFileConfig { catalogs: Vec::new() },
        auth: config.auth.clone(),
        tls: tls.cloned(),
    }
}

fn project_profile(profile: &AlgeonGpuDeviceProfile) -> ServerAdmissionDeviceProfileFileConfig {
    let bounds = profile.installation_bounds();
    ServerAdmissionDeviceProfileFileConfig {
        device_ordinal: profile.device_ordinal(),
        memory_limit_bytes: match profile.memory_limit() {
            GpuDeviceMemoryLimit::Explicit { limit_bytes } => Some(limit_bytes),
            _ => None,
        },
        backend_reserve_bytes: Some(profile.backend_reserve_bytes()),
        cache_cap_bytes: Some(profile.cache_cap_bytes()),
        default_query_limit_bytes: profile.default_query_limit_bytes(),
        source_work_per_attempt: Some(profile.source_work_per_attempt()),
        max_active_attempts: Some(profile.max_active_attempts()),
        stream_pool_size: bounds.map(|bounds| bounds.stream_pool_size),
        worker_count: bounds.map(|bounds| bounds.worker_count),
        lane_count: bounds.map(|bounds| bounds.lane_count),
        install_telemetry: Some(profile.install_telemetry()),
    }
}

fn saturating_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn effective_table(config: &ServerConfig, tls: Option<&ServerTlsFileConfig>) -> toml::Table {
    toml::Table::try_from(project(config, tls)).expect("the server file schema serializes to a TOML table")
}

fn default_table() -> &'static toml::Table {
    static DEFAULTS: OnceLock<toml::Table> = OnceLock::new();
    DEFAULTS.get_or_init(|| {
        let (config, _) = ServerConfig::from_file(ServerFile::default(), &ConfigSources::default())
            .expect("the built-in defaults form a valid configuration");
        effective_table(&config, None)
    })
}

/// Every scalar setting whose effective value differs from its built-in
/// default, as `(path, value, source)`. An unset value renders as `unset`.
pub(crate) fn non_default_settings(
    config: &ServerConfig,
    sources: &ConfigSources,
) -> Vec<(&'static str, String, ConfigValueSource)> {
    let effective = effective_table(config, sources.tls.as_ref());
    let defaults = default_table();
    scalar_leaves()
        .iter()
        .filter_map(|leaf| {
            let value = table_lookup(&effective, &leaf.path);
            (value != table_lookup(defaults, &leaf.path)).then(|| {
                (
                    leaf.path.as_str(),
                    value.map_or_else(|| "unset".to_string(), render_value),
                    sources.source(&leaf.path),
                )
            })
        })
        .collect()
}

pub(super) fn render(config: &ServerConfig, sources: &ConfigSources) -> String {
    let effective = effective_table(config, sources.tls.as_ref());
    let mut out = String::new();
    let config_file = sources.config_file().map_or_else(|| "none".to_string(), |path| path.display().to_string());
    let _ = writeln!(out, "# Effective Algeon server configuration. Config file: {config_file}");

    let mut sections: Vec<(&str, Vec<&super::env::ScalarLeaf>)> = Vec::new();
    for leaf in scalar_leaves() {
        let parent = leaf.path.rsplit_once('.').map_or("", |(parent, _)| parent);
        match sections.iter_mut().find(|(name, _)| *name == parent) {
            Some((_, leaves)) => leaves.push(leaf),
            None => sections.push((parent, vec![leaf])),
        }
    }
    sections.sort_by_key(|(name, _)| !name.is_empty());

    for (section, leaves) in sections {
        if !section.is_empty() {
            // A header over only unset keys would declare an empty table,
            // which `[tls]` for one does not accept.
            let any_set = leaves.iter().any(|leaf| table_lookup(&effective, &leaf.path).is_some());
            let _ = writeln!(out, "\n{}[{section}]", if any_set { "" } else { "# " });
        }
        for leaf in leaves {
            let key = leaf.path.rsplit_once('.').map_or(leaf.path.as_str(), |(_, key)| key);
            let source = sources.source(&leaf.path);
            match table_lookup(&effective, &leaf.path) {
                Some(value) => {
                    let _ = writeln!(out, "{key} = {}  # source: {source}", render_value(value));
                },
                None => {
                    let _ = writeln!(out, "# {key} is unset  # source: {source}");
                },
            }
        }
        match section {
            "admission" => render_device_profiles(&mut out, &effective, sources),
            "auth.mode" => render_users(&mut out, &effective),
            _ => {},
        }
    }
    render_iceberg_catalogs(&mut out, &config.iceberg_catalogs, sources);
    out
}

/// `[[iceberg.catalogs]]` from the typed config. `credential` and `token`
/// render as `is set` comments.
fn render_iceberg_catalogs(out: &mut String, catalogs: &[ServerIcebergCatalogConfig], sources: &ConfigSources) {
    let file_catalogs = table_lookup(&sources.file_table, "iceberg.catalogs").and_then(toml::Value::as_array);
    for (index, catalog) in catalogs.iter().enumerate() {
        let file_entry = file_catalogs.and_then(|entries| entries.get(index)).and_then(toml::Value::as_table);
        let file_auth = file_entry.and_then(|entry| entry.get("auth")).and_then(toml::Value::as_table);
        let _ = writeln!(out, "\n[[iceberg.catalogs]]");
        render_string(out, "name", Some(&catalog.name), file_entry);
        match &catalog.kind {
            ServerIcebergCatalogKind::Rest(rest) => {
                render_string(out, "kind", Some("rest"), file_entry);
                render_string(out, "uri", Some(&rest.uri), file_entry);
                render_string(out, "warehouse", rest.warehouse.as_deref(), file_entry);
                render_string(out, "prefix", rest.prefix.as_deref(), file_entry);
                let _ = writeln!(out, "\n[iceberg.catalogs.auth]");
                match &rest.auth {
                    ServerIcebergRestAuth::None => render_string(out, "kind", Some("none"), file_auth),
                    ServerIcebergRestAuth::OAuth2 { server_uri, scope, .. } => {
                        render_string(out, "kind", Some("oauth2"), file_auth);
                        let _ = writeln!(out, "# credential is set  # source: file");
                        render_string(out, "server_uri", server_uri.as_deref(), file_auth);
                        render_string(out, "scope", scope.as_deref(), file_auth);
                    },
                    ServerIcebergRestAuth::Token(_) => {
                        render_string(out, "kind", Some("token"), file_auth);
                        let _ = writeln!(out, "# token is set  # source: file");
                    },
                }
            },
            ServerIcebergCatalogKind::Glue(glue) => {
                render_string(out, "kind", Some("glue"), file_entry);
                render_string(out, "warehouse", Some(&glue.warehouse), file_entry);
                render_string(out, "catalog_id", glue.catalog_id.as_deref(), file_entry);
                render_string(out, "region", glue.region.as_deref(), file_entry);
                render_string(out, "endpoint", glue.endpoint.as_deref(), file_entry);
            },
        }
    }
}

fn render_string(out: &mut String, key: &str, value: Option<&str>, file: Option<&toml::Table>) {
    let source = if file.is_some_and(|file| file.contains_key(key)) { "file" } else { "default" };
    match value {
        Some(value) => {
            let _ = writeln!(out, "{key} = {}  # source: {source}", render_value(&toml::Value::from(value)));
        },
        None => {
            let _ = writeln!(out, "# {key} is unset  # source: {source}");
        },
    }
}

fn render_device_profiles(out: &mut String, effective: &toml::Table, sources: &ConfigSources) {
    let profiles = table_lookup(effective, "admission.device_profiles").and_then(toml::Value::as_array);
    let file_profiles = table_lookup(&sources.file_table, "admission.device_profiles").and_then(toml::Value::as_array);
    for (index, profile) in profiles.into_iter().flatten().filter_map(toml::Value::as_table).enumerate() {
        if sources.device_profiles_defaulted() {
            let _ = writeln!(out, "\n[[admission.device_profiles]]  # defaulted");
        } else {
            let _ = writeln!(out, "\n[[admission.device_profiles]]");
        }
        let file_profile = file_profiles.and_then(|profiles| profiles.get(index)).and_then(toml::Value::as_table);
        for key in device_profile_keys() {
            let source = if file_profile.is_some_and(|file| file.contains_key(key)) { "file" } else { "default" };
            match profile.get(key) {
                Some(value) => {
                    let _ = writeln!(out, "{key} = {}  # source: {source}", render_value(value));
                },
                None => {
                    let _ = writeln!(out, "# {key} is unset  # source: {source}");
                },
            }
        }
    }
}

fn device_profile_keys() -> &'static [String] {
    static KEYS: OnceLock<Vec<String>> = OnceLock::new();
    KEYS.get_or_init(|| {
        let template = toml::Table::try_from(ServerAdmissionDeviceProfileFileConfig::leaf_type_template())
            .expect("a device profile serializes to a TOML table");
        let mut keys = template.keys().cloned().collect::<Vec<_>>();
        keys.sort_by_key(|key| key != "device_ordinal");
        keys
    })
}

fn render_users(out: &mut String, effective: &toml::Table) {
    let users = table_lookup(effective, "auth.mode.users").and_then(toml::Value::as_array);
    for user in users.into_iter().flatten().filter_map(toml::Value::as_table) {
        let _ = writeln!(out, "\n[[auth.mode.users]]");
        if let Some(username) = user.get("username") {
            let _ = writeln!(out, "username = {}  # source: file", render_value(username));
        }
        let _ = writeln!(out, "# password_hash is set  # source: file");
    }
}

fn render_value(value: &toml::Value) -> String {
    value.to_string()
}
