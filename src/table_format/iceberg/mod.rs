//! User-facing Iceberg integration helpers.
//!
//! This module is available with the `iceberg` Cargo feature. Lower-level,
//! format-neutral scan facts remain under [`crate::table_format`].

mod adapter;
pub mod catalog;
mod planner;

pub use adapter::iceberg_scan_candidate_to_table_format_scan_plan;
pub use nexus_iceberg::{IcebergTableIdentifier, ResolvedIcebergTable, S3CredentialMode, SourceAccessPolicy};

pub use self::planner::{
    IcebergTableFormatPlanner, NexusIcebergTableProvider, iceberg_file_scan_tasks_to_table_format_scan_plan,
    iceberg_schema_to_datafusion_arrow_schema, source_access_policy_to_native_source_storage_profile,
};

/// Optional AWS profile name. Non-secret; preferred over raw env credentials.
pub const AWS_PROFILE_ENV: &str = "AWS_PROFILE";
/// Preferred AWS region env var.
pub const AWS_REGION_ENV: &str = "AWS_REGION";
/// Fallback AWS region env var when `AWS_REGION` is unset.
pub const AWS_DEFAULT_REGION_ENV: &str = "AWS_DEFAULT_REGION";
/// AWS access key id consumed at load time and never retained in diagnostics.
pub const AWS_ACCESS_KEY_ID_ENV: &str = "AWS_ACCESS_KEY_ID";
/// AWS secret key consumed at load time and never retained in diagnostics.
pub const AWS_SECRET_ACCESS_KEY_ENV: &str = "AWS_SECRET_ACCESS_KEY";
/// AWS session token consumed at load time and never retained in diagnostics.
pub const AWS_SESSION_TOKEN_ENV: &str = "AWS_SESSION_TOKEN";
/// AWS ecosystem-standard session-token alias accepted by AWS tooling.
pub const AWS_SECURITY_TOKEN_ENV: &str = "AWS_SECURITY_TOKEN";

/// Explicit S3 access key id for the REST FileIO path.
///
/// Deliberately distinct from `AWS_ACCESS_KEY_ID` so a local REST/RustFS stack
/// never silently picks up a developer's real AWS credentials. Consumed at
/// connect time and never retained in diagnostics.
pub const NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV: &str = "NEXUS_ICEBERG_S3_ACCESS_KEY_ID";
/// Explicit S3 secret access key for the REST FileIO path. Secret; consumed at
/// connect time and never retained in diagnostics. See
/// [`NEXUS_ICEBERG_S3_ACCESS_KEY_ID_ENV`] for why this is not `AWS_*`.
pub const NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY_ENV: &str = "NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY";
/// Optional explicit S3 session token for the REST FileIO path. Secret;
/// consumed at connect time and never retained in diagnostics.
pub const NEXUS_ICEBERG_S3_SESSION_TOKEN_ENV: &str = "NEXUS_ICEBERG_S3_SESSION_TOKEN";
/// Optional S3-compatible endpoint for Glue/REST FileIO.
pub const NEXUS_ICEBERG_S3_ENDPOINT_ENV: &str = "NEXUS_ICEBERG_S3_ENDPOINT";
/// Optional S3 region for Glue/REST FileIO.
pub const NEXUS_ICEBERG_S3_REGION_ENV: &str = "NEXUS_ICEBERG_S3_REGION";
/// Optional S3 path-style flag for Glue/REST FileIO.
pub const NEXUS_ICEBERG_S3_PATH_STYLE_ENV: &str = "NEXUS_ICEBERG_S3_PATH_STYLE";
/// Optional explicit S3 credential source for Glue/REST object storage.
///
/// The only accepted value is `default_chain`. Setting it is an explicit
/// operator opt-in to the ambient AWS default credential chain (env vars,
/// profile, IMDS instance role) for the configured object storage — including
/// when a custom `NEXUS_ICEBERG_S3_ENDPOINT` is set, where the default is
/// otherwise unsigned requests so a local S3-compatible stack never silently
/// picks up developer AWS credentials. Mutually exclusive with the static
/// `NEXUS_ICEBERG_S3_ACCESS_KEY_ID`/`NEXUS_ICEBERG_S3_SECRET_ACCESS_KEY` pair.
pub const NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV: &str = "NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE";
/// Accepted value of [`NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_ENV`] selecting the
/// ambient AWS default credential chain.
pub const NEXUS_ICEBERG_S3_CREDENTIAL_SOURCE_DEFAULT_CHAIN: &str = "default_chain";
