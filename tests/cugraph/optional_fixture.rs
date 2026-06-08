//! Shared skip-vs-required policy for local cuGraph fixtures and oracles.

pub const CUGRAPH_DATA_ROOT_ENV: &str = "CUDF_CUGRAPH_DATA_ROOT";
pub const CUGRAPH_FIXTURE_REQUIRED_ENV: &str = "CUDF_CUGRAPH_FIXTURE_REQUIRED";
pub const CUGRAPH_GRAPH500_RMAT_REQUIRED_ENV: &str = "CUDF_CUGRAPH_GRAPH500_RMAT_REQUIRED";
pub const CUGRAPH_ORACLE_REQUIRED_ENV: &str = "CUDF_CUGRAPH_ORACLE_REQUIRED";

pub fn handle_cugraph_fixture_unavailable(label: &str, message: impl AsRef<str>) {
    if let Some(required_env) = cugraph_fixture_required_by() {
        panic_optional_required(label, required_env, message.as_ref());
    }
    print_skip(label, CUGRAPH_FIXTURE_REQUIRED_ENV, message.as_ref());
}

pub fn handle_cugraph_graph500_rmat_unavailable(label: &str, message: impl AsRef<str>) {
    handle_optional_unavailable(label, CUGRAPH_GRAPH500_RMAT_REQUIRED_ENV, message);
}

pub fn handle_cugraph_python_oracle_unavailable(label: &str, message: impl AsRef<str>) {
    handle_optional_unavailable(label, CUGRAPH_ORACLE_REQUIRED_ENV, message);
}

fn handle_optional_unavailable(label: &str, required_env: &'static str, message: impl AsRef<str>) {
    if env_truthy(required_env) {
        panic_optional_required(label, required_env, message.as_ref());
    }
    print_skip(label, required_env, message.as_ref());
}

fn cugraph_fixture_required_by() -> Option<&'static str> {
    if env_truthy(CUGRAPH_FIXTURE_REQUIRED_ENV) {
        return Some(CUGRAPH_FIXTURE_REQUIRED_ENV);
    }
    if std::env::var_os(CUGRAPH_DATA_ROOT_ENV).is_some() {
        return Some(CUGRAPH_DATA_ROOT_ENV);
    }
    None
}

fn print_skip(label: &str, required_env: &'static str, message: &str) {
    eprintln!("skipping optional {label}: {} (set {required_env}=1 to require this fixture/oracle)", message.trim());
}

fn panic_optional_required(label: &str, required_env: &'static str, message: &str) -> ! {
    panic!("required optional {label} is unavailable because {required_env} is set: {}", message.trim());
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| truthy_value(&value))
}

fn truthy_value(value: &str) -> bool {
    matches!(value, "1" | "true" | "TRUE" | "yes" | "YES" | "required" | "REQUIRED")
}

#[cfg(test)]
mod tests {
    #[test]
    fn env_truthy_accepts_required_spellings() {
        for value in ["1", "true", "TRUE", "yes", "YES", "required", "REQUIRED"] {
            assert!(super::truthy_value(value));
        }
        for value in ["", "0", "false", "no", "optional"] {
            assert!(!super::truthy_value(value));
        }
    }
}
