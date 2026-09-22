use std::collections::BTreeMap;

use datafusion::physical_plan::metrics::Label;

pub(super) fn labels_by_name(labels: &[Label]) -> BTreeMap<&str, &str> {
    labels.iter().map(|label| (label.name(), label.value())).collect()
}

pub(super) fn string_label(labels: &BTreeMap<&str, &str>, name: &str) -> String {
    labels.get(name).copied().unwrap_or("unknown").to_string()
}

pub(super) fn optional_string_label(labels: &BTreeMap<&str, &str>, name: &str) -> Option<String> {
    let value = labels.get(name).copied()?;
    (value != "none").then(|| value.to_string())
}

pub(super) fn usize_label(labels: &BTreeMap<&str, &str>, name: &str) -> usize {
    labels.get(name).and_then(|value| value.parse().ok()).unwrap_or(0)
}

pub(super) fn bool_label(labels: &BTreeMap<&str, &str>, name: &str) -> bool {
    labels.get(name).and_then(|value| value.parse().ok()).unwrap_or(false)
}

pub(super) fn optional_usize_label(labels: &BTreeMap<&str, &str>, name: &str) -> Option<usize> {
    let value = labels.get(name).copied()?;
    (value != "none").then(|| value.parse().ok()).flatten()
}

pub(super) fn optional_i64_label(labels: &BTreeMap<&str, &str>, name: &str) -> Option<i64> {
    let value = labels.get(name).copied()?;
    (value != "none").then(|| value.parse().ok()).flatten()
}

pub(super) fn optional_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

pub(super) fn optional_i64(value: Option<i64>) -> String {
    value.map_or_else(|| "none".to_string(), |value| value.to_string())
}

pub(super) fn optional_str(value: Option<&str>) -> String {
    value.unwrap_or("none").to_string()
}
