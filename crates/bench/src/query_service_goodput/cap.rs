use std::{collections::BTreeMap, error::Error, fs, path::Path};

use super::workload::{WorkloadFamily, WorkloadItem};

/// Independent request-cap assignment input for a paired goodput trace.
///
/// The workload item shape and the public report schemas deliberately do not
/// carry this configuration. A manifest can assign caps by sequence, item or
/// family, with the most specific matching rule winning at arrival time.
pub(super) const CAP_ASSIGNMENT_SCHEMA: &str = "algeon-goodput-cap-assignment";

const COLUMNS: [&str; 3] = ["kind", "key", "cap_bytes"];

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CapAssignment {
    default: Option<u64>,
    sequences: BTreeMap<u64, u64>,
    items: BTreeMap<String, u64>,
    families: BTreeMap<WorkloadFamily, u64>,
}

impl CapAssignment {
    pub(super) fn from_path(path: &Path) -> Result<Self, Box<dyn Error>> {
        let text = fs::read_to_string(path)
            .map_err(|error| format!("failed to read request cap manifest {}: {error}", path.display()))?;
        Self::parse(&text)
            .map_err(|error| format!("failed to parse request cap manifest {}: {error}", path.display()).into())
    }

    pub(super) fn parse(text: &str) -> Result<Self, Box<dyn Error>> {
        let mut assignment = Self::default();
        let mut schema_seen = false;
        let mut columns_seen = false;
        let mut entries = 0_usize;

        for (line_number, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim_end();
            if line.is_empty() {
                continue;
            }
            if let Some(comment) = line.strip_prefix("# ") {
                let (key, value) = comment
                    .split_once('=')
                    .ok_or_else(|| format!("cap assignment line {} must be key=value", line_number + 1))?;
                if key != "schema" {
                    return Err(format!("unknown cap assignment header {key} on line {}", line_number + 1).into());
                }
                if schema_seen {
                    return Err(format!("duplicate cap assignment schema on line {}", line_number + 1).into());
                }
                if value != CAP_ASSIGNMENT_SCHEMA {
                    return Err(format!("cap assignment schema {value} is not {CAP_ASSIGNMENT_SCHEMA}").into());
                }
                schema_seen = true;
                continue;
            }

            let fields = line.split('\t').collect::<Vec<_>>();
            if !columns_seen {
                if fields != COLUMNS {
                    return Err(format!(
                        "cap assignment column header on line {} does not match {}",
                        line_number + 1,
                        COLUMNS.join("\t")
                    )
                    .into());
                }
                columns_seen = true;
                continue;
            }
            if fields.len() != COLUMNS.len() {
                return Err(format!(
                    "cap assignment row on line {} has {} fields, expected {}",
                    line_number + 1,
                    fields.len(),
                    COLUMNS.len()
                )
                .into());
            }
            let cap_bytes = fields[2]
                .parse::<u64>()
                .map_err(|error| format!("invalid cap on line {}: {error}", line_number + 1))?;
            if cap_bytes == 0 {
                return Err(format!("cap on line {} must be greater than zero", line_number + 1).into());
            }

            match fields[0] {
                "default" => {
                    if !fields[1].is_empty() {
                        return Err(format!("default cap on line {} must have an empty key", line_number + 1).into());
                    }
                    if assignment.default.replace(cap_bytes).is_some() {
                        return Err(format!("duplicate default cap on line {}", line_number + 1).into());
                    }
                },
                "sequence" => {
                    let sequence = fields[1]
                        .parse::<u64>()
                        .map_err(|error| format!("invalid sequence on line {}: {error}", line_number + 1))?;
                    if assignment.sequences.insert(sequence, cap_bytes).is_some() {
                        return Err(format!("duplicate sequence {sequence} on line {}", line_number + 1).into());
                    }
                },
                "item" => {
                    if fields[1].is_empty() {
                        return Err(format!("item cap on line {} requires an item id", line_number + 1).into());
                    }
                    let item_id = fields[1].to_string();
                    if assignment.items.insert(item_id.clone(), cap_bytes).is_some() {
                        return Err(format!("duplicate item {item_id} on line {}", line_number + 1).into());
                    }
                },
                "family" => {
                    let family = parse_family(fields[1])
                        .ok_or_else(|| format!("unknown workload family {} on line {}", fields[1], line_number + 1))?;
                    if assignment.families.insert(family, cap_bytes).is_some() {
                        return Err(format!("duplicate family {} on line {}", fields[1], line_number + 1).into());
                    }
                },
                kind => return Err(format!("unknown cap assignment kind {kind} on line {}", line_number + 1).into()),
            }
            entries += 1;
        }

        if !schema_seen {
            return Err("cap assignment file is missing its schema header".into());
        }
        if !columns_seen {
            return Err("cap assignment file is missing its column header".into());
        }
        if entries == 0 {
            return Err("cap assignment file has no assignments".into());
        }
        Ok(assignment)
    }

    /// Resolves one trace request. Sequence assignments are the most
    /// specific, followed by item, family and the manifest default.
    pub(super) fn cap_for(&self, sequence: u64, item_id: &str, fallback: u64) -> u64 {
        self.sequences
            .get(&sequence)
            .copied()
            .or_else(|| self.items.get(item_id).copied())
            .or_else(|| parse_family_from_item(item_id).and_then(|family| self.families.get(&family).copied()))
            .or(self.default)
            .unwrap_or(fallback)
    }

    /// Resolves caps for single-flight calibration rows.
    ///
    /// Because sequence-only assignments cannot determine a cap for calibration,
    /// a manifest containing sequence overrides must provide an item, family,
    /// or default rule. Tying latency baselines to explicit caps ensures
    /// reproducible calibration. When sequence overrides are present, any item
    /// lacking a matching rule results in an error.
    pub(super) fn calibration_caps(&self, items: &[WorkloadItem], fallback: u64) -> Result<Vec<u64>, Box<dyn Error>> {
        items
            .iter()
            .map(|item| {
                self.items
                    .get(&item.id)
                    .copied()
                    .or_else(|| self.families.get(&item.family).copied())
                    .or(self.default)
                    .or_else(|| self.sequences.is_empty().then_some(fallback))
                    .ok_or_else(|| {
                        format!(
                            "request cap manifest has no calibration cap for {}; add an item, family or default assignment",
                            item.id
                        )
                        .into()
                    })
            })
            .collect()
    }

    pub(super) fn validate(&self, device_capacity_bytes: u64) -> Result<(), Box<dyn Error>> {
        let caps =
            self.default.iter().chain(self.sequences.values()).chain(self.items.values()).chain(self.families.values());
        for cap in caps {
            if *cap == 0 {
                return Err("request cap assignments must be greater than zero".into());
            }
            if *cap > device_capacity_bytes {
                return Err(format!(
                    "request cap assignment {cap} must not exceed device capacity {device_capacity_bytes}"
                )
                .into());
            }
        }
        Ok(())
    }

    /// Stable, compact representation recorded with calibration evidence.
    pub(super) fn descriptor(&self) -> String {
        let mut entries = Vec::new();
        if let Some(cap) = self.default {
            entries.push(format!("default:{cap}"));
        }
        entries.extend(self.families.iter().map(|(family, cap)| format!("family={}:{cap}", family.as_str())));
        entries.extend(self.items.iter().map(|(item, cap)| format!("item={item}:{cap}")));
        entries.extend(self.sequences.iter().map(|(sequence, cap)| format!("sequence={sequence}:{cap}")));
        if entries.is_empty() { "global".to_string() } else { entries.join(",") }
    }
}

fn parse_family(value: &str) -> Option<WorkloadFamily> {
    match value {
        "tpch" => Some(WorkloadFamily::Tpch),
        "tpcds" => Some(WorkloadFamily::Tpcds),
        "adhoc" => Some(WorkloadFamily::Adhoc),
        "graph" => Some(WorkloadFamily::Graph),
        _ => None,
    }
}

fn parse_family_from_item(item_id: &str) -> Option<WorkloadFamily> {
    item_id.split_once('.').and_then(|(family, _)| parse_family(family))
}
