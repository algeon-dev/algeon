use std::{collections::BTreeMap, error::Error, fmt::Write as _, fs, path::Path};

use super::workload::WorkloadFamily;

pub(super) const CALIBRATION_SCHEMA: &str = "algeon-goodput-calibration";

const COLUMNS: &[&str] = &[
    "item_id",
    "family",
    "work_units",
    "reps",
    "p50_ms",
    "p95_ms",
    "attempt_peak_aligned_bytes_p50",
    "attempt_peak_aligned_bytes_max",
    "disposition",
    "output_rows",
    "output_fingerprint",
];

/// Environment facts a measurement run must match before it may reuse a
/// frozen calibration.
///
/// The gate is deliberately limited to facts this process can observe. Source
/// and native-library identity is recorded by the driver and carried through
/// `source_identity`, which is reported but never compared.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(super) struct CalibrationIdentity {
    pub(super) gpu_model: String,
    pub(super) tpch_fingerprint: String,
    pub(super) tpcds_fingerprint: String,
    pub(super) graph_fingerprint: String,
}

#[derive(Debug, Clone)]
pub(super) struct CalibrationRow {
    pub(super) item_id: String,
    pub(super) family: WorkloadFamily,
    pub(super) work_units: u64,
    pub(super) reps: usize,
    pub(super) p50_ms: f64,
    pub(super) p95_ms: f64,
    pub(super) attempt_peak_aligned_bytes_p50: Option<u64>,
    pub(super) attempt_peak_aligned_bytes_max: Option<u64>,
    pub(super) disposition: String,
    pub(super) output_rows: usize,
    pub(super) output_fingerprint: String,
}

#[derive(Debug, Clone)]
pub(super) struct CalibrationStore {
    pub(super) identity: CalibrationIdentity,
    pub(super) header: BTreeMap<String, String>,
    pub(super) rows: Vec<CalibrationRow>,
}

impl CalibrationStore {
    pub(super) fn row(&self, item_id: &str) -> Option<&CalibrationRow> {
        self.rows.iter().find(|row| row.item_id == item_id)
    }

    /// Fails closed when the live environment differs from the frozen one.
    ///
    /// An empty recorded fingerprint means the calibration did not exercise
    /// that family, so a measurement mix that needs it must be recalibrated.
    pub(super) fn assert_identity(
        &self,
        live: &CalibrationIdentity,
        required: &[WorkloadFamily],
    ) -> Result<(), Box<dyn Error>> {
        if self.identity.gpu_model != live.gpu_model {
            return Err(format!(
                "calibration gpu_model {:?} does not match the live device {:?}; re-run calibration",
                self.identity.gpu_model, live.gpu_model
            )
            .into());
        }
        for family in required {
            let (recorded, observed) = match family {
                WorkloadFamily::Tpch | WorkloadFamily::Adhoc => {
                    (&self.identity.tpch_fingerprint, &live.tpch_fingerprint)
                },
                WorkloadFamily::Tpcds => (&self.identity.tpcds_fingerprint, &live.tpcds_fingerprint),
                WorkloadFamily::Graph => (&self.identity.graph_fingerprint, &live.graph_fingerprint),
            };
            if recorded.is_empty() || recorded != observed {
                return Err(format!(
                    "calibration dataset fingerprint for {} is {:?} but the live dataset is {:?}; re-run calibration",
                    family.as_str(),
                    recorded,
                    observed
                )
                .into());
            }
        }
        Ok(())
    }

    pub(super) fn to_tsv(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(output, "# schema={CALIBRATION_SCHEMA}");
        for (key, value) in &self.header {
            let _ = writeln!(output, "# {key}={value}");
        }
        output.push_str(&COLUMNS.join("\t"));
        output.push('\n');
        for row in &self.rows {
            let fields = [
                row.item_id.clone(),
                row.family.as_str().to_string(),
                row.work_units.to_string(),
                row.reps.to_string(),
                format!("{:.3}", row.p50_ms),
                format!("{:.3}", row.p95_ms),
                optional_u64(row.attempt_peak_aligned_bytes_p50),
                optional_u64(row.attempt_peak_aligned_bytes_max),
                row.disposition.clone(),
                row.output_rows.to_string(),
                row.output_fingerprint.clone(),
            ];
            debug_assert_eq!(fields.len(), COLUMNS.len());
            output.push_str(&fields.join("\t"));
            output.push('\n');
        }
        output
    }

    pub(super) fn write_to(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, self.to_tsv())?;
        Ok(())
    }

    pub(super) fn parse(text: &str) -> Result<Self, Box<dyn Error>> {
        let mut header = BTreeMap::new();
        let mut schema_seen = false;
        let mut columns: Option<Vec<&str>> = None;
        let mut rows = Vec::new();
        for line in text.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            if let Some(comment) = line.strip_prefix("# ") {
                let (key, value) = comment.split_once('=').ok_or("calibration header lines must be key=value")?;
                if key == "schema" {
                    if value != CALIBRATION_SCHEMA {
                        return Err(format!("calibration schema {value} is not {CALIBRATION_SCHEMA}").into());
                    }
                    schema_seen = true;
                } else {
                    header.insert(key.to_string(), value.to_string());
                }
                continue;
            }
            let fields = line.split('\t').collect::<Vec<_>>();
            match &columns {
                None => {
                    if fields != COLUMNS {
                        return Err("calibration column header does not match this harness".into());
                    }
                    columns = Some(fields);
                },
                Some(_) => rows.push(parse_row(&fields)?),
            }
        }
        if !schema_seen {
            return Err("calibration file is missing its schema header".into());
        }
        if columns.is_none() {
            return Err("calibration file is missing its column header".into());
        }
        let identity = CalibrationIdentity {
            gpu_model: header.get("gpu_model").cloned().unwrap_or_default(),
            tpch_fingerprint: header.get("tpch_fingerprint").cloned().unwrap_or_default(),
            tpcds_fingerprint: header.get("tpcds_fingerprint").cloned().unwrap_or_default(),
            graph_fingerprint: header.get("graph_fingerprint").cloned().unwrap_or_default(),
        };
        Ok(Self { identity, header, rows })
    }

    pub(super) fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        let text = fs::read_to_string(path)
            .map_err(|error| format!("failed to read calibration {}: {error}", path.display()))?;
        Self::parse(&text)
    }
}

fn parse_row(fields: &[&str]) -> Result<CalibrationRow, Box<dyn Error>> {
    if fields.len() != COLUMNS.len() {
        return Err(format!("calibration row has {} fields, expected {}", fields.len(), COLUMNS.len()).into());
    }
    let family = match fields[1] {
        "tpch" => WorkloadFamily::Tpch,
        "tpcds" => WorkloadFamily::Tpcds,
        "adhoc" => WorkloadFamily::Adhoc,
        "graph" => WorkloadFamily::Graph,
        other => return Err(format!("unknown calibration family {other}").into()),
    };
    Ok(CalibrationRow {
        item_id: fields[0].to_string(),
        family,
        work_units: fields[2].parse()?,
        reps: fields[3].parse()?,
        p50_ms: fields[4].parse()?,
        p95_ms: fields[5].parse()?,
        attempt_peak_aligned_bytes_p50: parse_optional_u64(fields[6])?,
        attempt_peak_aligned_bytes_max: parse_optional_u64(fields[7])?,
        disposition: fields[8].to_string(),
        output_rows: fields[9].parse()?,
        output_fingerprint: fields[10].to_string(),
    })
}

fn optional_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "unavailable".to_string(), |value| value.to_string())
}

fn parse_optional_u64(value: &str) -> Result<Option<u64>, Box<dyn Error>> {
    if value == "unavailable" {
        return Ok(None);
    }
    Ok(Some(value.parse()?))
}
