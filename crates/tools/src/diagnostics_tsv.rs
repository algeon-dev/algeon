//! Reader for the `[section]` TSV tables in the server's diagnostic action
//! payloads (`GetAdmissionStats`, `GetQueryStats`).

use std::collections::BTreeMap;

pub(crate) type TsvRow = BTreeMap<String, String>;

/// Rows of the `[name]` section keyed by its header columns. The section ends
/// at a blank line or the next section; `#` lines inside it are skipped.
pub(crate) fn tsv_section(payload: &str, name: &str) -> Result<Vec<TsvRow>, String> {
    let marker = format!("[{name}]");
    let mut lines = payload.lines().skip_while(|line| *line != marker).skip(1);
    let header = lines.next().ok_or_else(|| format!("missing [{name}] section in payload:\n{payload}"))?;
    if header.is_empty() || header.starts_with('[') {
        return Err(format!("missing [{name}] TSV header in payload:\n{payload}"));
    }
    let columns = header.split('\t').collect::<Vec<_>>();
    let mut rows = Vec::new();
    for line in lines {
        if line.is_empty() || line.starts_with('[') {
            break;
        }
        if line.starts_with('#') {
            continue;
        }
        let values = line.split('\t').collect::<Vec<_>>();
        if values.len() != columns.len() {
            return Err(format!(
                "[{name}] TSV width mismatch: {} columns, {} values in {line:?}",
                columns.len(),
                values.len()
            ));
        }
        rows.push(
            columns.iter().zip(values).map(|(column, value)| ((*column).to_string(), value.to_string())).collect(),
        );
    }
    Ok(rows)
}
