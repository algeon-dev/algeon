use std::{fs, path::Path};

pub(crate) fn read_cargo_link_manifest(prefix: &Path) -> Result<Vec<String>, String> {
    let path = prefix.join("share/cuml_rust/cargo-link.txt");
    let content = fs::read_to_string(&path).map_err(|error| format!("could not read {}: {error}", path.display()))?;
    let directives =
        content.lines().filter(|line| line.starts_with("cargo:")).map(|line| line[6..].to_owned()).collect::<Vec<_>>();
    if directives.is_empty() {
        return Err(format!("{} contains no Cargo link directives", path.display()));
    }
    Ok(directives)
}
