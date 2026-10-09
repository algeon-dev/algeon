use std::{env, panic};

/// Runs cases in order on the current thread and reports all failing names.
///
/// `ALGEON_TEST_CASES` selects names containing any comma-separated substring.
/// An unset or empty selector runs every case. A selector matching no case
/// fails the table. Cases must restore any state needed by subsequent cases
/// during unwinding; a panic does not stop the remaining cases.
pub fn run_named_cases(cases: &[(&str, fn())]) {
    let selector = env::var_os("ALGEON_TEST_CASES")
        .map(|value| value.into_string().expect("ALGEON_TEST_CASES must contain Unicode"))
        .unwrap_or_default();
    run_selected_cases(cases, &selector);
}

fn run_selected_cases(cases: &[(&str, fn())], selector: &str) {
    let filters = selector.split(',').map(str::trim).filter(|filter| !filter.is_empty()).collect::<Vec<_>>();
    let selected = cases
        .iter()
        .filter(|(name, _)| selector.trim().is_empty() || filters.iter().any(|filter| name.contains(filter)))
        .collect::<Vec<_>>();
    assert!(!selected.is_empty(), "ALGEON_TEST_CASES={selector:?} matched no cases");
    let failed = selected
        .iter()
        .filter(|(name, case)| {
            eprintln!("case `{name}` running");
            let panicked = panic::catch_unwind(*case).is_err();
            if panicked {
                eprintln!("case `{name}` failed; its panic message is above");
            }
            panicked
        })
        .map(|(name, _)| *name)
        .collect::<Vec<_>>();
    assert!(failed.is_empty(), "{} of {} cases failed: {}", failed.len(), selected.len(), failed.join(", "));
}

#[cfg(test)]
mod tests;
