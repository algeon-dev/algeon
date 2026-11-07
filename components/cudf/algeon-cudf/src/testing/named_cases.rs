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
mod tests {
    use std::{cell::RefCell, panic};

    use super::run_selected_cases;

    thread_local! {
        static VISITED: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
    }

    fn first() {
        VISITED.with(|visited| visited.borrow_mut().push("first"));
    }

    fn second() {
        VISITED.with(|visited| visited.borrow_mut().push("second"));
    }

    fn fails_first() {
        first();
        panic!("first failure");
    }

    fn fails_second() {
        second();
        panic!("second failure");
    }

    #[test]
    fn selectors_match_substrings_and_preserve_case_order() {
        run_selected_cases(
            &[("case_first", first), ("unselected", || panic!("not selected")), ("case_second", second)],
            " second, first ",
        );
        VISITED.with(|visited| assert_eq!(*visited.borrow(), ["first", "second"]));
    }

    #[test]
    fn unmatched_selector_fails_without_running_cases() {
        for selector in ["missing", ",", " , "] {
            assert!(panic::catch_unwind(|| run_selected_cases(&[("first", first)], selector)).is_err());
        }
        VISITED.with(|visited| assert!(visited.borrow().is_empty()));
    }

    #[test]
    fn panicking_cases_are_aggregated_and_later_cases_run() {
        let failure = panic::catch_unwind(|| {
            run_selected_cases(&[("failed_first", fails_first), ("failed_second", fails_second), ("last", first)], "");
        })
        .unwrap_err();
        let message = failure.downcast_ref::<String>().unwrap();
        assert!(message.contains("2 of 3 cases failed: failed_first, failed_second"), "{message}");
        VISITED.with(|visited| assert_eq!(*visited.borrow(), ["first", "second", "first"]));
    }
}
