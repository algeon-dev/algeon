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
