//! Shared helper primitives for report and benchmark harnesses.

pub use nexus_query_planning::util::{StableFnv1a64, max_option};

#[cfg(test)]
mod tests {
    use super::max_option;

    #[test]
    fn max_option_returns_max_present_value() {
        assert_eq!(max_option(Some(1usize), Some(3)), Some(3));
        assert_eq!(max_option(Some(4usize), None), Some(4));
        assert_eq!(max_option(None, Some(5usize)), Some(5));
        assert_eq!(max_option::<usize>(None, None), None);
    }
}
