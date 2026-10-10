//! Nearest-name suggestions for misspelled configuration variables and GPU
//! function names.

/// The candidate most likely meant by `name`, or `None` when no candidate is
/// similar enough to name without misleading the caller.
///
/// Only candidates that share `prefix` with `name` qualify, and the parts
/// after the prefix are compared. A candidate qualifies when one part contains
/// the other and the shorter part has at least three characters (a truncated
/// name, an added suffix, or a renamed section such as `execution_mode` inside
/// `native_execution_mode`), or when the parts are within an edit distance of
/// `max(2, shorter / 4)`. Among qualifying candidates the smallest distance
/// wins, then the shortest candidate, then the lexically smallest one.
pub(crate) fn closest_name<'a>(
    name: &str,
    prefix: &str,
    candidates: impl IntoIterator<Item = &'a str>,
) -> Option<&'a str> {
    let stem = name.strip_prefix(prefix)?;
    candidates
        .into_iter()
        .filter_map(|candidate| {
            let candidate_stem = candidate.strip_prefix(prefix)?;
            let distance = edit_distance(stem, candidate_stem);
            let shorter = stem.len().min(candidate_stem.len());
            let contained = shorter >= 3 && (candidate_stem.contains(stem) || stem.contains(candidate_stem));
            (contained || distance <= (shorter / 4).max(2)).then_some((distance, candidate.len(), candidate))
        })
        .min()
        .map(|(_, _, candidate)| candidate)
}

/// Levenshtein distance over bytes; the names compared here are ASCII.
fn edit_distance(left: &str, right: &str) -> usize {
    let right = right.as_bytes();
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    let mut current = vec![0; right.len() + 1];
    for (row, left_byte) in left.bytes().enumerate() {
        current[0] = row + 1;
        for (column, right_byte) in right.iter().enumerate() {
            let substitution = previous[column] + usize::from(left_byte != *right_byte);
            current[column + 1] = substitution.min(previous[column + 1] + 1).min(current[column] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.len()]
}

#[cfg(test)]
mod tests {
    use super::closest_name;

    const ENV: &str = "ALGEON_SERVER_";
    const KNOWN: [&str; 6] = [
        "ALGEON_SERVER_BIND",
        "ALGEON_SERVER_NATIVE_EXECUTION_MODE",
        "ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS",
        "ALGEON_SERVER_TLS_CERT_PATH",
        "ALGEON_SERVER_CUGRAPH_ENABLED",
        "ALGEON_SERVER_CUGRAPH_CONSTRUCTION_POLICY",
    ];

    #[test]
    fn truncations_renamed_sections_and_typos_find_their_name() {
        assert_eq!(closest_name("ALGEON_SERVER_TLS_CERT", ENV, KNOWN), Some("ALGEON_SERVER_TLS_CERT_PATH"));
        assert_eq!(
            closest_name("ALGEON_SERVER_ADMISSION_MAX_QUEUED", ENV, KNOWN),
            Some("ALGEON_SERVER_ADMISSION_MAX_QUEUED_ATTEMPTS")
        );
        assert_eq!(
            closest_name("ALGEON_SERVER_EXECUTION_MODE", ENV, KNOWN),
            Some("ALGEON_SERVER_NATIVE_EXECUTION_MODE")
        );
        assert_eq!(
            closest_name("ALGEON_SERVER_NATIVE_EXECUTON_MODE", ENV, KNOWN),
            Some("ALGEON_SERVER_NATIVE_EXECUTION_MODE")
        );
    }

    #[test]
    fn distant_names_get_no_suggestion() {
        for name in ["ALGEON_SERVER_GPU_DEVICES", "ALGEON_SERVER_S3_REGION", "ALGEON_SERVER_CUGRAPH_DIRECTED"] {
            assert_eq!(closest_name(name, ENV, KNOWN), None, "{name}");
        }
    }

    #[test]
    fn gpu_function_names_prefer_the_nearest_name_then_a_contained_stem() {
        let registered = ["cugraph_pagerank", "cugraph_personalized_pagerank", "cuvs_brute_force_knn", "cuvs_kmeans"];
        assert_eq!(closest_name("cugraph_page_rank", "cugraph_", registered), Some("cugraph_pagerank"));
        assert_eq!(closest_name("cuvs_knn", "cuvs_", registered), Some("cuvs_brute_force_knn"));
        assert_eq!(closest_name("cuvs_kmeans_plus", "cuvs_", registered), Some("cuvs_kmeans"));
        assert_eq!(closest_name("gpu_frobnicate", "gpu_", registered), None);
        assert_eq!(closest_name("cuvs_xy", "cuvs_", registered), None);
    }
}
