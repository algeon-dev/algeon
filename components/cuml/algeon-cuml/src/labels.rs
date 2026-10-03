use cudf::{Exec, Series, TypeId};

use crate::{
    error::{ErrorCode, FailurePhase, Result, error, from_cudf},
    resources::CumlResources,
};

pub(crate) fn adopt_labels(resources: &CumlResources, labels: &Series, op: &'static str) -> Result<Series> {
    if labels.device() != resources.device() {
        return Err(error(
            ErrorCode::RuntimeBindingMismatch,
            FailurePhase::Preflight,
            op,
            "labels and resources use different CUDA devices",
        ));
    }
    // Native fit reads labels on the resource stream; relocation orders producer and consumer streams.
    labels.copy_into_runtime(resources.runtime()).map_err(|e| from_cudf(op, e))
}

pub(crate) fn validate_class_labels(labels: &Series, op: &'static str) -> Result<usize> {
    if labels.dtype().map_err(|e| from_cudf(op, e))?.id() != TypeId::Int32 {
        return Err(error(ErrorCode::UnsupportedDtype, FailurePhase::Preflight, op, "class labels must be Int32")
            .with_detail("reason", "unsupported_label_type"));
    }
    if labels.null_count() != 0 {
        return Err(error(
            ErrorCode::NullUnexpected,
            FailurePhase::Preflight,
            op,
            "class labels must not contain nulls",
        )
        .with_detail("reason", "label_nulls")
        .with_detail("null_count", labels.null_count()));
    }
    // Only scalar reduction results cross to host; class columns remain on device.
    let (min, max) = labels.on(Exec::RETAINED).minmax_values::<i32>().map_err(|e| from_cudf(op, e))?;
    let classes = labels.on(Exec::RETAINED).nunique().map_err(|e| from_cudf(op, e))?;
    if classes < 2 || min != Some(0) || max != i32::try_from(classes - 1).ok() {
        return Err(error(
            ErrorCode::LabelOutOfRange,
            FailurePhase::Preflight,
            op,
            "labels must contain contiguous classes 0..C-1 with C >= 2",
        )
        .with_detail("reason", if classes < 2 { "single_class" } else { "labels_not_contiguous" })
        .with_detail("observed_min", format!("{min:?}"))
        .with_detail("observed_max", format!("{max:?}"))
        .with_detail("distinct_count", classes));
    }
    Ok(classes)
}

pub(crate) fn validate_finite_targets(targets: &Series, op: &'static str) -> Result<()> {
    if targets.dtype().map_err(|e| from_cudf(op, e))?.id() != TypeId::Float32 {
        return Err(error(
            ErrorCode::UnsupportedDtype,
            FailurePhase::Preflight,
            op,
            "regression targets must be Float32",
        ));
    }
    if targets.null_count() != 0 {
        return Err(error(
            ErrorCode::NullUnexpected,
            FailurePhase::Preflight,
            op,
            "regression targets must not contain nulls",
        ));
    }
    let nan = targets
        .on(Exec::RETAINED)
        .is_nan()
        .map_err(|e| from_cudf(op, e))?
        .on(Exec::RETAINED)
        .any_value()
        .map_err(|e| from_cudf(op, e))?;
    let (min, max) = targets.on(Exec::RETAINED).minmax_values::<f32>().map_err(|e| from_cudf(op, e))?;
    if nan || min.is_none_or(|v| !v.is_finite()) || max.is_none_or(|v| !v.is_finite()) {
        return Err(error(ErrorCode::NonFiniteInput, FailurePhase::Preflight, op, "regression targets must be finite"));
    }
    Ok(())
}
