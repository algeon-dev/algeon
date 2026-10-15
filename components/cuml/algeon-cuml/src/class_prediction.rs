use cudf::{
    DataFrame, Exec, Expr, Runtime, Series,
    ops::expr::{col, lit, when},
};

use crate::error::{ErrorCode, FailurePhase, Result, error, from_cudf};

/// The most probable class of each row and that class's probability.
pub struct ClassPrediction {
    /// Int32 class index in `0..classes`. A tie between classes takes the
    /// lowest index.
    pub class: Series,
    /// Float32 probability of `class`, validated to lie in `[0, 1]`.
    pub probability: Series,
}

impl ClassPrediction {
    /// Reduces one probability column per class. The caller guarantees the
    /// columns are non-null; values outside `[0, 1]` fail validation.
    pub(crate) fn from_probabilities(runtime: &Runtime, columns: Vec<Series>, op: &'static str) -> Result<Self> {
        let classes = columns.len();
        let frame = with_row_max(class_frame(runtime, columns, op)?, classes, op)?;
        let (class, probability) = (column(&frame, "class", op)?, column(&frame, "best", op)?);
        Self::validated(class, probability, op)
    }

    /// Computes binary class predictions and probabilities from a class-1 decision score.
    /// Non-positive scores predict class 0 with probability `sigmoid(-z)`; positive
    /// scores predict class 1 with probability `sigmoid(z)`.
    pub(crate) fn from_logistic_score(runtime: &Runtime, score: Series, op: &'static str) -> Result<Self> {
        let frame = class_frame(runtime, vec![score], op)?;
        let class = when(col("c0").gt(0.0_f32)).then(1_i32).otherwise(0_i32);
        let probability = lit(1.0_f32) / (lit(1.0_f32) + (-col("c0").abs()).exp());
        Self::validated(eval(&class, &frame, op)?, eval(&probability, &frame, op)?, op)
    }

    /// Computes multiclass predictions and normalized probabilities from per-class decision scores.
    /// Subtracts row maxima before exponentiation to prevent floating-point overflow.
    pub(crate) fn from_softmax_scores(runtime: &Runtime, scores: Vec<Series>, op: &'static str) -> Result<Self> {
        let classes = scores.len();
        let frame = with_row_max(class_frame(runtime, scores, op)?, classes, op)?;
        let denominator = (0..classes)
            .map(|class| (col(class_column(class)) - col("best")).exp())
            .reduce(|sum, term| sum + term)
            .expect("softmax needs at least one class");
        let probability = lit(1.0_f32) / denominator;
        Self::validated(column(&frame, "class", op)?, eval(&probability, &frame, op)?, op)
    }

    fn validated(class: Series, probability: Series, op: &'static str) -> Result<Self> {
        let nan = probability
            .on(Exec::RETAINED)
            .is_nan()
            .and_then(|mask| mask.on(Exec::RETAINED).any_value())
            .map_err(|e| from_cudf(op, e))?;
        let (min, max) = probability.on(Exec::RETAINED).minmax_values::<f32>().map_err(|e| from_cudf(op, e))?;
        let in_range = |value: Option<f32>| value.is_some_and(|value| (0.0..=1.0).contains(&value));
        if nan || !in_range(min) || !in_range(max) {
            return Err(error(
                ErrorCode::OutputOutOfRange,
                FailurePhase::OutputValidation,
                op,
                "class probabilities must lie in [0, 1]",
            )
            .with_detail("observed_min", format!("{min:?}"))
            .with_detail("observed_max", format!("{max:?}")));
        }
        Ok(Self { class, probability })
    }
}

fn class_column(class: usize) -> String {
    format!("c{class}")
}

fn class_frame(runtime: &Runtime, columns: Vec<Series>, op: &'static str) -> Result<DataFrame> {
    runtime
        .empty_dataframe()
        .assign(columns.into_iter().enumerate().map(|(class, series)| (class_column(class), series)))
        .map_err(|e| from_cudf(op, e))
}

/// Appends `best` (row maximum across class columns) and `class` (lowest index containing `best`).
/// Each pass compares the candidate column against the running maximum.
fn with_row_max(frame: DataFrame, classes: usize, op: &'static str) -> Result<DataFrame> {
    let mut frame = frame
        .on(Exec::RETAINED)
        .with_columns([("best", col(class_column(0))), ("class", lit(0_i32))])
        .map_err(|e| from_cudf(op, e))?;
    for class in 1..classes {
        let value = col(class_column(class));
        let beats = value.clone().gt(col("best"));
        frame = frame
            .on(Exec::RETAINED)
            .with_columns([
                ("best", when(beats.clone()).then(value).otherwise(col("best"))),
                ("class", when(beats).then(class as i32).otherwise(col("class"))),
            ])
            .map_err(|e| from_cudf(op, e))?;
    }
    Ok(frame)
}

fn column(frame: &DataFrame, name: &str, op: &'static str) -> Result<Series> {
    frame.series(name).map_err(|e| from_cudf(op, e))
}

fn eval(expression: &Expr, frame: &DataFrame, op: &'static str) -> Result<Series> {
    expression.eval(frame).map_err(|e| from_cudf(op, e))
}
