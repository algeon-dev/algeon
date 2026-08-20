use std::collections::{BTreeMap, BTreeSet};

use super::{BinaryOperator, Expr};
use crate::core::{
    error::{Error, ErrorCode, FACT_TYPE_ID, Result},
    types::{DataType, TypeId, cudf_null_mask_allocation_size_bytes},
};

pub const DATAFRAME_EXPRESSION_ALLOCATION_PREFLIGHT_CONTRACT: &str =
    "libcudf_dataframe_expression_allocation_preflight_v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataFrameExpressionInputColumn {
    name: String,
    dtype: DataType,
    nullable: bool,
    retained_bytes: usize,
}

impl DataFrameExpressionInputColumn {
    pub fn new(name: impl Into<String>, dtype: DataType, nullable: bool, retained_bytes: usize) -> Self {
        Self { name: name.into(), dtype, nullable, retained_bytes }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn nullable(&self) -> bool {
        self.nullable
    }

    pub fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataFrameExpressionPeakGroup {
    referenced_input_relocation_bytes: usize,
    boolean_equality_output_bytes: usize,
    null_float64_literal_bytes: usize,
    copy_if_else_output_bytes: usize,
    unchanged_column_relocation_bytes: usize,
    peak_bytes: usize,
}

impl DataFrameExpressionPeakGroup {
    pub fn referenced_input_relocation_bytes(self) -> usize {
        self.referenced_input_relocation_bytes
    }

    pub fn boolean_equality_output_bytes(self) -> usize {
        self.boolean_equality_output_bytes
    }

    pub fn null_float64_literal_bytes(self) -> usize {
        self.null_float64_literal_bytes
    }

    pub fn copy_if_else_output_bytes(self) -> usize {
        self.copy_if_else_output_bytes
    }

    pub fn unchanged_column_relocation_bytes(self) -> usize {
        self.unchanged_column_relocation_bytes
    }

    pub fn peak_bytes(self) -> usize {
        self.peak_bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataFrameExpressionAllocationPreflight {
    boolean_equality_output_bytes: usize,
    null_float64_literal_bytes: usize,
    copy_if_else_output_bytes: usize,
    output_dataframe_retained_bytes: usize,
    unchanged_column_relocation_bytes: usize,
    temporary_workspace_peak_bytes: usize,
    peak_groups: [DataFrameExpressionPeakGroup; 2],
    active_phase_peak_bytes: usize,
}

impl DataFrameExpressionAllocationPreflight {
    pub fn boolean_equality_output_bytes(&self) -> usize {
        self.boolean_equality_output_bytes
    }

    pub fn null_float64_literal_bytes(&self) -> usize {
        self.null_float64_literal_bytes
    }

    pub fn copy_if_else_output_bytes(&self) -> usize {
        self.copy_if_else_output_bytes
    }

    pub fn output_dataframe_retained_bytes(&self) -> usize {
        self.output_dataframe_retained_bytes
    }

    pub fn unchanged_column_relocation_bytes(&self) -> usize {
        self.unchanged_column_relocation_bytes
    }

    pub fn temporary_workspace_peak_bytes(&self) -> usize {
        self.temporary_workspace_peak_bytes
    }

    pub fn peak_groups(&self) -> &[DataFrameExpressionPeakGroup; 2] {
        &self.peak_groups
    }

    pub fn active_phase_peak_bytes(&self) -> usize {
        self.active_phase_peak_bytes
    }
}

#[derive(Clone, Copy)]
struct ExpressionValue<'a> {
    dtype: DataType,
    nullable: bool,
    retained_bytes: usize,
    source: Option<&'a DataFrameExpressionInputColumn>,
}

#[derive(Clone, Copy)]
struct OutputAnalysis<'a> {
    value: ExpressionValue<'a>,
    temporary_workspace_peak_bytes: usize,
}

#[derive(Default)]
struct ExpressionComponents<'a> {
    referenced_columns: BTreeSet<&'a str>,
    boolean_equality_output_bytes: usize,
    null_float64_literal_bytes: usize,
    copy_if_else_output_bytes: usize,
}

fn preflight_error(reason: &'static str, message: &'static str) -> Error {
    Error::new_with(ErrorCode::UnsupportedOperation, "expr::dataframe_expression_allocation_preflight", message)
        .with_detail("resource", "expression_allocation")
        .with_detail("reason", reason)
        .with_detail("required_contract", DATAFRAME_EXPRESSION_ALLOCATION_PREFLIGHT_CONTRACT)
}

fn checked_add(lhs: usize, rhs: usize, resource: &'static str) -> Result<usize> {
    lhs.checked_add(rhs).ok_or_else(|| {
        Error::new_with(
            ErrorCode::SizeOverflow,
            "expr::dataframe_expression_allocation_preflight",
            "expression allocation component sum overflowed usize",
        )
        .with_detail("resource", resource)
        .with_detail("reason", "component_sum_overflowed_usize")
    })
}

fn fixed_width_column_bytes(rows: usize, dtype: DataType, nullable: bool) -> Result<usize> {
    let width = dtype.id().size_in_bytes()?.ok_or_else(|| {
        preflight_error("variable_width_expression_output", "expression preflight only supports fixed-width outputs")
    })?;
    let data_bytes = rows.checked_mul(width as usize).ok_or_else(|| {
        Error::new_with(
            ErrorCode::SizeOverflow,
            "expr::dataframe_expression_allocation_preflight",
            "expression output data bytes overflowed usize",
        )
        .with_detail("resource", "fixed_width_output")
        .with_detail("reason", "row_width_product_overflowed_usize")
    })?;
    checked_add(data_bytes, if nullable { cudf_null_mask_allocation_size_bytes(rows) } else { 0 }, "fixed_width_output")
}

fn unwrap_alias(expr: &Expr) -> &Expr {
    match expr {
        Expr::Alias { inner, .. } => unwrap_alias(inner),
        _ => expr,
    }
}

fn input_column<'a>(
    name: &str,
    inputs: &'a BTreeMap<&str, &'a DataFrameExpressionInputColumn>,
) -> Result<&'a DataFrameExpressionInputColumn> {
    inputs.get(name).copied().ok_or_else(|| {
        preflight_error("input_column_missing", "expression preflight input column is missing")
            .with_detail("column", name.to_string())
    })
}

fn direct_fixed_width_column<'a>(
    expr: &Expr,
    inputs: &'a BTreeMap<&str, &'a DataFrameExpressionInputColumn>,
    components: &mut ExpressionComponents<'a>,
) -> Result<ExpressionValue<'a>> {
    let Expr::Column(name) = unwrap_alias(expr) else {
        return Err(preflight_error(
            "fixed_width_column_required",
            "supported expression shape requires a direct fixed-width column",
        ));
    };
    let column = input_column(name, inputs)?;
    fixed_width_column_bytes(0, column.dtype, column.nullable)
        .map_err(|error| error.with_detail("column", name.clone()).with_fact(FACT_TYPE_ID, column.dtype.id()))?;
    components.referenced_columns.insert(column.name.as_str());
    Ok(ExpressionValue {
        dtype: column.dtype,
        nullable: column.nullable,
        retained_bytes: column.retained_bytes,
        source: Some(column),
    })
}

fn fixed_width_literal(
    expr: &Expr,
    rows: usize,
    components: &mut ExpressionComponents<'_>,
) -> Result<ExpressionValue<'static>> {
    let Expr::Literal(literal) = unwrap_alias(expr) else {
        return Err(preflight_error(
            "fixed_width_literal_required",
            "supported expression shape requires a fixed-width literal",
        ));
    };
    let Some((dtype, nullable)) = literal.fixed_width_metadata() else {
        return Err(preflight_error(
            "fixed_width_literal_required",
            "supported expression shape requires a fixed-width literal",
        ));
    };
    let retained_bytes = fixed_width_column_bytes(rows, dtype, nullable)?;
    components.null_float64_literal_bytes =
        checked_add(components.null_float64_literal_bytes, retained_bytes, "null_float64_literal_bytes")?;
    Ok(ExpressionValue { dtype, nullable, retained_bytes, source: None })
}

fn comparison_operand<'a>(
    expr: &Expr,
    rows: usize,
    inputs: &'a BTreeMap<&str, &'a DataFrameExpressionInputColumn>,
    components: &mut ExpressionComponents<'a>,
) -> Result<ExpressionValue<'a>> {
    match unwrap_alias(expr) {
        Expr::Column(_) => direct_fixed_width_column(expr, inputs, components),
        Expr::Literal(_) => fixed_width_literal(expr, rows, components),
        _ => Err(preflight_error(
            "comparison_operand_unsupported",
            "comparison preflight requires a direct fixed-width column or literal",
        )),
    }
}

fn fixed_width_comparison<'a>(
    expr: &Expr,
    rows: usize,
    inputs: &'a BTreeMap<&str, &'a DataFrameExpressionInputColumn>,
    components: &mut ExpressionComponents<'a>,
) -> Result<OutputAnalysis<'a>> {
    let Expr::BinaryOp { lhs, op, rhs } = unwrap_alias(expr) else {
        return Err(preflight_error(
            "fixed_width_comparison_required",
            "supported expression shape requires fixed-width equality or inequality",
        ));
    };
    if !matches!(op, BinaryOperator::Equal | BinaryOperator::NotEqual) {
        return Err(preflight_error(
            "comparison_operator_unsupported",
            "expression preflight supports equality and inequality comparisons",
        ));
    }
    let lhs = comparison_operand(lhs, rows, inputs, components)?;
    let rhs = comparison_operand(rhs, rows, inputs, components)?;
    if lhs.dtype != rhs.dtype {
        return Err(preflight_error(
            "comparison_operand_type_mismatch",
            "comparison operands must have the same fixed-width dtype",
        ));
    }
    let dtype = DataType::new(TypeId::Bool8);
    let nullable = lhs.nullable || rhs.nullable;
    let retained_bytes = fixed_width_column_bytes(rows, dtype, nullable)?;
    components.boolean_equality_output_bytes =
        checked_add(components.boolean_equality_output_bytes, retained_bytes, "boolean_equality_output_bytes")?;
    let temporary_workspace_peak_bytes =
        [lhs, rhs].into_iter().filter(|value| value.source.is_none()).try_fold(0usize, |bytes, value| {
            checked_add(bytes, value.retained_bytes, "comparison_temporary_workspace_peak_bytes")
        })?;
    Ok(OutputAnalysis {
        value: ExpressionValue { dtype, nullable, retained_bytes, source: None },
        temporary_workspace_peak_bytes,
    })
}

fn null_check<'a>(
    input: &Expr,
    rows: usize,
    inputs: &'a BTreeMap<&str, &'a DataFrameExpressionInputColumn>,
    components: &mut ExpressionComponents<'a>,
) -> Result<OutputAnalysis<'a>> {
    let Expr::Column(name) = unwrap_alias(input) else {
        return Err(preflight_error(
            "null_check_input_unsupported",
            "null-check preflight requires a direct input column",
        ));
    };
    let column = input_column(name, inputs)?;
    components.referenced_columns.insert(column.name.as_str());
    let dtype = DataType::new(TypeId::Bool8);
    let retained_bytes = fixed_width_column_bytes(rows, dtype, false)?;
    Ok(OutputAnalysis {
        value: ExpressionValue { dtype, nullable: false, retained_bytes, source: None },
        temporary_workspace_peak_bytes: 0,
    })
}

fn supported_output<'a>(
    expr: &Expr,
    rows: usize,
    inputs: &'a BTreeMap<&str, &'a DataFrameExpressionInputColumn>,
    components: &mut ExpressionComponents<'a>,
) -> Result<OutputAnalysis<'a>> {
    match unwrap_alias(expr) {
        Expr::Column(name) => {
            let column = input_column(name, inputs)?;
            components.referenced_columns.insert(column.name.as_str());
            Ok(OutputAnalysis {
                value: ExpressionValue {
                    dtype: column.dtype,
                    nullable: column.nullable,
                    retained_bytes: column.retained_bytes,
                    source: Some(column),
                },
                temporary_workspace_peak_bytes: 0,
            })
        },
        Expr::BinaryOp { .. } => fixed_width_comparison(expr, rows, inputs, components),
        Expr::IsNull(input) | Expr::IsNotNull(input) => null_check(input, rows, inputs, components),
        Expr::IfElse { condition, true_val, false_val } => {
            let condition = fixed_width_comparison(condition, rows, inputs, components)?;
            if condition.value.dtype.id() != TypeId::Bool8 {
                return Err(preflight_error("boolean_condition_required", "IfElse condition must be Boolean"));
            }
            let true_val = comparison_operand(true_val, rows, inputs, components)?;
            let false_val = comparison_operand(false_val, rows, inputs, components)?;
            if true_val.dtype != false_val.dtype {
                return Err(preflight_error(
                    "if_else_branch_type_mismatch",
                    "IfElse branches must have the same fixed-width dtype",
                ));
            }
            let nullable = true_val.nullable || false_val.nullable;
            let retained_bytes = fixed_width_column_bytes(rows, true_val.dtype, nullable)?;
            components.copy_if_else_output_bytes =
                checked_add(components.copy_if_else_output_bytes, retained_bytes, "copy_if_else_output_bytes")?;
            let branch_workspace_bytes = [true_val, false_val]
                .into_iter()
                .filter(|value| value.source.is_none())
                .try_fold(condition.value.retained_bytes, |bytes, value| {
                    checked_add(bytes, value.retained_bytes, "if_else_temporary_workspace_peak_bytes")
                })?;
            let condition_workspace_bytes = checked_add(
                condition.value.retained_bytes,
                condition.temporary_workspace_peak_bytes,
                "if_else_condition_workspace_peak_bytes",
            )?;
            Ok(OutputAnalysis {
                value: ExpressionValue { dtype: true_val.dtype, nullable, retained_bytes, source: None },
                temporary_workspace_peak_bytes: condition_workspace_bytes.max(branch_workspace_bytes),
            })
        },
        _ => Err(preflight_error(
            "expression_shape_unsupported",
            "expression allocation preflight does not support this expression shape",
        )),
    }
}

pub fn dataframe_expression_allocation_preflight_for_metadata(
    rows: usize,
    input_columns: &[DataFrameExpressionInputColumn],
    outputs: &[(String, Expr)],
    relocation_required: bool,
) -> Result<DataFrameExpressionAllocationPreflight> {
    let mut inputs = BTreeMap::new();
    for column in input_columns {
        if inputs.insert(column.name.as_str(), column).is_some() {
            return Err(preflight_error("duplicate_input_column", "expression preflight input names must be unique")
                .with_detail("column", column.name.clone()));
        }
    }

    let mut components = ExpressionComponents::default();
    let mut output_values = BTreeMap::new();
    let mut temporary_workspace_peak_bytes = 0usize;
    for (name, expr) in outputs {
        let analysis = supported_output(expr, rows, &inputs, &mut components)?;
        temporary_workspace_peak_bytes = temporary_workspace_peak_bytes.max(analysis.temporary_workspace_peak_bytes);
        output_values.insert(name.as_str(), analysis.value);
    }

    let mut output_dataframe_retained_bytes = 0usize;
    let mut unchanged_column_relocation_bytes = 0usize;
    for column in input_columns {
        if output_values.contains_key(column.name.as_str()) {
            continue;
        }
        output_dataframe_retained_bytes =
            checked_add(output_dataframe_retained_bytes, column.retained_bytes, "output_dataframe_retained_bytes")?;
        if relocation_required {
            unchanged_column_relocation_bytes = checked_add(
                unchanged_column_relocation_bytes,
                column.retained_bytes,
                "unchanged_column_relocation_bytes",
            )?;
        }
    }
    let mut completed_output_bytes = 0usize;
    for value in output_values.values() {
        output_dataframe_retained_bytes =
            checked_add(output_dataframe_retained_bytes, value.retained_bytes, "output_dataframe_retained_bytes")?;
        if value.source.is_none() {
            completed_output_bytes =
                checked_add(completed_output_bytes, value.retained_bytes, "completed_output_bytes")?;
        }
    }

    let referenced_input_relocation_bytes = if relocation_required {
        components.referenced_columns.iter().try_fold(0usize, |bytes, name| {
            checked_add(bytes, input_column(name, &inputs)?.retained_bytes, "referenced_input_relocation_bytes")
        })?
    } else {
        0
    };
    let evaluation_peak_bytes = [
        referenced_input_relocation_bytes,
        components.boolean_equality_output_bytes,
        components.null_float64_literal_bytes,
        components.copy_if_else_output_bytes,
    ]
    .into_iter()
    .try_fold(0usize, |bytes, component| checked_add(bytes, component, "evaluation_peak_bytes"))?;
    let assembly_peak_bytes =
        [referenced_input_relocation_bytes, completed_output_bytes, unchanged_column_relocation_bytes]
            .into_iter()
            .try_fold(0usize, |bytes, component| checked_add(bytes, component, "assembly_peak_bytes"))?;
    let evaluation_peak_group = DataFrameExpressionPeakGroup {
        referenced_input_relocation_bytes,
        boolean_equality_output_bytes: components.boolean_equality_output_bytes,
        null_float64_literal_bytes: components.null_float64_literal_bytes,
        copy_if_else_output_bytes: components.copy_if_else_output_bytes,
        unchanged_column_relocation_bytes: 0,
        peak_bytes: evaluation_peak_bytes,
    };
    let assembly_peak_group = DataFrameExpressionPeakGroup {
        referenced_input_relocation_bytes,
        boolean_equality_output_bytes: 0,
        null_float64_literal_bytes: 0,
        copy_if_else_output_bytes: 0,
        unchanged_column_relocation_bytes,
        peak_bytes: assembly_peak_bytes,
    };

    Ok(DataFrameExpressionAllocationPreflight {
        boolean_equality_output_bytes: components.boolean_equality_output_bytes,
        null_float64_literal_bytes: components.null_float64_literal_bytes,
        copy_if_else_output_bytes: components.copy_if_else_output_bytes,
        output_dataframe_retained_bytes,
        unchanged_column_relocation_bytes,
        temporary_workspace_peak_bytes,
        peak_groups: [evaluation_peak_group, assembly_peak_group],
        active_phase_peak_bytes: evaluation_peak_bytes.max(assembly_peak_bytes),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::expr::{col, lit, lit_null, when};

    #[test]
    fn case_preflight_reports_owner_components_and_phase_groups() {
        let rows = 1_000;
        let f64_bytes = fixed_width_column_bytes(rows, DataType::new(TypeId::Float64), false).unwrap();
        let inputs = [
            DataFrameExpressionInputColumn::new("score", DataType::new(TypeId::Float64), false, f64_bytes),
            DataFrameExpressionInputColumn::new("fallback", DataType::new(TypeId::Float64), false, f64_bytes),
        ];
        let outputs = [(
            "derived_score".to_string(),
            when(col("score").eq(col("fallback"))).then(col("score")).otherwise(lit_null(TypeId::Float64)),
        )];

        let same_domain = dataframe_expression_allocation_preflight_for_metadata(rows, &inputs, &outputs, false)
            .expect("supported CASE shape must preflight");
        assert_eq!(same_domain.boolean_equality_output_bytes(), rows);
        assert_eq!(same_domain.null_float64_literal_bytes(), f64_bytes + cudf_null_mask_allocation_size_bytes(rows));
        assert_eq!(same_domain.copy_if_else_output_bytes(), f64_bytes + cudf_null_mask_allocation_size_bytes(rows));
        assert_eq!(same_domain.unchanged_column_relocation_bytes(), 0);
        assert_eq!(same_domain.peak_groups()[0].boolean_equality_output_bytes(), rows);
        assert_eq!(same_domain.peak_groups()[1].unchanged_column_relocation_bytes(), 0);

        let relocated = dataframe_expression_allocation_preflight_for_metadata(rows, &inputs, &outputs, true)
            .expect("relocation-required CASE shape must preflight");
        assert_eq!(relocated.unchanged_column_relocation_bytes(), f64_bytes * 2);
        assert!(relocated.active_phase_peak_bytes() > same_domain.active_phase_peak_bytes());
    }

    #[test]
    fn fixed_width_case_outputs_share_one_workspace_peak() {
        let rows = 1_000;
        let int64_bytes = fixed_width_column_bytes(rows, DataType::new(TypeId::Int64), false).unwrap();
        let nullable_int64_bytes = fixed_width_column_bytes(rows, DataType::new(TypeId::Int64), true).unwrap();
        let bool_bytes = fixed_width_column_bytes(rows, DataType::new(TypeId::Bool8), false).unwrap();
        let inputs = [
            DataFrameExpressionInputColumn::new("distance", DataType::new(TypeId::Int64), false, int64_bytes),
            DataFrameExpressionInputColumn::new("predecessor", DataType::new(TypeId::Int64), false, int64_bytes),
        ];
        let outputs = [
            (
                "distance".to_string(),
                when(col("distance").eq(lit(i64::MAX))).then(lit_null(TypeId::Int64)).otherwise(col("distance")),
            ),
            (
                "predecessor".to_string(),
                when(col("predecessor").eq(lit(-1_i64))).then(lit_null(TypeId::Int64)).otherwise(col("predecessor")),
            ),
            ("reachable".to_string(), col("distance").ne(lit(i64::MAX))),
        ];

        let preflight = dataframe_expression_allocation_preflight_for_metadata(rows, &inputs, &outputs, false).unwrap();

        assert_eq!(preflight.temporary_workspace_peak_bytes(), bool_bytes + nullable_int64_bytes);
        assert_eq!(preflight.output_dataframe_retained_bytes(), nullable_int64_bytes * 2 + bool_bytes);
    }

    #[test]
    fn unsupported_arithmetic_fails_closed() {
        let inputs = [DataFrameExpressionInputColumn::new("score", DataType::new(TypeId::Float64), false, 8)];
        let error = dataframe_expression_allocation_preflight_for_metadata(
            1,
            &inputs,
            &[("derived_score".to_string(), col("score") + col("score"))],
            false,
        )
        .expect_err("unsupported arithmetic must remain unresolved");
        assert_eq!(error.detail_value("reason"), Some("comparison_operator_unsupported"));
    }

    #[test]
    fn null_checks_preflight_non_nullable_boolean_outputs() {
        let rows = 1_000;
        let bool_bytes = fixed_width_column_bytes(rows, DataType::new(TypeId::Bool8), false).unwrap();
        for nullable in [false, true] {
            let inputs =
                [DataFrameExpressionInputColumn::new("vertexId", DataType::new(TypeId::String), nullable, 32_000)];
            for expression in [col("vertexId").is_null(), col("vertexId").is_not_null()] {
                let preflight = dataframe_expression_allocation_preflight_for_metadata(
                    rows,
                    &inputs,
                    &[("metadata_missing".to_string(), expression)],
                    false,
                )
                .expect("direct null checks must have a finite owner preflight");
                assert_eq!(preflight.output_dataframe_retained_bytes(), 32_000 + bool_bytes);
                assert_eq!(preflight.temporary_workspace_peak_bytes(), 0);
            }
        }
    }

    #[test]
    fn null_check_preflight_handles_zero_rows_and_checked_output_overflow() {
        let inputs = [DataFrameExpressionInputColumn::new("vertexId", DataType::new(TypeId::String), true, 0)];
        let empty = dataframe_expression_allocation_preflight_for_metadata(
            0,
            &inputs,
            &[("metadata_missing".to_string(), col("vertexId").is_null())],
            false,
        )
        .expect("zero-row null checks must remain allocation-free");
        assert_eq!(empty.output_dataframe_retained_bytes(), 0);
        assert_eq!(empty.active_phase_peak_bytes(), 0);

        let error = dataframe_expression_allocation_preflight_for_metadata(
            usize::MAX,
            &inputs,
            &[
                ("is_null".to_string(), col("vertexId").is_null()),
                ("is_not_null".to_string(), col("vertexId").is_not_null()),
            ],
            false,
        )
        .expect_err("the retained output sum must use checked arithmetic");
        assert_eq!(error.detail_value("reason"), Some("component_sum_overflowed_usize"));

        let nested = dataframe_expression_allocation_preflight_for_metadata(
            1,
            &inputs,
            &[("nested".to_string(), col("vertexId").is_null().is_not_null())],
            false,
        )
        .expect_err("nested null checks remain outside the owner contract");
        assert_eq!(nested.detail_value("reason"), Some("null_check_input_unsupported"));
    }
}
