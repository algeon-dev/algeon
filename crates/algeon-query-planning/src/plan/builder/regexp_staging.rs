//! Projections that compute `REGEXP_REPLACE` placements a native projection
//! does not evaluate.
//!
//! The projection executor sizes each replacement from its input column before
//! it runs the replacement, so an output may be a replacement over a column and
//! nothing else ([`Expr::projection_evaluates_regexp_replace`]).
//! [`stage_regexp_replace`] moves every other placement into projections that
//! run first. A replacement whose input reads the result of another replacement
//! runs one level later. Each level has an input projection, which computes the
//! replacement inputs that are not columns, and a replacement projection. Every
//! projection passes through the columns that later projections read. Equal
//! inputs and equal replacements are computed once, and an output that is
//! already a replacement over a column reads the staged column of an equal one.

use std::collections::BTreeSet;

use crate::{
    expr::{Expr, alias_wrapped_column},
    naming::first_available_numeric_suffix_from,
    plan::{DataType, Field, ProjectionOutput, SchemaRef},
};

const REPLACEMENT_COLUMN_BASE: &str = "__algeon_native_regexp_replace";
const REPLACEMENT_INPUT_COLUMN_BASE: &str = "__algeon_native_regexp_replace_input";

/// A projection split into the projections that run before it and its own
/// outputs.
pub(super) struct StagedProjection {
    /// Outputs of the projections that run first, in order.
    pub(super) stages: Vec<Vec<ProjectionOutput>>,
    pub(super) outputs: Vec<ProjectionOutput>,
}

/// Moves every `REGEXP_REPLACE` that `outputs` place where a native projection
/// does not evaluate it into earlier projections over `input_schema`.
///
/// Outputs a projection already evaluates are returned unchanged with no
/// stages, as are outputs naming a column `input_schema` lacks; plan
/// validation reports that column against the projection that names it. When
/// staging is needed, an output that is a replacement over a column and equals
/// a staged replacement becomes a rename of the staged column, so the
/// replacement runs once.
pub(super) fn stage_regexp_replace(input_schema: &SchemaRef, mut outputs: Vec<ProjectionOutput>) -> StagedProjection {
    let unresolved_column = || {
        outputs
            .iter()
            .flat_map(|output| output.expr.columns())
            .any(|column| input_schema.field_with_name(column.name()).is_err())
    };
    if outputs.iter().all(|output| output.expr.projection_evaluates_regexp_replace()) || unresolved_column() {
        return StagedProjection { stages: Vec::new(), outputs };
    }
    let mut staging = Staging { input_schema, stages: Vec::new(), computed: Vec::new() };
    for output in &mut outputs {
        if !output.expr.projection_evaluates_regexp_replace() {
            staging.extract(&mut output.expr);
        }
    }
    for output in &mut outputs {
        staging.reuse_staged_replacement(&mut output.expr);
    }
    StagedProjection { stages: staging.finish(&outputs), outputs }
}

struct Staging<'a> {
    input_schema: &'a SchemaRef,
    /// Columns each stage computes. Level `n` owns stage `2n`, which computes
    /// replacement inputs, and stage `2n + 1`, which computes replacements.
    stages: Vec<Vec<ProjectionOutput>>,
    computed: Vec<ComputedColumn>,
}

struct ComputedColumn {
    field: Field,
    stage: usize,
    /// The replacement or replacement input this column holds, so that an equal
    /// one of the same type reads it. An input never contains a replacement,
    /// because extraction replaces those first, so the two kinds cannot be equal.
    source: Expr,
}

impl Staging<'_> {
    /// Replaces each `REGEXP_REPLACE` in `expr`, innermost first, with the
    /// column that computes it.
    fn extract(&mut self, expr: &mut Expr) {
        for child in expr.children_mut() {
            self.extract(child);
        }
        if matches!(expr, Expr::RegexpReplace { .. }) {
            *expr = Expr::column(self.replacement_column(expr));
        }
    }

    /// Makes an output that is itself a replacement over a column read the
    /// column that already computes an equal replacement.
    fn reuse_staged_replacement(&self, expr: &mut Expr) {
        if let Expr::Alias { expr, .. } = expr {
            self.reuse_staged_replacement(expr);
        } else if let Expr::RegexpReplace { output_type, .. } = &*expr
            && let Some(column) = self.computed_column(expr, output_type)
        {
            *expr = Expr::column(column);
        }
    }

    fn replacement_column(&mut self, replacement: &Expr) -> String {
        let Expr::RegexpReplace { expr: input, pattern, replacement: template, output_type } = replacement else {
            unreachable!("only REGEXP_REPLACE expressions are staged");
        };
        let latest_read = input.columns().filter_map(|column| self.stage_of(column.name())).max();
        let input_stage = latest_read.map_or(0, |stage| stage + 2 - stage % 2);
        let input = if alias_wrapped_column(input).is_some() {
            input.as_ref().clone()
        } else {
            // A replacement input is a string. String representations share
            // one cuDF type and validation admits the implicit cast between
            // them, so the input column takes the replacement's string type.
            Expr::column(self.compute(
                input_stage,
                REPLACEMENT_INPUT_COLUMN_BASE,
                output_type,
                input,
                input.as_ref().clone(),
            ))
        };
        self.compute(
            input_stage + 1,
            REPLACEMENT_COLUMN_BASE,
            output_type,
            replacement,
            Expr::regexp_replace(input, pattern.clone(), template.clone(), output_type.clone()),
        )
    }

    /// The column that computes `expr` as `source`, adding it to `stage` unless
    /// an equal `source` of the same type already has one.
    fn compute(&mut self, stage: usize, base: &str, data_type: &DataType, source: &Expr, expr: Expr) -> String {
        if let Some(existing) = self.computed_column(source, data_type) {
            return existing;
        }
        let suffix = first_available_numeric_suffix_from(0, |suffix| self.is_taken(&format!("{base}_{suffix}")));
        let field = Field::new(format!("{base}_{suffix}"), data_type.clone(), true);
        if self.stages.len() <= stage {
            self.stages.resize_with(stage + 1, Vec::new);
        }
        self.stages[stage].push(ProjectionOutput::new(expr, field.clone()));
        let name = field.name().clone();
        self.computed.push(ComputedColumn { field, stage, source: source.clone() });
        name
    }

    fn computed_column(&self, source: &Expr, data_type: &DataType) -> Option<String> {
        self.computed
            .iter()
            .find(|column| column.source == *source && column.field.data_type() == data_type)
            .map(|column| column.field.name().clone())
    }

    fn is_taken(&self, name: &str) -> bool {
        self.input_schema.field_with_name(name).is_ok()
            || self.computed.iter().any(|column| column.field.name() == name)
    }

    fn stage_of(&self, name: &str) -> Option<usize> {
        self.computed.iter().find(|column| column.field.name() == name).map(|column| column.stage)
    }

    /// The non-empty stages, each passing through the columns that later
    /// stages and `outputs` read ahead of the columns it computes. Passed
    /// columns keep input-schema order, then computation order.
    fn finish(self, outputs: &[ProjectionOutput]) -> Vec<Vec<ProjectionOutput>> {
        let known = self
            .input_schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .chain(self.computed.iter().map(|column| column.field.clone()))
            .collect::<Vec<_>>();
        let mut read = outputs
            .iter()
            .flat_map(|output| output.expr.columns())
            .map(|column| column.name().to_string())
            .collect::<BTreeSet<_>>();
        let mut stages = Vec::new();
        for computed in self.stages.into_iter().rev().filter(|stage| !stage.is_empty()) {
            let defined = computed.iter().map(|output| output.field.name().as_str()).collect::<BTreeSet<_>>();
            let mut stage = known
                .iter()
                .filter(|field| read.contains(field.name()) && !defined.contains(field.name().as_str()))
                .map(|field| ProjectionOutput::new(Expr::column(field.name()), field.clone()))
                .collect::<Vec<_>>();
            read = stage
                .iter()
                .map(|output| output.field.name().clone())
                .chain(computed.iter().flat_map(|output| output.expr.columns()).map(|column| column.name().to_string()))
                .collect();
            stage.extend(computed);
            stages.push(stage);
        }
        stages.reverse();
        stages
    }
}
