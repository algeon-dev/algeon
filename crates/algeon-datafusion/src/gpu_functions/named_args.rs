use std::collections::{BTreeMap, BTreeSet};

use datafusion::{
    common::utils::datafusion_strsim::levenshtein,
    sql::sqlparser::ast::{
        Expr as SqlExpr, FunctionArg, FunctionArgExpr, FunctionArgOperator, Ident, Query as SqlQuery, TableFunctionArgs,
    },
};

use crate::error::{Error, ErrorCode, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NamedArgumentKind {
    Relation,
    Value,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NamedArgumentDefinition {
    pub(crate) name: &'static str,
    pub(crate) kind: NamedArgumentKind,
    pub(crate) required: bool,
}

#[derive(Debug)]
pub(crate) enum NamedArgumentValue {
    Relation(Box<SqlQuery>),
    Value(Box<SqlExpr>),
}

#[derive(Debug)]
pub(crate) struct BoundNamedArguments {
    values: BTreeMap<&'static str, NamedArgumentValue>,
}

impl BoundNamedArguments {
    pub(crate) fn take_relation(&mut self, name: &'static str) -> Option<Box<SqlQuery>> {
        match self.values.remove(name) {
            Some(NamedArgumentValue::Relation(query)) => Some(query),
            Some(NamedArgumentValue::Value(_)) => unreachable!("named argument kind was validated"),
            None => None,
        }
    }

    pub(crate) fn take_value(&mut self, name: &'static str) -> Option<SqlExpr> {
        match self.values.remove(name) {
            Some(NamedArgumentValue::Value(value)) => Some(*value),
            Some(NamedArgumentValue::Relation(_)) => unreachable!("named argument kind was validated"),
            None => None,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

pub(crate) fn bind_named_arguments(
    function_name: &str,
    arguments: TableFunctionArgs,
    definitions: &[NamedArgumentDefinition],
    normalize_ident: impl Fn(Ident) -> String,
) -> Result<BoundNamedArguments> {
    if arguments.settings.is_some() {
        return Err(binding_error(
            function_name,
            "unsupported_table_function_modifier",
            "GPU relation functions do not accept a SETTINGS clause",
        ));
    }

    let definitions_by_name =
        definitions.iter().map(|definition| (definition.name, *definition)).collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    let mut values = BTreeMap::new();
    for argument in arguments.args {
        let FunctionArg::Named { name, arg, operator } = argument else {
            return Err(binding_error(
                function_name,
                "positional_argument_not_supported",
                "GPU relation functions require every argument to use name => value syntax",
            ));
        };
        if operator != FunctionArgOperator::RightArrow {
            return Err(binding_error(
                function_name,
                "invalid_named_argument_operator",
                "GPU relation function arguments must use the => operator",
            ));
        }
        let normalized = normalize_ident(name);
        let Some(definition) = definitions_by_name.get(normalized.as_str()).copied() else {
            return Err(binding_error(
                function_name,
                "unknown_argument",
                unknown_name_message("argument", &normalized, function_name, definitions.iter().map(|d| d.name)),
            ));
        };
        if !seen.insert(definition.name) {
            return Err(binding_error(
                function_name,
                "duplicate_argument",
                format!("argument '{}' is specified more than once for {function_name}", definition.name),
            ));
        }
        let FunctionArgExpr::Expr(expression) = arg else {
            return Err(binding_error(
                function_name,
                "invalid_argument_shape",
                format!("argument '{}' must be an expression", definition.name),
            ));
        };
        let value = match (definition.kind, expression) {
            (NamedArgumentKind::Relation, SqlExpr::Subquery(query)) => NamedArgumentValue::Relation(query),
            (NamedArgumentKind::Relation, _) => {
                return Err(binding_error(
                    function_name,
                    "relation_argument_not_subquery",
                    format!("relation argument '{}' must be a parenthesized query", definition.name),
                ));
            },
            (NamedArgumentKind::Value, expression) => NamedArgumentValue::Value(Box::new(expression)),
        };
        values.insert(definition.name, value);
    }

    if let Some(missing) = definitions.iter().find(|definition| definition.required && !seen.contains(definition.name))
    {
        return Err(binding_error(
            function_name,
            "missing_argument",
            format!("required argument '{}' is missing for {function_name}", missing.name),
        ));
    }
    Ok(BoundNamedArguments { values })
}

/// Message for a name that the function does not declare: the closest declared
/// name when one is near enough, then every declared name.
pub(crate) fn unknown_name_message<'a>(
    kind: &str,
    supplied: &str,
    function_name: &str,
    declared: impl IntoIterator<Item = &'a str> + Clone,
) -> String {
    let mut message = format!("unknown {kind} '{supplied}' for {function_name}.");
    if let Some(closest) = closest_name(supplied, declared.clone()) {
        message.push_str(&format!(" Did you mean '{closest}'?"));
    }
    message.push_str(&format!(" Valid {kind}s: {}.", declared.into_iter().collect::<Vec<_>>().join(", ")));
    message
}

/// The declared name nearest to `supplied`, or `None` when none is near.
///
/// A name is near when it is within half its length in edits, the threshold
/// DataFusion applies to its "Did you mean" column hints. The comparison runs
/// against the whole declared name and against each leading run of its
/// `_`-separated words, so `source` and `sources` both reach `source_vertex`.
/// Ties go to the shorter name.
fn closest_name<'a>(supplied: &str, declared: impl IntoIterator<Item = &'a str>) -> Option<&'a str> {
    let supplied = supplied.to_lowercase();
    declared
        .into_iter()
        .filter_map(|name| {
            let lowered = name.to_lowercase();
            let prefix_ends = lowered.match_indices('_').map(|(index, _)| index).chain([lowered.len()]);
            let distance = prefix_ends
                .filter_map(|end| {
                    let prefix = &lowered[..end];
                    let distance = levenshtein(&supplied, prefix);
                    let longest = supplied.chars().count().max(prefix.chars().count());
                    (distance * 2 <= longest).then_some(distance)
                })
                .min()?;
            Some((distance, name.len(), name))
        })
        .min()
        .map(|(_, _, name)| name)
}

fn binding_error(function_name: &str, reason: &'static str, message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidRequest, "algeon_datafusion.gpu_functions.bind", message)
        .with_log_safe_fact("function_name", function_name)
        .with_log_safe_fact("reason", reason)
}
