//! DataFusion contracts shared by dense-relation SQL families.

use arrow::datatypes::SchemaRef;
use datafusion::{
    catalog::Session,
    common::{DFSchema, DataFusionError, Result as DataFusionResult, TableReference},
    datasource::provider_as_source,
    execution::SessionState,
    logical_expr::{FetchType, LogicalPlan, LogicalPlanBuilder, SkipType, SortExpr, planner::RelationPlannerContext},
    sql::sqlparser::ast::{Expr as SqlExpr, TableAlias, TableFactor},
};
use query_planning::plan::{DenseVectorBinding, GpuOptionDefinition, GpuOptionKind};
use serde_json::{Map, Value, json};

use super::{
    NamedArgumentDefinition, NamedArgumentKind, bind_named_arguments,
    dense_relation_errors::DenseRelationErrorContext,
    resolve_input_ordering,
    scalar_arguments::{scalar_json, scalar_literal},
};
use crate::error::{DataFusionErrorExt, Error, ErrorCode, Result};

/// Typed reads of one call's canonical options. Canonicalization fills every
/// option a definition declares with its value or default and checks its kind,
/// so a failed read means the canonical object and the definitions disagree.
pub(crate) struct CanonicalOptions<'a> {
    values: &'a Map<String, Value>,
    errors: &'static DenseRelationErrorContext,
    sql_function_name: &'static str,
}

/// The ID field, feature vector, and optional label a relation binding names.
#[derive(Debug)]
pub(crate) struct CanonicalRelationBinding {
    pub(crate) id: String,
    pub(crate) vector: DenseVectorBinding,
    pub(crate) label: Option<String>,
}

impl<'a> CanonicalOptions<'a> {
    pub(crate) fn new(
        values: &'a Map<String, Value>,
        errors: &'static DenseRelationErrorContext,
        sql_function_name: &'static str,
    ) -> Self {
        Self { values, errors, sql_function_name }
    }

    pub(crate) fn u32(&self, name: &'static str) -> Result<u32> {
        self.value(name)?.as_u64().and_then(|value| u32::try_from(value).ok()).ok_or_else(|| self.mismatch(name))
    }

    pub(crate) fn i32(&self, name: &'static str) -> Result<i32> {
        self.value(name)?.as_u64().and_then(|value| i32::try_from(value).ok()).ok_or_else(|| self.mismatch(name))
    }

    pub(crate) fn u64(&self, name: &'static str) -> Result<u64> {
        self.value(name)?.as_u64().ok_or_else(|| self.mismatch(name))
    }

    pub(crate) fn usize(&self, name: &'static str) -> Result<usize> {
        self.value(name)?.as_u64().and_then(|value| usize::try_from(value).ok()).ok_or_else(|| self.mismatch(name))
    }

    pub(crate) fn optional_u32(&self, name: &'static str) -> Result<Option<u32>> {
        self.values.get(name).map(|_| self.u32(name)).transpose()
    }

    pub(crate) fn f64(&self, name: &'static str) -> Result<f64> {
        self.value(name)?.as_f64().ok_or_else(|| self.mismatch(name))
    }

    pub(crate) fn bool(&self, name: &'static str) -> Result<bool> {
        self.value(name)?.as_bool().ok_or_else(|| self.mismatch(name))
    }

    pub(crate) fn str(&self, name: &'static str) -> Result<&'a str> {
        self.value(name)?.as_str().ok_or_else(|| self.mismatch(name))
    }

    /// The typed value of an enumerated option, from its canonical spellings.
    pub(crate) fn choice<T: Copy>(&self, name: &'static str, choices: &[(&str, T)]) -> Result<T> {
        self.parsed(name, |value| choices.iter().find(|(spelling, _)| *spelling == value).map(|(_, choice)| *choice))
    }

    /// The typed value `parse` reads from a string option.
    pub(crate) fn parsed<T>(&self, name: &'static str, parse: impl FnOnce(&str) -> Option<T>) -> Result<T> {
        parse(self.str(name)?).ok_or_else(|| self.mismatch(name))
    }

    /// The binding of the relation argument `role`.
    pub(crate) fn relation_binding(&self, role: &'static str) -> Result<CanonicalRelationBinding> {
        let binding = self.value(role)?.as_object().ok_or_else(|| self.mismatch(role))?;
        let text = |key: &str| binding.get(key).and_then(Value::as_str).map(str::to_owned);
        let id = text("id").ok_or_else(|| self.mismatch(role))?;
        let vector = binding.get("vector").and_then(Value::as_object).ok_or_else(|| self.mismatch(role))?;
        let vector =
            match (vector.get("columns").and_then(Value::as_array), vector.get("column").and_then(Value::as_str)) {
                (Some(columns), None) => DenseVectorBinding::Columns(
                    columns
                        .iter()
                        .map(|column| column.as_str().map(str::to_owned))
                        .collect::<Option<_>>()
                        .ok_or_else(|| self.mismatch(role))?,
                ),
                (None, Some(column)) => DenseVectorBinding::VectorColumn(column.to_owned()),
                _ => return Err(self.mismatch(role)),
            };
        Ok(CanonicalRelationBinding { id, vector, label: text("label") })
    }

    /// The value options, without relation bindings, by name.
    pub(crate) fn value_options(&self, definitions: &[GpuOptionDefinition]) -> Map<String, Value> {
        definitions
            .iter()
            .filter(|option| !matches!(option.kind, GpuOptionKind::RelationBinding))
            .filter_map(|option| self.values.get(option.name).map(|value| (option.name.to_string(), value.clone())))
            .collect()
    }

    fn value(&self, name: &'static str) -> Result<&'a Value> {
        self.values.get(name).ok_or_else(|| self.mismatch(name))
    }

    fn mismatch(&self, name: &'static str) -> Error {
        Error::new(
            ErrorCode::Internal,
            self.errors.validate_operation,
            format!("canonical {} option '{name}' is missing or has another type", self.errors.family),
        )
        .with_log_safe_fact(self.errors.operation_fact, self.sql_function_name)
        .with_log_safe_fact("argument", name)
    }
}

/// How a relation argument reads a column named `label`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LabelColumn {
    /// The training target; the relation must have it.
    Required,
    /// The relation is predicted for and must not have it.
    Forbidden,
    /// An ordinary feature field.
    Feature,
}

/// One relation argument of a dense-relation function.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DenseRelationArgument {
    pub(crate) role: &'static str,
    pub(crate) label: LabelColumn,
}

/// A dense-relation SQL table function as its family declares it: the error
/// context, the SQL name, the relation arguments in role order, and the
/// options.
pub(crate) struct DenseRelationFunction {
    pub(crate) errors: &'static DenseRelationErrorContext,
    pub(crate) sql_function_name: &'static str,
    pub(crate) relations: Vec<DenseRelationArgument>,
    pub(crate) options: &'static [GpuOptionDefinition],
}

/// A call's relation inputs and options, planned before its logical node.
pub(crate) struct DenseRelationCall {
    pub(crate) inputs: Vec<LogicalPlan>,
    /// Top-level `ORDER BY` of each relation argument, in role order.
    pub(crate) input_orderings: Vec<Vec<SortExpr>>,
    pub(crate) options_json: String,
    pub(crate) alias: Option<TableAlias>,
}

/// The normalized name of the single-part table function `relation` calls.
pub(crate) fn table_function_name(relation: &TableFactor, context: &dyn RelationPlannerContext) -> Option<String> {
    let TableFactor::Table { name, args: Some(_), .. } = relation else {
        return None;
    };
    let [part] = name.0.as_slice() else {
        return None;
    };
    Some(context.normalize_ident(part.as_ident()?.clone()))
}

impl DenseRelationFunction {
    pub(crate) fn invalid_call(&self, reason: &'static str, message: impl Into<String>) -> Error {
        self.errors.invalid_call(self.sql_function_name, reason, message)
    }

    pub(crate) fn canonical_options<'a>(&self, values: &'a Map<String, Value>) -> CanonicalOptions<'a> {
        CanonicalOptions::new(values, self.errors, self.sql_function_name)
    }

    /// Plans the relation arguments and literal option values of
    /// `relation`, a call of this function.
    pub(crate) fn plan_call(
        &self,
        relation: TableFactor,
        context: &mut dyn RelationPlannerContext,
    ) -> DataFusionResult<DenseRelationCall> {
        let unsupported_modifier = || {
            self.invalid_call(
                "unsupported_table_function_modifier",
                format!("{} relation functions accept named arguments and a result alias", self.errors.family),
            )
        };
        let TableFactor::Table {
            alias,
            args: Some(function_args),
            with_hints,
            version,
            with_ordinality,
            partitions,
            json_path,
            sample,
            index_hints,
            ..
        } = relation
        else {
            return Err(unsupported_modifier().into());
        };
        if !with_hints.is_empty()
            || version.is_some()
            || with_ordinality
            || !partitions.is_empty()
            || json_path.is_some()
            || sample.is_some()
            || !index_hints.is_empty()
        {
            return Err(unsupported_modifier().into());
        }
        let definitions =
            dense_relation_argument_definitions(self.relations.iter().map(|relation| relation.role), self.options);
        let mut arguments = bind_named_arguments(self.sql_function_name, function_args, &definitions, |ident| {
            context.normalize_ident(ident)
        })?;
        let mut inputs = Vec::with_capacity(self.relations.len());
        let mut input_orderings = Vec::with_capacity(self.relations.len());
        for relation in &self.relations {
            // The named binder rejects a call without every required relation.
            let Some(query) = arguments.take_relation(relation.role) else {
                return Err(self
                    .invalid_call(
                        "missing_relation_argument",
                        format!("{} requires the relation argument '{}'", self.sql_function_name, relation.role),
                    )
                    .into());
            };
            let input = context.plan(TableFactor::Derived {
                lateral: false,
                subquery: query.clone(),
                alias: None,
                sample: None,
            })?;
            input_orderings.push(resolve_input_ordering(&query, &input, context, |key| {
                self.errors.input_ordering_key_not_projected(self.sql_function_name, relation.role, key)
            })?);
            inputs.push(input);
        }
        let mut values = Map::new();
        for option in self.options.iter().filter(|option| !matches!(option.kind, GpuOptionKind::RelationBinding)) {
            if let Some(expression) = arguments.take_value(option.name) {
                values.insert(option.name.to_string(), self.literal_value(option.name, expression, context)?);
            }
        }
        debug_assert!(arguments.is_empty());
        let options_json = self.options_json(&inputs, values)?;
        Ok(DenseRelationCall { inputs, input_orderings, options_json, alias })
    }

    /// The options JSON of a call with `inputs` and the value options
    /// `values`: the values plus each relation's binding, inferred from its
    /// schema.
    pub(crate) fn options_json(&self, inputs: &[LogicalPlan], mut values: Map<String, Value>) -> Result<String> {
        if inputs.len() != self.relations.len() {
            return Err(self.invalid_call(
                "invalid_relation_count",
                format!("{} requires {} relation argument(s)", self.sql_function_name, self.relations.len()),
            ));
        }
        for (relation, input) in self.relations.iter().zip(inputs) {
            if values.contains_key(relation.role) {
                return Err(self.invalid_call(
                    "unknown_argument",
                    format!(
                        "{} relation role '{}' cannot appear in the value options object",
                        self.errors.family, relation.role
                    ),
                ));
            }
            values.insert(relation.role.to_string(), self.relation_binding(input, *relation)?);
        }
        Ok(Value::Object(values).to_string())
    }

    /// The binding of `input`: its `id`, its feature fields, and its `label`
    /// when the relation trains on one.
    fn relation_binding(&self, input: &LogicalPlan, relation: DenseRelationArgument) -> Result<Value> {
        let schema = input.schema();
        let role = relation.role;
        let family = self.errors.family;
        schema
            .field_with_unqualified_name("id")
            .map_err(|_| self.errors.field_not_found(self.sql_function_name, role, "id"))?;
        let has_label = schema.field_with_unqualified_name("label").is_ok();
        match relation.label {
            LabelColumn::Required if !has_label => {
                return Err(
                    self.invalid_call("missing_label", format!("{family} {role} relation requires a 'label' column"))
                );
            },
            LabelColumn::Forbidden if has_label => {
                return Err(self.invalid_call(
                    "predict_relation_has_label",
                    format!("{family} {role} relation must omit 'label'"),
                ));
            },
            LabelColumn::Required | LabelColumn::Forbidden | LabelColumn::Feature => {},
        }
        let feature_names = schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .filter(|name| *name != "id" && (relation.label == LabelColumn::Feature || *name != "label"))
            .collect::<Vec<_>>();
        if feature_names.is_empty() {
            return Err(self.invalid_call(
                "missing_vector_features",
                format!("{family} {role} relation requires at least one feature field after 'id'"),
            ));
        }
        let vector = if feature_names.contains(&"vector") {
            if feature_names.len() != 1 {
                return Err(self.invalid_call(
                    "vector_field_with_additional_features",
                    format!("{family} {role} relation cannot combine 'vector' with other feature fields"),
                ));
            }
            json!({"column": "vector"})
        } else {
            json!({"columns": feature_names})
        };
        Ok(if relation.label == LabelColumn::Required {
            json!({"id": "id", "vector": vector, "label": "label"})
        } else {
            json!({"id": "id", "vector": vector})
        })
    }

    fn literal_value(
        &self,
        name: &'static str,
        expression: SqlExpr,
        context: &mut dyn RelationPlannerContext,
    ) -> DataFusionResult<Value> {
        let expression = context.sql_expr_to_logical_expr(expression, &DFSchema::empty())?;
        let invalid = || {
            self.invalid_call(
                "value_argument_not_literal",
                format!("{} argument '{name}' must be a scalar literal", self.errors.family),
            )
        };
        scalar_literal(&invalid, expression)
            .and_then(|value| scalar_json(&invalid, value))
            .map_err(DataFusionError::from)
    }
}

pub(crate) struct ResolvedSchemaOnlyInput {
    pub(crate) plan: LogicalPlan,
    pub(crate) schema: SchemaRef,
}

pub(crate) async fn resolve_schema_only_input(
    session: &dyn Session,
    name: &str,
    table_reference: TableReference,
) -> std::result::Result<ResolvedSchemaOnlyInput, String> {
    let state = session.as_any().downcast_ref::<SessionState>().ok_or_else(|| {
        "GPU validation requires a DataFusion SessionState so named relations can be resolved".to_string()
    })?;
    let catalog_options = &state.config_options().catalog;
    let resolved = table_reference.clone().resolve(&catalog_options.default_catalog, &catalog_options.default_schema);
    let catalog =
        state.catalog_list().catalog(&resolved.catalog).ok_or_else(|| format!("table '{name}' was not found"))?;
    let schema = catalog.schema(&resolved.schema).ok_or_else(|| format!("table '{name}' was not found"))?;
    let provider = schema
        .table(&resolved.table)
        .await
        .map_err(|error| format!("could not resolve table '{name}': {error}"))?
        .ok_or_else(|| format!("table '{name}' was not found"))?;
    let provider_schema = provider.schema();
    let plan = LogicalPlanBuilder::scan(table_reference, provider_as_source(provider), None)
        .map_err(|error| format!("could not construct schema-only input for table '{name}': {error}"))?
        .build()
        .map_err(|error| format!("could not construct schema-only input for table '{name}': {error}"))?;
    Ok(ResolvedSchemaOnlyInput { plan, schema: provider_schema })
}

pub(crate) fn dense_relation_argument_definitions(
    roles: impl Iterator<Item = &'static str>,
    options: &[GpuOptionDefinition],
) -> Vec<NamedArgumentDefinition> {
    roles
        .map(|name| NamedArgumentDefinition { name, kind: NamedArgumentKind::Relation, required: true })
        .chain(options.iter().filter_map(|option| {
            (!matches!(option.kind, GpuOptionKind::RelationBinding)).then_some(NamedArgumentDefinition {
                name: option.name,
                kind: NamedArgumentKind::Value,
                required: option.required,
            })
        }))
        .collect()
}

pub(crate) fn known_row_count(plan: &LogicalPlan) -> Option<usize> {
    match plan {
        LogicalPlan::EmptyRelation(empty) => Some(usize::from(empty.produce_one_row)),
        LogicalPlan::Values(values) => Some(values.values.len()),
        LogicalPlan::Projection(projection) => known_row_count(&projection.input),
        LogicalPlan::SubqueryAlias(alias) => known_row_count(&alias.input),
        LogicalPlan::Limit(limit) => {
            let input_rows = match limit.get_skip_type().ok()? {
                SkipType::Literal(skip) => known_row_count(&limit.input)?.saturating_sub(skip),
                SkipType::UnsupportedExpr => return None,
            };
            match limit.get_fetch_type().ok()? {
                FetchType::Literal(fetch) => Some(input_rows.min(fetch.unwrap_or(input_rows))),
                FetchType::UnsupportedExpr => None,
            }
        },
        _ => None,
    }
}

pub(crate) fn validation_request_schema<'a>(
    roles: impl Iterator<Item = &'a str> + Clone,
    options_schema: &Value,
) -> Value {
    let relation_properties = roles
        .clone()
        .map(|role| {
            (
                role.to_string(),
                json!({
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["table"],
                    "properties": {"table": {"type": "string"}}
                }),
            )
        })
        .collect::<Map<_, _>>();
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": false,
        "required": ["relations", "options"],
        "properties": {
            "relations": {
                "type": "object",
                "additionalProperties": false,
                "required": roles.collect::<Vec<_>>(),
                "properties": relation_properties,
            },
            "options": options_schema,
        }
    })
}

/// Projects a DataFusion error into validation details; an error without an
/// adapter cause reports `fallback_code` with the DataFusion message.
pub(crate) fn validation_error(
    error: &DataFusionError,
    details: &mut Value,
    fallback_code: &'static str,
) -> (String, String) {
    let Some(adapter) = error.find_algeon_error() else {
        details["path"] = Value::String("options".to_string());
        return (fallback_code.to_string(), error.to_string());
    };
    adapter_validation_error(adapter, details, fallback_code)
}

/// Projects the typed rejection and its argument or relation locator into validation details.
pub(crate) fn adapter_validation_error(
    error: &Error,
    details: &mut Value,
    fallback_code: &'static str,
) -> (String, String) {
    let code = match error.code() {
        ErrorCode::FieldNotFound => "column_not_found",
        ErrorCode::UnsupportedType => "unsupported_type",
        _ => error.facts().get("reason").unwrap_or(fallback_code),
    };
    let facts = error.facts();
    // The validation result goes to the caller, so it carries the caller's
    // field detail along with the facts.
    let detail = |key| facts.get(key).or_else(|| error.client_detail().and_then(|detail| detail.value(key)));
    let path = match (facts.get("argument"), facts.get("relation_role"), detail("field")) {
        (Some(argument), _, _) => format!("options.{argument}"),
        (None, Some(role), Some(field)) => format!("relations.{role}.fields.{field}"),
        (None, Some(role), None) => format!("relations.{role}"),
        (None, None, _) => "options".to_string(),
    };
    details["path"] = Value::String(path);
    for key in ["argument", "relation_role", "field", "actual_type", "expected_type"] {
        if let Some(value) = detail(key) {
            details[key] = Value::String(value.to_string());
        }
    }
    (code.to_string(), error.message().to_string())
}

pub(crate) fn schema_json(schema: &SchemaRef) -> Value {
    json!({
        "fields": schema.fields().iter().map(|field| json!({
            "name": field.name(),
            "data_type": format!("{:?}", field.data_type()),
            "nullable": field.is_nullable(),
            "metadata": field.metadata(),
        })).collect::<Vec<_>>(),
        "metadata": schema.metadata(),
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const ERRORS: DenseRelationErrorContext = DenseRelationErrorContext {
        family: "cuVS",
        operation_fact: "cuvs_operation",
        validate_operation: "algeon_datafusion.cuvs_sql.validate",
        validate_schema_operation: "algeon_datafusion.cuvs_sql.validate_schema",
        execute_operation: "algeon_datafusion.cuvs_sql.execute",
        required_feature: "cuvs",
        feature_disabled_message: "cuVS vector algorithm execution requires the cuvs Cargo feature",
    };

    #[test]
    fn canonical_options_that_disagree_with_their_definitions_are_typed_errors() {
        let Value::Object(values) = json!({
            "k": "ten",
            "n_lists": 1_u64 << 40,
            "dataset": {"id": "id", "vector": {"columns": ["x", 7]}},
        }) else {
            unreachable!("a JSON object literal");
        };
        let options = CanonicalOptions::new(&values, &ERRORS, "cuvs_ivf_flat");

        for error in [
            options.usize("k").unwrap_err(),
            options.u32("n_lists").unwrap_err(),
            options.f64("missing").unwrap_err(),
            options.relation_binding("dataset").unwrap_err(),
        ] {
            assert_eq!(error.code(), ErrorCode::Internal);
            assert_eq!(error.fact_value("cuvs_operation"), Some("cuvs_ivf_flat"));
        }
        assert_eq!(options.usize("k").unwrap_err().fact_value("argument"), Some("k"));
    }
}
