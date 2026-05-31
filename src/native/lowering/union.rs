use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::physical_plan::ExecutionPlan;
use nexus_query_engine::{Expr as NativeExpr, QueryPlanBuilder as NativeQueryPlanBuilder};

use super::{
    FieldBinding, LoweredNativeInput, RequiredFields, executable_schema_and_bindings, expr::alias_if_needed,
    lower_native_input, projection_outputs,
};
use crate::{
    native::{normalize::NormalizedNativeUnion, validate::validate_schema},
    policy::FallbackReason,
};

pub(super) fn lower_union_input(
    union: &NormalizedNativeUnion<'_>,
    builder: &mut NativeQueryPlanBuilder,
    required_output_fields: &RequiredFields,
) -> Result<LoweredNativeInput, FallbackReason> {
    if union.inputs.is_empty() {
        return Err(FallbackReason::UnsupportedOperator);
    }
    validate_schema(union.union.schema())?;
    validate_union_input_schemas(union)?;

    let (schema, bindings) = executable_schema_and_bindings(&union.union.schema(), required_output_fields)?;
    let inputs = union
        .inputs
        .iter()
        .map(|input| {
            let lowered = lower_native_input(input, builder, required_output_fields)?;
            project_union_child_to_output_schema(builder, lowered, Arc::clone(&schema), &bindings)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let node = builder.add_union(inputs.iter().map(|input| input.node).collect(), Arc::clone(&schema));
    Ok(LoweredNativeInput::with_bindings(node, schema, bindings))
}

fn validate_union_input_schemas(union: &NormalizedNativeUnion<'_>) -> Result<(), FallbackReason> {
    let output_schema = union.union.schema();
    for input in union.union.inputs() {
        let input_schema = input.schema();
        if input_schema.fields().len() != output_schema.fields().len() {
            return Err(FallbackReason::UnsupportedOperator);
        }
        for (input_field, output_field) in input_schema.fields().iter().zip(output_schema.fields().iter()) {
            if input_field.data_type() != output_field.data_type() {
                return Err(FallbackReason::UnsupportedType);
            }
        }
        validate_schema(input_schema)?;
    }
    Ok(())
}

fn project_union_child_to_output_schema(
    builder: &mut NativeQueryPlanBuilder,
    input: LoweredNativeInput,
    schema: SchemaRef,
    bindings: &[FieldBinding],
) -> Result<LoweredNativeInput, FallbackReason> {
    if input.schema.as_ref() == schema.as_ref()
        && input.bindings.iter().zip(bindings.iter()).all(|(input, output)| {
            input.frontend_index == output.frontend_index && input.native_name == output.native_name
        })
    {
        return Ok(LoweredNativeInput::with_bindings(input.node, input.schema, bindings.to_vec()));
    }

    let expressions = bindings
        .iter()
        .map(|binding| {
            let input_binding = input.binding_for_frontend_index(binding.frontend_index)?;
            Ok(alias_if_needed(NativeExpr::column(input_binding.native_name.as_str()), binding.native_name.as_str()))
        })
        .collect::<Result<Vec<_>, FallbackReason>>()?;
    let node = builder.add_projection(input.node, projection_outputs(expressions, &schema)?);
    Ok(LoweredNativeInput::with_bindings(node, schema, bindings.to_vec()))
}
