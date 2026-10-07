use std::{cmp::Ordering, fmt, hash::Hash, sync::Arc};

use arrow::datatypes::SchemaRef;
use datafusion::{
    common::{DFSchema, DFSchemaRef, Result as DataFusionResult},
    logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore},
};
use serde_json::{Map, Value};

#[cfg(feature = "cugraph")]
use super::CugraphAlgoExec;
use super::{
    CugraphAlgorithm, CugraphAlgorithmParams, CugraphAlgorithmSpec, CugraphBfsSource,
    CugraphForceAtlas2InitialPositionsTable, CugraphForceAtlas2VertexAttributesTable, CugraphGraphEdgePredicateTable,
    CugraphGraphPredicateTable, CugraphSqlConfig,
    definitions::{CugraphArgumentKind, arguments_for_algorithm},
    error::invalid_option,
    options, plan,
    planning::{validate_edge_source_schema, validate_predicate_source_schema},
    result_schema_for_params, result_schema_for_vertex_domain,
};
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CugraphRelationRole {
    Edges,
    Personalization,
    Sources,
    IncludeVertices,
    ExcludeVertices,
    TargetVertices,
    IncludeEdgeIds,
    IncludeEdges,
    VertexAttributes,
    InitialPositions,
    VertexPairs,
}

impl CugraphRelationRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Edges => "edges",
            Self::Personalization => "personalization",
            Self::Sources => "sources",
            Self::IncludeVertices => "include_vertices",
            Self::ExcludeVertices => "exclude_vertices",
            Self::TargetVertices => "target_vertices",
            Self::IncludeEdgeIds => "include_edge_ids",
            Self::IncludeEdges => "include_edges",
            Self::VertexAttributes => "vertex_attributes",
            Self::InitialPositions => "initial_positions",
            Self::VertexPairs => "vertex_pairs",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CugraphLogicalNode {
    spec: CugraphAlgorithmSpec,
    inputs: Vec<LogicalPlan>,
    roles: Vec<CugraphRelationRole>,
    binding_key: String,
    gpu_result_input_name: String,
    schema: DFSchemaRef,
}

impl CugraphLogicalNode {
    pub(crate) fn try_new(
        algorithm: CugraphAlgorithm,
        inputs: Vec<LogicalPlan>,
        roles: Vec<CugraphRelationRole>,
        values: Map<String, Value>,
        source: Option<CugraphBfsSource>,
        config: &CugraphSqlConfig,
    ) -> DataFusionResult<Self> {
        let spec = validated_spec(algorithm, &inputs, &roles, &values, source, config)?;
        let schema = Arc::new(DFSchema::try_from(spec.output_schema.as_ref().clone())?);
        let binding_key = serde_json::to_string(&values).expect("cuGraph named values are JSON serializable");
        #[cfg(feature = "cugraph")]
        let gpu_result_input_name = CugraphAlgoExec::next_gpu_result_input_name();
        #[cfg(not(feature = "cugraph"))]
        let gpu_result_input_name = String::new();
        Ok(Self { spec, inputs, roles, binding_key, gpu_result_input_name, schema })
    }

    pub fn spec(&self) -> &CugraphAlgorithmSpec {
        &self.spec
    }

    pub fn output_schema(&self) -> SchemaRef {
        Arc::clone(&self.spec.output_schema)
    }

    #[cfg(feature = "cugraph")]
    pub(crate) fn gpu_result_input_name(&self) -> &str {
        &self.gpu_result_input_name
    }

    /// The relation each logical input supplies, in input order.
    #[cfg(feature = "cugraph")]
    pub(crate) fn roles(&self) -> &[CugraphRelationRole] {
        &self.roles
    }
}

fn validated_spec(
    algorithm: CugraphAlgorithm,
    inputs: &[LogicalPlan],
    roles: &[CugraphRelationRole],
    values: &Map<String, Value>,
    source: Option<CugraphBfsSource>,
    config: &CugraphSqlConfig,
) -> Result<CugraphAlgorithmSpec> {
    if roles.first() != Some(&CugraphRelationRole::Edges) || roles.len() != inputs.len() {
        return Err(plan("cuGraph logical relation roles must begin with edges and match the child count"));
    }
    if algorithm == CugraphAlgorithm::Bfs {
        let selector_count = usize::from(source.is_some()) + usize::from(roles.contains(&CugraphRelationRole::Sources));
        if selector_count != 1 {
            return Err(invalid_option(
                "cugraph_bfs requires exactly one of source_vertex, source_vertices, or sources",
            ));
        }
    }
    validate_contract_column_uniqueness(algorithm, inputs, roles)?;
    let (mut graph_options, mut params) = options::default_options(algorithm, config, source)?;
    let edge_schema = inputs[0].schema();
    let has_weight = edge_schema.field_with_unqualified_name("weight").is_ok();
    let has_edge_id = edge_schema.field_with_unqualified_name("edge_id").is_ok();
    options::apply_options(algorithm, values, &mut graph_options, &mut params)?;
    apply_relation_roles(roles, inputs, &mut params, has_edge_id)?;
    options::validate_graph_directedness(algorithm, &graph_options)
        .map_err(|error| error.with_fact("argument", "directed"))?;
    if algorithm.native_algorithm().weight_policy().requires_weight() && !has_weight {
        return Err(plan(format!(
            "{} requires the edges relation to contain a weight field",
            algorithm.function_name()
        ))
        .with_fact("relation_role", CugraphRelationRole::Edges.as_str()));
    }
    let mut spec = CugraphAlgorithmSpec {
        algorithm,
        has_weight,
        has_edge_id,
        graph_options,
        output_schema: result_schema_for_params(algorithm, &params),
        params,
    };
    let vertex_domain = validate_inputs(&spec, inputs, roles)?;
    spec.output_schema = result_schema_for_vertex_domain(algorithm, &spec.params, Some(&vertex_domain));
    Ok(spec)
}

fn validate_inputs(
    spec: &CugraphAlgorithmSpec,
    inputs: &[LogicalPlan],
    roles: &[CugraphRelationRole],
) -> Result<super::CugraphVertexIdDomain> {
    validate_contract_column_uniqueness(spec.algorithm, inputs, roles)?;
    let edge_schema = Arc::new(inputs[0].schema().as_arrow().clone());
    let vertex_domain = validate_edge_source_schema(spec, &edge_schema)
        .map_err(|error| error.with_fact("relation_role", CugraphRelationRole::Edges.as_str()))?;
    let source_specs = spec.predicate_source_specs();
    if source_specs.len() != inputs.len() - 1 {
        return Err(plan(format!(
            "{} expected {} side-input relation(s), received {}",
            spec.algorithm.function_name(),
            source_specs.len(),
            inputs.len() - 1
        )));
    }
    for ((source, input), role) in source_specs.iter().zip(&inputs[1..]).zip(&roles[1..]) {
        let schema = Arc::new(input.schema().as_arrow().clone());
        validate_predicate_source_schema(spec, source, &schema, &vertex_domain)
            .map_err(|error| error.with_fact("relation_role", role.as_str()))?;
    }
    Ok(vertex_domain)
}

/// Contract columns are matched by unqualified name, so a relation child that
/// exposes the same name under two qualifiers (for example `a.weight` and
/// `b.weight` from a join) has no single binding and is rejected here rather
/// than silently bound to the first match.
fn validate_contract_column_uniqueness(
    algorithm: CugraphAlgorithm,
    inputs: &[LogicalPlan],
    roles: &[CugraphRelationRole],
) -> Result<()> {
    for (role, input) in roles.iter().zip(inputs) {
        let Some(columns) = arguments_for_algorithm(algorithm).iter().find_map(|argument| match argument.kind {
            CugraphArgumentKind::Relation { columns } if argument.name == role.as_str() => Some(columns),
            _ => None,
        }) else {
            continue;
        };
        for column in columns {
            let matches = input.schema().qualified_fields_with_unqualified_name(column.name);
            if matches.len() > 1 {
                let qualifiers = matches
                    .iter()
                    .map(|(qualifier, _)| qualifier.map_or_else(|| "<unqualified>".to_string(), ToString::to_string))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(plan(format!(
                    "cuGraph {} relation column '{}' is ambiguous across qualifiers [{qualifiers}]; project one column",
                    role.as_str(),
                    column.name
                ))
                .with_fact("relation_role", role.as_str()));
            }
        }
    }
    Ok(())
}

fn apply_relation_roles(
    roles: &[CugraphRelationRole],
    inputs: &[LogicalPlan],
    params: &mut CugraphAlgorithmParams,
    has_edge_id: bool,
) -> Result<()> {
    for (role, input) in roles.iter().copied().zip(inputs).skip(1) {
        apply_relation_role(role, input, params, has_edge_id)
            .map_err(|error| error.with_fact("relation_role", role.as_str()))?;
    }
    validate_relation_combinations(params)
}

fn apply_relation_role(
    role: CugraphRelationRole,
    input: &LogicalPlan,
    params: &mut CugraphAlgorithmParams,
    has_edge_id: bool,
) -> Result<()> {
    match role {
        CugraphRelationRole::Personalization => {},
        CugraphRelationRole::Sources => {
            let CugraphAlgorithmParams::Bfs(bfs) = params else {
                return Err(plan("sources relation is only valid for cugraph_bfs"));
            };
            bfs.source = CugraphBfsSource::Relation(CugraphGraphPredicateTable::new());
        },
        CugraphRelationRole::IncludeVertices => {
            let CugraphAlgorithmParams::Bfs(bfs) = params else {
                return Err(plan("include_vertices relation is only valid for cugraph_bfs"));
            };
            bfs.predicates.include_vertices = Some(CugraphGraphPredicateTable::new());
        },
        CugraphRelationRole::ExcludeVertices => {
            let CugraphAlgorithmParams::Bfs(bfs) = params else {
                return Err(plan("exclude_vertices relation is only valid for cugraph_bfs"));
            };
            bfs.predicates.exclude_vertices = Some(CugraphGraphPredicateTable::new());
        },
        CugraphRelationRole::TargetVertices => {
            let CugraphAlgorithmParams::Bfs(bfs) = params else {
                return Err(plan("target_vertices relation is only valid for cugraph_bfs"));
            };
            bfs.predicates.target_vertices = Some(CugraphGraphPredicateTable::new());
        },
        CugraphRelationRole::IncludeEdgeIds => {
            let CugraphAlgorithmParams::Bfs(bfs) = params else {
                return Err(plan("include_edge_ids relation is only valid for cugraph_bfs"));
            };
            if !has_edge_id {
                return Err(plan("cugraph_bfs include_edge_ids requires edges.edge_id"));
            }
            bfs.predicates.include_edge_ids = Some(CugraphGraphPredicateTable::new());
        },
        CugraphRelationRole::IncludeEdges => {
            let CugraphAlgorithmParams::Bfs(bfs) = params else {
                return Err(plan("include_edges relation is only valid for cugraph_bfs"));
            };
            let predicate_has_edge_id = input.schema().field_with_unqualified_name("edge_id").is_ok();
            if predicate_has_edge_id && !has_edge_id {
                return Err(plan("cugraph_bfs include_edges.edge_id requires edges.edge_id"));
            }
            bfs.predicates.include_edges = Some(CugraphGraphEdgePredicateTable::new(predicate_has_edge_id));
        },
        CugraphRelationRole::VertexAttributes => {
            let CugraphAlgorithmParams::ForceAtlas2(force) = params else {
                return Err(plan("vertex_attributes relation is only valid for cugraph_force_atlas2"));
            };
            let attributes = CugraphForceAtlas2VertexAttributesTable {
                has_radius: input.schema().field_with_unqualified_name("radius").is_ok(),
                has_mobility: input.schema().field_with_unqualified_name("mobility").is_ok(),
                has_mass: input.schema().field_with_unqualified_name("mass").is_ok(),
            };
            if !attributes.has_radius && !attributes.has_mobility && !attributes.has_mass {
                return Err(plan("cugraph_force_atlas2 vertex_attributes requires radius, mobility, or mass"));
            }
            force.vertex_attributes = Some(attributes);
        },
        CugraphRelationRole::InitialPositions => {
            let CugraphAlgorithmParams::ForceAtlas2(force) = params else {
                return Err(plan("initial_positions relation is only valid for cugraph_force_atlas2"));
            };
            force.initial_positions = Some(CugraphForceAtlas2InitialPositionsTable::new());
        },
        CugraphRelationRole::VertexPairs => {},
        CugraphRelationRole::Edges => return Err(plan("edges may only be the first cuGraph relation role")),
    }
    Ok(())
}

fn validate_relation_combinations(params: &CugraphAlgorithmParams) -> Result<()> {
    match params {
        CugraphAlgorithmParams::Bfs(bfs) => {
            if bfs.predicates.include_vertices.is_some() && bfs.predicates.exclude_vertices.is_some() {
                return Err(plan("BFS include_vertices and exclude_vertices are mutually exclusive"));
            }
            if bfs.output.return_target_info && bfs.predicates.target_vertices.is_none() {
                return Err(plan("BFS return_target_info requires target_vertices"));
            }
            if bfs.output.mode == super::CugraphBfsOutputMode::Path && bfs.output.return_target_info {
                return Err(plan("BFS path output cannot combine with return_target_info=true"));
            }
            if bfs.output.mode == super::CugraphBfsOutputMode::Path && bfs.predicates.target_vertices.is_none() {
                return Err(plan("BFS path output requires target_vertices"));
            }
        },
        CugraphAlgorithmParams::ForceAtlas2(force)
            if force.prevent_overlapping && force.vertex_attributes.as_ref().is_none_or(|input| !input.has_radius) =>
        {
            return Err(plan("ForceAtlas2 prevent_overlapping=true requires vertex_attributes.radius"));
        },
        _ => {},
    }
    Ok(())
}

impl PartialEq for CugraphLogicalNode {
    fn eq(&self, other: &Self) -> bool {
        self.spec.algorithm == other.spec.algorithm
            && self.inputs == other.inputs
            && self.roles == other.roles
            && self.binding_key == other.binding_key
            && self.gpu_result_input_name == other.gpu_result_input_name
    }
}

impl Eq for CugraphLogicalNode {}

impl Hash for CugraphLogicalNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.spec.algorithm.hash(state);
        self.inputs.hash(state);
        self.roles.hash(state);
        self.binding_key.hash(state);
        self.gpu_result_input_name.hash(state);
    }
}

impl PartialOrd for CugraphLogicalNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (&self.spec.algorithm, &self.inputs, &self.roles, &self.binding_key, &self.gpu_result_input_name).partial_cmp(
            &(&other.spec.algorithm, &other.inputs, &other.roles, &other.binding_key, &other.gpu_result_input_name),
        )
    }
}

impl UserDefinedLogicalNodeCore for CugraphLogicalNode {
    fn name(&self) -> &str {
        "CugraphLogicalNode"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        #[cfg(feature = "cugraph")]
        {
            self.inputs.iter().collect()
        }
        #[cfg(not(feature = "cugraph"))]
        {
            Vec::new()
        }
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        Vec::new()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "CugraphLogicalNode: algorithm={}, relation_roles=[{}], options={}",
            self.spec.algorithm.function_name(),
            self.roles.iter().map(|role| role.as_str()).collect::<Vec<_>>().join(","),
            self.binding_key
        )
    }

    fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> DataFusionResult<Self> {
        if !exprs.is_empty() {
            return Err(plan("cuGraph logical nodes do not expose optimizer expressions").into());
        }
        #[cfg(not(feature = "cugraph"))]
        {
            if !inputs.is_empty() {
                return Err(plan("feature-disabled cuGraph logical nodes must not receive input children").into());
            }
            Ok(self.clone())
        }
        #[cfg(feature = "cugraph")]
        {
            let mut node = self.clone();
            let vertex_domain = validate_inputs(&node.spec, &inputs, &node.roles)?;
            node.spec.output_schema =
                result_schema_for_vertex_domain(node.spec.algorithm, &node.spec.params, Some(&vertex_domain));
            node.schema = Arc::new(DFSchema::try_from(node.spec.output_schema.as_ref().clone())?);
            node.inputs = inputs;
            Ok(node)
        }
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}
