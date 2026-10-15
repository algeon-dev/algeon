//! Plan element blocking a native candidate, and the shared description
//! rendered across report projections.

use datafusion::logical_expr::WindowFrame;
use query_planning::datatypes::diagnostic_arrow_type;
use serde_json::{Map, Value};

use super::{NotSupportedDetail, UnsupportedColumnDetail};
use crate::native::{ExpressionCapability, ExpressionRejection, ExpressionRejectionSubject};

/// Plan element, outside the expression contract, that blocked a candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PlanBlocker {
    /// A physical operator with no native counterpart.
    Operator(String),
    /// A DataFusion operator that keeps a contract the native fragment below
    /// it would break.
    ParentOperator(String),
    Aggregate {
        function: String,
        defect: AggregateDefect,
    },
    /// Partial and final aggregate stages that native lowering cannot lower as
    /// one aggregate; the clause says what differs between them.
    DistributedAggregate(String),
    Window {
        function: String,
        defect: WindowDefect,
    },
    /// An ordering the native plan cannot reproduce: the declared output
    /// ordering of the replaced subtree, or the input ordering of a window.
    Ordering {
        required_by: &'static str,
        ordering: Option<String>,
    },
}

/// What planning the query for one partition (`SET
/// datafusion.execution.target_partitions = 1`) does to a candidate that
/// DataFusion shaped for several partitions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum SinglePartitionOutcome {
    /// DataFusion plans a shape that native lowering handles.
    #[default]
    Lowers,
    /// The blocker remains at one partition: the plan fixes its partitions
    /// itself, or the rejected expression has no native form in a single
    /// stage either.
    StillRejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AggregateDefect {
    UnsupportedFunction,
    Distinct,
    OrderBy,
    /// The filters of the final and partial stages differ.
    Filter,
    ArgumentCount(usize),
    /// The function has no partial-stage native kernel.
    PartialStage,
    /// `MEDIAN` next to a floating-point statistical aggregate.
    MedianWithStatistical,
    /// The function has no native kernel for grouping sets.
    GroupingSets,
    /// DataFusion reads the aggregate input through more than one partition,
    /// so its floating-point result depends on partition arrival order.
    PartitionedInput,
}

impl AggregateDefect {
    const fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedFunction => "unsupported_function",
            Self::Distinct => "distinct",
            Self::OrderBy => "order_by",
            Self::Filter => "filter",
            Self::ArgumentCount(_) => "argument_count",
            Self::PartialStage => "partial_stage",
            Self::MedianWithStatistical => "median_with_statistical",
            Self::GroupingSets => "grouping_sets",
            Self::PartitionedInput => "partitioned_input",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WindowDefect {
    UnsupportedFunction,
    MissingOrderBy,
    UnsupportedFrame(Box<WindowFrame>),
    /// An offset or default argument that is not a constant.
    Arguments,
}

impl WindowDefect {
    const fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedFunction => "unsupported_function",
            Self::MissingOrderBy => "missing_order_by",
            Self::UnsupportedFrame(_) => "unsupported_frame",
            Self::Arguments => "arguments",
        }
    }
}

/// What blocked one native candidate: a stable machine kind and a one-line
/// human text, in the form every report projection carries.
///
/// The text and type names can quote caller identifiers and SQL; error
/// boundaries treat this text as client detail and exclude it from operational
/// logs. The kind uses a fixed vocabulary and is log-safe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blocking {
    kind: &'static str,
    text: String,
    fields: Vec<(&'static str, String)>,
}

impl Blocking {
    fn new(kind: &'static str, text: String, fields: Vec<(&'static str, String)>) -> Self {
        Self { kind, text, fields }
    }

    /// Stable machine kind: `operator`, `parent_operator`, `aggregate_function`,
    /// `distributed_aggregate`, `window_function`, `scalar_function`, `cast`,
    /// `binary_operator`, `unary_operator`, `literal`, `expression`,
    /// `column_type`, `ordering`, or `compiled_contract`.
    pub const fn kind(&self) -> &'static str {
        self.kind
    }

    /// One-line description that reads after "does not support", such as
    /// ``cast from Int64 to Int32 in `CAST(buys AS INT)` ``.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The kind, the text, and the structured fields of the subject.
    pub fn to_json(&self) -> Value {
        let mut object = Map::new();
        object.insert("kind".to_string(), Value::String(self.kind.to_string()));
        object.insert("text".to_string(), Value::String(self.text.clone()));
        for (key, value) in &self.fields {
            object.insert((*key).to_string(), Value::String(value.clone()));
        }
        Value::Object(object)
    }
}

impl NotSupportedDetail {
    /// Element that blocked this candidate, when reported by the planning site.
    /// Report projections read this description directly.
    ///
    /// A subject named by the expression contract (a cast, function, or operator)
    /// is the most specific cause. An unsupported column type follows, then
    /// the plan element, then an unsegmented expression rejection, then
    /// admission rejection.
    pub fn blocking(&self) -> Option<Blocking> {
        if let Some(rejection) = self.named_expression() {
            return Some(expression_blocking(rejection));
        }
        if let Some(column) = &self.column {
            return Some(column_blocking(column));
        }
        if let Some(blocker) = &self.blocker {
            return Some(plan_blocking(blocker));
        }
        if let Some(rejection) = &self.expression
            && rejection.sql().is_some()
        {
            return Some(expression_blocking(rejection));
        }
        self.admission.as_ref().map(|admission| {
            let reason = admission.reason.as_str();
            let boundary = admission.boundary();
            Blocking::new(
                "compiled_contract",
                format!("compiled contract rule `{reason}` at the {boundary} boundary"),
                vec![("reason", reason.to_string()), ("boundary", boundary.to_string())],
            )
        })
    }
}

impl NotSupportedDetail {
    fn named_expression(&self) -> Option<&ExpressionRejection> {
        self.expression
            .as_deref()
            .filter(|rejection| !matches!(rejection.subject(), ExpressionRejectionSubject::Unnamed))
    }

    /// The unsupported column, when it is the subject [`Self::blocking`] reports.
    pub(crate) fn blocking_column(&self) -> Option<&UnsupportedColumnDetail> {
        self.column.as_ref().filter(|_| self.named_expression().is_none())
    }
}

fn column_blocking(column: &UnsupportedColumnDetail) -> Blocking {
    Blocking::new(
        "column_type",
        format!("{} column `{}` with Arrow type `{}`", column.role, column.name, column.data_type),
        vec![
            ("column", column.name.clone()),
            ("data_type", column.data_type.clone()),
            ("role", column.role.to_string()),
        ],
    )
}

fn expression_blocking(rejection: &ExpressionRejection) -> Blocking {
    let sql = rejection.sql();
    let location = sql.map(|sql| format!(" in `{sql}`")).unwrap_or_default();
    let mut fields = Vec::new();
    let (kind, text) = match rejection.subject() {
        ExpressionRejectionSubject::Cast { source, target } => {
            let (source, target) = (diagnostic_arrow_type(source), diagnostic_arrow_type(target));
            let word = if rejection.capability() == ExpressionCapability::TryCastPair { "try cast" } else { "cast" };
            fields.push(("source_type", source.clone()));
            fields.push(("target_type", target.clone()));
            ("cast", format!("{word} from {source} to {target}{location}"))
        },
        ExpressionRejectionSubject::ScalarFunction(name) => {
            fields.push(("function", name.clone()));
            ("scalar_function", format!("scalar function {name}{location}"))
        },
        ExpressionRejectionSubject::BinaryOperator { operator, left, right } => {
            let (left, right) = (diagnostic_arrow_type(left), diagnostic_arrow_type(right));
            fields.push(("operator", operator.to_string()));
            fields.push(("left_type", left.clone()));
            fields.push(("right_type", right.clone()));
            ("binary_operator", format!("binary operator `{operator}` over {left} and {right}{location}"))
        },
        ExpressionRejectionSubject::UnaryOperator { operator, operand } => {
            let operand = diagnostic_arrow_type(operand);
            fields.push(("operator", (*operator).to_string()));
            fields.push(("operand_type", operand.clone()));
            ("unary_operator", format!("unary operator `{operator}` over {operand}{location}"))
        },
        ExpressionRejectionSubject::ValueType(data_type) => {
            let data_type = diagnostic_arrow_type(data_type);
            fields.push(("data_type", data_type.clone()));
            let text = match sql {
                Some(sql) => format!("expression `{sql}` of type {data_type}"),
                None => format!("expression of type {data_type}"),
            };
            ("expression", text)
        },
        ExpressionRejectionSubject::LiteralType(data_type) => {
            let data_type = diagnostic_arrow_type(data_type);
            fields.push(("data_type", data_type.clone()));
            ("literal", format!("literal of type {data_type}{location}"))
        },
        ExpressionRejectionSubject::Unnamed | ExpressionRejectionSubject::DynamicFilterError(_) => {
            ("expression", format!("expression{location}"))
        },
    };
    if let Some(sql) = sql {
        fields.push(("expression", sql.to_string()));
    }
    Blocking::new(kind, text, fields)
}

fn plan_blocking(blocker: &PlanBlocker) -> Blocking {
    match blocker {
        PlanBlocker::Operator(name) => {
            Blocking::new("operator", format!("operator {name}"), vec![("operator", name.clone())])
        },
        PlanBlocker::ParentOperator(name) => Blocking::new(
            "parent_operator",
            format!("a native fragment below operator {name}"),
            vec![("operator", name.clone())],
        ),
        PlanBlocker::Aggregate { function, defect } => {
            let text = match defect {
                AggregateDefect::UnsupportedFunction => format!("aggregate function {function}"),
                AggregateDefect::Distinct => format!("aggregate function {function} with DISTINCT"),
                AggregateDefect::OrderBy => format!("aggregate function {function} with ORDER BY"),
                AggregateDefect::Filter => {
                    format!("aggregate function {function} with FILTER clauses that differ between aggregation stages")
                },
                AggregateDefect::ArgumentCount(count) => {
                    format!("aggregate function {function} with {count} arguments")
                },
                AggregateDefect::PartialStage => {
                    format!("aggregate function {function} in a partial aggregation stage")
                },
                AggregateDefect::MedianWithStatistical => {
                    format!("aggregate function {function} next to a standard deviation or variance aggregate")
                },
                AggregateDefect::GroupingSets => {
                    format!("aggregate function {function} in a grouping set aggregate")
                },
                AggregateDefect::PartitionedInput => {
                    format!("aggregate function {function} over input that DataFusion reads in several partitions")
                },
            };
            Blocking::new(
                "aggregate_function",
                text,
                vec![("function", function.clone()), ("defect", defect.code().to_string())],
            )
        },
        PlanBlocker::DistributedAggregate(difference) => Blocking::new(
            "distributed_aggregate",
            format!("distributed aggregate where {difference}"),
            vec![("difference", difference.clone())],
        ),
        PlanBlocker::Window { function, defect } => {
            let mut fields = vec![("function", function.clone()), ("defect", defect.code().to_string())];
            let text = match defect {
                WindowDefect::UnsupportedFunction => format!("window function {function}"),
                WindowDefect::MissingOrderBy => format!("window function {function} without ORDER BY"),
                WindowDefect::UnsupportedFrame(frame) => {
                    fields.push(("frame", frame.to_string()));
                    format!("window function {function} over frame {frame}")
                },
                WindowDefect::Arguments => {
                    format!("window function {function} with an offset or default that is not a constant")
                },
            };
            Blocking::new("window_function", text, fields)
        },
        PlanBlocker::Ordering { required_by, ordering } => {
            let mut fields = vec![("required_by", (*required_by).to_string())];
            let text = match ordering {
                Some(ordering) => {
                    fields.push(("ordering", ordering.clone()));
                    format!("{required_by} `{ordering}`")
                },
                None => (*required_by).to_string(),
            };
            Blocking::new("ordering", text, fields)
        },
    }
}
