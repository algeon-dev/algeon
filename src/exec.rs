use std::{any::Any, sync::Arc};

use arrow::record_batch::RecordBatch;
use cudf::{
    Expr, Runtime,
    ops::{col, lit},
};
use datafusion::{
    common::{Result as DataFusionResult, ScalarValue},
    execution::context::TaskContext,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr,
        expressions::{BinaryExpr, Column, Literal, NotExpr},
    },
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
        filter::FilterExec,
        projection::{ProjectionExec, ProjectionExpr},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::StreamExt;

use crate::error::{Error, ErrorCode, Result as AdapterResult};

#[derive(Debug, Clone)]
pub struct CudfGpuIslandExec {
    input: Arc<dyn ExecutionPlan>,
    filter: Option<Arc<dyn PhysicalExpr>>,
    projection: Vec<ProjectionExpr>,
    cache: Arc<PlanProperties>,
}

impl CudfGpuIslandExec {
    pub fn try_new_from_plan(plan: Arc<dyn ExecutionPlan>) -> AdapterResult<Self> {
        if let Some(projection) = plan.as_any().downcast_ref::<ProjectionExec>() {
            if let Some(filter) = projection.input().as_any().downcast_ref::<FilterExec>() {
                return Ok(Self {
                    input: Arc::clone(filter.input()),
                    filter: Some(Arc::clone(filter.predicate())),
                    projection: projection.expr().to_vec(),
                    cache: Arc::clone(plan.properties()),
                });
            }
            return Ok(Self {
                input: Arc::clone(projection.input()),
                filter: None,
                projection: projection.expr().to_vec(),
                cache: Arc::clone(plan.properties()),
            });
        }

        if let Some(filter) = plan.as_any().downcast_ref::<FilterExec>() {
            return Ok(Self {
                input: Arc::clone(filter.input()),
                filter: Some(Arc::clone(filter.predicate())),
                projection: Vec::new(),
                cache: Arc::clone(plan.properties()),
            });
        }

        Err(Error::new(
            ErrorCode::InvalidExecutionPlan,
            "datafusion_cudf.gpu_island.try_new",
            "cannot build GPU island from physical plan",
        )
        .with_fact("plan_node", plan.name()))
    }
}

impl DisplayAs for CudfGpuIslandExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "CudfGpuIslandExec: filter={}, projection_exprs={}, boundary=host_to_device_then_device_to_host",
                    self.filter.is_some(),
                    self.projection.len()
                )
            },
            DisplayFormatType::TreeRender => {
                writeln!(f, "filter={}", self.filter.is_some())?;
                writeln!(f, "projection_exprs={}", self.projection.len())?;
                writeln!(f, "input_boundary=host_to_device")?;
                writeln!(f, "output_boundary=device_to_host")
            },
        }
    }
}

impl ExecutionPlan for CudfGpuIslandExec {
    fn name(&self) -> &str {
        "CudfGpuIslandExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(Error::new(
                ErrorCode::InvalidChildCount,
                "datafusion_cudf.gpu_island.with_new_children",
                "GPU island expected exactly one child",
            )
            .with_fact("actual_child_count", children.len())
            .into());
        }
        Ok(Arc::new(Self {
            input: children.swap_remove(0),
            filter: self.filter.clone(),
            projection: self.projection.clone(),
            cache: Arc::clone(&self.cache),
        }))
    }

    fn execute(&self, partition: usize, context: Arc<TaskContext>) -> DataFusionResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let schema = self.schema();
        let filter = self.filter.clone();
        let projection = self.projection.clone();
        let runtime = Arc::new(Runtime::builder().build().map_err(|error| {
            Error::from_cudf("datafusion_cudf.runtime.build", "failed to initialize cudf runtime", error)
        })?);

        let output = input.map(move |batch| {
            let batch = batch?;
            execute_batch(&runtime, &filter, &projection, batch)
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, output)))
    }
}

fn execute_batch(
    runtime: &Runtime,
    filter: &Option<Arc<dyn PhysicalExpr>>,
    projection: &[ProjectionExpr],
    batch: RecordBatch,
) -> DataFusionResult<RecordBatch> {
    let ctx =
        cudf::ExecutionOverrides::new().with_stream(runtime.stream()).with_memory_resource(runtime.memory_resource());
    let mut frame = cudf::DataFrame::from_record_batch(runtime, &batch).map_err(|error| {
        Error::from_cudf("datafusion_cudf.record_batch.upload", "failed to upload RecordBatch to cudf", error)
    })?;

    if let Some(filter) = filter {
        let predicate = lower_expr(filter)?;
        frame = frame.with_execution(&ctx).filter_expr(predicate).map_err(|error| {
            Error::from_cudf("datafusion_cudf.filter.execute", "failed to execute cudf filter expression", error)
        })?;
    }

    if !projection.is_empty() {
        let mut exprs = Vec::with_capacity(projection.len());
        let mut output_names = Vec::with_capacity(projection.len());
        for projection_expr in projection {
            let expr = lower_expr(&projection_expr.expr)?;
            exprs.push((projection_expr.alias.as_str(), expr));
            output_names.push(projection_expr.alias.as_str());
        }
        frame = frame.with_execution(&ctx).with_columns(exprs).map_err(|error| {
            Error::from_cudf(
                "datafusion_cudf.projection.execute",
                "failed to execute cudf projection expressions",
                error,
            )
        })?;
        frame = frame.select(output_names).map_err(|error| {
            Error::from_cudf(
                "datafusion_cudf.projection.select",
                "failed to select cudf projection output columns",
                error,
            )
        })?;
    }

    frame.to_record_batch().map_err(|error| {
        Error::from_cudf(
            "datafusion_cudf.record_batch.materialize",
            "failed to materialize cudf DataFrame to RecordBatch",
            error,
        )
        .into()
    })
}

fn lower_expr(expr: &Arc<dyn PhysicalExpr>) -> AdapterResult<Expr> {
    if let Some(column) = expr.as_any().downcast_ref::<Column>() {
        return Ok(col(column.name()));
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return lower_literal(literal.value());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        let left = lower_expr(binary.left())?;
        let right = lower_expr(binary.right())?;
        return lower_binary(left, binary.op(), right);
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return Ok(lower_expr(not.arg())?.not());
    }
    Err(Error::new(
        ErrorCode::UnsupportedExpression,
        "datafusion_cudf.expr.lower",
        "cannot lower unsupported physical expression",
    )
    .with_fact("expression", expr))
}

fn lower_literal(value: &ScalarValue) -> AdapterResult<Expr> {
    match value {
        ScalarValue::Boolean(Some(value)) => Ok(lit(*value)),
        ScalarValue::Int8(Some(value)) => Ok(lit(*value)),
        ScalarValue::Int16(Some(value)) => Ok(lit(*value)),
        ScalarValue::Int32(Some(value)) => Ok(lit(*value)),
        ScalarValue::Int64(Some(value)) => Ok(lit(*value)),
        ScalarValue::UInt8(Some(value)) => Ok(lit(*value)),
        ScalarValue::UInt16(Some(value)) => Ok(lit(*value)),
        ScalarValue::UInt32(Some(value)) => Ok(lit(*value)),
        ScalarValue::UInt64(Some(value)) => Ok(lit(*value)),
        ScalarValue::Float32(Some(value)) => Ok(lit(*value)),
        ScalarValue::Float64(Some(value)) => Ok(lit(*value)),
        _ => Err(Error::new(
            ErrorCode::UnsupportedExpression,
            "datafusion_cudf.expr.lower_literal",
            "cannot lower unsupported or null literal",
        )
        .with_fact("literal", format!("{value:?}"))),
    }
}

fn lower_binary(left: Expr, op: &Operator, right: Expr) -> AdapterResult<Expr> {
    match op {
        Operator::Plus => Ok(left + right),
        Operator::Minus => Ok(left - right),
        Operator::Multiply => Ok(left * right),
        Operator::Divide => Ok(left / right),
        Operator::Modulo => Ok(left % right),
        Operator::Eq => Ok(left.eq(right)),
        Operator::NotEq => Ok(left.ne(right)),
        Operator::Lt => Ok(left.lt(right)),
        Operator::LtEq => Ok(left.le(right)),
        Operator::Gt => Ok(left.gt(right)),
        Operator::GtEq => Ok(left.ge(right)),
        Operator::And => Ok(left & right),
        Operator::Or => Ok(left | right),
        _ => Err(Error::new(
            ErrorCode::UnsupportedOperator,
            "datafusion_cudf.expr.lower_binary",
            "cannot lower unsupported binary operator",
        )
        .with_fact("operator", format!("{op:?}"))),
    }
}
