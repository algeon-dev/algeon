use super::*;

pub(super) fn expression_category(expr: &Arc<dyn PhysicalExpr>) -> Option<String> {
    if expr.as_any().downcast_ref::<Column>().is_some() {
        return None;
    }
    if let Some(literal) = expr.as_any().downcast_ref::<Literal>() {
        return (!literal_is_supported(literal.value())).then(|| "expression_literal".to_string());
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        if expression_contains_interval_literal(binary.left()) || expression_contains_interval_literal(binary.right()) {
            return interval_arithmetic_category(binary);
        }
        if !binary_operator_is_supported(binary.op()) {
            return Some(format!("expression_binary_{}", binary_operator_identifier(binary.op())));
        }
        return expression_category(binary.left()).or_else(|| expression_category(binary.right()));
    }
    if let Some(not) = expr.as_any().downcast_ref::<NotExpr>() {
        return expression_category(not.arg());
    }
    if let Some(is_null) = expr.as_any().downcast_ref::<IsNullExpr>() {
        return expression_category(is_null.arg());
    }
    if let Some(is_not_null) = expr.as_any().downcast_ref::<IsNotNullExpr>() {
        return expression_category(is_not_null.arg());
    }
    if let Some(cast) = expr.as_any().downcast_ref::<CastExpr>() {
        return expression_category(cast.expr());
    }
    if let Some(try_cast) = expr.as_any().downcast_ref::<TryCastExpr>() {
        return expression_category(try_cast.expr()).or_else(|| Some("expression_try_cast".to_string()));
    }
    if let Some(like) = expr.as_any().downcast_ref::<LikeExpr>() {
        if like.case_insensitive() {
            return Some("expression_like_case_insensitive".to_string());
        }
        return expression_category(like.expr()).or_else(|| {
            expression_category(like.pattern())
                .or_else(|| literal_string(like.pattern()).is_none().then(|| "expression_like_pattern".to_string()))
        });
    }
    if let Some(in_list) = expr.as_any().downcast_ref::<InListExpr>() {
        if in_list.is_empty() {
            return Some("expression_in_list".to_string());
        }
        if in_list
            .list()
            .iter()
            .any(|expr| expr.as_any().downcast_ref::<Literal>().is_none_or(|literal| literal.value().is_null()))
        {
            return Some("expression_in_list".to_string());
        }
        return expression_category(in_list.expr()).or_else(|| in_list.list().iter().find_map(expression_category));
    }
    if let Some(case) = expr.as_any().downcast_ref::<CaseExpr>() {
        if let Some(expr) = case.expr()
            && let Some(category) = expression_category(expr)
        {
            return Some(category);
        }
        for (when, then) in case.when_then_expr() {
            if let Some(category) = expression_category(when).or_else(|| expression_category(then)) {
                return Some(category);
            }
        }
        return case.else_expr().and_then(expression_category);
    }
    if let Some(function) = expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        let function_name = stable_identifier(function.name());
        match function_name.as_str() {
            "substr" | "substring" => {
                return function.args().iter().find_map(expression_category).or_else(|| {
                    (!substring_arguments_are_supported(function)).then(|| "expression_substring_argument".to_string())
                });
            },
            "date_part" | "extract" => {
                return function.args().iter().find_map(expression_category).or_else(|| {
                    (!extract_field_is_supported(function)).then(|| "expression_extract_field".to_string())
                });
            },
            _ => return Some(format!("expression_scalar_function_{function_name}")),
        }
    }
    Some("expression".to_string())
}

fn literal_is_supported(value: &ScalarValue) -> bool {
    is_supported_arrow_type(&value.data_type())
}

fn expression_contains_interval_literal(expr: &Arc<dyn PhysicalExpr>) -> bool {
    if expr
        .as_any()
        .downcast_ref::<Literal>()
        .is_some_and(|literal| matches!(literal.value(), ScalarValue::IntervalMonthDayNano(Some(_))))
    {
        return true;
    }
    if let Some(binary) = expr.as_any().downcast_ref::<BinaryExpr>() {
        return expression_contains_interval_literal(binary.left())
            || expression_contains_interval_literal(binary.right());
    }
    false
}

fn interval_arithmetic_category(binary: &BinaryExpr) -> Option<String> {
    let left_interval = is_interval_literal_expr(binary.left());
    let right_interval = is_interval_literal_expr(binary.right());
    let supported_shape = matches!(
        (binary.op(), left_interval, right_interval),
        (Operator::Plus, true, false) | (Operator::Plus | Operator::Minus, false, true)
    );
    if !supported_shape {
        return Some("expression_interval_arithmetic".to_string());
    }
    if left_interval { expression_category(binary.right()) } else { expression_category(binary.left()) }
}

fn is_interval_literal_expr(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.as_any()
        .downcast_ref::<Literal>()
        .is_some_and(|literal| matches!(literal.value(), ScalarValue::IntervalMonthDayNano(Some(_))))
}

fn literal_string(expr: &Arc<dyn PhysicalExpr>) -> Option<String> {
    let literal = expr.as_any().downcast_ref::<Literal>()?;
    match literal.value() {
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) | ScalarValue::Utf8View(Some(value)) => {
            Some(value.clone())
        },
        _ => None,
    }
}

fn substring_arguments_are_supported(function: &ScalarFunctionExpr) -> bool {
    let args = function.args();
    if !(2..=3).contains(&args.len()) {
        return false;
    }
    let Some(start) = literal_i32(&args[1]) else {
        return false;
    };
    if start < 1 {
        return false;
    }
    if args.len() == 3 {
        let Some(length) = literal_i32(&args[2]) else {
            return false;
        };
        return length >= 0 && substring_stop_position(start, length).is_some();
    }
    true
}

fn substring_stop_position(start: i32, length: i32) -> Option<i32> {
    let zero_based_start = start.checked_sub(1)?;
    zero_based_start.checked_add(length)
}

fn extract_field_is_supported(function: &ScalarFunctionExpr) -> bool {
    let [field, _expr] = function.args() else {
        return false;
    };
    literal_string(field)
        .map(|field| matches!(field.to_ascii_lowercase().as_str(), "year" | "month" | "day"))
        .unwrap_or(false)
}

fn literal_i32(expr: &Arc<dyn PhysicalExpr>) -> Option<i32> {
    let literal = expr.as_any().downcast_ref::<Literal>()?;
    match literal.value() {
        ScalarValue::Int8(Some(value)) => Some(i32::from(*value)),
        ScalarValue::Int16(Some(value)) => Some(i32::from(*value)),
        ScalarValue::Int32(Some(value)) => Some(*value),
        ScalarValue::Int64(Some(value)) => i32::try_from(*value).ok(),
        ScalarValue::UInt8(Some(value)) => Some(i32::from(*value)),
        ScalarValue::UInt16(Some(value)) => Some(i32::from(*value)),
        ScalarValue::UInt32(Some(value)) => i32::try_from(*value).ok(),
        ScalarValue::UInt64(Some(value)) => i32::try_from(*value).ok(),
        _ => None,
    }
}

fn binary_operator_is_supported(op: &Operator) -> bool {
    matches!(
        op,
        Operator::Plus
            | Operator::Minus
            | Operator::Multiply
            | Operator::Divide
            | Operator::Modulo
            | Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
            | Operator::And
            | Operator::Or
    )
}

fn binary_operator_identifier(op: &Operator) -> &'static str {
    match op {
        Operator::Plus => "add",
        Operator::Minus => "subtract",
        Operator::Multiply => "multiply",
        Operator::Divide => "divide",
        Operator::Modulo => "modulo",
        Operator::Eq => "eq",
        Operator::NotEq => "not_eq",
        Operator::Lt => "lt",
        Operator::LtEq => "lt_eq",
        Operator::Gt => "gt",
        Operator::GtEq => "gt_eq",
        Operator::And => "and",
        Operator::Or => "or",
        _ => "other",
    }
}
