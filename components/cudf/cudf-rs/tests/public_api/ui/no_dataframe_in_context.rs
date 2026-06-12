use cudf::{DataFrame, ExecutionOverrides};

fn require_removed_method(df: &DataFrame, execution: &ExecutionOverrides<'_>) {
    let _ = df.in_context(execution);
}

fn main() {}
