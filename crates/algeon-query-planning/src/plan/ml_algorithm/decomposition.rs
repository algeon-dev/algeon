use std::fmt;

use super::{
    common::MlRelationBinding,
    definition::{CumlOperation, CumlOperationDescriptor},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MlDecompositionSolver {
    CovEigDq,
    CovEigJacobi,
}

impl MlDecompositionSolver {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CovEigDq => "cov_eig_dq",
            Self::CovEigJacobi => "cov_eig_jacobi",
        }
    }
}

impl fmt::Display for MlDecompositionSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MlTsvdParams {
    pub input: MlRelationBinding,
    pub n_components: usize,
    pub solver: MlDecompositionSolver,
    pub tolerance: f64,
    pub n_iterations: u32,
}

impl MlTsvdParams {
    pub fn new(input: MlRelationBinding, n_components: usize) -> Self {
        let (solver, tolerance, n_iterations) = decomposition_defaults(CumlOperation::Tsvd.descriptor());
        Self { input, n_components, solver, tolerance, n_iterations }
    }
}

fn decomposition_defaults(descriptor: &CumlOperationDescriptor) -> (MlDecompositionSolver, f64, u32) {
    let solver = match descriptor.option("solver").and_then(|option| option.kind.default_str()) {
        Some("cov_eig_dq") => MlDecompositionSolver::CovEigDq,
        Some("cov_eig_jacobi") => MlDecompositionSolver::CovEigJacobi,
        _ => panic!("canonical cuML solver default"),
    };
    let tolerance =
        descriptor.option("tol").and_then(|option| option.kind.default_f64()).expect("cuML tolerance default");
    let iterations =
        descriptor.option("n_iterations").and_then(|option| option.kind.default_u64()).expect("cuML iteration default")
            as u32;
    (solver, tolerance, iterations)
}
