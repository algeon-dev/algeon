use std::fmt;

use super::{common::VectorRelationBinding, definition::CuvsOperation};

/// PCA solver selected by the SQL options contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorPcaSolver {
    CovEigDq,
    CovEigJacobi,
}

impl VectorPcaSolver {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::CovEigDq => "cov_eig_dq",
            Self::CovEigJacobi => "cov_eig_jacobi",
        }
    }
}

impl fmt::Display for VectorPcaSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// PCA fit-transform parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorPcaParams {
    /// Input relation to fit and transform.
    pub input: VectorRelationBinding,
    /// Number of output principal components.
    pub n_components: usize,
    /// Whether transformed components are whitened.
    pub whiten: bool,
    /// Covariance eigensolver.
    pub solver: VectorPcaSolver,
    /// Nonnegative eigensolver tolerance.
    pub tolerance: f64,
    /// Maximum eigensolver iterations.
    pub n_iterations: u32,
    /// Native sign-fixing option.
    pub flip_signs_based_on_u: bool,
}

impl VectorPcaParams {
    /// Construct PCA fit-transform parameters.
    pub fn new(input: VectorRelationBinding, n_components: usize) -> Self {
        let descriptor = CuvsOperation::Pca.descriptor();
        Self {
            input,
            n_components,
            whiten: descriptor.option("whiten").and_then(|option| option.kind.default_bool()).expect("PCA whiten"),
            solver: match descriptor.option("solver").and_then(|option| option.kind.default_str()) {
                Some("cov_eig_jacobi") => VectorPcaSolver::CovEigJacobi,
                Some("cov_eig_dq") => VectorPcaSolver::CovEigDq,
                _ => panic!("canonical PCA solver default"),
            },
            tolerance: descriptor.option("tol").and_then(|option| option.kind.default_f64()).expect("PCA tolerance"),
            n_iterations: descriptor
                .option("n_iterations")
                .and_then(|option| option.kind.default_u64())
                .expect("canonical PCA n_iterations default") as u32,
            flip_signs_based_on_u: descriptor
                .option("flip_signs_based_on_u")
                .and_then(|option| option.kind.default_bool())
                .expect("PCA sign flip"),
        }
    }
}
