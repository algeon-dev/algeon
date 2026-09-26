use super::common::MlRelationBinding;
use crate::plan::MlForestModelSnapshot;

/// Scores one relation with a pretrained forest bound at planning.
#[derive(Debug, Clone, PartialEq)]
pub struct MlForestPredictParams {
    pub input: MlRelationBinding,
    pub model: MlForestModelSnapshot,
}
