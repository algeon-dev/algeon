use super::common::MlRelationBinding;
use crate::plan::MlForestModelSnapshot;

/// Scores one relation with a pretrained forest bound at planning.
#[derive(Debug, Clone, PartialEq)]
pub struct MlForestPredictParams {
    /// Dense feature rows scored against the pretrained model.
    pub input: MlRelationBinding,
    /// Immutable checkpoint, source digest, and output facts bound at planning.
    pub model: MlForestModelSnapshot,
}
