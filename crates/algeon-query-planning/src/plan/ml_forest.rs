//! Backend-neutral facts for a pretrained tree-ensemble model bound at planning.

use std::{cmp::Ordering, fmt, hash, sync::Arc};

use arrow::datatypes::{DataType, Field, Schema};
use sha2::{Digest as _, Sha256};

use super::SchemaRef;
use crate::error::{QueryError, QueryErrorCode, Result};

const SNAPSHOT: &str = "query_engine.plan.ml_forest";

/// SHA-256 of the model file bytes a statement was planned against.
///
/// The digest covers the source file, not the derived checkpoint, so operators
/// can compare it with `sha256sum <model file>`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MlForestModelDigest([u8; 32]);

impl MlForestModelDigest {
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::digest(bytes).into())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// 64 lowercase hexadecimal characters.
    pub fn to_hex(&self) -> String {
        const DIGITS: &[u8; 16] = b"0123456789abcdef";
        self.0
            .iter()
            .flat_map(|byte| [DIGITS[usize::from(byte >> 4)], DIGITS[usize::from(byte & 0x0f)]])
            .map(char::from)
            .collect()
    }

    /// First 12 hexadecimal characters, used in plan display and stage labels.
    pub fn short_hex(&self) -> String {
        let mut hex = self.to_hex();
        hex.truncate(12);
        hex
    }
}

impl fmt::Debug for MlForestModelDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for MlForestModelDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", self.to_hex())
    }
}

/// Serialized tree-ensemble format named by the `model_format` option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MlForestModelFormat {
    XgboostJson,
    XgboostUbjson,
    XgboostLegacy,
    LightGbm,
    TreeliteCheckpoint,
}

impl MlForestModelFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::XgboostJson => "xgboost_json",
            Self::XgboostUbjson => "xgboost_ubjson",
            Self::XgboostLegacy => "xgboost_legacy",
            Self::LightGbm => "lightgbm",
            Self::TreeliteCheckpoint => "treelite",
        }
    }

    pub fn from_option_value(value: &str) -> Option<Self> {
        match value {
            "xgboost_json" => Some(Self::XgboostJson),
            "xgboost_ubjson" => Some(Self::XgboostUbjson),
            "xgboost_legacy" => Some(Self::XgboostLegacy),
            "lightgbm" => Some(Self::LightGbm),
            "treelite" => Some(Self::TreeliteCheckpoint),
            _ => None,
        }
    }
}

impl fmt::Display for MlForestModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Floating-point type of the imported forest. Double forests score Float64
/// features, so execution widens the Float32 input matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MlForestPrecision {
    Single,
    Double,
}

impl MlForestPrecision {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Double => "double",
        }
    }
}

impl fmt::Display for MlForestPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Row shape of a forest prediction result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MlForestOutputContract {
    /// One Float32 `prediction` per row.
    Score,
    /// One Int32 `prediction` per row in `0..classes`.
    ClassIndex { classes: usize },
    /// `output_0..output_{n-1}` Float32 values per row.
    Vector(usize),
}

impl MlForestOutputContract {
    /// Number of prediction columns per row.
    pub const fn width(self) -> usize {
        match self {
            Self::Score | Self::ClassIndex { .. } => 1,
            Self::Vector(width) => width,
        }
    }

    /// Arrow type of every prediction column.
    pub const fn prediction_type(self) -> DataType {
        match self {
            Self::ClassIndex { .. } => DataType::Int32,
            Self::Score | Self::Vector(_) => DataType::Float32,
        }
    }
}

impl fmt::Display for MlForestOutputContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Score => f.write_str("score"),
            Self::ClassIndex { classes } => write!(f, "class_index({classes})"),
            Self::Vector(width) => write!(f, "vector({width})"),
        }
    }
}

/// Facts nvForest's importer reported for the planned model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MlForestModelFacts {
    pub num_features: usize,
    pub num_trees: usize,
    /// Native per-row output stride.
    pub native_output_stride: usize,
    pub precision: MlForestPrecision,
}

/// Immutable model bound to one statement at planning.
///
/// Holds the backend checkpoint derived from the source file, the source
/// file's digest, and the facts execution re-proves at import. `Debug`,
/// equality, ordering, and hashing cover the digest and facts, never the
/// checkpoint bytes, so plan fingerprints change with the source file without
/// embedding it.
#[derive(Clone)]
pub struct MlForestModelSnapshot {
    checkpoint: Arc<[u8]>,
    digest: MlForestModelDigest,
    format: MlForestModelFormat,
    facts: MlForestModelFacts,
    output: MlForestOutputContract,
}

impl MlForestModelSnapshot {
    /// Digests `source` and checks that `output` is a shape `facts` can produce.
    pub fn new(
        source: &[u8],
        checkpoint: impl Into<Arc<[u8]>>,
        format: MlForestModelFormat,
        facts: MlForestModelFacts,
        output: MlForestOutputContract,
    ) -> Result<Self> {
        let checkpoint = checkpoint.into();
        if source.is_empty() || checkpoint.is_empty() {
            return Err(QueryError::new(QueryErrorCode::InvalidPlan, SNAPSHOT, "forest model bytes are empty")
                .with_fact("reason", "empty_forest_model"));
        }
        if facts.num_features == 0 || facts.num_trees == 0 || facts.native_output_stride == 0 {
            return Err(QueryError::new(QueryErrorCode::InvalidPlan, SNAPSHOT, "forest model facts must be positive")
                .with_fact("reason", "invalid_forest_facts")
                .with_fact("num_features", facts.num_features)
                .with_fact("num_trees", facts.num_trees)
                .with_fact("native_output_stride", facts.native_output_stride));
        }
        let stride = facts.native_output_stride;
        let consistent = match output {
            MlForestOutputContract::Score => stride == 1,
            MlForestOutputContract::Vector(width) => width == stride && width >= 2,
            MlForestOutputContract::ClassIndex { classes } => classes == stride && classes >= 2,
        };
        if !consistent {
            return Err(QueryError::new(
                QueryErrorCode::InvalidPlan,
                SNAPSHOT,
                "forest output contract differs from the model output stride",
            )
            .with_fact("reason", "forest_output_contract_mismatch")
            .with_fact("output", output)
            .with_fact("native_output_stride", stride));
        }
        Ok(Self { checkpoint, digest: MlForestModelDigest::of(source), format, facts, output })
    }

    /// Backend checkpoint uploaded at execution.
    pub fn checkpoint(&self) -> &[u8] {
        &self.checkpoint
    }

    pub fn digest(&self) -> MlForestModelDigest {
        self.digest
    }

    pub fn format(&self) -> MlForestModelFormat {
        self.format
    }

    pub fn facts(&self) -> MlForestModelFacts {
        self.facts
    }

    pub fn num_features(&self) -> usize {
        self.facts.num_features
    }

    pub fn output(&self) -> MlForestOutputContract {
        self.output
    }

    pub fn precision(&self) -> MlForestPrecision {
        self.facts.precision
    }

    fn identity(&self) -> (MlForestModelDigest, MlForestModelFormat, MlForestModelFacts, MlForestOutputContract) {
        (self.digest, self.format, self.facts, self.output)
    }
}

impl fmt::Debug for MlForestModelSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MlForestModelSnapshot")
            .field("digest", &self.digest)
            .field("checkpoint_len", &self.checkpoint.len())
            .field("format", &self.format)
            .field("facts", &self.facts)
            .field("output", &self.output)
            .finish()
    }
}

impl PartialEq for MlForestModelSnapshot {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
}

impl Eq for MlForestModelSnapshot {}

impl PartialOrd for MlForestModelSnapshot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MlForestModelSnapshot {
    fn cmp(&self, other: &Self) -> Ordering {
        self.identity().cmp(&other.identity())
    }
}

impl hash::Hash for MlForestModelSnapshot {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.identity().hash(state);
    }
}

/// Stable forest prediction schema for the input ID type and output contract.
pub fn ml_forest_output_schema(id_type: DataType, output: MlForestOutputContract) -> SchemaRef {
    let mut fields = vec![Field::new("row_ordinal", DataType::UInt64, false), Field::new("id", id_type, false)];
    match output {
        MlForestOutputContract::Score | MlForestOutputContract::ClassIndex { .. } => {
            fields.push(Field::new("prediction", output.prediction_type(), false));
        },
        MlForestOutputContract::Vector(width) => {
            fields.extend((0..width).map(|index| Field::new(format!("output_{index}"), DataType::Float32, false)));
        },
    }
    Arc::new(Schema::new(fields))
}

#[cfg(test)]
mod tests {
    use sha2::{Digest as _, Sha256};

    use super::{
        MlForestModelDigest, MlForestModelFacts, MlForestModelFormat, MlForestModelSnapshot, MlForestOutputContract,
        MlForestPrecision,
    };

    const FACTS: MlForestModelFacts = MlForestModelFacts {
        num_features: 4,
        num_trees: 3,
        native_output_stride: 1,
        precision: MlForestPrecision::Single,
    };

    fn snapshot(source: &[u8], checkpoint: &[u8]) -> MlForestModelSnapshot {
        MlForestModelSnapshot::new(
            source,
            checkpoint.to_vec(),
            MlForestModelFormat::XgboostJson,
            FACTS,
            MlForestOutputContract::Score,
        )
        .unwrap()
    }

    #[test]
    fn digest_is_sha256_of_source_bytes_and_formats_as_prefixed_hex() {
        let source = b"{\"learner\": {}}";
        let digest = snapshot(source, b"checkpoint").digest();
        let expected: [u8; 32] = Sha256::digest(source).into();
        assert_eq!(digest.as_bytes(), &expected);
        assert_eq!(digest, MlForestModelDigest::of(source));
        let hex = expected.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        assert_eq!(digest.to_hex(), hex);
        assert_eq!(digest.short_hex(), hex[..12]);
        assert_eq!(digest.to_string(), format!("sha256:{hex}"));
    }

    #[test]
    fn debug_and_equality_cover_digest_and_facts_but_never_checkpoint_bytes() {
        let checkpoint = b"SECRET-CHECKPOINT-PAYLOAD";
        let model = snapshot(b"source", checkpoint);
        let debug = format!("{model:#?}");
        assert!(debug.contains(&model.digest().to_hex()), "{debug}");
        assert!(debug.contains(&format!("checkpoint_len: {}", checkpoint.len())), "{debug}");
        assert!(!debug.contains("SECRET"), "{debug}");
        assert!(!debug.contains(&format!("{:?}", checkpoint.as_slice())), "{debug}");
        assert_eq!(model, snapshot(b"source", b"other checkpoint bytes"));
        assert_ne!(model, snapshot(b"source2", checkpoint));
    }

    #[test]
    fn constructor_rejects_empty_bytes_zero_facts_and_contracts_the_stride_cannot_produce() {
        let rejected = |source: &[u8], checkpoint: &[u8], facts: MlForestModelFacts, output| {
            MlForestModelSnapshot::new(source, checkpoint.to_vec(), MlForestModelFormat::LightGbm, facts, output)
                .unwrap_err()
                .fact_value("reason")
                .map(str::to_owned)
        };
        let score = MlForestOutputContract::Score;
        let with_stride = |stride| MlForestModelFacts { native_output_stride: stride, ..FACTS };
        assert_eq!(rejected(b"", b"c", FACTS, score).as_deref(), Some("empty_forest_model"));
        assert_eq!(rejected(b"s", b"", FACTS, score).as_deref(), Some("empty_forest_model"));
        for facts in [
            MlForestModelFacts { num_features: 0, ..FACTS },
            MlForestModelFacts { num_trees: 0, ..FACTS },
            with_stride(0),
        ] {
            assert_eq!(rejected(b"s", b"c", facts, score).as_deref(), Some("invalid_forest_facts"));
        }
        for (facts, output) in [
            (with_stride(3), score),
            (with_stride(1), MlForestOutputContract::Vector(1)),
            (with_stride(3), MlForestOutputContract::Vector(2)),
            (with_stride(1), MlForestOutputContract::ClassIndex { classes: 1 }),
            (with_stride(3), MlForestOutputContract::ClassIndex { classes: 2 }),
        ] {
            assert_eq!(
                rejected(b"s", b"c", facts, output).as_deref(),
                Some("forest_output_contract_mismatch"),
                "{output}"
            );
        }
        for (stride, output) in
            [(1, score), (3, MlForestOutputContract::Vector(3)), (3, MlForestOutputContract::ClassIndex { classes: 3 })]
        {
            let model = MlForestModelSnapshot::new(
                b"s",
                b"c".to_vec(),
                MlForestModelFormat::XgboostJson,
                with_stride(stride),
                output,
            )
            .unwrap();
            assert_eq!(model.output(), output);
        }
    }
}
