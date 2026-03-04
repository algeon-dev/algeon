use super::*;

pub(super) fn fingerprint_schema_only(schema: &Schema) -> String {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("native-tpcds-parity-v1");
    hash_schema(&mut hasher, schema);
    hasher.update_field("rows=0");
    format!("output-fnv1a64:{:016x}", hasher.finish())
}

pub(super) fn fingerprint_batches(schema: &Schema, batches: &[RecordBatch]) -> String {
    match collect_canonical_rows(schema, batches) {
        Ok(rows) => fingerprint_rows(schema, &rows),
        Err(_) => fingerprint_schema_only(schema),
    }
}

pub(super) fn fingerprint_rows(schema: &Schema, rows: &[CanonicalRow]) -> String {
    let mut hasher = StableFnv1a64::new();
    hasher.update_field("native-tpcds-parity-v1");
    hash_comparison_schema(&mut hasher, schema);
    hasher.update_field(&format!("rows={}", rows.len()));
    for row in rows {
        for field in &row.fields {
            hash_field(&mut hasher, field);
        }
    }
    format!("output-fnv1a64:{:016x}", hasher.finish())
}

pub(super) fn hash_schema(hasher: &mut StableFnv1a64, schema: &Schema) {
    hasher.update_field("schema");
    hasher.update_field(&schema.fields().len().to_string());
    hash_metadata(hasher, schema.metadata());
    for field in schema.fields() {
        hasher.update_field("field");
        hasher.update_field(field.name());
        hasher.update_field(&format!("{:?}", field.data_type()));
        hasher.update_field(if field.is_nullable() { "nullable" } else { "non_nullable" });
        hash_metadata(hasher, field.metadata());
    }
}

pub(super) fn hash_comparison_schema(hasher: &mut StableFnv1a64, schema: &Schema) {
    hasher.update_field("comparison_schema");
    hasher.update_field(&schema.fields().len().to_string());
    for field in schema.fields() {
        hasher.update_field("field");
        hasher.update_field(&comparison_data_type_key(field.data_type()));
    }
}

fn comparison_data_type_key(data_type: &DataType) -> String {
    match data_type {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "utf8".to_string(),
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => "binary".to_string(),
        other => format!("{other:?}"),
    }
}

pub(super) fn hash_metadata(hasher: &mut StableFnv1a64, metadata: &HashMap<String, String>) {
    hasher.update_field(&metadata.len().to_string());
    let mut keys = metadata.keys().collect::<Vec<_>>();
    keys.sort();
    for key in keys {
        hasher.update_field(key);
        if let Some(value) = metadata.get(key) {
            hasher.update_field(value);
        }
    }
}

pub(super) fn hash_field(hasher: &mut StableFnv1a64, field: &CanonicalField) {
    match field {
        CanonicalField::Null => hasher.update_field("null"),
        CanonicalField::Bool(value) => {
            hasher.update_field("bool");
            hasher.update_field(if *value { "1" } else { "0" });
        },
        CanonicalField::Int { width, value } => {
            hasher.update_field("int");
            hasher.update_field(&width.to_string());
            hasher.update_field(&value.to_string());
        },
        CanonicalField::UInt { width, value } => {
            hasher.update_field("uint");
            hasher.update_field(&width.to_string());
            hasher.update_field(&value.to_string());
        },
        CanonicalField::Float { width, bits } => {
            hasher.update_field("float");
            hasher.update_field(&width.to_string());
            hasher.update_field(&format!("{bits:016x}"));
        },
        CanonicalField::Decimal128 { precision, scale, value } => {
            hasher.update_field("decimal128");
            hasher.update_field(&format!("{precision}:{scale}"));
            hasher.update_field(&value.to_string());
        },
        CanonicalField::Decimal256 { precision, scale, value } => {
            hasher.update_field("decimal256");
            hasher.update_field(&format!("{precision}:{scale}"));
            hasher.update_bytes(value);
        },
        CanonicalField::Date32(value) => {
            hasher.update_field("date32");
            hasher.update_field(&value.to_string());
        },
        CanonicalField::Date64(value) => {
            hasher.update_field("date64");
            hasher.update_field(&value.to_string());
        },
        CanonicalField::Timestamp { unit, tz, value } => {
            hasher.update_field("timestamp");
            hasher.update_field(&format!("{unit:?}"));
            hasher.update_field(tz.as_deref().unwrap_or(""));
            hasher.update_field(&value.to_string());
        },
        CanonicalField::Utf8(value) => {
            hasher.update_field("utf8");
            hasher.update_field(value);
        },
        CanonicalField::Binary(bytes) => {
            hasher.update_field("binary");
            hasher.update_bytes(bytes);
        },
        CanonicalField::Unsupported(text) => {
            hasher.update_field("unsupported");
            hasher.update_field(text);
        },
    }
}

// Fixed FNV-1a implementation so fingerprints do not depend on std hasher choices.
pub(super) struct StableFnv1a64 {
    state: u64,
}

impl StableFnv1a64 {
    fn new() -> Self {
        Self { state: 0xcbf29ce484222325 }
    }

    fn update_field(&mut self, value: &str) {
        self.update_bytes(value.as_bytes());
        self.update_bytes(&[0xff]);
    }

    fn update_bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state ^= u64::from(*byte);
            self.state = self.state.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(&self) -> u64 {
        self.state
    }
}
