//! Immutable GPU UDTF catalog and catalog-wide invariants.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    sync::Arc,
};

use super::{GpuFunctionDescriptor, GpuFunctionFamily};
use crate::error::{Error, ErrorCode, Result};

/// Immutable lookup entry coupling a descriptor to its owning family adapter.
#[derive(Clone)]
pub(crate) struct GpuFunctionCatalogEntry {
    pub(crate) descriptor: GpuFunctionDescriptor,
    pub(crate) family: Arc<dyn GpuFunctionFamily>,
}

/// Per-session immutable inventory of installed GPU UDTF families.
#[derive(Clone)]
pub(crate) struct GpuFunctionCatalog {
    families: Vec<Arc<dyn GpuFunctionFamily>>,
    entries: Vec<GpuFunctionCatalogEntry>,
    by_function_name: HashMap<String, usize>,
    family_ids: BTreeSet<&'static str>,
}

impl std::fmt::Debug for GpuFunctionCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuFunctionCatalog")
            .field("function_count", &self.entries.len())
            .field("family_ids", &self.family_ids)
            .finish()
    }
}

impl GpuFunctionCatalog {
    /// Construct a checked, lexically ordered immutable catalog.
    pub(crate) fn try_new(families: Vec<Arc<dyn GpuFunctionFamily>>) -> Result<Self> {
        let mut family_ids = BTreeSet::new();
        let mut providers = BTreeMap::new();
        let mut entries = Vec::new();
        for family in &families {
            let family_id = family.family_id();
            if !is_identifier(family_id) || !family_ids.insert(family_id) {
                return Err(catalog_error(format!("duplicate or invalid GPU function family id '{family_id}'")));
            }
            for descriptor in family.descriptors() {
                validate_descriptor(&descriptor)?;
                match providers.entry(descriptor.provider.clone()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert(family_id);
                    },
                    std::collections::btree_map::Entry::Occupied(entry) if *entry.get() == family_id => {},
                    std::collections::btree_map::Entry::Occupied(_) => {
                        return Err(catalog_error(format!(
                            "GPU provider '{}' is declared by more than one family",
                            descriptor.provider
                        )));
                    },
                }
                entries.push(GpuFunctionCatalogEntry { descriptor, family: Arc::clone(family) });
            }
        }
        entries.sort_by(|left, right| left.descriptor.function_name.cmp(&right.descriptor.function_name));
        let mut by_function_name = HashMap::with_capacity(entries.len());
        for (index, entry) in entries.iter().enumerate() {
            let function_name = entry.descriptor.function_name.clone();
            if by_function_name.insert(function_name.clone(), index).is_some() {
                return Err(catalog_error(format!("duplicate GPU function name '{function_name}'")));
            }
        }
        Ok(Self { families, entries, by_function_name, family_ids })
    }

    /// Return descriptors in deterministic canonical function-name order.
    pub(crate) fn entries(&self) -> &[GpuFunctionCatalogEntry] {
        &self.entries
    }

    /// Look up an execution function by exact canonical name.
    pub(crate) fn get(&self, function_name: &str) -> Option<&GpuFunctionCatalogEntry> {
        self.by_function_name.get(function_name).map(|index| &self.entries[*index])
    }

    /// Return whether this snapshot contains a family identity.
    pub(crate) fn contains_family(&self, family_id: &str) -> bool {
        self.family_ids.contains(family_id)
    }

    /// Return every family adapter represented by this snapshot.
    ///
    /// The separate family list deliberately preserves installed families with
    /// no descriptors (for example, a feature-disabled family that remains
    /// visible to a higher-level session builder).
    pub(crate) fn families(&self) -> &[Arc<dyn GpuFunctionFamily>] {
        &self.families
    }
}

fn validate_descriptor(descriptor: &GpuFunctionDescriptor) -> Result<()> {
    for (field, value) in
        [("function_name", descriptor.function_name.as_str()), ("provider", descriptor.provider.as_str())]
    {
        if !is_identifier(value) {
            return Err(catalog_error(format!("GPU descriptor {field} must be a non-empty lowercase identifier")));
        }
    }
    if !descriptor.function_name.starts_with(&format!("{}_", descriptor.provider)) {
        return Err(catalog_error("GPU descriptor function_name must be provider-qualified"));
    }
    if descriptor.signature.trim().is_empty() || descriptor.summary.trim().is_empty() {
        return Err(catalog_error("GPU descriptor signature and summary must be non-empty"));
    }
    if let Some(reason) = descriptor.availability.unavailable_reason()
        && !is_identifier(reason)
    {
        return Err(catalog_error("GPU descriptor unavailable reason must be a lowercase identifier"));
    }
    if !descriptor.argument_descriptions.is_array()
        || !descriptor.options_schema.is_object()
        || !descriptor.result_schemas.is_object()
        || !descriptor.validation_request_schema.is_object()
        || !descriptor.examples.is_array()
        || !descriptor.limitations.is_array()
    {
        return Err(catalog_error("GPU descriptor JSON values must use the common array/object contract"));
    }
    let roles = descriptor.relation_role_names().map_err(catalog_error)?;
    let mut unique_roles = BTreeSet::new();
    for role in roles {
        if !is_identifier(role) || !unique_roles.insert(role) {
            return Err(catalog_error("GPU descriptor relation roles must be unique non-empty lowercase identifiers"));
        }
    }
    Ok(())
}

fn is_identifier(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

fn catalog_error(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidRequest, "algeon_datafusion.gpu_functions.catalog", message)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::catalog::Session;
    use serde_json::json;

    use super::*;
    use crate::gpu_functions::{GpuCallRequest, GpuCallValidation, GpuFunctionAvailability};

    #[derive(Debug)]
    struct FakeFamily {
        id: &'static str,
        names: Vec<&'static str>,
    }

    #[async_trait::async_trait]
    impl GpuFunctionFamily for FakeFamily {
        fn family_id(&self) -> &'static str {
            self.id
        }

        fn descriptors(&self) -> Vec<GpuFunctionDescriptor> {
            self.names.iter().map(|name| descriptor(name)).collect()
        }

        async fn validate(&self, _: &dyn Session, function_name: &str, request: &GpuCallRequest) -> GpuCallValidation {
            GpuCallValidation::invalid("fake", "fake", function_name, request.normalized_json())
        }
    }

    fn descriptor(name: &str) -> GpuFunctionDescriptor {
        GpuFunctionDescriptor {
            function_name: format!("fake_{name}"),
            provider: "fake".to_string(),
            signature: format!("fake_{name}()"),
            summary: "test function".to_string(),
            relation_roles: json!([]),
            argument_descriptions: json!([]),
            options_schema: json!({}),
            result_schemas: json!({}),
            validation_request_schema: json!({}),
            examples: json!([]),
            limitations: json!([]),
            availability: GpuFunctionAvailability::Available,
        }
    }

    #[test]
    fn catalog_sorts_and_rejects_duplicate_names_and_families() {
        let alpha = Arc::new(FakeFamily { id: "alpha", names: vec!["z", "a"] });
        let catalog = GpuFunctionCatalog::try_new(vec![alpha]).unwrap();
        assert_eq!(
            catalog.entries().iter().map(|entry| entry.descriptor.function_name.as_str()).collect::<Vec<_>>(),
            vec!["fake_a", "fake_z"]
        );
        assert!(catalog.get("fake_a").is_some());
        assert!(catalog.get("a").is_none());

        let duplicate_name = GpuFunctionCatalog::try_new(vec![
            Arc::new(FakeFamily { id: "one", names: vec!["one"] }),
            Arc::new(FakeFamily { id: "two", names: vec!["one"] }),
        ]);
        assert!(duplicate_name.is_err());
        let duplicate_provider = GpuFunctionCatalog::try_new(vec![
            Arc::new(FakeFamily { id: "one", names: vec!["one"] }),
            Arc::new(FakeFamily { id: "two", names: vec!["two"] }),
        ]);
        assert!(duplicate_provider.is_err());
        let duplicate_family = GpuFunctionCatalog::try_new(vec![
            Arc::new(FakeFamily { id: "one", names: vec!["one"] }),
            Arc::new(FakeFamily { id: "one", names: vec!["two"] }),
        ]);
        assert!(duplicate_family.is_err());
    }
}
