//! Session-builder installation of immutable GPU UDTF catalog snapshots.

use std::sync::Arc;

use datafusion::{execution::SessionStateBuilder, logical_expr::planner::RelationPlanner};

use super::{
    GpuFunctionCatalog, GpuFunctionFamily, gpu_metadata_table_functions, unknown_function::UnknownGpuFunctionPlanner,
};
use crate::error::{Error, ErrorCode, Result};

/// Install one family into the builder's immutable GPU UDTF catalog.
///
/// Reinstalling the same family identity is idempotent only when it has the
/// same descriptor projection; changing a family's session configuration after
/// it has been installed is a local setup error. Every successful installation
/// replaces the catalog extension and rebinds the three shared UDTFs and the
/// unknown-function relation planner to the new immutable snapshot. That
/// planner stays last in the builder's relation planners, after every family
/// planner, which DataFusion tries first because installation prepends them.
pub(crate) fn install_gpu_function_family(
    builder: &mut SessionStateBuilder,
    family: Arc<dyn GpuFunctionFamily>,
) -> Result<()> {
    let config = builder.config().get_or_insert_with(Default::default);
    let prior = config.get_extension::<GpuFunctionCatalog>();
    let mut families = prior.as_ref().map(|catalog| catalog.families().to_vec()).unwrap_or_default();

    if let Some(existing) = families.iter().find(|existing| existing.family_id() == family.family_id()) {
        if family.same_installation(existing.as_ref()) {
            return Ok(());
        }
        return Err(install_error(format!(
            "conflicting GPU function family installation for '{}'",
            family.family_id()
        )));
    }
    families.push(family);
    let catalog = Arc::new(GpuFunctionCatalog::try_new(families)?);
    config.set_extension(Arc::clone(&catalog));
    let prior_planner = config.get_extension::<UnknownGpuFunctionPlanner>();
    let unknown_function_planner = Arc::new(UnknownGpuFunctionPlanner::new(Arc::clone(&catalog)));
    config.set_extension(Arc::clone(&unknown_function_planner));
    let table_functions = builder.table_functions().get_or_insert_with(Default::default);
    for function in gpu_metadata_table_functions(catalog) {
        table_functions.insert(function.name().to_string(), function);
    }
    let relation_planners = builder.relation_planners().get_or_insert_with(Vec::new);
    if let Some(prior_planner) = prior_planner {
        let prior_planner: Arc<dyn RelationPlanner> = prior_planner;
        relation_planners.retain(|planner| !Arc::ptr_eq(planner, &prior_planner));
    }
    relation_planners.push(unknown_function_planner);
    Ok(())
}

fn install_error(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidRequest, "algeon_datafusion.gpu_functions.install", message)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::{catalog::Session, execution::SessionStateBuilder};
    use serde_json::json;

    use super::*;
    use crate::gpu_functions::{GpuCallRequest, GpuCallValidation, GpuFunctionAvailability, GpuFunctionDescriptor};

    #[derive(Debug)]
    struct FakeFamily {
        id: &'static str,
        name: &'static str,
    }

    #[async_trait::async_trait]
    impl GpuFunctionFamily for FakeFamily {
        fn family_id(&self) -> &'static str {
            self.id
        }

        fn descriptors(&self) -> Vec<GpuFunctionDescriptor> {
            vec![GpuFunctionDescriptor {
                function_name: self.name.to_string(),
                provider: self.id.to_string(),
                signature: format!("{}()", self.name),
                summary: "test".to_string(),
                relation_roles: json!([]),
                argument_descriptions: json!([]),
                options_schema: json!({}),
                result_schemas: json!({}),
                validation_request_schema: json!({}),
                examples: json!([]),
                limitations: json!([]),
                availability: GpuFunctionAvailability::Available,
            }]
        }

        async fn validate(&self, _: &dyn Session, function_name: &str, request: &GpuCallRequest) -> GpuCallValidation {
            GpuCallValidation::invalid("fake", "fake", function_name, request.normalized_json())
        }
    }

    #[test]
    fn builder_extension_is_immutable_snapshot_and_composes_families() {
        let mut builder = SessionStateBuilder::new();
        install_gpu_function_family(&mut builder, Arc::new(FakeFamily { id: "one", name: "one_function" })).unwrap();
        let first = builder.config().as_ref().unwrap().get_extension::<GpuFunctionCatalog>().unwrap();
        install_gpu_function_family(&mut builder, Arc::new(FakeFamily { id: "two", name: "two_function" })).unwrap();
        let second = builder.config().as_ref().unwrap().get_extension::<GpuFunctionCatalog>().unwrap();
        assert_eq!(first.entries().len(), 1);
        assert_eq!(second.entries().len(), 2);
        assert!(second.contains_family("one"));
        assert!(second.contains_family("two"));
        assert!(builder.table_functions().as_ref().unwrap().contains_key("gpu_list_functions"));
    }

    #[test]
    fn identical_install_is_idempotent_and_conflicting_install_is_rejected() {
        let mut builder = SessionStateBuilder::new();
        install_gpu_function_family(&mut builder, Arc::new(FakeFamily { id: "one", name: "one_function" })).unwrap();
        install_gpu_function_family(&mut builder, Arc::new(FakeFamily { id: "one", name: "one_function" })).unwrap();
        assert_eq!(
            builder.config().as_ref().unwrap().get_extension::<GpuFunctionCatalog>().unwrap().entries().len(),
            1
        );
        assert!(
            install_gpu_function_family(&mut builder, Arc::new(FakeFamily { id: "one", name: "changed_function" }))
                .is_err()
        );
    }
}
