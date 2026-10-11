//! Column reference IR shared across native expressions.
//!
//! See [`ColumnRef`] for the lookup/display-name split and qualifier rules.

use std::fmt;

/// Column identity in the native expression IR.
///
/// `name` is the physical/logical field name used to resolve the column against
/// an input schema. `qualifier` disambiguates which input namespace owns that
/// field when a node has more than one input. `display_name` is deliberately
/// separate: it is the externally visible or projected name used for output
/// contracts and selected diagnostics, and it may differ from `name` when the
/// same source column is projected under a distinct alias. `Display`
/// formatting uses the qualifier and lookup name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ColumnRef {
    /// Optional namespace for resolving the column in multi-input or graph contexts.
    qualifier: Option<ColumnQualifier>,
    /// Source/input field name used for schema lookup and execution lowering.
    name: String,
    /// User-visible or projected field name retained for output contracts.
    display_name: String,
}

impl ColumnRef {
    /// Build a column reference with no namespace qualifier.
    ///
    /// Unqualified references resolve against the current single-input schema.
    /// The display name is initialized to the same value as the lookup name.
    pub fn unqualified(name: impl Into<String>) -> Self {
        let name = name.into();
        Self { qualifier: None, display_name: name.clone(), name }
    }

    /// Build a column reference scoped to an input ordinal.
    ///
    /// This is the preferred qualifier for binary-input nodes such as joins.
    /// `input` is zero-based and identifies the node input, not a field index.
    /// Equality and hashing include the qualifier and both names, so a join
    /// key spelled with a different `display_name` is a different key.
    pub fn input(input: usize, name: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            qualifier: Some(ColumnQualifier::Input { index: input }),
            name: name.into(),
            display_name: display_name.into(),
        }
    }

    pub fn qualifier(&self) -> Option<&ColumnQualifier> {
        self.qualifier.as_ref()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn display_name(&self) -> &str {
        &self.display_name
    }

    pub fn is_qualified(&self) -> bool {
        self.qualifier.is_some()
    }
}

impl fmt::Display for ColumnRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(qualifier) = &self.qualifier {
            write!(f, "{qualifier}.{}", self.name)
        } else {
            f.write_str(&self.name)
        }
    }
}

/// Namespace used to disambiguate column references.
///
/// The formatted qualifier is part of expression diagnostics and plan display.
/// It is not necessarily the field name exposed by downstream schemas; use
/// [`ColumnRef::display_name`] for that.
///
/// Only [`Self::Input`] takes part in resolution: join validation, capability
/// analysis and the optimizer match `Input { index: 0 | 1 }` to a join side and
/// treat an unqualified reference as single-input. `Node` and `Relation` are
/// carried for display and are matched nowhere in this crate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ColumnQualifier {
    /// Positional input namespace, formatted as `input{index}`.
    Input {
        /// Zero-based input ordinal for a multi-input node.
        index: usize,
    },
    /// Query-plan node namespace.
    Node {
        /// Stable node identifier used by the plan graph.
        id: String,
    },
    /// Named relation namespace.
    Relation {
        /// Relation or table name.
        name: String,
    },
}

impl fmt::Display for ColumnQualifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input { index } => write!(f, "input{index}"),
            Self::Node { id } => write!(f, "{id}"),
            Self::Relation { name } => f.write_str(name),
        }
    }
}
