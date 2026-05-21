use super::*;

// ===========================================================================
// StringOperator enum
// ===========================================================================

/// String operations that can be applied within an [`Expr`].
#[derive(Debug, Clone)]
pub enum StringOperator {
    Lower,
    Upper,
    Strip,
    LStrip,
    RStrip,
    Len,
    ByteCount,
    Contains(String),
    StartsWith(String),
    EndsWith(String),
    Like { pattern: String, escape: Option<String> },
    ReplaceWithBackrefs { pattern: String, replacement: String },
    Slice(StringSliceOptions),
}

// ===========================================================================
// String expression methods
// ===========================================================================

/// String expression methods on string columns.
///
/// Construct via [`Expr::str`]. Methods on this type produce row-aligned
/// expressions that preserve the input column name.
pub struct ExprStringNamespace(pub(super) Expr);

impl ExprStringNamespace {
    /// Convert each string element to lowercase.
    pub fn lower(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Lower }
    }

    /// Convert each string element to uppercase.
    pub fn upper(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Upper }
    }

    /// Strip whitespace from both sides.
    pub fn strip(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Strip }
    }

    /// Strip leading whitespace.
    pub fn lstrip(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::LStrip }
    }

    /// Strip trailing whitespace.
    pub fn rstrip(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::RStrip }
    }

    /// Character count per string element.
    pub fn len(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Len }
    }

    /// Byte count per string element.
    pub fn byte_count(self) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::ByteCount }
    }

    /// Literal substring containment test. Returns BOOL8.
    pub fn contains(self, pat: impl Into<String>) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Contains(pat.into()) }
    }

    /// Starts-with test. Returns BOOL8.
    pub fn starts_with(self, pat: impl Into<String>) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::StartsWith(pat.into()) }
    }

    /// Ends-with test. Returns BOOL8.
    pub fn ends_with(self, pat: impl Into<String>) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::EndsWith(pat.into()) }
    }

    /// SQL `LIKE` pattern match. Returns BOOL8.
    ///
    /// `%` matches zero or more characters and `_` matches one character.
    /// This is not a regex API.
    pub fn like(self, pattern: impl Into<String>) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Like { pattern: pattern.into(), escape: None } }
    }

    /// SQL `LIKE` pattern match with an explicit escape character.
    pub fn like_escape(self, pattern: impl Into<String>, escape: impl Into<String>) -> Expr {
        Expr::StringOp {
            input: Box::new(self.0),
            op: StringOperator::Like { pattern: pattern.into(), escape: Some(escape.into()) },
        }
    }

    /// Regex replacement using a back-reference replacement template.
    pub fn replace_with_backrefs(self, pattern: impl Into<String>, replacement: impl Into<String>) -> Expr {
        Expr::StringOp {
            input: Box::new(self.0),
            op: StringOperator::ReplaceWithBackrefs { pattern: pattern.into(), replacement: replacement.into() },
        }
    }

    /// Slice strings using zero-based character positions.
    pub fn slice(self, options: StringSliceOptions) -> Expr {
        Expr::StringOp { input: Box::new(self.0), op: StringOperator::Slice(options) }
    }
}
