use std::{fs, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RustToken<'a> {
    Identifier(&'a str),
    Punctuation(u8),
}

/// Return the Rust tokens needed by source architecture guards, excluding
/// comments and string or character literals.
///
/// The architecture guards only look for ASCII module and keyword names. A
/// small lexer keeps those checks independent of formatting, so grouped or
/// aliased imports and declarations split across lines cannot bypass them.
fn rust_tokens(source: &str) -> Vec<RustToken<'_>> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;

    while index < bytes.len() {
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'/') {
            index += 2;
            while index < bytes.len() && bytes[index] != b'\n' {
                index += 1;
            }
            continue;
        }
        if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
            index += 2;
            let mut depth = 1usize;
            while index < bytes.len() && depth > 0 {
                if bytes[index] == b'/' && bytes.get(index + 1) == Some(&b'*') {
                    depth += 1;
                    index += 2;
                } else if bytes[index] == b'*' && bytes.get(index + 1) == Some(&b'/') {
                    depth -= 1;
                    index += 2;
                } else {
                    index += 1;
                }
            }
            continue;
        }

        if let Some((content_start, hashes)) = raw_string_start(bytes, index) {
            index = content_start;
            while index < bytes.len() {
                if bytes[index] == b'"'
                    && bytes
                        .get(index + 1..index + 1 + hashes)
                        .is_some_and(|closing_hashes| closing_hashes.iter().all(|byte| *byte == b'#'))
                {
                    index += 1 + hashes;
                    break;
                }
                index += 1;
            }
            continue;
        }

        if bytes[index] == b'"' {
            index = quoted_literal_end(bytes, index + 1, b'"');
            continue;
        }
        if bytes[index] == b'\'' {
            if let Some(end) = character_literal_end(bytes, index) {
                index = end;
                continue;
            }
            index += 1;
            continue;
        }

        if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
            let start = index;
            index += 1;
            while index < bytes.len() && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_') {
                index += 1;
            }
            tokens.push(RustToken::Identifier(&source[start..index]));
            continue;
        }

        if bytes[index].is_ascii_punctuation() {
            tokens.push(RustToken::Punctuation(bytes[index]));
        }
        index += 1;
    }

    tokens
}

fn raw_string_start(bytes: &[u8], index: usize) -> Option<(usize, usize)> {
    let mut cursor = index;
    if bytes.get(cursor) == Some(&b'b') {
        cursor += 1;
    }
    if bytes.get(cursor) != Some(&b'r') {
        return None;
    }
    cursor += 1;
    let hashes_start = cursor;
    while bytes.get(cursor) == Some(&b'#') {
        cursor += 1;
    }
    (bytes.get(cursor) == Some(&b'"')).then_some((cursor + 1, cursor - hashes_start))
}

fn quoted_literal_end(bytes: &[u8], mut index: usize, delimiter: u8) -> usize {
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            index = (index + 2).min(bytes.len());
        } else if bytes[index] == delimiter {
            return index + 1;
        } else {
            index += 1;
        }
    }
    index
}

fn character_literal_end(bytes: &[u8], index: usize) -> Option<usize> {
    let content = index + 1;
    if bytes.get(content) == Some(&b'\\') {
        let end = quoted_literal_end(bytes, content + 1, b'\'');
        return (end <= bytes.len() && bytes.get(end.saturating_sub(1)) == Some(&b'\'')).then_some(end);
    }
    (bytes.get(content + 1) == Some(&b'\'')).then_some(content + 2)
}

pub fn rust_identifier_count(source: &str, identifier: &str) -> usize {
    rust_tokens(source)
        .into_iter()
        .filter(|token| matches!(token, RustToken::Identifier(candidate) if *candidate == identifier))
        .count()
}

fn contains_public_use(tokens: &[RustToken<'_>]) -> bool {
    tokens.windows(2).any(|tokens| matches!(tokens, [RustToken::Identifier("pub"), RustToken::Identifier("use")]))
}

fn contains_public_glob_use(tokens: &[RustToken<'_>]) -> bool {
    let mut index = 0;
    while index + 1 < tokens.len() {
        if matches!(&tokens[index..index + 2], [RustToken::Identifier("pub"), RustToken::Identifier("use")]) {
            index += 2;
            while index < tokens.len() && tokens[index] != RustToken::Punctuation(b';') {
                if tokens[index] == RustToken::Punctuation(b'*') {
                    return true;
                }
                index += 1;
            }
        }
        index += 1;
    }
    false
}

pub fn assert_no_removed_public_surface(dir: &Path) {
    for entry in fs::read_dir(dir).unwrap_or_else(|error| panic!("failed to read {}: {error}", dir.display())) {
        let path = entry.unwrap_or_else(|error| panic!("failed to read directory entry: {error}")).path();
        if path.is_dir() {
            assert_no_removed_public_surface(&path);
            continue;
        }
        if path.extension().is_none_or(|extension| extension != "rs") {
            continue;
        }

        let source =
            fs::read_to_string(&path).unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        let tokens = rust_tokens(&source);
        for declaration in tokens.windows(3) {
            let [RustToken::Identifier("pub"), RustToken::Identifier("mod"), RustToken::Identifier(module)] =
                declaration
            else {
                continue;
            };
            assert!(*module != "prelude", "{} must not reintroduce a prelude facade", path.display());
            assert!(!module.ends_with("_contracts"), "{} must not reintroduce a *_contracts facade", path.display());
        }
        assert!(!contains_public_glob_use(&tokens), "{} must not reintroduce a glob re-export", path.display());
    }
}

pub fn assert_no_crate_root_reexports(crate_root: &Path) {
    let source = fs::read_to_string(crate_root)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", crate_root.display()));
    assert!(
        !contains_public_use(&rust_tokens(&source)),
        "{} must not reintroduce crate-root flat re-exports; canonical paths are module paths",
        crate_root.display()
    );
}

#[allow(dead_code)] // adapter facade tests; pulled via path into engine contracts too
pub fn assert_modules_are_private(module_root: &Path, modules: &[&str]) {
    let source = fs::read_to_string(module_root)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", module_root.display()));
    let tokens = rust_tokens(&source);
    for module in modules {
        assert!(
            !tokens.windows(3).any(|declaration| {
                matches!(
                    declaration,
                    [RustToken::Identifier("pub"), RustToken::Identifier("mod"), RustToken::Identifier(candidate)]
                        if candidate == module
                )
            }),
            "{} must keep `{module}` private so its root re-exports remain the only public paths",
            module_root.display()
        );
    }
}

#[test]
fn rust_token_scan_is_format_independent_and_ignores_non_code() {
    let source = r###"
        // pub use crate::ignored;
        const DOC: &str = "exec cugraph_sql pub use";
        pub
        use crate::{kept, *};
        use crate::{exec as executor, cugraph_sql as graph, runtime as native_runtime};
    "###;
    let tokens = rust_tokens(source);

    assert!(contains_public_use(&tokens));
    assert!(contains_public_glob_use(&tokens));
    assert_eq!(rust_identifier_count(source, "exec"), 1);
    assert_eq!(rust_identifier_count(source, "cugraph_sql"), 1);
    assert_eq!(rust_identifier_count(source, "runtime"), 1);
}
