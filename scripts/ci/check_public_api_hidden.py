#!/usr/bin/env python3
"""Fail closed on unclassified `#[doc(hidden)] pub` surface in one crate.

`cargo public-api` omits `#[doc(hidden)]` items, so the snapshot gate in
``check_public_api_surface.sh`` cannot see them. This check closes that gap:
every `#[doc(hidden)]`-attributed public item under the crate's source
directory must resolve to exactly one row in the crate's hidden ledger
(``.config/policies/public-api/<package>-hidden.tsv``) naming its feature
contract, owner, consumer, and removal condition, and every row must match a
site. Module rows cover unattributed children inside that module.

Symbols are ``<root>::<module path>::<Type>::<item>``. Methods of an inherent
`impl` record under the implementing type; methods of a trait `impl` record as
``<Type as Trait>::<item>`` so two impls with a same-named method stay distinct.
The scanner follows `mod` declarations from ``lib.rs``, so the module path
comes from the declaration tree and not from file names.

The `feature` column is derived from the item's `cfg` predicates (its own plus
those of every enclosing module and `impl`), evaluated for a non-test build:

  * `bench-support`, `qualification`, `server`, `test-hooks`: the item compiles
    only when that Cargo feature is on.
  * `cross-crate`: the item requires none of them; the row's `consumer` names
    the workspace crates that reach it.

Capability features (`cugraph`, `cuvs`, `cuml`, `nvml`) do not change the
value. Items that cannot compile outside `cfg(test)` are not part of the
surface and are skipped.

Struct fields record as ``<Type>.<field>`` (tuple fields by position), enum
variants as ``<Enum>::<Variant>``, and a `pub use` group records one symbol per
name it introduces, or ``*(<path>)`` for a glob.

Every `#[doc(hidden)]` attribute in a scanned file must be classified as a
site, as a non-public item, or as test-only; an attribute in any other position
(macro body, `cfg_attr`, a hidden `impl` block) is an error, so a new shape
cannot bypass the ledger. A file that no `mod` declaration reaches fails when it
contains a hidden attribute.

A public item that compiles only with `test-hooks` must be `#[doc(hidden)]`,
so every test seam a consumer can name has a row; items inside a hidden module
are covered by the module row.

A hidden cross-crate module lists every public child separately: each one
carries its own `#[doc(hidden)]` or is narrowed, so no row stands in for
children nobody recorded.

A row whose symbol no crate names records `exposed-by: <symbols>`; each
symbol must resolve in the crate, a member through its type.

Each row names one public path. A module-level hidden item fails when some
enclosing module is not `pub mod`, because no consumer can name that path; the
attribute belongs on the `pub use` that publishes it. Two hidden symbols that
resolve to one definition (an item and its re-export, or two re-exports) fail
until one path remains. Members (fields, variants, methods) record under their
type's defining module.

Usage:
  python3 scripts/ci/check_public_api_hidden.py SRC_DIR ROOT_MODULE LEDGER

Exits nonzero on any ledger/source mismatch, with the offending symbols
printed to stderr.
"""

from __future__ import annotations

import csv
import itertools
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

HEADER = ["symbol", "feature", "owner", "consumer", "removal_condition"]
SEAM_FEATURES = ("bench-support", "qualification", "server", "test-hooks")
CROSS_CRATE = "cross-crate"
ALLOWED_FEATURES = {*SEAM_FEATURES, CROSS_CRATE}

TOKEN_RE = re.compile(
    r"""
      (?P<ws>\s+)
    | (?P<line_comment>//[^\n]*)
    | (?P<block_open>/\*)
    | (?P<raw_string>(?:br|cr|r)(?P<hashes>\#*)")
    | (?P<string>(?:b|c)?"(?:\\.|[^"\\])*")
    | (?P<char>b?'(?:\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f_]+\}|[^xu])|[^\\'])')
    | (?P<lifetime>'[A-Za-z_][A-Za-z0-9_]*)
    | (?P<ident>r\#[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*)
    | (?P<number>[0-9][0-9A-Za-z_]*(?:\.[0-9][0-9A-Za-z_]*)?)
    | (?P<punct>::|->|=>|.)
    """,
    re.VERBOSE | re.DOTALL,
)

OPENERS = {"(": ")", "[": "]", "{": "}"}
CLOSERS = set(OPENERS.values())
ITEM_QUALIFIERS = {"const", "async", "unsafe", "extern", "default", "safe"}


class ScanError(Exception):
    pass


class Token(NamedTuple):
    kind: str
    text: str
    line: int
    pos: int


def tokenize(text: str, path: Path) -> list[Token]:
    tokens: list[Token] = []
    pos, line, size = 0, 1, len(text)
    while pos < size:
        match = TOKEN_RE.match(text, pos)
        kind = match.lastgroup
        end = match.end()
        if kind == "block_open":
            depth = 1
            end = match.end()
            while depth:
                open_at, close_at = text.find("/*", end), text.find("*/", end)
                if close_at < 0:
                    raise ScanError(f"{path}:{line}: unterminated block comment")
                if 0 <= open_at < close_at:
                    depth, end = depth + 1, open_at + 2
                else:
                    depth, end = depth - 1, close_at + 2
        elif kind == "raw_string":
            terminator = '"' + match.group("hashes")
            close_at = text.find(terminator, end)
            if close_at < 0:
                raise ScanError(f"{path}:{line}: unterminated raw string")
            end = close_at + len(terminator)
        if kind not in {"ws", "line_comment", "block_open"}:
            tokens.append(Token(kind, text[pos:end], line, pos))
        line += text.count("\n", pos, end)
        pos = end
    return tokens


def group_end(tokens: list[Token], start: int) -> int:
    """Return the index after the bracket group opening at `start`."""
    depth = 0
    for index in range(start, len(tokens)):
        text = tokens[index].text
        if tokens[index].kind == "punct":
            if text in OPENERS:
                depth += 1
            elif text in CLOSERS:
                depth -= 1
                if depth == 0:
                    return index + 1
    raise ScanError(f"line {tokens[start].line}: unbalanced bracket")


def find_top_level(tokens: list[Token], start: int, stops: set[str], nested: str = "([") -> int:
    """Return the first index at bracket depth zero whose text is in `stops`."""
    closers = {OPENERS[opener] for opener in nested}
    depth = 0
    for index in range(start, len(tokens)):
        token = tokens[index]
        if depth == 0 and token.text in stops:
            return index
        if token.kind == "punct":
            if token.text in nested:
                depth += 1
            elif token.text in closers:
                depth -= 1
    raise ScanError(f"line {tokens[start].line}: item is not terminated")


def member_end(tokens: list[Token], index: int, end: int) -> int:
    """Return the index after the field or variant at `index`, past its comma."""
    depth, angle, in_expression = 0, 0, False
    while index < end:
        text = tokens[index].text
        if tokens[index].kind == "punct":
            if text in OPENERS:
                depth += 1
            elif text in CLOSERS:
                depth -= 1
            elif depth == 0 and not in_expression:
                if text == "<":
                    angle += 1
                elif text == ">":
                    angle -= 1
                elif text == "=" and angle == 0:
                    in_expression = True
            if text == "," and depth == 0 and angle <= 0:
                return index + 1
        index += 1
    return end


# A cfg predicate is ("atom", name), ("kv", name, value), ("not", p), ("all", ps), or ("any", ps).
Cfg = tuple


def parse_cfg(tokens: list[Token], index: int) -> tuple[Cfg, int]:
    name = tokens[index].text
    index += 1
    if index < len(tokens) and tokens[index].text == "(":
        index += 1
        args: list[Cfg] = []
        while tokens[index].text != ")":
            arg, index = parse_cfg(tokens, index)
            args.append(arg)
            if tokens[index].text == ",":
                index += 1
        index += 1
        if name == "not":
            return ("not", args[0]), index
        return (name, tuple(args)), index
    if index < len(tokens) and tokens[index].text == "=":
        return ("kv", name, tokens[index + 1].text.strip('"')), index + 2
    return ("atom", name), index


def cfg_variables(cfg: Cfg, found: set[tuple]) -> None:
    kind = cfg[0]
    if kind == "atom" and cfg[1] != "test":
        found.add(("atom", cfg[1]))
    elif kind == "kv":
        found.add(("kv", cfg[1], cfg[2]))
    elif kind == "not":
        cfg_variables(cfg[1], found)
    elif kind in {"all", "any"}:
        for child in cfg[1]:
            cfg_variables(child, found)


def cfg_value(cfg: Cfg, assignment: dict[tuple, bool]) -> bool:
    kind = cfg[0]
    if kind == "atom":
        return False if cfg[1] == "test" else assignment[("atom", cfg[1])]
    if kind == "kv":
        return assignment[("kv", cfg[1], cfg[2])]
    if kind == "not":
        return not cfg_value(cfg[1], assignment)
    if kind == "all":
        return all(cfg_value(child, assignment) for child in cfg[1])
    return any(cfg_value(child, assignment) for child in cfg[1])


_FEATURE_CACHE: dict[tuple[Cfg, ...], str | None] = {}


def derive_feature(cfgs: tuple[Cfg, ...]) -> str | None:
    """Return the ledger feature for a conjunction of cfgs, or None if test-only."""
    if cfgs in _FEATURE_CACHE:
        return _FEATURE_CACHE[cfgs]
    found: set[tuple] = set()
    for cfg in cfgs:
        cfg_variables(cfg, found)
    variables = sorted(found)
    if len(variables) > 16:
        raise ScanError(f"cfg conjunction has {len(variables)} variables: {cfgs}")
    satisfying = []
    for values in itertools.product((False, True), repeat=len(variables)):
        assignment = dict(zip(variables, values))
        if all(cfg_value(cfg, assignment) for cfg in cfgs):
            satisfying.append(assignment)
    if not satisfying:
        result = None
    else:
        required = [
            feature
            for feature in SEAM_FEATURES
            if all(assignment.get(("kv", "feature", feature), False) for assignment in satisfying)
        ]
        result = CROSS_CRATE if not required else "+".join(required)
    _FEATURE_CACHE[cfgs] = result
    return result


@dataclass(frozen=True)
class Scope:
    kind: str
    module: tuple[str, ...]
    cfgs: tuple[Cfg, ...]
    child_dir: Path
    owner: str | None = None
    trait_impl: bool = False


@dataclass
class Attrs:
    cfgs: list[Cfg]
    hidden_at: list[int]
    path: str | None
    inner_cfgs: list[Cfg]


class Site(NamedTuple):
    path: Path
    line: int
    # Offset in `path` where an outer attribute can precede the item.
    pos: int
    module: tuple[str, ...]
    # The path a `pub use` names, resolved against `module`; None for a definition.
    source: tuple[str, ...] | None = None


def compact(tokens: list[Token]) -> str:
    out: list[str] = []
    previous: Token | None = None
    for token in tokens:
        wordlike = token.kind in {"ident", "lifetime", "number", "string", "char"}
        if previous is not None:
            previous_word = previous.kind in {"ident", "lifetime", "number", "string", "char"}
            if (wordlike and previous_word) or previous.text == ",":
                out.append(" ")
        out.append(token.text)
        previous = token
    return "".join(out)


def skip_generics(tokens: list[Token], index: int) -> int:
    if index >= len(tokens) or tokens[index].text != "<":
        return index
    depth = 0
    for cursor in range(index, len(tokens)):
        if tokens[cursor].kind == "punct":
            if tokens[cursor].text == "<":
                depth += 1
            elif tokens[cursor].text == ">":
                depth -= 1
                if depth == 0:
                    return cursor + 1
    raise ScanError(f"line {tokens[index].line}: unbalanced generics")


def type_name(tokens: list[Token]) -> str:
    """Name the implementing type: the last path segment before its generics."""
    index = 0
    while index < len(tokens) and (
        tokens[index].text in {"&", "mut", "dyn", "const", "*"} or tokens[index].kind == "lifetime"
    ):
        index += 1
    stop = next((at for at in range(index, len(tokens)) if tokens[at].text == "<"), len(tokens))
    segments = tokens[index:stop]
    if segments and all(token.kind == "ident" or token.text == "::" for token in segments):
        return segments[-1].text if segments[-1].kind == "ident" else compact(tokens)
    return compact(tokens)


def trait_name(tokens: list[Token]) -> str:
    stop = next((at for at in range(len(tokens)) if tokens[at].text == "<"), len(tokens))
    start = 0
    for at in range(stop):
        if tokens[at].text == "::":
            start = at + 1
    return compact(tokens[start:])


def impl_owner(header: list[Token]) -> tuple[str, bool]:
    """Return (owner label, is trait impl) for tokens between `impl` and `{`."""
    index = skip_generics(header, 0)
    if index < len(header) and header[index].text == "!":
        index += 1
    where = next((at for at in range(index, len(header)) if header[at].text == "where"), len(header))
    body = header[index:where]
    depth, split = 0, None
    for at, token in enumerate(body):
        if token.text == "<":
            depth += 1
        elif token.text == ">":
            depth -= 1
        elif depth == 0 and token.text == "for" and not (at + 1 < len(body) and body[at + 1].text == "<"):
            split = at
            break
    if split is None:
        return type_name(body), False
    return f"<{type_name(body[split + 1:])} as {trait_name(body[:split])}>", True


def use_bindings(tokens: list[Token], start: int, end: int) -> list[tuple[str, tuple[str, ...]]]:
    """Return `(name, source path)` for each binding a `use` tree introduces; a glob's name is `*`."""

    def tree(index: int, prefix: list[str]) -> tuple[list[tuple[str, tuple[str, ...]]], int]:
        bindings: list[tuple[str, tuple[str, ...]]] = []
        segments = list(prefix)
        while index < end:
            token = tokens[index]
            if token.text == "{":
                index += 1
                while tokens[index].text != "}":
                    sub, index = tree(index, segments)
                    bindings += sub
                    if tokens[index].text == ",":
                        index += 1
                return bindings, index + 1
            if token.text == "*":
                return [("*", tuple(segments))], index + 1
            if token.text == "as":
                source = segments[:-1] if segments[-1] == "self" else segments
                return [(tokens[index + 1].text, tuple(source))], index + 2
            if token.text in {",", "}"}:
                break
            if token.kind == "ident":
                segments.append(token.text)
            index += 1
        if segments[-1] == "self":
            segments.pop()
        return [(segments[-1], tuple(segments))], index

    bindings, _ = tree(start, [])
    return bindings


class Item(NamedTuple):
    kind: str
    name: str | None
    stop: int
    body: tuple[int, int] | None = None
    header: tuple[int, int] | None = None
    tuple_body: bool = False


def read_item(path: Path, tokens: list[Token], index: int, end: int) -> Item:
    """Read one item starting after its attributes and visibility."""
    first = tokens[index]
    if first.text == "macro_rules" and tokens[index + 1].text == "!":
        stop = group_end(tokens, index + 3)
        return Item("skip", None, stop + (stop < end and tokens[stop].text == ";"))
    cursor = index
    while cursor < end and (tokens[cursor].text in ITEM_QUALIFIERS or tokens[cursor].kind == "string"):
        cursor += 1
    keyword = tokens[cursor].text if cursor < end else ""

    if keyword in {"fn", "struct", "enum", "union"}:
        name = tokens[cursor + 1].text
        stop = find_top_level(tokens, cursor + 2, {"{", ";"})
        if tokens[stop].text != "{":
            after_generics = skip_generics(tokens, cursor + 2)
            if keyword == "struct" and tokens[after_generics].text == "(":
                close = group_end(tokens, after_generics)
                return Item(keyword, name, stop + 1, body=(after_generics + 1, close - 1), tuple_body=True)
            return Item(keyword, name, stop + 1)
        close = group_end(tokens, stop)
        return Item(keyword, name, close, body=None if keyword == "fn" else (stop + 1, close - 1))
    if keyword == "type":
        return Item("type", tokens[cursor + 1].text, find_top_level(tokens, cursor + 1, {";"}, "([{") + 1)
    if keyword == "static":
        name_at = cursor + 2 if tokens[cursor + 1].text == "mut" else cursor + 1
        return Item("static", tokens[name_at].text, find_top_level(tokens, name_at, {";"}, "([{") + 1)
    if keyword == "use":
        semicolon = find_top_level(tokens, cursor + 1, {";"}, "([{")
        return Item("use", None, semicolon + 1, header=(cursor + 1, semicolon))
    if keyword == "crate":
        return Item("skip", None, find_top_level(tokens, cursor + 1, {";"}) + 1)
    if first.text == "const" and keyword not in {"fn", "impl", "trait"}:
        return Item("const", tokens[index + 1].text, find_top_level(tokens, index + 1, {";"}, "([{") + 1)
    if keyword == "mod":
        name = tokens[cursor + 1].text
        if tokens[cursor + 2].text == ";":
            return Item("mod", name, cursor + 3)
        return Item("mod", name, group_end(tokens, cursor + 2), body=(cursor + 3, group_end(tokens, cursor + 2) - 1))
    if keyword in {"trait", "impl"}:
        open_at = find_top_level(tokens, cursor + 1, {"{"})
        stop = group_end(tokens, open_at)
        name = tokens[cursor + 1].text if keyword == "trait" else None
        return Item(keyword, name, stop, body=(open_at + 1, stop - 1), header=(cursor + 1, open_at))
    if keyword == "{" and cursor > index:
        return Item("skip", None, group_end(tokens, cursor))
    if first.kind == "ident":
        cursor = index
        while cursor < end and (tokens[cursor].kind == "ident" or tokens[cursor].text == "::"):
            cursor += 1
        if cursor < end and tokens[cursor].text == "!":
            cursor += 1
            if tokens[cursor].kind == "ident":
                cursor += 1
            stop = group_end(tokens, cursor)
            return Item("skip", None, stop + (stop < end and tokens[stop].text == ";"))
    raise ScanError(f"{path}:{first.line}: unrecognized item starting at `{first.text}`")


class CrateScanner:
    def __init__(self, src: Path, root: str) -> None:
        self.src = src
        self.root = root
        self.sites: dict[tuple[str, str], list[Site]] = {}
        self.visited: set[Path] = set()
        self.classified: set[tuple[Path, int]] = set()
        # Modules every ancestor of which is declared `pub mod`.
        self.public_modules: set[tuple[str, ...]] = {(root,)}
        # Module-level hidden items whose own path no consumer can name.
        self.unnameable: list[tuple[str, Site]] = []
        # Modules carrying `#[doc(hidden)]`; their row covers unattributed children.
        self.hidden_modules: set[tuple[str, ...]] = set()
        # Public test-hooks items without `#[doc(hidden)]`, as
        # (symbol, module, owner type, path, line); main keeps the reachable ones.
        self.seam_candidates: list[tuple[str, tuple[str, ...], str | None, Path, int]] = []
        # Uncovered `pub use` bindings in public modules: (module, name, source).
        # A hidden `pub use` has its own row and a hidden module's row covers
        # its re-exports, so neither publishes a seam without a row.
        self.public_reexports: list[tuple[tuple[str, ...], str, tuple[str, ...]]] = []
        # Hidden types; their row covers their members.
        self.hidden_types: set[tuple[tuple[str, ...], str]] = set()
        # Public module-level children per module: (name, hidden, path, line,
        # source path for a `pub use`, else None).
        self.public_children: dict[
            tuple[str, ...], list[tuple[str, bool, Path, int, tuple[str, ...] | None]]
        ] = defaultdict(list)

    def scan(self) -> dict[tuple[str, str], list[Site]]:
        lib = self.src / "lib.rs"
        if not lib.is_file():
            raise ScanError(f"{lib} does not exist")
        self.scan_file(lib, (self.root,), (), self.src)
        for path in sorted(self.src.rglob("*.rs")):
            if path not in self.visited and "doc(hidden" in path.read_text():
                raise ScanError(f"{path}: contains doc(hidden) but no `mod` declaration reaches it")
        return self.sites

    def scan_file(self, path: Path, module: tuple[str, ...], cfgs: tuple[Cfg, ...], child_dir: Path) -> None:
        self.visited.add(path)
        tokens = tokenize(path.read_text(), path)
        self.parse_items(path, tokens, 0, len(tokens), Scope("mod", module, cfgs, child_dir))
        for index in range(len(tokens) - 2):
            if tokens[index].text == "doc" and tokens[index + 1].text == "(" and tokens[index + 2].text == "hidden":
                if (path, index) not in self.classified:
                    raise ScanError(
                        f"{path}:{tokens[index].line}: #[doc(hidden)] in a position the scanner does not classify"
                    )

    def parse_attrs(self, path: Path, tokens: list[Token], index: int, end: int) -> tuple[Attrs, int]:
        attrs = Attrs([], [], None, [])
        while index < end and tokens[index].text == "#":
            cursor = index + 1
            inner = tokens[cursor].text == "!"
            if inner:
                cursor += 1
            if tokens[cursor].text != "[":
                raise ScanError(f"{path}:{tokens[index].line}: malformed attribute")
            stop = group_end(tokens, cursor)
            body = cursor + 1
            name = tokens[body].text
            if name == "cfg" and tokens[body + 1].text == "(":
                cfg, _ = parse_cfg(tokens, body + 2)
                (attrs.inner_cfgs if inner else attrs.cfgs).append(cfg)
            elif name == "doc" and tokens[body + 1].text == "(" and not inner:
                depth = 0
                for at in range(body + 1, stop - 1):
                    depth += (tokens[at].text == "(") - (tokens[at].text == ")")
                    if depth == 1 and tokens[at].text == "hidden":
                        attrs.hidden_at.append(body)
            elif name == "path" and tokens[body + 1].text == "=":
                attrs.path = tokens[body + 2].text.strip('"')
            index = stop
        return attrs, index

    def mark_classified(self, path: Path, tokens: list[Token], start: int, stop: int) -> None:
        for index in range(start, stop - 2):
            if tokens[index].text == "doc" and tokens[index + 1].text == "(" and tokens[index + 2].text == "hidden":
                self.classified.add((path, index))

    def record(
        self,
        path: Path,
        scope: Scope,
        name: str,
        feature: str,
        line: int,
        pos: int,
        source: tuple[str, ...] | None = None,
    ) -> None:
        prefix = "::".join([*scope.module, *([scope.owner] if scope.owner else [])])
        separator = "." if scope.kind == "fields" else "::"
        symbol = f"{prefix}{separator}{name}"
        site = Site(path, line, pos, scope.module, source)
        self.sites.setdefault((symbol, feature), []).append(site)
        if scope.kind == "mod" and scope.module not in self.public_modules:
            self.unnameable.append((symbol, site))

    def parse_items(self, path: Path, tokens: list[Token], index: int, end: int, scope: Scope) -> None:
        while index < end:
            if tokens[index].text == ";":
                index += 1
                continue
            start = index
            attrs, index = self.parse_attrs(path, tokens, index, end)
            if attrs.inner_cfgs:
                scope = replace(scope, cfgs=scope.cfgs + tuple(attrs.inner_cfgs))
            if index >= end:
                break
            pos = tokens[index].pos
            visibility = None
            if tokens[index].text == "pub":
                index += 1
                visibility = "pub"
                if index < end and tokens[index].text == "(":
                    visibility = "restricted"
                    index = group_end(tokens, index)
            item = read_item(path, tokens, index, end)
            self.parse_item(path, tokens, scope, attrs, visibility, item, start, tokens[index].line, pos)
            index = item.stop

    def parse_item(
        self,
        path: Path,
        tokens: list[Token],
        scope: Scope,
        attrs: Attrs,
        visibility: str | None,
        item: Item,
        start: int,
        line: int,
        pos: int,
    ) -> None:
        cfgs = scope.cfgs + tuple(attrs.cfgs)
        feature = derive_feature(cfgs)
        if scope.kind == "mod" and visibility == "pub" and feature is not None and item.kind not in {"impl", "skip"}:
            bindings = use_bindings(tokens, *item.header) if item.kind == "use" else [(item.name, None)]
            for name, source in bindings:
                self.public_children[scope.module].append((name, bool(attrs.hidden_at), path, line, source))
        under_hidden = any(scope.module[:depth] in self.hidden_modules for depth in range(1, len(scope.module) + 1))
        if attrs.hidden_at and item.kind in {"struct", "enum", "union", "trait", "type"}:
            self.hidden_types.add((scope.module, item.name))
        if (
            scope.kind == "mod"
            and visibility == "pub"
            and item.kind == "use"
            and scope.module in self.public_modules
            and not attrs.hidden_at
            and not under_hidden
        ):
            for name, source in use_bindings(tokens, *item.header):
                self.public_reexports.append((scope.module, name, source))
        if (
            feature == "test-hooks"
            and not attrs.hidden_at
            and not under_hidden
            and (scope.module, scope.owner) not in self.hidden_types
            and scope.kind in {"mod", "impl"}
            and not scope.trait_impl
            and visibility == "pub"
            # A module is a namespace; its public children are checked one by one.
            and item.kind not in {"impl", "skip", "use", "mod"}
        ):
            owner = f"::{scope.owner}" if scope.owner else ""
            symbol = f"{'::'.join(scope.module)}{owner}::{item.name}"
            self.seam_candidates.append((symbol, scope.module, scope.owner, path, line))
        if feature is None:
            self.mark_classified(path, tokens, start, item.stop)
            if item.kind == "mod" and item.body is None:
                self.mod_file(path, tokens, scope, attrs, item, start, required=False)
            return
        if item.kind == "skip":
            return
        if item.kind == "impl":
            owner, trait_impl = impl_owner(tokens[slice(*item.header)])
            body = Scope("impl", scope.module, cfgs, scope.child_dir, owner, trait_impl)
            self.parse_items(path, tokens, *item.body, body)
            return

        member = scope.kind == "trait" or (scope.kind == "impl" and scope.trait_impl)
        if attrs.hidden_at:
            self.classified.update((path, at) for at in attrs.hidden_at)
            if visibility == "pub" or (member and visibility is None):
                if item.kind == "use":
                    for name, source in use_bindings(tokens, *item.header):
                        leaf = f"*({'::'.join(source)})" if name == "*" else name
                        self.record(path, scope, leaf, feature, line, pos, source)
                else:
                    self.record(path, scope, item.name, feature, line, pos)

        if item.kind in {"struct", "enum", "union"} and item.body is not None:
            start, stop = item.body
            if any(tokens[at].text == "doc" and tokens[at + 1].text == "(" for at in range(start, stop - 1)):
                kind = "tuple" if item.tuple_body else item.kind
                owner = Scope("variants" if kind == "enum" else "fields", scope.module, cfgs, scope.child_dir, item.name)
                self.parse_members(path, tokens, owner, start, stop, kind, visibility == "pub")
        if item.kind == "mod":
            if attrs.hidden_at:
                self.hidden_modules.add((*scope.module, item.name))
            if visibility == "pub" and scope.module in self.public_modules:
                self.public_modules.add((*scope.module, item.name))
            if item.body is None:
                file = self.mod_file(path, tokens, scope, attrs, item, start)
                child_dir = file.parent if file.name == "mod.rs" or attrs.path else file.parent / file.stem
                self.scan_file(file, (*scope.module, item.name), cfgs, child_dir)
            else:
                body = Scope("mod", (*scope.module, item.name), cfgs, scope.child_dir / item.name)
                self.parse_items(path, tokens, *item.body, body)
        elif item.kind == "trait":
            body = Scope("trait", scope.module, cfgs, scope.child_dir, item.name)
            if visibility == "restricted":
                self.mark_classified(path, tokens, *item.body)
            else:
                self.parse_items(path, tokens, *item.body, body)

    def parse_members(
        self, path: Path, tokens: list[Token], scope: Scope, start: int, end: int, kind: str, public: bool
    ) -> None:
        """Classify hidden attributes on struct fields, tuple fields, or enum variants."""
        is_enum = kind == "enum"
        index = start
        position = -1
        while index < end:
            position += 1
            attrs, index = self.parse_attrs(path, tokens, index, end)
            pos = tokens[index].pos if index < end else -1
            visibility = None
            if not is_enum and index < end and tokens[index].text == "pub":
                visibility = "pub"
                index += 1
                if index < end and tokens[index].text == "(":
                    visibility = "restricted"
                    index = group_end(tokens, index)
            if index >= end:
                break
            name, line = tokens[index].text, tokens[index].line
            if kind == "tuple":
                name = str(position)
            index = member_end(tokens, index, end)
            feature = derive_feature(scope.cfgs + tuple(attrs.cfgs))
            if not attrs.hidden_at:
                continue
            self.classified.update((path, at) for at in attrs.hidden_at)
            if feature is not None and public and (is_enum or visibility == "pub"):
                self.record(path, scope, name, feature, line, pos)

    def mod_file(
        self, path: Path, tokens: list[Token], scope: Scope, attrs: Attrs, item: Item, start: int, required: bool = True
    ) -> Path | None:
        if attrs.path is not None:
            file = path.parent / attrs.path
        else:
            file = next(
                (
                    candidate
                    for candidate in (scope.child_dir / f"{item.name}.rs", scope.child_dir / item.name / "mod.rs")
                    if candidate.is_file()
                ),
                None,
            )
        if file is None or not file.is_file():
            if required:
                raise ScanError(f"{path}:{tokens[start].line}: no file for `mod {item.name};`")
            return None
        self.visited.add(file)
        return file


class DefinitionIndex(CrateScanner):
    """Hidden sites plus every module-level name, so a re-export resolves to its definition."""

    def __init__(self, src: Path, root: str) -> None:
        super().__init__(src, root)
        self.modules: set[tuple[str, ...]] = {(root,)}
        self.definitions: dict[tuple[str, ...], dict[str, list[tuple[Path, int]]]] = defaultdict(dict)
        self.uses: dict[tuple[str, ...], dict[str, list[tuple[str, ...]]]] = defaultdict(dict)
        self.globs: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)

    def parse_item(
        self,
        path: Path,
        tokens: list[Token],
        scope: Scope,
        attrs: Attrs,
        visibility: str | None,
        item: Item,
        start: int,
        line: int,
        pos: int,
    ) -> None:
        if scope.kind == "mod" and derive_feature(scope.cfgs + tuple(attrs.cfgs)) is not None:
            module = scope.module
            if item.kind == "use":
                for name, source in use_bindings(tokens, *item.header):
                    if name == "*":
                        self.globs[module].append(source)
                    else:
                        self.uses[module].setdefault(name, []).append(source)
            elif item.name is not None and item.kind not in {"impl", "skip"}:
                self.definitions[module].setdefault(item.name, []).append((path, pos))
                if item.kind == "mod":
                    self.modules.add((*module, item.name))
        super().parse_item(path, tokens, scope, attrs, visibility, item, start, line, pos)

    def resolve(self, module: tuple[str, ...], source: tuple[str, ...], seen: frozenset = frozenset()):
        """Definition sites a path names from `module`, or None when it leaves the crate."""
        key = (module, source)
        if key in seen:
            return None
        seen = seen | {key}
        base, rest = self.anchor(module, source)
        if base is None or not rest:
            return None
        for segment in rest[:-1]:
            base = self.child_module(base, segment, seen)
            if base is None:
                return None
        return self.lookup(base, rest[-1], seen)

    def anchor(self, module: tuple[str, ...], source: tuple[str, ...]):
        first, rest = source[0], list(source[1:])
        if first == "crate":
            return (self.root,), rest
        if first in {"self", "super"}:
            base = module if first == "self" else module[:-1]
            while rest and rest[0] == "super":
                base, rest = base[:-1], rest[1:]
            return base, rest
        names = self.definitions[module].keys() | self.uses[module].keys()
        return (module, list(source)) if first in names else (None, [])

    def child_module(self, module: tuple[str, ...], name: str, seen: frozenset):
        if (*module, name) in self.modules:
            return (*module, name)
        for source in self.uses[module].get(name, []):
            base, rest = self.anchor(module, source)
            for segment in rest if base is not None else []:
                base = self.child_module(base, segment, seen)
                if base is None:
                    break
            if base is not None:
                return base
        return None

    def lookup(self, module: tuple[str, ...], name: str, seen: frozenset):
        if name in self.definitions[module]:
            return self.definitions[module][name]
        for source in self.uses[module].get(name, []):
            found = self.resolve(module, source, seen)
            if found:
                return found
        for source in self.globs[module]:
            found = self.resolve(module, (*source, name), seen)
            if found:
                return found
        return None


def fail(message: str) -> None:
    print(f"ERROR: hidden public API policy: {message}", file=sys.stderr)


CONSUMER_KINDS = ("prod", "tools", "tests", "exposed-by")
# A test-hooks item serves tests and developer tools, and a bench-support item
# serves bench and tools binaries; a production consumer of either would ship
# the seam.
FEATURE_BARRED_CONSUMERS = {"test-hooks": frozenset({"prod"}), "bench-support": frozenset({"prod"})}
# Plans live in the untracked docs/plan directory, so a condition that cites one
# cannot be checked from the repository.
PLAN_REFERENCE = re.compile(r"(?i)docs/plan|\bplan\s+P\d|\bP\d+\b|\b(?:step|stage)\s+\d")


def parse_consumers(value: str) -> dict[str, tuple[str, ...]]:
    """Parse `prod: a, b; tools: c; tests: d; exposed-by: <symbol>`.

    Clauses appear in that order, each at most once, with sorted unique
    names. `prod`, `tools`, and `tests` list workspace packages; `exposed-by`
    lists public symbols whose signatures reach a symbol no package names.
    """
    clauses: dict[str, tuple[str, ...]] = {}
    for clause in value.split("; "):
        kind, separator, names = clause.partition(": ")
        if not separator or kind not in CONSUMER_KINDS:
            raise ValueError(f"clause {clause!r} is not `<{'|'.join(CONSUMER_KINDS)}>: <names>`")
        if clauses and CONSUMER_KINDS.index(kind) <= CONSUMER_KINDS.index(next(reversed(clauses))):
            raise ValueError(f"clause {kind!r} is out of order or repeated")
        listed = tuple(names.split(", "))
        if not all(listed) or list(listed) != sorted(set(listed)):
            raise ValueError(f"{kind} names must be non-empty, sorted, and unique")
        clauses[kind] = listed
    return clauses


def read_ledger(ledger_path: Path) -> dict[tuple[str, str], dict[str, str]] | None:
    with ledger_path.open(newline="") as ledger_file:
        reader = csv.DictReader((row for row in ledger_file if not row.startswith("#")), delimiter="\t")
        if reader.fieldnames != HEADER:
            fail(f"{ledger_path} header must be {' '.join(HEADER)}")
            return None
        rows = list(reader)
    ledger: dict[tuple[str, str], dict[str, str]] = {}
    ok = True
    for line_number, row in enumerate(rows, 2):
        symbol, feature = row["symbol"], row["feature"]
        if not all(row[column] for column in HEADER):
            fail(f"{ledger_path}:{line_number} has an empty field")
            ok = False
        if feature not in ALLOWED_FEATURES:
            fail(f"{ledger_path}:{line_number} has unapproved feature {feature!r}")
            ok = False
        if (symbol, feature) in ledger:
            fail(f"{ledger_path}:{line_number} duplicates {(symbol, feature)}")
            ok = False
        try:
            consumers = parse_consumers(row["consumer"])
        except ValueError as error:
            fail(f"{ledger_path}:{line_number} consumer {row['consumer']!r}: {error}")
            ok = False
        else:
            barred = FEATURE_BARRED_CONSUMERS.get(feature, ()) & consumers.keys()
            if barred:
                fail(f"{ledger_path}:{line_number} {feature} item cannot have {', '.join(sorted(barred))} consumers")
                ok = False
        if PLAN_REFERENCE.search(row["removal_condition"]):
            fail(f"{ledger_path}:{line_number} removal_condition cites a plan document; state the condition itself")
            ok = False
        ledger[(symbol, feature)] = row
    return ledger if ok else None


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    src, root, ledger_path = Path(argv[0]), argv[1], Path(argv[2])
    if not ledger_path.is_file():
        fail(f"missing ledger {ledger_path}")
        return 1
    ledger = read_ledger(ledger_path)
    if ledger is None:
        return 1
    index = DefinitionIndex(src, root)
    try:
        sites = index.scan()
    except ScanError as error:
        fail(str(error))
        return 1

    status = 0
    reexported = {
        target
        for module, name, source in index.public_reexports
        if name != "*"
        for target in index.resolve(module, source) or []
    }
    for symbol, module, owner, path, line in index.seam_candidates:
        name = owner or symbol.rsplit("::", 1)[-1]
        definitions = index.definitions[module].get(name, [])
        if module in index.public_modules or any(definition in reexported for definition in definitions):
            fail(f"{path}:{line}: {symbol} compiles only with test-hooks but has no #[doc(hidden)] row")
            status = 1
    for symbol, site in index.unnameable:
        fail(
            f"{site.path}:{site.line}: {symbol} is hidden on a path no consumer can name; "
            "put #[doc(hidden)] on the public re-export instead"
        )
        status = 1
    paths: dict[tuple[Path, int], list[str]] = defaultdict(list)
    for (symbol, _), where in sites.items():
        site = where[0]
        targets = [(site.path, site.pos)] if site.source is None else index.resolve(site.module, site.source) or []
        for target in targets:
            paths[target].append(symbol)
    for symbols in paths.values():
        if len(symbols) > 1:
            fail(f"{' and '.join(sorted(symbols))} are hidden public paths to one definition; keep one")
            status = 1
    for (symbol, feature), _ in sorted(sites.items()):
        module = tuple(symbol.split("::"))
        if feature != CROSS_CRATE or module not in index.modules:
            continue
        for name, hidden, path, line, _ in index.public_children[module]:
            if not hidden:
                fail(
                    f"{path}:{line}: {symbol}::{name} is a public child of a hidden cross-crate module without "
                    "its own row; hide it or narrow it"
                )
                status = 1
    for (symbol, feature), where in sorted(sites.items()):
        if feature not in ALLOWED_FEATURES:
            fail(f"{symbol} resolves to unapproved feature contract {feature!r}")
            status = 1
        if len(where) > 1:
            places = ", ".join(f"{site.path}:{site.line}" for site in where)
            fail(f"{symbol} [{feature}] is recorded at several sites ({places}); the symbol is ambiguous")
            status = 1
    for (symbol, _), row in sorted(ledger.items()):
        for exposer in parse_consumers(row["consumer"]).get("exposed-by", ()):
            segments = exposer.replace(".", "::").split("::")
            path = ("crate", *segments[1:])
            # `Type::member` resolves through its type; members are not indexed.
            names_item = index.resolve((root,), path) or (len(path) > 2 and index.resolve((root,), path[:-1]))
            if segments[0] != root or not names_item:
                fail(f"{symbol} is exposed-by {exposer}, which names nothing in the crate")
                status = 1
    for symbol, feature in sorted(sites):
        if (symbol, feature) not in ledger:
            fail(f"unlisted hidden symbol {symbol} [{feature}] has no ledger entry")
            status = 1
    for symbol, feature in sorted(ledger):
        if (symbol, feature) not in sites:
            fail(f"stale ledger entry {symbol} [{feature}] matches no hidden site")
            status = 1
    if status == 0:
        print(f"PASS: {len(sites)} hidden symbols license-checked against {ledger_path.name}")
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
