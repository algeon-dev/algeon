#!/usr/bin/env python3
"""Check test module ownership, source boundaries, and current clippy coverage.

Usage: check_test_layout.py [ROOT] [--clippy-artifacts JSONL]
Coverage combines successful clippy streams from this checkout: the full
workspace and the feature-off owner targets. It never searches the shared target for older units.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from check_public_api_hidden import (
    CrateScanner, ScanError, Scope, derive_feature, group_end, tokenize, use_bindings,
)


class LayoutScanner(CrateScanner):
    def __init__(self, package, targets):
        self.package = package
        self.base = Path(package['manifest_path']).parent.resolve()
        super().__init__(self.base / 'src', package['name'])
        self.targets = targets
        self.owners = defaultdict(set)
        self.build_helpers = set()
        self.unit_support = set()
        self.aliases = []
        self.errors = []
        self.active = set()

    def error(self, path, line, message):
        self.errors.append(f'{path}:{line}: {message}')

    def scan_target(self, target):
        self.target = target
        self.seen = set()
        root = Path(target['src_path']).resolve()
        self.visit(root, (), root.parent)
        return self.seen

    def visit(self, path, cfgs, child_dir):
        path = path.resolve()
        if not path.is_relative_to(self.base):
            raise ScanError(f'{path}: cross-crate Rust module reference')
        if path in self.active:
            raise ScanError(f'{path}: recursive module reference')
        self.seen.add(path)
        self.owners[path].add((tuple(self.target['kind']), self.target['name']))
        self.active.add(path)
        tokens = tokenize(path.read_text(), path)
        self.check_includes(path, tokens)
        if path in self.unit_support:
            for i, token in enumerate(tokens[:-1]):
                if token.text in {'crate', 'super'} and tokens[i + 1].text == '::':
                    self.error(path, token.line, 'unit-test support must use public library names and its own imports')
        self.parse_items(path, tokens, 0, len(tokens), Scope('mod', (), cfgs, child_dir))
        self.active.remove(path)

    def check_includes(self, path, tokens):
        for i in range(len(tokens) - 2):
            if tokens[i].text != 'include' or tokens[i + 1].text != '!':
                continue
            stop = group_end(tokens, i + 2)
            body = tokens[i + 3:stop - 1]
            literal = len(body) == 1 and body[0].kind == 'string'
            generated = False
            if literal:
                source = (path.parent / body[0].text.strip('"')).resolve()
                if not source.is_relative_to(self.base):
                    self.error(path, tokens[i].line, 'cross-crate Rust include reference')
                    continue
                if source.is_file() and source.name == 'bindings.rs':
                    generated = 'automatically generated' in source.read_text()[:500]
            else:
                texts = [t.text for t in body]
                generated = texts[:9] == ['concat', '!', '(', 'env', '!', '(', '"OUT_DIR"', ')', ',']
            if not generated:
                self.error(path, tokens[i].line, 'handwritten include! is prohibited')

    def parse_attrs(self, path, tokens, index, end):
        start = index
        attrs, stop = super().parse_attrs(path, tokens, index, end)
        # Conditional path attributes need both module trees. A cfg_attr that
        # wraps #[path] is rejected; each branch is a separate #[cfg] and
        # #[path] declaration that the scan visits.
        cursor = start
        while cursor < stop:
            opener = cursor + 1 + (tokens[cursor + 1].text == '!')
            close = group_end(tokens, opener)
            body = tokens[opener + 1:close - 1]
            if body and body[0].text == 'cfg_attr' and any(t.kind == 'ident' and t.text == 'path' for t in body):
                raise ScanError(f'{path}:{tokens[cursor].line}: express cfg-selected paths with separate #[cfg] and #[path] declarations')
            cursor = close
        return attrs, stop

    def parse_item(self, path, tokens, scope, attrs, visibility, item, start, line, pos):
        cfgs = scope.cfgs + tuple(attrs.cfgs)
        if item.kind == 'skip':
            text = [t.text for t in tokens[start:item.stop]]
            for i in range(len(text) - 4):
                if text[i:i + 4] == ['extern', 'crate', 'self', 'as']:
                    public_names = {t['name'] for t in self.targets if 'lib' in t['kind']}
                    if path != self.base / 'src/lib.rs' or derive_feature(cfgs) is not None or text[i + 4] not in public_names:
                        self.error(path, line, 'self alias requires cfg(test) in lib.rs and the public library name')
                    self.aliases.append((path, line))
        if item.kind == 'use' and path in self.unit_support:
            for _, source in use_bindings(tokens, *item.header):
                if source[0] in {'super', 'crate'}:
                    self.error(path, line, 'unit-test support must use public library names and its own imports')
        if item.kind != 'mod':
            return
        if item.body is not None:
            self.parse_items(path, tokens, *item.body, replace(scope, cfgs=cfgs, child_dir=scope.child_dir / item.name))
            return
        file = self.mod_file(path, tokens, scope, attrs, item, start).resolve()
        if not file.is_relative_to(self.base):
            self.error(path, line, 'cross-crate Rust module reference')
            return
        support = file.is_relative_to(self.base / 'tests/support')
        build = file in self.build_helpers
        source_to_test = path.is_relative_to(self.base / 'src') and file.is_relative_to(self.base / 'tests')
        test_to_source = path.is_relative_to(self.base / 'tests') and file.is_relative_to(self.base / 'src')
        if source_to_test:
            if support and derive_feature(cfgs) is None:
                self.unit_support.add(file)
            else:
                self.error(path, line, 'src may load only shared tests/support from cfg(test)')
        if test_to_source:
            self.error(path, line, 'tests may not load src modules')
        if attrs.path and not (support or build or self.target['kind'] == ['custom-build']):
            # An alias selects a differently named implementation under cfg.
            implementation_name = file.parent.name if file.name == 'mod.rs' else file.stem
            alias = bool(attrs.cfgs) and implementation_name != item.name and not file.is_relative_to(self.base / 'tests')
            if not alias:
                self.error(path, line, '#[path] requires shared support, a build helper, or a cfg implementation alias')
        child_dir = file.parent if attrs.path or file.name == 'mod.rs' else file.parent / file.stem
        if path in self.unit_support:
            self.unit_support.add(file)
        self.visit(file, cfgs, child_dir)

    def check(self):
        # Build-script traversal identifies the helper exception from its owner,
        # without a filename allowlist.
        for target in self.targets:
            if target['kind'] == ['custom-build']:
                self.build_helpers.update(self.scan_target(target) - {Path(target['src_path']).resolve()})
        for target in self.targets:
            if target['kind'] != ['custom-build']:
                self.scan_target(target)
        test_roots = {Path(t['src_path']).resolve() for t in self.targets if t['kind'] == ['test']}
        tests = self.base / 'tests'
        if tests.is_dir():
            for path in tests.iterdir():
                if path.is_file() and path.suffix == '.rs' and path.resolve() not in test_roots:
                    self.error(path, 1, 'tests first level contains a non-target Rust file')
                if path.is_dir() and path.name != 'support' and any(path.rglob('*.rs')):
                    if (path / 'main.rs').resolve() not in test_roots:
                        self.error(path, 1, 'test target directory requires its own main.rs')
        for path, owners in self.owners.items():
            if len(owners) > 1 and path not in self.build_helpers and not path.is_relative_to(tests / 'support'):
                self.error(path, 1, 'source belongs to multiple targets: ' + ', '.join('/'.join((*kind, name)) for kind, name in sorted(owners)))
        for root in test_roots:
            name = next(t['name'] for t in self.targets if Path(t['src_path']).resolve() == root)
            if root not in {(tests / f'{name}.rs').resolve(), (tests / name / 'main.rs').resolve()}:
                self.error(root, 1, 'test root must match its Cargo target name')
            owned = {p for p, names in self.owners.items() if next((tuple(t['kind']), t['name']) for t in self.targets if Path(t['src_path']).resolve() == root) in names}
            owned = {p for p in owned if p not in self.build_helpers and not p.is_relative_to(tests / 'support')}
            if len(owned) > 1 and root.name != 'main.rs':
                self.error(root, 1, 'multi-file target requires tests/<target>/main.rs')
        if self.aliases and not self.unit_support:
            for path, line in self.aliases:
                self.error(path, line, 'self alias requires unit tests that load shared tests/support')
        return self.errors


def dep_info(artifact):
    paths = set()
    for name in artifact['filenames']:
        file = Path(name)
        if file.name == 'build-script-build':
            tag = file.parent.name.rsplit('-', 1)[1]
            candidate = file.with_name(f'build_script_build-{tag}.d')
        elif file.suffix in {'.rmeta', '.rlib'}:
            candidate = file.with_name(file.stem.removeprefix('lib') + '.d')
        else:
            candidate = file.with_suffix('.d')
        if candidate.is_file():
            paths.add(candidate)
    if not paths:
        raise ScanError(f"no dep-info for current artifact {artifact['target']['src_path']}")
    return paths


def compiled_sources(root, packages, stream):
    ids = {p['id'] for p in packages}
    compiled = set()
    artifacts = 0
    finished = []
    for line in stream.read_text().splitlines():
        message = json.loads(line)
        if message.get('reason') == 'build-finished':
            finished.append(message['success'])
        if message.get('reason') != 'compiler-artifact' or message['package_id'] not in ids:
            continue
        artifacts += 1
        for path in dep_info(message):
            # rustc emits Make escaping for spaces, backslashes, and '#'. Only
            # dependency sides count; phony rules have no dependencies.
            for rule in path.read_text().replace('\\\n', '').splitlines():
                if rule.startswith('#') or ': ' not in rule:
                    continue
                rhs = rule.split(': ', 1)[1]
                words = re.findall(r'(?:\\.|[^\s\\])+', rhs)
                for word in words:
                    name = re.sub(r'\\(.)', r'\1', word)
                    source = Path(name)
                    if source.suffix == '.rs':
                        compiled.add((root / source).resolve())
    if finished != [True] or not artifacts:
        raise ScanError('clippy artifact stream must end in a successful build and contain workspace artifacts')
    return compiled


def check(root, metadata, artifacts=None):
    packages = [p for p in metadata['packages'] if p['id'] in metadata['workspace_members']]
    errors = []
    for package in packages:
        errors.extend(LayoutScanner(package, package['targets']).check())
    if artifacts:
        streams = [artifacts] if isinstance(artifacts, Path) else artifacts
        compiled = set().union(*(compiled_sources(root, packages, stream) for stream in streams))
        tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', '*.rs']).decode().split('\0')
        bases = [Path(p['manifest_path']).parent.resolve() for p in packages]
        for name in filter(None, tracked):
            path = (root / name).resolve()
            if path.is_file() and any(path.is_relative_to(base) for base in bases) and path not in compiled:
                errors.append(f'{name}: tracked Rust source was not compiled by this clippy run')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path.cwd())
    parser.add_argument('--clippy-artifacts', type=Path, action='append')
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        metadata = json.loads(subprocess.check_output([
            'cargo', 'metadata', '--locked', '--all-features', '--no-deps', '--format-version', '1',
        ], cwd=root))
        errors = check(root, metadata, args.clippy_artifacts)
    except (ScanError, OSError, ValueError, subprocess.CalledProcessError) as error:
        errors = [str(error)]
    for error in errors:
        print('ERROR: test layout: ' + error, file=sys.stderr)
    if errors:
        return 1
    print('PASS: test layout' + (' and current clippy source coverage' if args.clippy_artifacts else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
