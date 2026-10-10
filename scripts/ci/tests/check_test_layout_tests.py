#!/usr/bin/env python3
"""Fault injection for module boundaries and clippy source coverage."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_test_layout import ScanError, check, compiled_sources


class TestLayout(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='algeon-test-layout-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        self.targets = []
        self.write('src/lib.rs', '')
        self.target('src/lib.rs', 'fixture', 'lib')

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def target(self, source, name, kind='test'):
        self.targets.append({'name': name, 'kind': [kind], 'src_path': str(self.root / source)})

    def metadata(self):
        return {'workspace_members': ['fixture'], 'packages': [{
            'id': 'fixture', 'name': 'fixture', 'manifest_path': str(self.root / 'Cargo.toml'),
            'targets': self.targets,
        }]}

    def errors(self, artifacts=None):
        return '\n'.join(check(self.root, self.metadata(), artifacts))

    def test_single_target_ordinary_modules_and_shared_support(self):
        self.write('tests/cases_cpu/main.rs', 'mod cases; #[path="../support/shared.rs"] mod shared;')
        self.write('tests/cases_cpu/cases.rs', 'use super::shared;')
        self.write('tests/support/shared.rs', 'use fixture::Public;')
        self.target('tests/cases_cpu/main.rs', 'cases_cpu')
        self.assertEqual(self.errors(), '')

    def test_unregistered_first_level_helper_is_rejected(self):
        self.write('tests/helper.rs', '')
        self.assertIn('non-target Rust file', self.errors())

    def test_multi_file_root_and_redundant_path_are_rejected(self):
        self.write('tests/cases_cpu.rs', '#[path="leaf.rs"] mod leaf;')
        self.write('tests/leaf.rs', '')
        self.target('tests/cases_cpu.rs', 'cases_cpu')
        errors = self.errors()
        self.assertIn('multi-file target', errors)
        self.assertIn('#[path] requires', errors)

    def test_cross_target_owned_file_is_rejected(self):
        self.write('tests/a_cpu/main.rs', 'mod shared;')
        self.write('tests/a_cpu/shared.rs', '')
        self.write('tests/b_cpu/main.rs', '#[path="../a_cpu/shared.rs"] mod shared;')
        self.target('tests/a_cpu/main.rs', 'a_cpu')
        self.target('tests/b_cpu/main.rs', 'b_cpu')
        self.assertIn('multiple targets', self.errors())

    def test_unit_support_requires_test_cfg_and_public_imports(self):
        self.write('src/lib.rs', '#[cfg(test)] #[path="../tests/support/fixture.rs"] mod support_fixture;')
        support = self.write('tests/support/fixture.rs', 'use super::*;')
        self.assertIn('own imports', self.errors())
        support.write_text('use fixture::Public;')
        self.assertEqual(self.errors(), '')
        self.write('src/lib.rs', '#[path="../tests/support/fixture.rs"] mod support_fixture;')
        self.assertIn('cfg(test)', self.errors())

    def test_unit_support_cannot_call_private_includer_items(self):
        self.write('src/lib.rs', '#[cfg(test)] #[path="../tests/support/fixture.rs"] mod support_fixture;')
        self.write('tests/support/fixture.rs', 'fn setup() { crate::private_setup(); }')
        self.assertIn('public library names', self.errors())

    def test_target_directory_must_match_cargo_name(self):
        self.write('tests/other/main.rs', '')
        self.target('tests/other/main.rs', 'cases_cpu')
        self.assertIn('Cargo target name', self.errors())

    def test_test_to_source_and_cross_crate_references_are_rejected(self):
        self.write('tests/a_cpu/main.rs', '#[path="../../src/private.rs"] mod private;')
        self.write('src/private.rs', '')
        self.target('tests/a_cpu/main.rs', 'a_cpu')
        self.assertIn('tests may not load src', self.errors())
        foreign = self.root.parent / (self.root.name + '-foreign.rs')
        foreign.write_text('')
        self.addCleanup(foreign.unlink)
        self.write('tests/a_cpu/main.rs', f'#[path="{foreign}"] mod private;')
        self.assertIn('cross-crate', self.errors())

    def test_build_helper_exception_is_discovered_from_build_owner(self):
        self.write('build.rs', 'mod build_support;')
        self.write('build_support.rs', '')
        self.target('build.rs', 'build-script-build', 'custom-build')
        self.write('tests/build_cpu.rs', '#[path="../build_support.rs"] mod build_support;')
        self.target('tests/build_cpu.rs', 'build_cpu')
        self.assertEqual(self.errors(), '')

    def test_cfg_implementation_alias_is_allowed(self):
        self.write('src/lib.rs', '#[cfg(unix)] #[path="unix.rs"] mod implementation;')
        self.write('src/unix.rs', '')
        self.assertEqual(self.errors(), '')

    def test_handwritten_include_is_rejected_but_generated_and_literals_are_allowed(self):
        self.write('src/lib.rs', 'include!("fragment.rs");')
        self.write('src/fragment.rs', '')
        self.assertIn('handwritten include!', self.errors())
        self.write('src/bindings.rs', '/* automatically generated by bindgen */')
        self.write('src/lib.rs', '''include!("bindings.rs");
include!(concat!(env!("OUT_DIR"), "/generated.rs"));
const TEXT: &str = r###"include!(\"fragment.rs\"); #[path=\"foreign.rs\"]"###;
/* include!("fragment.rs"); /* nested comment */ */''')
        self.assertEqual(self.errors(), '')

    def test_conditional_path_attribute_cannot_hide_a_branch(self):
        self.write('src/lib.rs', '#[cfg_attr(unix, path="../foreign.rs")] mod implementation;')
        self.write('src/implementation.rs', '')
        with self.assertRaisesRegex(ScanError, r'separate #\[cfg\] and #\[path\]'):
            self.errors()

    def test_generated_bindings_cannot_cross_crate_boundaries(self):
        foreign = self.root.parent / (self.root.name + '-foreign') / 'bindings.rs'
        foreign.parent.mkdir()
        foreign.write_text('/* automatically generated by bindgen */')
        self.addCleanup(foreign.parent.rmdir)
        self.addCleanup(foreign.unlink)
        self.write('src/lib.rs', f'include!("{foreign}");')
        self.assertIn('cross-crate Rust include', self.errors())

    def test_self_alias_requires_actual_shared_unit_support(self):
        self.write('src/lib.rs', '#[cfg(test)] extern crate self as fixture;')
        self.assertIn('load shared tests/support', self.errors())
        self.write('tests/support/fixture.rs', 'use fixture::Public;')
        self.write('src/lib.rs', '#[cfg(test)] extern crate self as fixture; '
                   '#[cfg(test)] #[path="../tests/support/fixture.rs"] mod support_fixture;')
        self.assertEqual(self.errors(), '')
        self.write('src/lib.rs', 'extern crate self as fixture; '
                   '#[cfg(test)] #[path="../tests/support/fixture.rs"] mod support_fixture;')
        self.assertIn('requires cfg(test)', self.errors())

    def compile_fixture(self):
        source = self.write('src/lib.rs', 'mod compiled;')
        self.write('src/compiled.rs', 'pub fn answer() -> u8 { 42 }')
        artifact = self.root / 'target/debug/deps/libfixture-current.rmeta'
        artifact.parent.mkdir(parents=True)
        dep = artifact.with_name('fixture-current.d')
        subprocess.run(['rustc', '--crate-type=lib', '--emit', f'metadata={artifact},dep-info={dep}',
                        str(source)], cwd=self.root, check=True, capture_output=True)
        message = {'reason': 'compiler-artifact', 'package_id': 'fixture',
                   'target': self.targets[0], 'filenames': [str(artifact)]}
        stream = self.write('artifacts.jsonl', json.dumps(message) + '\n' +
                            json.dumps({'reason': 'build-finished', 'success': True}) + '\n')
        subprocess.run(['git', '-C', str(self.root), 'add', 'src'], check=True)
        return stream, dep

    def test_real_dep_info_includes_compiled_sources(self):
        stream, _ = self.compile_fixture()
        self.assertEqual(self.errors(stream), '')

    def test_old_dep_info_cannot_hide_a_tracked_orphan(self):
        stream, _ = self.compile_fixture()
        orphan = self.write('src/orphan.rs', 'pub fn missing() {}')
        self.write('target/debug/deps/fixture-stale.d', f'old: {orphan}\n')
        subprocess.run(['git', '-C', str(self.root), 'add', 'src/orphan.rs'], check=True)
        self.assertIn('src/orphan.rs: tracked Rust source was not compiled', self.errors(stream))

    def test_feature_off_stream_covers_its_own_conditional_source(self):
        stream, dep = self.compile_fixture()
        source = self.write('src/lib.rs', 'mod compiled; #[cfg(not(feature="native"))] mod fallback;')
        self.write('src/fallback.rs', 'pub fn unavailable() {}')
        artifact = dep.with_name('libfixture-current.rmeta')
        subprocess.run(['rustc', '--crate-type=lib', '--cfg', 'feature="native"', '--emit',
                        f'metadata={artifact},dep-info={dep}', str(source)], check=True, capture_output=True)
        extra = artifact.with_name('libfixture-feature-off.rmeta')
        extra_dep = dep.with_name('fixture-feature-off.d')
        subprocess.run(['rustc', '--crate-type=lib', '--emit', f'metadata={extra},dep-info={extra_dep}',
                        str(source)], check=True, capture_output=True)
        message = {'reason': 'compiler-artifact', 'package_id': 'fixture',
                   'target': self.targets[0], 'filenames': [str(extra)]}
        supplement = self.write('feature-off.jsonl', json.dumps(message) + '\n' +
                                json.dumps({'reason': 'build-finished', 'success': True}) + '\n')
        subprocess.run(['git', '-C', str(self.root), 'add', 'src'], check=True)
        self.assertIn('fallback.rs', self.errors(stream))
        self.assertEqual(self.errors([stream, supplement]), '')

    def test_failed_or_missing_current_artifacts_fail_closed(self):
        stream, dep = self.compile_fixture()
        dep.unlink()
        with self.assertRaisesRegex(ScanError, 'no dep-info'):
            self.errors(stream)
        stream.write_text(json.dumps({'reason': 'build-finished', 'success': False}) + '\n')
        with self.assertRaisesRegex(ScanError, 'successful build'):
            self.errors(stream)

    def test_make_escaping_and_foreign_checkout_paths(self):
        stream, dep = self.compile_fixture()
        special = self.root / 'src/space # slash\\.rs'
        escaped = str(special).replace('\\', '\\\\').replace(' ', '\\ ').replace('#', '\\#')
        dep.write_text(f'current: {escaped} /other/checkout/src/lib.rs\n')
        sources = compiled_sources(self.root, self.metadata()['packages'], stream)
        self.assertIn(special, sources)
        self.assertIn('src/lib.rs: tracked Rust source was not compiled', self.errors(stream))


if __name__ == '__main__':
    unittest.main()
