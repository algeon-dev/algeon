#!/usr/bin/env python3
"""Exercise documentation reachability with real rustdoc JSON and gate shims."""

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/ci/check_public_api_docs.py"
SPEC = importlib.util.spec_from_file_location("documentation_policy", SCRIPT)
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)

OWNER = '''//! Workspace implementation dependency.
extern crate query_planning;
mod private {
    /// Sealed report.
    pub struct Report {
        /// Observation state.
        pub state: Status,
        /// Details with an inaccessible defining path.
        pub nested: Nested,
    }
    /// State of one observation.
    pub enum Status {
        /// Ready with a scalar count.
        Ready(usize),
        /// Completed with details.
        Complete {
            /// Completion detail.
            detail: query_planning::Detail,
        },
    }
    /// Report nested through a field, without a public reexport.
    pub struct Nested;
    /// Item reachable only through a trait-associated type.
    pub struct AssociatedReport;
    /// Input reachable only through a trait implementation signature.
    pub struct TraitInput;
    impl From<TraitInput> for Report {
        fn from(_: TraitInput) -> Self { panic!("fixture constructor is not executed") }
    }
    impl Iterator for Report {
        type Item = AssociatedReport;
        fn next(&mut self) -> Option<Self::Item> { None }
    }
    impl Report {
        /// Obtain nested details from a public method.
        pub fn nested(&self) -> Nested { Nested }
        #[doc(hidden)]
        pub fn internal(&self) -> Hidden { Hidden }
    }
    #[doc(hidden)]
    pub struct Hidden;
    pub struct UnrelatedUndocumented;
}
pub use private::Report;
mod nested_methods {
    impl super::private::Nested {
        /// Read the planning detail.
        pub fn detail(&self) -> query_planning::Detail { query_planning::Detail }
    }
}
'''
PLANNING = '''//! Planning dependency.
/// Trait used by macro-generated implementation details.
pub trait PublicFlags {
    /// Concrete representation supplied by the macro.
    type Internal;
}
mod private {
    /// Planning detail.
    pub struct Detail;
}
pub use private::Detail;
pub struct UnrelatedUndocumented;
'''
APPLICATION = '''//! Application entry point.
extern crate query_runtime;
/// Reports for applications.
pub mod reports { pub use query_runtime::Report; }
'''


class DocumentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="algeon-public-doc-tests-")
        cls.directory = Path(cls.temporary.name)
        cls.sources = {}
        for crate, source in {"query_planning": PLANNING, "query_runtime": OWNER,
                              "algeon_datafusion": APPLICATION}.items():
            cls.sources[crate] = cls.directory / f"{crate}.rs"
            cls.sources[crate].write_text(source)
            command = ["rustc", "+" + POLICY.NIGHTLY_TOOLCHAIN, "--crate-type", "lib", "--crate-name", crate,
                       str(cls.sources[crate]), "-o", str(cls.directory / f"lib{crate}.rlib"),
                       "-L", f"dependency={cls.directory}"]
            if crate != "query_planning":
                dependency = "query_runtime" if crate == "algeon_datafusion" else "query_planning"
                command.extend(["--extern", f"{dependency}={cls.directory}/lib{dependency}.rlib"])
            subprocess.run(command, check=True, capture_output=True, text=True)
        cls.baseline = {crate: cls.document(crate) for crate in cls.sources}
        packages = [{"id": crate, "name": package, "manifest_path": str(cls.directory / crate / "Cargo.toml"),
                     "targets": [{"kind": ["lib"], "name": crate}]} for crate, package in {
                         "algeon_datafusion": "algeon-datafusion", "query_runtime": "algeon-query-runtime",
                         "query_planning": "algeon-query-planning"}.items()]
        cls.metadata = {"workspace_members": list(cls.sources), "packages": packages,
                        "workspace_root": str(cls.directory), "target_directory": str(cls.directory / "target")}
        cls.metadata_path = cls.directory / "metadata.json"
        cls.metadata_path.write_text(json.dumps(cls.metadata))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def document(cls, crate):
        command = ["rustdoc", "+" + POLICY.NIGHTLY_TOOLCHAIN, "--crate-type", "lib", "--crate-name", crate,
                   "-Z", "unstable-options", "--output-format", "json", "--document-private-items",
                   "--document-hidden-items", str(cls.sources[crate]), "-o", str(cls.directory),
                   "-L", f"dependency={cls.directory}"]
        if crate != "query_planning":
            dependency = "query_runtime" if crate == "algeon_datafusion" else "query_planning"
            command.extend(["--extern", f"{dependency}={cls.directory}/lib{dependency}.rlib"])
        subprocess.run(command, check=True, capture_output=True, text=True)
        return json.loads((cls.directory / f"{crate}.json").read_text())

    def setUp(self):
        for crate, source in {"query_planning": PLANNING, "query_runtime": OWNER,
                              "algeon_datafusion": APPLICATION}.items():
            self.sources[crate].write_text(source)
            (self.directory / f"{crate}.json").write_text(json.dumps(self.baseline[crate]))

    def check(self):
        return subprocess.run(["python3", str(SCRIPT), "check", "--metadata", str(self.metadata_path),
                               "--json-dir", str(self.directory)], capture_output=True, text=True)

    def remove_documentation(self, crate, text):
        source = self.sources[crate]
        source.write_text(source.read_text().replace(text, "", 1))
        self.document(crate)
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        return result.stderr

    def test_documented_closure_ignores_unrelated_and_hidden_items(self):
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_reexported_owner_named_field_requires_docs(self):
        self.assertIn("Report::state", self.remove_documentation("query_runtime", "/// Observation state."))

    def test_reexported_owner_variant_requires_docs(self):
        self.assertIn("Status::Ready", self.remove_documentation("query_runtime", "/// Ready with a scalar count."))

    def test_tuple_payload_reaches_transitive_workspace_owner(self):
        source = self.sources["query_runtime"]
        source.write_text(source.read_text().replace("Ready(usize)", "Ready(query_planning::Detail)"))
        self.document("query_runtime")
        self.assertIn("query_planning::private::Detail",
                      self.remove_documentation("query_planning", "/// Planning detail."))

    def test_method_on_type_without_public_reexport_requires_docs(self):
        self.assertIn("Nested::detail", self.remove_documentation("query_runtime", "/// Read the planning detail."))

    def test_trait_associated_type_requires_docs_without_requiring_impl_member_docs(self):
        self.assertIn("AssociatedReport", self.remove_documentation(
            "query_runtime", "/// Item reachable only through a trait-associated type."))

    def test_trait_implementation_signature_type_requires_docs(self):
        self.assertIn("TraitInput", self.remove_documentation(
            "query_runtime", "/// Input reachable only through a trait implementation signature."))

    def test_whitespace_docs_do_not_satisfy_presence(self):
        document = self.baseline["query_runtime"]
        document = json.loads(json.dumps(document))
        next(item for item in document["index"].values() if item["name"] == "Report")["docs"] = " \n\t"
        (self.directory / "query_runtime.json").write_text(json.dumps(document))
        self.assertEqual(self.check().returncode, 1)

    def test_missing_workspace_owner_and_incomplete_json_fail(self):
        (self.directory / "query_planning.json").unlink()
        result = self.check()
        self.assertIn("reachable workspace owner algeon-query-planning", result.stderr)
        self.assertEqual(result.returncode, 1)
        document = json.loads(json.dumps(self.baseline["algeon_datafusion"]))
        document["includes_private"] = False
        (self.directory / "algeon_datafusion.json").write_text(json.dumps(document))
        self.assertIn("omits private items", self.check().stderr)

    def test_missing_member_record_fails(self):
        document = json.loads(json.dumps(self.baseline["query_runtime"]))
        field = next(item_id for item_id, item in document["index"].items() if item["name"] == "state")
        del document["index"][field]
        (self.directory / "query_runtime.json").write_text(json.dumps(document))
        self.assertIn("omits child item", self.check().stderr)

    def test_build_preserves_feature_arguments(self):
        args = ["--all-features", "--features", "one", "--features=two", "-F", "three",
                "--target", "x86_64-unknown-linux-gnu", "--target-dir", "/tmp/docs-target", "--color", "never"]
        self.assertEqual(POLICY.build_arguments(args), args[:-2])

    def test_anonymous_const_macro_associated_type_has_no_documentation_target(self):
        source = self.sources["query_runtime"]
        macro = '''
    macro_rules! flags {
        ($owner:ty) => {
            const _: () = {
                pub struct InternalFlags;
                impl query_planning::PublicFlags for $owner {
                    type Internal = InternalFlags;
                }
            };
        };
    }
    flags!(Report);
'''
        source.write_text(source.read_text().replace("pub use private::Report;", macro + "pub use private::Report;"))
        document = self.document("query_runtime")
        internal = next(item for item in document["index"].values() if item["name"] == "Internal")
        reference = str(internal["inner"]["assoc_type"]["type"]["resolved_path"]["id"])
        self.assertNotIn(reference, document["index"])
        self.assertNotIn(reference, document["paths"])
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_hidden_parent_traits_and_private_traits_do_not_extend_public_surface(self):
        source = self.sources["query_runtime"]
        private_traits = '''
#[doc(hidden)]
pub mod testing {
    pub trait Probe {
        type Output;
        fn probe(&self) -> Self::Output;
    }
    pub struct ProbeOutput;
    impl Probe for crate::Report {
        type Output = ProbeOutput;
        fn probe(&self) -> Self::Output { ProbeOutput }
    }
}
pub(crate) trait Internal {
    type Output;
}
pub struct InternalOutput;
impl Internal for Report { type Output = InternalOutput; }
'''
        source.write_text(source.read_text() + private_traits)
        self.document("query_runtime")
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_visible_reexport_out_of_hidden_parent_still_requires_owner_docs(self):
        source = self.sources["query_runtime"]
        hidden_definition = '''
#[doc(hidden)]
pub mod implementation {
    pub struct Visible;
}
pub use implementation::Visible;
impl private::Report {
    /// Visible application type defined under a hidden module.
    pub fn visible(&self) -> Visible { Visible }
}
'''
        source.write_text(source.read_text() + hidden_definition)
        self.document("query_runtime")
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("implementation::Visible", result.stderr)

    def test_module_alias_out_of_hidden_parent_exposes_descendants(self):
        source = self.sources["query_runtime"]
        alias = '''
#[doc(hidden)]
mod implementation {
    /// Reports exposed through a module alias.
    pub mod reports {
        pub struct Visible;
    }
}
pub use implementation::reports as exposed;
impl private::Report {
    /// Visible application type defined under a hidden ancestor.
    pub fn visible(&self) -> exposed::Visible { exposed::Visible }
}
'''
        source.write_text(source.read_text() + alias)
        self.document("query_runtime")
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("implementation::reports::Visible", result.stderr)

    def test_public_generic_impl_keeps_method_docs_without_private_trait_members(self):
        source = self.sources["query_runtime"]
        generic = '''
pub(crate) trait InternalBound {
    type Hidden;
}
pub struct HiddenBoundType;
/// A concrete parameter accepted by the public generic wrapper.
pub struct Concrete;
impl InternalBound for Concrete { type Hidden = HiddenBoundType; }
/// Generic wrapper with a private implementation bound.
pub struct Generic<T>(T);
impl<T: InternalBound> Generic<T> {
    pub fn count(&self) -> usize { 0 }
}
impl private::Report {
    /// Construct the generic wrapper.
    pub fn generic(&self) -> Generic<Concrete> { Generic(Concrete) }
}
'''
        source.write_text(source.read_text() + generic)
        self.document("query_runtime")
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("Generic::count", result.stderr)
        self.assertNotIn("HiddenBoundType", result.stderr)
        self.assertNotIn("InternalBound", result.stderr)
        source.write_text(source.read_text().replace("pub fn count", "/// Count observed values.\n    pub fn count"))
        self.document("query_runtime")
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_generated_source_exception_does_not_cover_handwritten_owner_items(self):
        bindings = self.directory / "bindings.rs"
        bindings.write_text("pub struct GeneratedRecord { pub value: u64 }\n")
        source = self.sources["query_runtime"]
        source.write_text("mod bindings;\n" + source.read_text().replace("//! Workspace implementation dependency.\n", "")
                          .replace("pub nested: Nested", "pub nested: crate::bindings::GeneratedRecord"))
        self.document("query_runtime")
        policy = POLICY.DocumentationPolicy(POLICY.workspace_libraries(self.metadata), self.directory,
                                             {bindings.resolve()}, self.directory)
        failures = policy.check()
        self.assertEqual(failures, [])
        source.write_text(source.read_text().replace("/// Observation state.", ""))
        self.document("query_runtime")
        policy = POLICY.DocumentationPolicy(POLICY.workspace_libraries(self.metadata), self.directory,
                                             {bindings.resolve()}, self.directory)
        self.assertTrue(any("Report::state" in failure for failure in policy.check()))

    def test_existing_snapshot_gate_rejects_dependency_docs_without_signature_drift(self):
        binaries = self.directory / "bin"
        binaries.mkdir(exist_ok=True)
        cargo = binaries / "cargo"
        cargo.write_text('''#!/usr/bin/env python3
import json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "metadata":
    print(Path(os.environ["DOC_FIXTURE_METADATA"]).read_text())
elif args == ["public-api", "--version"]:
    print("cargo-public-api fixture")
elif args[:2] == [os.environ["DOC_FIXTURE_TOOLCHAIN"], "public-api"]:
    print("pub mod " + args[args.index("-p") + 1].replace("-", "_"))
elif args[:2] == [os.environ["DOC_FIXTURE_TOOLCHAIN"], "doc"]:
    Path(os.environ["DOC_FIXTURE_BUILD_LOG"]).write_text(json.dumps({"args": args, "flags": os.environ["RUSTDOCFLAGS"]}))
    if os.environ.get("DOC_FIXTURE_BUILD_FAIL"):
        sys.exit(1)
    metadata = json.loads(Path(os.environ["DOC_FIXTURE_METADATA"]).read_text())
    destination = Path(args[args.index("--target-dir") + 1]) if "--target-dir" in args else Path(metadata["target_directory"])
    if "--target" in args:
        destination /= args[args.index("--target") + 1]
    destination /= "doc"
    destination.mkdir(parents=True, exist_ok=True)
    for crate in metadata["workspace_members"]:
        shutil.copyfile(Path(os.environ["DOC_FIXTURE_JSON"]) / (crate + ".json"), destination / (crate + ".json"))
else:
    sys.exit("unexpected cargo invocation " + repr(args))
''')
        cargo.chmod(0o755)
        for name, contents in {"cargo-public-api": "#!/bin/sh\nexit 0\n",
                               "rustc": "#!/bin/sh\nprintf 'rustc fixture\\n'\n"}.items():
            path = binaries / name
            path.write_text(contents)
            path.chmod(0o755)
        environment = os.environ.copy()
        environment.update({
            "PATH": f"{binaries}:{environment['PATH']}", "PUBLIC_API_SNAPSHOT_DIR": str(self.directory / "snapshots"),
            "PUBLIC_API_SKIP_NATIVE_COMPONENT_POLICY": "1", "PUBLIC_API_SKIP_FACADE_POLICY": "1",
            "PUBLIC_API_SKIP_HIDDEN_POLICY": "1", "PUBLIC_API_SKIP_DOCUMENTATION_POLICY": "0",
            "DOC_FIXTURE_METADATA": str(self.metadata_path), "DOC_FIXTURE_JSON": str(self.directory),
            "DOC_FIXTURE_BUILD_LOG": str(self.directory / "build.json"), "RUSTDOCFLAGS": "-D warnings",
            "DOC_FIXTURE_TOOLCHAIN": "+" + POLICY.NIGHTLY_TOOLCHAIN,
        })
        script = ROOT / "scripts/ci/check_public_api_surface.sh"
        cargo_args = ["--all-features", "--target-dir", str(self.directory / "custom-target"),
                      "--target", "x86_64-unknown-linux-gnu"]
        result = subprocess.run(["bash", str(script), "write", *cargo_args], env=environment,
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        build = json.loads((self.directory / "build.json").read_text())
        self.assertIn("--all-features", build["args"])
        self.assertIn("-D warnings", build["flags"])
        self.assertNotIn("-A warnings", build["flags"])
        self.assertIn("--document-private-items", build["flags"])
        self.remove_documentation("query_runtime", "/// Observation state.")
        result = subprocess.run(["bash", str(script), "check", *cargo_args], env=environment,
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("public API matches", result.stdout)
        self.assertIn("Report::state", result.stderr)
        environment["DOC_FIXTURE_BUILD_FAIL"] = "1"
        result = subprocess.run(["bash", str(script), "check", *cargo_args], env=environment,
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("complete workspace rustdoc JSON build failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
