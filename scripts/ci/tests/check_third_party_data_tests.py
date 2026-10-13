#!/usr/bin/env python3
"""Exercise the third-party data policy in temporary repositories."""

import json
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "check_third_party_data.py"
LICENSE = "CC-BY-NC-4.0; derived from the MerRec dataset"


class ThirdPartyDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="algeon-third-party-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.data = self.root / "website/static/data"
        self.write("NOTICE", "- MerRec (website/static/data/merrec-*.json): CC BY-NC 4.0.\n")
        self.write("website/static/data/LICENSE-MERREC.md", "Files covered:\n\n- `merrec-sample.json`\n")
        self.write("website/static/data/merrec-sample.json", json.dumps({"license": LICENSE, "rows": 1}))
        self.write("docker/Dockerfile", "FROM scratch\nCOPY dist/bin/algeon_server /app/algeon_server\n")

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        subprocess.run(["git", "-C", str(self.root), "add", name], check=True)

    def check(self):
        return subprocess.run(["python3", str(SCRIPT), str(self.root)], text=True, capture_output=True)

    def test_declared_capture_passes(self):
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_license_field_fails(self):
        self.write("website/static/data/merrec-sample.json", json.dumps({"rows": 1}))
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn('top-level "license" must start with CC-BY-NC-4.0', result.stderr)

    def test_capture_outside_data_directory_fails(self):
        self.write("crates/demo/merrec-copy.json", json.dumps({"license": LICENSE}))
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("derived MerRec data must live under website/static/data", result.stderr)

    def test_license_file_list_must_match_tracked_captures(self):
        self.write("website/static/data/merrec-extra.json", json.dumps({"license": LICENSE}))
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("LICENSE-MERREC.md: missing merrec-extra.json", result.stderr)
        self.write("website/static/data/LICENSE-MERREC.md", "- `merrec-sample.json`\n- `merrec-extra.json`\n- `merrec-gone.json`\n")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("lists untracked merrec-gone.json", result.stderr)

    def test_packaging_reference_to_website_data_fails(self):
        self.write("docker/stage.sh", "cp -r website/static/data dist/\n")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("docker/stage.sh: packaging must not reference website/static", result.stderr)

    def test_notice_must_declare_the_glob(self):
        self.write("NOTICE", "Algeon\n")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("NOTICE: must declare website/static/data/merrec-*.json", result.stderr)


if __name__ == "__main__":
    unittest.main()
