#!/usr/bin/env python3
"""Exercise indexed and working-tree branding checks in temporary repositories."""

from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "check_single_brand.sh"
LEGACY = "nex" + "us"


class SingleBrandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="algeon-brand-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.file = self.root / "source.rs"

    def stage(self, content):
        self.file.write_text(content)
        subprocess.run(["git", "-C", str(self.root), "add", "source.rs"], check=True)

    def check(self):
        return subprocess.run(["bash", str(SCRIPT), str(self.root)], text=True, capture_output=True)

    def test_clean_index_does_not_hide_dirty_brand_violation(self):
        self.stage("Algeon\n")
        self.file.write_text(LEGACY + "\n")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("worktree content contains the legacy brand", result.stderr)

    def test_indexed_violation_is_checked_even_after_deletion(self):
        self.stage(LEGACY + "\n")
        self.file.unlink()
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("tracked content contains the legacy brand", result.stderr)

    def test_pending_deletion_of_clean_file_is_accepted(self):
        self.stage("Algeon\n")
        self.file.unlink()
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_dirty_symlink_target_is_checked(self):
        self.file.symlink_to("algeon-target")
        subprocess.run(["git", "-C", str(self.root), "add", "source.rs"], check=True)
        self.file.unlink()
        self.file.symlink_to(LEGACY + "-target")
        result = self.check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("tracked symlink target contains the legacy brand", result.stderr)


if __name__ == "__main__":
    unittest.main()
