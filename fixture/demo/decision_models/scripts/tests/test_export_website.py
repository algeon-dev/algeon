"""The website export must carry no decision backend or model identity."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import export_website  # noqa: E402


class ExportWebsiteTests(unittest.TestCase):
    def test_strip_removes_backend_fields_at_every_depth(self):
        data = {
            "backend": "x", "model_ids": ["m"], "totals": {"api_calls": 1},
            "decisions": [{"model_id": "m", "content_hash": "h", "answers": {"q": {"value": 0.5}}}],
        }
        stripped = export_website.strip_backend_fields(data)
        self.assertEqual(stripped, {"totals": {"api_calls": 1}, "decisions": [{"answers": {"q": {"value": 0.5}}}]})
        export_website.check_no_backend_fields(stripped)

    def test_check_names_the_offending_path(self):
        with self.assertRaises(ValueError) as caught:
            export_website.check_no_backend_fields({"windows": {"all": [{"model_id": "m"}]}})
        self.assertIn("/windows/all[0]", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
