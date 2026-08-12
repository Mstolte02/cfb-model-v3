import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.publish import fingerprint_assets


class PublishFingerprintTests(unittest.TestCase):
    def test_js_and_css_revisions_are_content_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            (site / "app.js").write_text("console.log('v4')")
            (site / "style.css").write_text("body { color: white; }")
            (site / "index.html").write_text(
                '<link rel="stylesheet" href="style.css?v=old">\n'
                '<script src="app.js?v=27"></script>')
            versions = fingerprint_assets(site)
            html = (site / "index.html").read_text()
            for name in ("app.js", "style.css"):
                expected = hashlib.sha256((site / name).read_bytes()).hexdigest()[:12]
                self.assertEqual(versions[name], expected)
                self.assertIn(f"{name}?v={expected}", html)
            self.assertNotIn("v=old", html)
            self.assertNotIn("v=27", html)


if __name__ == "__main__":
    unittest.main()
