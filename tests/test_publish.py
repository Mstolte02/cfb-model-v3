import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.publish import fingerprint_assets


class PublishFingerprintTests(unittest.TestCase):
    def test_js_and_css_are_renamed_with_content_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp)
            (site / "app.js").write_text("console.log('v4')")
            (site / "style.css").write_text("body { color: white; }")
            (site / "index.html").write_text(
                '<link rel="stylesheet" href="style.css?v=old">\n'
                '<script src="app.js?v=27"></script>')
            versions = fingerprint_assets(site)
            html = (site / "index.html").read_text()
            app_digest = hashlib.sha256(b"console.log('v4')").hexdigest()[:12]
            self.assertEqual(versions["app.js"], f"inline:{app_digest}")
            self.assertIn("<script>\nconsole.log('v4')\n</script>", html)
            self.assertFalse((site / "app.js").exists())

            css_digest = hashlib.sha256(b"body { color: white; }").hexdigest()[:12]
            css_target = f"style.{css_digest}.css"
            self.assertEqual(versions["style.css"], css_target)
            self.assertIn(css_target, html)
            self.assertTrue((site / css_target).is_file())
            self.assertFalse((site / "style.css").exists())
            self.assertNotIn("v=old", html)
            self.assertNotIn("v=27", html)


if __name__ == "__main__":
    unittest.main()
