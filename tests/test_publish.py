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
            (site / "team-card.js").write_text("window.TeamCard = {}")
            (site / "ranking-history.js").write_text("window.RankingHistory = {}")
            (site / "style.css").write_text("body { color: white; }")
            (site / "index.html").write_text(
                '<link rel="stylesheet" href="style.css?v=old">\n'
                '<script src="ranking-history.js"></script>\n<script src="team-card.js"></script>\n<script src="app.js?v=27"></script>')
            versions = fingerprint_assets(site)
            html = (site / "index.html").read_text()
            expected = {
                "ranking-history.js": hashlib.sha256(b"window.RankingHistory = {}").hexdigest()[:12],
                "app.js": hashlib.sha256(b"console.log('v4')").hexdigest()[:12],
                "team-card.js": hashlib.sha256(b"window.TeamCard = {}").hexdigest()[:12],
                "style.css": hashlib.sha256(b"body { color: white; }").hexdigest()[:12],
            }
            for name, digest in expected.items():
                source = Path(name)
                target = f"{source.stem}.{digest}{source.suffix}"
                self.assertEqual(versions[name], target)
                self.assertIn(target, html)
                self.assertTrue((site / target).is_file())
                self.assertFalse((site / name).exists())
            self.assertNotIn("v=old", html)
            self.assertNotIn("v=27", html)


if __name__ == "__main__":
    unittest.main()
