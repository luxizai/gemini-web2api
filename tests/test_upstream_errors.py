"""Upstream error detection and parsing edge cases."""
import json
import unittest

from gemini_web2api.gemini import extract_response_text, _extract_texts_from_line


class UpstreamErrorTests(unittest.TestCase):
    def test_bard_error_formats(self):
        # New JSPB format (2026-08): application.BardErrorInfo",[1060]]
        for code in (1060, 1037, 1013, 1050):
            raw = (
                ')]}\n\n121\n[["wrb.fr",null,null,null,null,[9,null,'
                '[["type.googleapis.com/assistant.boq.bard.application.BardErrorInfo",['
                + str(code) + ']]]]]]'
            )
            with self.assertRaises(RuntimeError) as ctx:
                extract_response_text(raw)
            self.assertIn(f"[{code}]", str(ctx.exception))
        # Old format still detected
        with self.assertRaises(RuntimeError):
            extract_response_text("junk BardErrorInfo [1037] junk")

    def test_short_wrb_line_parsed(self):
        # Regression: lines under 200 chars used to be skipped entirely
        inner = [None, ["c_1", "r_1"], None, None, [["cid", ["hi"]]]]
        line = '[["wrb.fr",null,' + json.dumps(json.dumps(inner)) + ']]'
        texts = _extract_texts_from_line(line)
        self.assertEqual(texts, ["hi"])

    def test_empty_raw(self):
        self.assertEqual(extract_response_text(""), "")


if __name__ == "__main__":
    unittest.main()
