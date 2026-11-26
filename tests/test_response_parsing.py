"""Regression tests for StreamGenerate response parsing.

These simulate the multi-frame wire format that Gemini Web actually returns:
the main answer frame (carrying conversation/response ids) plus thought
summaries, alternative drafts and follow-up chip frames that do NOT carry
those ids.

The old parser picked the longest text found anywhere in the response, so a
long thought-summary / draft / chip frame won over a short real answer and
the API returned content unrelated to the user's question.
"""
import importlib.util
import json
import os
import unittest
from unittest import mock

from gemini_web2api.config import CONFIG
from gemini_web2api.gemini import (
    HAS_HTTPX,
    StreamTextCleaner,
    clean_text,
    extract_response_text,
    generate_stream,
    _next_reqid,
    _build_headers,
)
from gemini_web2api.models import MODEL_IDS, build_model_header, resolve_model

SINGLE_FILE = os.path.join(os.path.dirname(__file__), "..", "gemini_web2api.py")


def load_single_file():
    spec = importlib.util.spec_from_file_location("gemini_web2api_single", SINGLE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ─── wire-format builders ────────────────────────────────────────────────────

def make_line(*inners):
    """Serialize one StreamGenerate chunk line containing several frames."""
    return json.dumps([["wrb.fr", "f%d" % i, json.dumps(inner)] for i, inner in enumerate(inners)])


def make_raw(*lines):
    body = "".join("%d\n%s\n" % (len(l), l) for l in lines)
    return ")]}'\n\n" + body


def answer_inner(segments, drafts=(), convo="c_abc123", resp="r_def456"):
    """Main answer frame: carries conversation id at [1] and response id at [2]."""
    inner = [None] * 6
    inner[1] = convo
    inner[2] = resp
    candidates = [[None, list(segments)]]
    for draft in drafts:
        candidates.append([None, list(draft)])
    inner[4] = candidates
    return inner


def noise_inner(segments):
    """A frame without conversation/response ids: thought summary, follow-up
    chip, image-agent or search frame."""
    inner = [None] * 6
    inner[4] = [[None, list(segments)]]
    return inner


ANATOMY_NOISE = (
    "The pharynx, larynx, and trachea form a continuous vertical pathway "
    "connecting your nasal cavity and mouth down to your lungs. " * 8
)
SHORT_ANSWER = "I'm not sure what you mean. Could you clarify?"


# ─── non-streaming parser ────────────────────────────────────────────────────

class ExtractResponseTextTests(unittest.TestCase):
    def test_thought_frame_longer_than_answer_is_ignored(self):
        raw = make_raw(
            make_line(noise_inner([ANATOMY_NOISE])),
            make_line(answer_inner([SHORT_ANSWER])),
        )
        self.assertEqual(extract_response_text(raw), SHORT_ANSWER)

    def test_chip_frame_after_answer_is_ignored(self):
        chip = "Would you like a closer look at how the epiglottis prevents choking during swallowing? " * 6
        raw = make_raw(
            make_line(answer_inner([SHORT_ANSWER])),
            make_line(noise_inner([chip])),
        )
        self.assertEqual(extract_response_text(raw), SHORT_ANSWER)

    def test_multi_segment_answer_is_joined(self):
        raw = make_raw(make_line(answer_inner(["你好，", "我是一段", "被切開的答案", "請重新組合"])))
        self.assertEqual(extract_response_text(raw), "你好，我是一段被切開的答案請重新組合")

    def test_longer_alternative_draft_does_not_override_primary(self):
        draft = "This alternative draft is much longer than the primary answer " * 5
        raw = make_raw(make_line(answer_inner([SHORT_ANSWER], drafts=[(draft,)])))
        self.assertEqual(extract_response_text(raw), SHORT_ANSWER)

    def test_progressive_updates_return_final_text(self):
        growing = ["Hel", "Hello", "Hello, world!"]
        raw = make_raw(*[make_line(answer_inner([t])) for t in growing])
        self.assertEqual(extract_response_text(raw), "Hello, world!")

    def test_fallback_when_no_frame_carries_ids(self):
        # Protocol drift: no answer frame ids anywhere. Longest joined
        # candidate is returned so the API stays functional.
        raw = make_raw(
            make_line(noise_inner(["short"])),
            make_line(noise_inner([SHORT_ANSWER])),
        )
        self.assertEqual(extract_response_text(raw), SHORT_ANSWER)

    def test_multi_frame_line_all_frames_parsed(self):
        # A single line carrying both noise and the answer frame.
        raw = make_raw(make_line(noise_inner([ANATOMY_NOISE]), answer_inner([SHORT_ANSWER])))
        self.assertEqual(extract_response_text(raw), SHORT_ANSWER)

    def test_bard_error_raises(self):
        with self.assertRaises(RuntimeError):
            extract_response_text(")]}'\n\n[3,[\"er\",null,\"BardErrorInfo [1037]\"]]\n")

    def test_garbage_returns_empty(self):
        self.assertEqual(extract_response_text(")]}'\n\n12\n[\"e\",4]\n"), "")


class SingleFileParityTests(unittest.TestCase):
    """The single-file build must behave identically."""

    @classmethod
    def setUpClass(cls):
        cls.single = load_single_file()

    def test_thought_frame_longer_than_answer_is_ignored(self):
        raw = make_raw(
            make_line(noise_inner([ANATOMY_NOISE])),
            make_line(answer_inner([SHORT_ANSWER])),
        )
        self.assertEqual(self.single.extract_response_text(raw), SHORT_ANSWER)

    def test_multi_segment_answer_is_joined(self):
        raw = make_raw(make_line(answer_inner(["a ", "b", "c"])))
        self.assertEqual(self.single.extract_response_text(raw), "a bc")

    def test_longer_alternative_draft_does_not_override_primary(self):
        draft = "This alternative draft is much longer than the primary answer " * 5
        raw = make_raw(make_line(answer_inner([SHORT_ANSWER], drafts=[(draft,)])))
        self.assertEqual(self.single.extract_response_text(raw), SHORT_ANSWER)

    def test_single_file_build_model_header_parity(self):
        for name in ("gemini-4-hydrogen", "gemini-4-hydrogen-thinking", "gemini-3.8-flash"):
            header_modular = build_model_header(name, 1)
            header_single = self.single.build_model_header(name, 1)
            parsed_mod = json.loads(header_modular)
            parsed_sin = json.loads(header_single)
            self.assertEqual(parsed_mod[:16], parsed_sin[:16])
        self.assertIsNone(self.single.build_model_header("gemini-auto", 1))

    def test_single_file_ui_tag_cleaning_parity(self):
        sample = (
            '排查步骤\n\n'
            '<Steps> <Step subtitle="终端操作" title="结束卡住的 Git"> 输入 `n` 并回车。 </Step> '
            '<Step title="重新拉取"> 运行 `git pull`。 </Step> </Steps>\n\n'
            '<ElicitationsGroup message="继续排查："> '
            '<Elicitation label="macOS 或 Linux" query="macOS 排查" /> </ElicitationsGroup>'
        )
        self.assertEqual(self.single.clean_gemini_text(sample), clean_text(sample))


# ─── model routing header ───────────────────────────────────────────────────

class ModelRoutingHeaderTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def test_known_models_generate_valid_header_json(self):
        for name in [
            "gemini-4-hydrogen",
            "gemini-4-hydrogen-thinking",
            "gemini-hydrogen",
            "gemini-3.8-flash",
            "gemini-3.1-pro",
            "gemini-flash-lite",
        ]:
            header_str = build_model_header(name, 1)
            self.assertIsNotNone(header_str)
            parsed = json.loads(header_str)
            self.assertEqual(parsed[0], 1)
            self.assertEqual(parsed[4], MODEL_IDS[name])
            self.assertEqual(parsed[14], 1)
            self.assertEqual(parsed[15], 0)
            self.assertTrue(len(parsed[16]) > 0)

    def test_model_tickets_config_override(self):
        CONFIG["model_tickets"] = {
            "gemini-4-hydrogen": "deadbeef12345678",
            "gemini-3.8-flash": None,
        }
        parsed = json.loads(build_model_header("gemini-4-hydrogen", 1))
        self.assertEqual(parsed[4], "deadbeef12345678")
        self.assertIsNone(build_model_header("gemini-3.8-flash", 1))

    def test_resolve_gemini_4_hydrogen_variants(self):
        name, mode, think, err, extra = resolve_model("gemini-4-hydrogen")
        self.assertEqual((name, mode, think, err, extra), ("gemini-4-hydrogen", 1, 4, None, None))
        name, mode, think, err, _ = resolve_model("gemini-4-hydrogen-thinking")
        self.assertEqual((name, mode, think, err), ("gemini-4-hydrogen-thinking", 2, 0, None))
        name, mode, think, err, _ = resolve_model("google/gemini-4-hydrogen@think=2")
        self.assertEqual((name, mode, think, err), ("gemini-4-hydrogen", 1, 2, None))
        # Unknown models fall back to configured default (gemini-4-hydrogen)
        name, mode, think, err, _ = resolve_model("non-existent-model")
        self.assertEqual((name, mode, think, err), ("gemini-4-hydrogen", 1, 4, None))

    def test_auto_and_unknown_models_return_none(self):
        self.assertIsNone(build_model_header("gemini-auto", 1))
        self.assertIsNone(build_model_header("non-existent-model", 1))

    def test_build_headers_injects_model_header_when_present(self):
        headers = _build_headers(model_name="gemini-4-hydrogen", model_id=1)
        self.assertIn("x-goog-ext-525001261-jspb", headers)
        headers_auto = _build_headers(model_name="gemini-auto", model_id=1)
        self.assertNotIn("x-goog-ext-525001261-jspb", headers_auto)


# ─── Gemini UI tag & artifact formatting ────────────────────────────────────

SCREENSHOT_SAMPLE = (
    "排查与解决步骤\n\n"
    '<Steps> <Step subtitle="终端操作" title="先结束当前卡住的 Git 操作"> '
    "在当前提示 `Should I try again? (y/n)` 的终端中输入 `n` 并回车。 "
    "</Step> "
    '<Step subtitle="Windows / 编辑器" title="释放可能占用 .git 文件的进程"> '
    "关闭 IDE 和后台 `git.exe` 进程。 "
    "</Step> </Steps>\n\n"
    "补充处理：重启系统。\n\n"
    '<ElicitationsGroup message="如果问题仍未解决，可以从以下方向继续排查："> '
    '<Elicitation label="我使用的是 macOS 或 Linux" query="macOS 排查" /> '
    '<Elicitation label="提示 Permission denied" query="拒绝访问排查" /> '
    "</ElicitationsGroup>"
)


class UiTagCleaningTests(unittest.TestCase):
    def test_steps_converted_and_elicitations_stripped_by_default(self):
        cleaned = clean_text(SCREENSHOT_SAMPLE)
        expected = (
            "排查与解决步骤\n\n"
            "#### 1. 先结束当前卡住的 Git 操作 — 终端操作\n\n"
            "在当前提示 `Should I try again? (y/n)` 的终端中输入 `n` 并回车。\n\n"
            "#### 2. 释放可能占用 .git 文件的进程 — Windows / 编辑器\n\n"
            "关闭 IDE 和后台 `git.exe` 进程。\n\n"
            "补充处理：重启系统。"
        )
        self.assertEqual(cleaned, expected)

    def test_elicitations_markdown_mode(self):
        cleaned = clean_text(SCREENSHOT_SAMPLE, elicitation_mode="markdown")
        self.assertIn("---\n\n**如果问题仍未解决，可以从以下方向继续排查：**", cleaned)
        self.assertIn("- 我使用的是 macOS 或 Linux", cleaned)
        self.assertIn("- 提示 Permission denied", cleaned)
        self.assertNotIn("<ElicitationsGroup", cleaned)

    def test_followup_tag_strip_and_markdown(self):
        raw = '完成。<FollowUp label="查看详情" query="给我详细说明" />'
        self.assertEqual(clean_text(raw), "完成。")
        self.assertEqual(clean_text(raw, elicitation_mode="markdown"), "完成。\n\n- 查看详情")

    def test_code_blocks_and_inline_code_preserve_raw_tags(self):
        raw = (
            "不要直接输出 `<Steps>` 和 `<ElicitationsGroup>`。\n\n"
            "```jsx\n"
            '<Steps>\n  <Step title="Demo">keep me</Step>\n</Steps>\n'
            '<ElicitationsGroup message="keep"><Elicitation label="A" /></ElicitationsGroup>\n'
            "```\n\n"
            '<Steps><Step title="Real">convert me</Step></Steps>'
        )
        cleaned = clean_text(raw)
        self.assertIn("不要直接输出 `<Steps>` 和 `<ElicitationsGroup>`。", cleaned)
        self.assertIn('<Step title="Demo">keep me</Step>', cleaned)
        self.assertIn('<ElicitationsGroup message="keep">', cleaned)
        self.assertIn("#### 1. Real\n\nconvert me", cleaned)

    def test_html_entities_and_gt_inside_quoted_attr(self):
        raw = '<Steps><Step title="Compare a &gt; b &amp; c" subtitle="x > y">Body</Step></Steps>'
        self.assertEqual(clean_text(raw), "#### 1. Compare a > b & c — x > y\n\nBody")

    def test_code_exec_and_card_content_stripped(self):
        raw = (
            "Start\n"
            "```python?code_reference&code_event_index=0\nprint(1)\n```\n"
            "http://googleusercontent.com/card_content/0\n"
            "End"
        )
        self.assertEqual(clean_text(raw), "Start\nEnd")

    def test_clean_ui_tags_disabled_leaves_xml_untouched(self):
        raw = '<Steps><Step title="One">Body</Step></Steps>'
        self.assertEqual(clean_text(raw, clean_ui_tags=False), raw)

    def test_streaming_char_by_char_matches_non_streaming(self):
        expected = clean_text(SCREENSHOT_SAMPLE, strip=False)
        cleaner = StreamTextCleaner()
        streamed = "".join(cleaner.feed(ch) for ch in SCREENSHOT_SAMPLE) + cleaner.flush()
        self.assertEqual(streamed, expected)

        # Also verify markdown mode and code blocks split char-by-char
        md_expected = clean_text(SCREENSHOT_SAMPLE, strip=False, elicitation_mode="markdown")
        cleaner_md = StreamTextCleaner(elicitation_mode="markdown")
        md_streamed = "".join(cleaner_md.feed(ch) for ch in SCREENSHOT_SAMPLE) + cleaner_md.flush()
        self.assertEqual(md_streamed, md_expected)

    def test_truncated_elicitation_block_dropped_in_strip_mode(self):
        cleaner = StreamTextCleaner()
        out = cleaner.feed('正文结束。\n\n<ElicitationsGroup message="继续"> <Elicitation label="A"')
        out += cleaner.flush()
        self.assertEqual(out, "正文结束。")



# ─── request ids ─────────────────────────────────────────────────────────────

class ReqidTests(unittest.TestCase):
    def test_reqids_unique_and_increasing_even_within_same_second(self):
        ids = [_next_reqid() for _ in range(50)]
        self.assertEqual(len(set(ids)), 50)
        self.assertEqual(ids, sorted(ids))


# ─── streaming ───────────────────────────────────────────────────────────────

class FakeStreamResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def raise_for_status(self):
        pass

    def iter_text(self):
        yield from self._chunks


class FakeStreamContext:
    def __init__(self, chunks):
        self._chunks = chunks

    def __enter__(self):
        return FakeStreamResponse(self._chunks)

    def __exit__(self, *args):
        return False


class FakeHttpClient:
    def __init__(self, chunk_iterables):
        self._chunk_iterables = chunk_iterables
        self.calls = 0

    def stream(self, method, url, content=None, headers=None):
        chunks = self._chunk_iterables[min(self.calls, len(self._chunk_iterables) - 1)]
        self.calls += 1
        return FakeStreamContext(chunks)


@unittest.skipUnless(HAS_HTTPX, "httpx not installed")
class GenerateStreamTests(unittest.TestCase):
    def setUp(self):
        self.original_config = dict(CONFIG)
        CONFIG["retry_attempts"] = 1
        CONFIG["retry_delay_sec"] = 0
        CONFIG["log_requests"] = False

    def tearDown(self):
        CONFIG.clear()
        CONFIG.update(self.original_config)

    def _run(self, client):
        with mock.patch("gemini_web2api.gemini._get_httpx_client", return_value=client):
            return "".join(generate_stream("prompt", 1, 4))

    def test_stream_ignores_thought_frames_and_emits_answer_deltas(self):
        chunks = [
            ")]}'\n\n",
            make_line(noise_inner([ANATOMY_NOISE])) + "\n",
            make_line(answer_inner(["Hello"])) + "\n",
            make_line(answer_inner(["Hello, wor"])) + "\n",
            make_line(answer_inner(["Hello, world!"])) + "\n",
        ]
        self.assertEqual(self._run(FakeHttpClient([chunks])), "Hello, world!")

    def test_stream_tolerates_answer_rewrite_without_splicing(self):
        chunks = [
            make_line(answer_inner(["Original answer, already emitted."])) + "\n",
            make_line(answer_inner(["A completely different replacement text."])) + "\n",
        ]
        out = self._run(FakeHttpClient([chunks]))
        self.assertEqual(out, "Original answer, already emitted.")

    def test_stream_falls_back_when_no_answer_frame_ids(self):
        chunks = [make_line(noise_inner([SHORT_ANSWER])) + "\n"]
        self.assertEqual(self._run(FakeHttpClient([chunks])), SHORT_ANSWER)

    def test_stream_retries_when_first_attempt_is_empty(self):
        CONFIG["retry_attempts"] = 2
        empty = ["[\"e\",4]\n"]
        good = [make_line(answer_inner(["Recovered on retry"])) + "\n"]
        client = FakeHttpClient([empty, good])
        with mock.patch("gemini_web2api.gemini._get_httpx_client", return_value=client):
            out = "".join(generate_stream("prompt", 1, 4))
        self.assertEqual(out, "Recovered on retry")
        self.assertEqual(client.calls, 2)

    def test_stream_raises_after_all_attempts_empty(self):
        with self.assertRaises(RuntimeError):
            self._run(FakeHttpClient([["[\"e\",4]\n"]]))

    def test_stream_cleans_split_ui_tags_end_to_end(self):
        # Progressive StreamGenerate updates cutting right through <Step> and <ElicitationsGroup>
        snapshots = [
            "步骤如下：\n\n<Ste",
            '步骤如下：\n\n<Steps> <Step subtitle="终端" title="第一',
            '步骤如下：\n\n<Steps> <Step subtitle="终端" title="第一步"> 运行 `git ',
            '步骤如下：\n\n<Steps> <Step subtitle="终端" title="第一步"> 运行 `git pull`。 </St',
            '步骤如下：\n\n<Steps> <Step subtitle="终端" title="第一步"> 运行 `git pull`。 </Step> </Steps>\n\n<ElicitationsGroup message="追问">',
            '步骤如下：\n\n<Steps> <Step subtitle="终端" title="第一步"> 运行 `git pull`。 </Step> </Steps>\n\n<ElicitationsGroup message="追问"><Elicitation label="A" query="B" /></ElicitationsGroup>',
        ]
        chunks = [")]}\'\n\n"] + [make_line(answer_inner([s])) + "\n" for s in snapshots]
        out = self._run(FakeHttpClient([chunks]))
        self.assertEqual(out, "步骤如下：\n\n#### 1. 第一步 — 终端\n\n运行 `git pull`。")


if __name__ == "__main__":
    unittest.main()
