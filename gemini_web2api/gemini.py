"""Gemini StreamGenerate protocol implementation with httpx streaming."""
import json
import time
import uuid
import re
import html
import random
import threading
import urllib.request
import urllib.parse
import urllib.error
import ssl
import os
import hashlib
from typing import Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .config import CONFIG
from .models import build_model_header

_ssl_ctx = None
_cookie_cache = {"str": "", "sapisid": None, "mtime": 0}
_httpx_client = None

# Browser-like per-session RPC counter. The real Gemini web app sends an
# ever-increasing _reqid (+100000 per RPC) for every request in a page
# session. Deriving it from a timestamp made concurrent requests (e.g. a
# cronjob firing while the user is chatting) send identical ids.
_REQID_LOCK = threading.Lock()
_REQID_NEXT = random.randrange(10000, 99999)


def _next_reqid() -> int:
    global _REQID_NEXT
    with _REQID_LOCK:
        _REQID_NEXT += 100000
        return _REQID_NEXT


def log(msg: str):
    if CONFIG["log_requests"]:
        import sys
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()


def _get_ssl_ctx():
    global _ssl_ctx
    if _ssl_ctx is None:
        _ssl_ctx = ssl.create_default_context()
    return _ssl_ctx


def _get_httpx_client():
    global _httpx_client
    if _httpx_client is None and HAS_HTTPX:
        proxy = CONFIG.get("proxy")
        transport = httpx.HTTPTransport(proxy=proxy) if proxy else None
        # Connection: close disables keep-alive reuse — the proxy/upstream kills
        # idle connections, and reusing a dead one fails mid-stream with
        # "[SSL] record layer failure", which corrupts streaming responses.
        _httpx_client = httpx.Client(transport=transport, timeout=CONFIG["request_timeout_sec"], verify=True,
                                     headers={"Connection": "close"})
    return _httpx_client


def load_cookie() -> tuple:
    """Load cookie from file with mtime-based caching."""
    cookie_file = CONFIG.get("cookie_file")
    if not cookie_file or not os.path.exists(cookie_file):
        return "", None
    try:
        st = os.stat(cookie_file)
        if (st.st_mtime, st.st_size) == (_cookie_cache["mtime"], _cookie_cache.get("size", -1)) and _cookie_cache["str"]:
            return _cookie_cache["str"], _cookie_cache["sapisid"]
        with open(cookie_file, "r") as f:
            content = f.read().strip()
        if content.startswith("{"):
            data = json.loads(content)
            cookie_str = data.get("cookie", "")
            sapisid = data.get("sapisid", "")
        elif "# Netscape HTTP Cookie File" in content or content.startswith("#HttpOnly_"):
            # Netscape cookies.txt (tab-separated: domain, flag, path, secure, expiry, name, value)
            pairs = {}
            for line in content.splitlines():
                if line.startswith("#HttpOnly_"):
                    line = line[len("#HttpOnly_"):]
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 7:
                    continue
                pairs[parts[5]] = parts[6]
            cookie_str = "; ".join(f"{k}={v}" for k, v in pairs.items())
            sapisid = pairs.get("SAPISID", "")
        else:
            cookie_str = content
            pairs = dict(p.split("=", 1) for p in cookie_str.split("; ") if "=" in p)
            sapisid = pairs.get("SAPISID", "")
        _cookie_cache.update({"str": cookie_str, "sapisid": sapisid or None, "mtime": st.st_mtime, "size": st.st_size})
        return cookie_str, sapisid if sapisid else None
    except Exception as e:
        log(f"Cookie load error: {e}")
        return _cookie_cache["str"], _cookie_cache["sapisid"]


def refresh_bl_and_xsrf() -> bool:
    """Fetch the app page with cookies; refresh gemini_bl and xsrf_token (SNlM0e).
    Returns True if either value changed."""
    old_bl = CONFIG["gemini_bl"]
    old_xsrf = CONFIG.get("xsrf_token")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        cookie_str, _ = load_cookie()
        if cookie_str:
            headers["Cookie"] = cookie_str
        req = urllib.request.Request("https://gemini.google.com/app", headers=headers)
        proxy = CONFIG.get("proxy")
        ctx = _get_ssl_ctx()
        if proxy:
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                urllib.request.HTTPSHandler(context=ctx))
            resp = opener.open(req, timeout=15)
        else:
            resp = urllib.request.urlopen(req, context=ctx, timeout=15)
        html = resp.read().decode("utf-8", errors="replace")
        m = re.search(r'"SNlM0e":"([^"]+)"', html)
        if m:
            CONFIG["xsrf_token"] = m.group(1)
        m = re.search(r'(boq_assistant-bard-web-server_\d+\.\d+_p\d+)', html)
        if m:
            CONFIG["gemini_bl"] = m.group(1)
    except Exception as e:
        log(f"BL/XSRF refresh failed: {e}")
    changed = CONFIG["gemini_bl"] != old_bl or CONFIG.get("xsrf_token") != old_xsrf
    if changed:
        log(f"BL/XSRF refreshed: xsrf {'new' if CONFIG.get('xsrf_token') != old_xsrf else 'unchanged'}, bl {old_bl} -> {CONFIG['gemini_bl']}")
    return changed


def make_sapisidhash(sapisid: str) -> str:
    ts = int(time.time())
    h = hashlib.sha1(f"{ts} {sapisid} https://gemini.google.com".encode()).hexdigest()
    return f"SAPISIDHASH {ts}_{h}"


def _account_prefix() -> str:
    """Return the Gemini account path prefix for non-default Google accounts."""
    auth_user = CONFIG.get("auth_user")
    if auth_user is None or auth_user == "":
        return ""
    return f"/u/{auth_user}"


def _build_headers(model_name: str = None, model_id: int = None) -> dict:
    account_prefix = _account_prefix()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{account_prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if account_prefix:
        headers["X-Goog-AuthUser"] = str(CONFIG["auth_user"])
    cookie_str, sapisid = load_cookie()
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)
    if model_name and model_id is not None:
        hdr = build_model_header(model_name, model_id)
        if hdr:
            headers["x-goog-ext-525001261-jspb"] = hdr
    return headers


def _apply_chat_persistence_flags(inner: list) -> None:
    """Apply Gemini Web persistence flags to an outgoing request payload."""
    if CONFIG.get("temporary_chats", False):
        # Match Gemini Web temporary-chat requests.
        inner[41] = [1]
        inner[45] = 1
    else:
        inner[41] = [2]


def _build_payload(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None) -> str:
    inner = [None] * 102
    if file_refs:
        refs = [[None, None, ref] for ref in file_refs]
        inner[0] = [prompt, 0, None, refs, None, None, 0]
    else:
        inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    _apply_chat_persistence_flags(inner)
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id
    if extra_fields:
        for k, v in extra_fields.items():
            inner[k] = v
    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    if CONFIG.get("xsrf_token"):
        params["at"] = CONFIG["xsrf_token"]
    return urllib.parse.urlencode(params)


def _get_url() -> str:
    account_prefix = _account_prefix()
    return (
        f"https://gemini.google.com{account_prefix}/_/BardChatUi/data/"
        "assistant.lamda.BardFrontendService/StreamGenerate"
        f"?bl={CONFIG['gemini_bl']}&hl=en&_reqid={_next_reqid()}&rt=c"
    )


_UI_TAG_NAMES = ("Steps", "Step", "ElicitationsGroup", "Elicitation", "FollowUp")
_CARD_URL_PREFIXES = (
    "http://googleusercontent.com/card_content/",
    "https://googleusercontent.com/card_content/",
)
_CODE_EXEC_INFO_RE = re.compile(
    r"^(?:python|javascript|text)\?code_(?:reference|stdout)&code_event_index=\d+\s*$"
)
_CODE_EXEC_LANG_PREFIXES = ("python?code_", "javascript?code_", "text?code_")
_ATTR_RE = re.compile(r'([A-Za-z_][\w:-]*)\s*=\s*(?:"([^"]*)"|\'([^\']*)\')')
_PROSE_SPECIAL_RE = re.compile(r"[ \t\r\n<`~h]")
_PROSE_SPECIAL_NO_UI_RE = re.compile(r"[ \t\r\n`~h]")
_MAX_UI_TAG_LEN = 2048
_MAX_UI_BLOCK_LEN = 32768


def _match_ui_tag_prefix(s: str) -> str:
    """Return 'match', 'need_more', or 'no' for a string starting with '<'."""
    pos = 2 if s.startswith("</") else 1
    if pos == len(s):
        return "need_more"
    j = pos
    while j < len(s) and s[j].isascii() and s[j].isalpha():
        j += 1
    name = s[pos:j]
    if j == len(s):
        if name and any(t.startswith(name) for t in _UI_TAG_NAMES):
            return "need_more"
        return "no"
    if name in _UI_TAG_NAMES and s[j] in (" ", "\t", "\r", "\n", ">", "/"):
        return "match"
    return "no"


def _find_tag_end(s: str) -> int:
    """Find closing '>' of an XML tag starting at index 0, ignoring quoted '>'."""
    quote = None
    for pos in range(1, len(s)):
        ch = s[pos]
        if quote:
            if ch == quote:
                quote = None
        elif ch in ('"', "'"):
            quote = ch
        elif ch == ">":
            return pos
    return -1


def _parse_ui_tag(tag_str: str) -> tuple:
    """Parse '<Name attr="val" ...>' into (name, is_closing, is_self_closing, attrs)."""
    inner = tag_str[1:-1].strip()
    if inner.startswith("/"):
        return inner[1:].strip(), True, False, {}
    is_self_closing = inner.endswith("/")
    if is_self_closing:
        inner = inner[:-1].rstrip()
    m = re.match(r"^([A-Za-z]+)(.*)$", inner, re.DOTALL)
    name = m.group(1) if m else inner
    attr_str = m.group(2) if m else ""
    attrs = {
        k: html.unescape(v1 if v1 is not None else v2)
        for k, v1, v2 in _ATTR_RE.findall(attr_str)
    }
    return name, False, is_self_closing, attrs


def _is_closing_fence(line: str, fence_char: str, fence_len: int) -> bool:
    stripped = line.rstrip(" \t\r")
    indent = len(stripped) - len(stripped.lstrip(" "))
    if indent > 3:
        return False
    rest = stripped[indent:]
    return len(rest) >= fence_len and all(c == fence_char for c in rest)


def _format_elicitation_block(message: str, inner: str) -> str:
    """Convert an <ElicitationsGroup> body into a Markdown suggestion list."""
    items = []
    pos = 0
    while pos < len(inner):
        idx = inner.find("<Elicitation", pos)
        if idx == -1:
            break
        after = idx + len("<Elicitation")
        if after >= len(inner) or inner[after] not in (" ", "\t", "\r", "\n", ">", "/"):
            pos = after
            continue
        end = _find_tag_end(inner[idx:])
        if end == -1:
            break
        _, is_closing, _, attrs = _parse_ui_tag(inner[idx : idx + end + 1])
        if not is_closing:
            item = (attrs.get("label") or attrs.get("query") or "").strip()
            if item:
                items.append(item)
        pos = idx + end + 1
    lines = []
    if message:
        lines.append(f"**{message}**")
    for item in items:
        lines.append(f"- {item}")
    if not lines:
        return ""
    return "---\n\n" + "\n".join(lines)


class StreamTextCleaner:
    """Incremental cleaner for Gemini Web response text.

    Strips internal code-execution blocks and card_content URLs, converts
    Gemini Web <Steps>/<Step> UI widgets into clean Markdown headings, and
    strips (or converts to Markdown) <ElicitationsGroup>/<Elicitation>/<FollowUp>
    chips — safely across arbitrary SSE chunk boundaries while leaving fenced
    and inline code blocks untouched.
    """

    def __init__(self, clean_ui_tags: bool = None, elicitation_mode: str = None):
        if clean_ui_tags is None:
            clean_ui_tags = CONFIG.get("clean_ui_tags", True)
        if elicitation_mode is None:
            elicitation_mode = CONFIG.get("elicitation_mode", "strip")
        self.clean_ui_tags = bool(clean_ui_tags)
        self.elicitation_mode = str(elicitation_mode or "strip").lower()
        self._buf = ""
        self._pending_ws = ""
        self._skip_ws = False
        self._need_block_sep = False
        self._has_output = False
        self._trailing_newlines = 0
        self._in_code_fence = False
        self._suppress_fence = False
        self._fence_char = ""
        self._fence_len = 0
        self._fence_line_buf = ""
        self._inline_ticks = 0
        self._step_index = 0
        self._block_tag = None
        self._block_attrs = {}
        self._block_buf = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        return self._process(final=False)

    def flush(self) -> str:
        return self._process(final=True)

    def _append_out(self, out: list, text: str) -> None:
        if not text:
            return
        out.append(text)
        self._has_output = True
        nl = 0
        all_nl = True
        for ch in reversed(text):
            if ch == "\n":
                nl += 1
            elif ch == "\r":
                continue
            else:
                all_nl = False
                break
        self._trailing_newlines = min(
            2, (self._trailing_newlines + nl) if all_nl else nl
        )

    def _emit_block_sep(self, out: list) -> None:
        if self._has_output:
            need = 2 - self._trailing_newlines
            if need > 0:
                self._append_out(out, "\n" * need)

    def _flush_ws_before_visible(self, out: list) -> None:
        if self._need_block_sep:
            self._emit_block_sep(out)
            self._need_block_sep = False
            self._pending_ws = ""
            self._skip_ws = False
        elif self._pending_ws:
            self._append_out(out, self._pending_ws)
            self._pending_ws = ""
        self._skip_ws = False

    def _at_fence_start(self) -> bool:
        if self._need_block_sep:
            return True
        if "\n" in self._pending_ws:
            tail = self._pending_ws.rsplit("\n", 1)[1]
            return len(tail) <= 3 and all(c == " " for c in tail)
        if not self._has_output or self._trailing_newlines >= 1:
            return len(self._pending_ws) <= 3 and all(c == " " for c in self._pending_ws)
        return False

    def _finish_block(self, out: list, tag: str, attrs: dict, inner: str) -> None:
        if self.elicitation_mode == "markdown":
            if tag == "ElicitationsGroup":
                md = _format_elicitation_block((attrs.get("message") or "").strip(), inner)
            else:
                label = (attrs.get("label") or attrs.get("query") or inner).strip()
                md = f"- {label}" if label else ""
            if md:
                self._pending_ws = ""
                self._emit_block_sep(out)
                self._append_out(out, md)
                self._skip_ws = True
                self._need_block_sep = True
                return
        self._pending_ws = ""
        self._skip_ws = True
        if self._has_output:
            self._need_block_sep = True

    def _process(self, final: bool) -> str:
        out = []
        special_re = _PROSE_SPECIAL_RE if self.clean_ui_tags else _PROSE_SPECIAL_NO_UI_RE

        while self._buf or (final and self._block_tag):
            # 1. Buffering inside <ElicitationsGroup>, <Elicitation>, or <FollowUp>
            if self._block_tag:
                if self._buf:
                    self._block_buf += self._buf
                    self._buf = ""
                m = re.search(rf"</{self._block_tag}\s*>", self._block_buf)
                if m:
                    inner = self._block_buf[: m.start()]
                    rest = self._block_buf[m.end() :]
                    tag, attrs = self._block_tag, self._block_attrs
                    self._block_tag = None
                    self._block_attrs = {}
                    self._block_buf = ""
                    self._buf = rest
                    self._finish_block(out, tag, attrs, inner)
                    continue
                if final or len(self._block_buf) > _MAX_UI_BLOCK_LEN:
                    tag, attrs, inner = self._block_tag, self._block_attrs, self._block_buf
                    self._block_tag = None
                    self._block_attrs = {}
                    self._block_buf = ""
                    self._finish_block(out, tag, attrs, inner)
                    continue
                break

            # 2. Inside a fenced code block
            if self._in_code_fence:
                nl = self._buf.find("\n")
                if self._suppress_fence:
                    if nl != -1:
                        line = self._buf[:nl]
                        self._buf = self._buf[nl + 1 :]
                        if _is_closing_fence(line, self._fence_char, self._fence_len):
                            self._in_code_fence = False
                            self._suppress_fence = False
                        continue
                    if final:
                        self._buf = ""
                        self._in_code_fence = False
                        self._suppress_fence = False
                    break
                if nl != -1:
                    part = self._buf[: nl + 1]
                    self._buf = self._buf[nl + 1 :]
                    full_line = self._fence_line_buf + part[:-1]
                    self._fence_line_buf = ""
                    self._append_out(out, part)
                    if _is_closing_fence(full_line, self._fence_char, self._fence_len):
                        self._in_code_fence = False
                    continue
                chunk = self._buf
                self._buf = ""
                if len(self._fence_line_buf) < 64:
                    self._fence_line_buf += chunk[: 64 - len(self._fence_line_buf)]
                self._append_out(out, chunk)
                if final and _is_closing_fence(
                    self._fence_line_buf, self._fence_char, self._fence_len
                ):
                    self._in_code_fence = False
                    self._fence_line_buf = ""
                break

            # 3. Inside inline code (`...`)
            if self._inline_ticks > 0:
                m = re.search(r"[`\n]", self._buf)
                if not m:
                    self._append_out(out, self._buf)
                    self._buf = ""
                    break
                if m.start() > 0:
                    self._append_out(out, self._buf[: m.start()])
                    self._buf = self._buf[m.start() :]
                    continue
                if self._buf[0] == "\n":
                    self._append_out(out, "\n")
                    self._buf = self._buf[1:]
                    if self._trailing_newlines >= 2:
                        self._inline_ticks = 0
                    continue
                count = 0
                while count < len(self._buf) and self._buf[count] == "`":
                    count += 1
                if count == len(self._buf) and not final:
                    break
                self._append_out(out, self._buf[:count])
                self._buf = self._buf[count:]
                if count == self._inline_ticks:
                    self._inline_ticks = 0
                continue

            # 4. Normal prose
            ch = self._buf[0]
            if ch in (" ", "\t", "\r", "\n"):
                pos = 1
                while pos < len(self._buf) and self._buf[pos] in (" ", "\t", "\r", "\n"):
                    pos += 1
                ws = self._buf[:pos]
                self._buf = self._buf[pos:]
                if not self._skip_ws:
                    self._pending_ws += ws
                continue

            if ch == "h":
                if any(p.startswith(self._buf) for p in _CARD_URL_PREFIXES):
                    if not final:
                        break
                else:
                    matched_prefix = next(
                        (p for p in _CARD_URL_PREFIXES if self._buf.startswith(p)), None
                    )
                    if matched_prefix:
                        d_end = len(matched_prefix)
                        while d_end < len(self._buf) and self._buf[d_end].isdigit():
                            d_end += 1
                        if d_end == len(self._buf) and not final:
                            break
                        if d_end > len(matched_prefix):
                            if d_end < len(self._buf) and self._buf[d_end] == "\r":
                                if d_end + 1 == len(self._buf) and not final:
                                    break
                                if d_end + 1 < len(self._buf) and self._buf[d_end + 1] == "\n":
                                    d_end += 2
                            elif d_end < len(self._buf) and self._buf[d_end] == "\n":
                                d_end += 1
                            self._buf = self._buf[d_end:]
                            continue
                self._flush_ws_before_visible(out)
                self._append_out(out, "h")
                self._buf = self._buf[1:]
                continue

            if ch in ("`", "~"):
                count = 0
                while count < len(self._buf) and self._buf[count] == ch:
                    count += 1
                if count == len(self._buf) and not final:
                    break
                if ch == "~":
                    if count >= 3 and self._at_fence_start():
                        nl = self._buf.find("\n", count)
                        if nl == -1 and not final:
                            break
                        end = nl + 1 if nl != -1 else len(self._buf)
                        self._flush_ws_before_visible(out)
                        self._append_out(out, self._buf[:end])
                        self._buf = self._buf[end:]
                        self._in_code_fence = True
                        self._suppress_fence = False
                        self._fence_char = "~"
                        self._fence_len = count
                        self._fence_line_buf = ""
                        continue
                    self._flush_ws_before_visible(out)
                    self._append_out(out, self._buf[:count])
                    self._buf = self._buf[count:]
                    continue
                # ch == '`'
                if count < 3:
                    self._flush_ws_before_visible(out)
                    self._append_out(out, self._buf[:count])
                    self._buf = self._buf[count:]
                    self._inline_ticks = count
                    continue
                if self._at_fence_start():
                    nl = self._buf.find("\n", count)
                    if nl == -1 and not final:
                        break
                    info = self._buf[count : nl if nl != -1 else len(self._buf)]
                    end = nl + 1 if nl != -1 else len(self._buf)
                    if _CODE_EXEC_INFO_RE.match(info):
                        self._buf = self._buf[end:]
                        self._in_code_fence = True
                        self._suppress_fence = True
                        self._fence_char = "`"
                        self._fence_len = count
                        continue
                    self._flush_ws_before_visible(out)
                    self._append_out(out, self._buf[:end])
                    self._buf = self._buf[end:]
                    self._in_code_fence = True
                    self._suppress_fence = False
                    self._fence_char = "`"
                    self._fence_len = count
                    self._fence_line_buf = ""
                    continue
                # Mid-line ```: still suppress Gemini code_reference/code_stdout blocks
                after = self._buf[count:]
                if any(p.startswith(after) for p in _CODE_EXEC_LANG_PREFIXES) and not final:
                    break
                if any(after.startswith(p) for p in _CODE_EXEC_LANG_PREFIXES):
                    nl = self._buf.find("\n", count)
                    if nl == -1 and not final:
                        break
                    info = self._buf[count : nl if nl != -1 else len(self._buf)]
                    if _CODE_EXEC_INFO_RE.match(info):
                        end = nl + 1 if nl != -1 else len(self._buf)
                        self._buf = self._buf[end:]
                        self._in_code_fence = True
                        self._suppress_fence = True
                        self._fence_char = "`"
                        self._fence_len = count
                        continue
                self._flush_ws_before_visible(out)
                self._append_out(out, self._buf[:count])
                self._buf = self._buf[count:]
                self._inline_ticks = count
                continue

            if ch == "<" and self.clean_ui_tags:
                status = _match_ui_tag_prefix(self._buf)
                if status == "need_more":
                    if not final:
                        break
                elif status == "match":
                    end_pos = _find_tag_end(self._buf)
                    if end_pos == -1:
                        if not final and len(self._buf) <= _MAX_UI_TAG_LEN:
                            break
                        if (
                            final
                            and self.elicitation_mode == "strip"
                            and re.match(
                                r"^</?(?:ElicitationsGroup|Elicitation|FollowUp)\b",
                                self._buf,
                            )
                        ):
                            self._buf = ""
                            self._pending_ws = ""
                            self._skip_ws = True
                            break
                    else:
                        tag_str = self._buf[: end_pos + 1]
                        self._buf = self._buf[end_pos + 1 :]
                        name, is_closing, is_self_closing, attrs = _parse_ui_tag(tag_str)
                        if name == "Steps":
                            self._step_index = 0
                            self._pending_ws = ""
                            self._skip_ws = True
                            if self._has_output:
                                self._need_block_sep = True
                        elif name == "Step":
                            if is_closing:
                                self._pending_ws = ""
                                self._skip_ws = True
                                if self._has_output:
                                    self._need_block_sep = True
                            else:
                                self._step_index += 1
                                title = (attrs.get("title") or "").strip()
                                subtitle = (attrs.get("subtitle") or "").strip()
                                if title and subtitle:
                                    heading = f"#### {self._step_index}. {title} — {subtitle}"
                                elif title:
                                    heading = f"#### {self._step_index}. {title}"
                                elif subtitle:
                                    heading = f"#### {self._step_index}. {subtitle}"
                                else:
                                    heading = f"#### Step {self._step_index}"
                                self._pending_ws = ""
                                self._emit_block_sep(out)
                                self._append_out(out, heading)
                                self._need_block_sep = True
                                self._skip_ws = True
                        elif name in ("ElicitationsGroup", "Elicitation", "FollowUp"):
                            if is_closing:
                                self._pending_ws = ""
                                self._skip_ws = True
                                if self._has_output:
                                    self._need_block_sep = True
                            elif is_self_closing:
                                if (
                                    self.elicitation_mode == "markdown"
                                    and name in ("Elicitation", "FollowUp")
                                ):
                                    label = (attrs.get("label") or attrs.get("query") or "").strip()
                                    if label:
                                        self._pending_ws = ""
                                        self._emit_block_sep(out)
                                        self._append_out(out, f"- {label}")
                                        self._skip_ws = True
                                        self._need_block_sep = True
                                    else:
                                        self._pending_ws = ""
                                        self._skip_ws = True
                                else:
                                    self._pending_ws = ""
                                    self._skip_ws = True
                                    if self._has_output:
                                        self._need_block_sep = True
                            else:
                                self._block_tag = name
                                self._block_attrs = attrs
                                self._block_buf = ""
                        continue
                self._flush_ws_before_visible(out)
                self._append_out(out, "<")
                self._buf = self._buf[1:]
                continue

            # Regular non-special text run
            m = special_re.search(self._buf, 1)
            end = m.start() if m else len(self._buf)
            self._flush_ws_before_visible(out)
            self._append_out(out, self._buf[:end])
            self._buf = self._buf[end:]

        if final and self._pending_ws and not self._skip_ws:
            self._append_out(out, self._pending_ws)
            self._pending_ws = ""
        return "".join(out)


def clean_text(
    text: str,
    strip: bool = True,
    clean_ui_tags: bool = None,
    elicitation_mode: str = None,
) -> str:
    cleaner = StreamTextCleaner(
        clean_ui_tags=clean_ui_tags, elicitation_mode=elicitation_mode
    )
    result = cleaner.feed(text) + cleaner.flush()
    return result.strip() if strip else result


def _iter_frames(line: str):
    """Yield the parsed inner payload of every wrb.fr frame in one response line.

    A single StreamGenerate chunk line can carry several frames; the answer,
    thought summaries, alternative drafts, follow-up chips and image-agent
    updates each arrive as separate frames.
    """
    if '"wrb.fr"' not in line or len(line) < 40:
        return
    try:
        frames = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(frames, list):
        return
    for frame in frames:
        if not (isinstance(frame, list) and len(frame) > 2 and frame[0] == "wrb.fr"):
            continue
        payload = frame[2]
        if not (isinstance(payload, str) and len(payload) >= 20):
            continue
        try:
            inner = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(inner, list) and len(inner) > 4 and inner[4]:
            yield inner


def _candidate_texts(inner: list) -> list:
    """Return [(candidate_index, joined_text)] for one frame payload.

    Each candidate's text is a *list* of segments in the wire format and must
    be joined; treating segments as standalone answers returned fragments.
    """
    out = []
    for idx, cand in enumerate(inner[4]):
        if isinstance(cand, list) and len(cand) > 1 and isinstance(cand[1], list):
            text = "".join(t for t in cand[1] if isinstance(t, str))
            if text:
                out.append((idx, text))
    return out


def _extract_texts_from_line(line: str) -> list:
    """Extract candidate text segments from a line (used for legacy callers and tests)."""
    texts = []
    for inner in _iter_frames(line):
        for _, text in _candidate_texts(inner):
            texts.append(text)
    return texts


def _is_answer_frame(inner: list) -> bool:
    """True for the main answer frame, which carries the new conversation id
    (inner[1], e.g. "c_...") and response id (inner[2], e.g. "r_...").

    Thought summaries, follow-up chips, search and image-agent frames do not
    carry these ids -- that is what tells them apart from the real answer.
    """
    if len(inner) > 2:
        conv_id, resp_id = inner[1], inner[2]
        return (isinstance(conv_id, str) and bool(conv_id)
                and isinstance(resp_id, str) and bool(resp_id))
    return False


def _best_main_answer(line: str):
    """Best primary-candidate answer text found in one line, or None."""
    best = ""
    for inner in _iter_frames(line):
        if not _is_answer_frame(inner):
            continue
        for idx, text in _candidate_texts(inner):
            if idx == 0 and len(text) > len(best):
                best = text
    return best or None


def extract_response_text(raw: str) -> str:
    """Parse full response to get final text.

    Selection order:
      1. Primary candidate (index 0) of the main answer frame -- the frame
         carrying conversation/response ids. Longest update wins (streaming
         re-sends this frame as the answer grows).
      2. Longest joined candidate from any frame (protocol drift fallback).
    Never the thought/draft/chip frames, which previously won the "longest
    text anywhere" heuristic and produced unrelated answers.
    """
    bard_err = re.search(r'BardErrorInfo"?,?\s*\[(\d+)\]', raw)
    if bard_err:
        code = int(bard_err.group(1))
        hints = {
            1060: "IP temporarily blocked or region not supported - use a proxy/different network or wait",
            1037: "usage limit exceeded",
            1013: "temporary upstream error, retry later",
            1185: "upstream rejected request",
        }
        raise RuntimeError(f"Gemini upstream error [{code}]: {hints.get(code, 'upstream rejected request')}")
    main_best = ""
    any_best = ""
    for line in raw.split("\n"):
        for inner in _iter_frames(line):
            cands = _candidate_texts(inner)
            if not cands:
                continue
            is_main = _is_answer_frame(inner)
            for idx, text in cands:
                if len(text) > len(any_best):
                    any_best = text
                if is_main and idx == 0 and len(text) > len(main_best):
                    main_best = text
    return clean_text(main_best or any_best)


def generate(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, model_name: str = None) -> str:
    """Non-streaming generation with retry."""
    ctx = _get_ssl_ctx()
    proxy = CONFIG.get("proxy")

    last_err = None
    for attempt in range(CONFIG["retry_attempts"]):
        try:
            body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields).encode()
            headers = _build_headers(model_name=model_name, model_id=model_id)
            req = urllib.request.Request(_get_url(), data=body, headers=headers, method="POST")
            if proxy:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                    urllib.request.HTTPSHandler(context=ctx)
                )
                resp = opener.open(req, timeout=CONFIG["request_timeout_sec"])
            else:
                resp = urllib.request.urlopen(req, context=ctx, timeout=CONFIG["request_timeout_sec"])
            raw = resp.read().decode("utf-8", errors="replace")
            return extract_response_text(raw)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RuntimeError("Gemini upstream rate-limited this IP (HTTP 429); retrying immediately would extend the block")
            if e.code in (400, 405) and refresh_bl_and_xsrf():
                log("Retrying with refreshed BL/XSRF...")
                last_err = e
                continue
            last_err = e
            if attempt < CONFIG["retry_attempts"] - 1:
                log(f"Retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
        except Exception as e:
            last_err = e
            if attempt < CONFIG["retry_attempts"] - 1:
                log(f"Retry {attempt+1}/{CONFIG['retry_attempts']}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err


def generate_stream(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None, model_name: str = None):
    """Streaming generation via httpx with retry on connection failure.

    Only the primary candidate of the main answer frame is streamed.
    Thought summaries, drafts and chip frames are ignored, deltas are only
    emitted when the new answer extends what was already emitted, and a
    mid-stream answer rewrite freezes the already-emitted text instead of
    splicing unrelated frames into the output.
    """
    if not HAS_HTTPX:
        text = generate(prompt, model_id, think_mode, file_refs, extra_fields, model_name=model_name)
        if text:
            yield text
        return

    client = _get_httpx_client()

    last_err = None
    for attempt in range(CONFIG["retry_attempts"]):
        emitted_raw_text = ""
        emitted_any = False
        raw_lines = []
        cleaner = StreamTextCleaner()
        try:
            body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields)
            headers = _build_headers(model_name=model_name, model_id=model_id)
            with client.stream("POST", _get_url(), content=body, headers=headers) as resp:
                resp.raise_for_status()
                buf = ""
                for chunk in resp.iter_text():
                    buf += chunk
                    if "BardErrorInfo" in buf:
                        bard_err = re.search(r'BardErrorInfo"?,?\s*\[(\d+)\]', buf)
                        if bard_err:
                            raise RuntimeError(
                                f"Gemini upstream error [{bard_err.group(1)}]"
                            )
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        raw_lines.append(line)
                        current = _best_main_answer(line)
                        if not current:
                            continue
                        if current == emitted_raw_text or emitted_raw_text.startswith(current):
                            continue  # duplicate or older update
                        if not current.startswith(emitted_raw_text):
                            if emitted_any:
                                log("Stream answer replaced mid-flight; keeping already-emitted text")
                                continue
                            emitted_raw_text = ""  # first frame was noise; adopt the real answer
                            cleaner = StreamTextCleaner()
                        delta = cleaner.feed(current[len(emitted_raw_text):])
                        emitted_raw_text = current
                        if delta:
                            emitted_any = True
                            yield delta
            tail = cleaner.flush()
            if tail:
                emitted_any = True
                yield tail
            if emitted_any:
                return
            # Strict parser matched nothing (protocol drift): rescan the full
            # buffered response with the fallback tiers before giving up.
            if raw_lines:
                fallback = extract_response_text("\n".join(raw_lines))
                if fallback:
                    yield fallback
                    return
            last_err = RuntimeError("no answer frame in Gemini stream response")
        except Exception as e:
            if emitted_raw_text or emitted_any:
                # Partial content already streamed: a retry would either fail the
                # prefix check or splice two different generations together, so end
                # the stream cleanly and let the client ask to continue.
                tail = cleaner.flush()
                if tail:
                    yield tail
                log(f"Stream interrupted after partial output ({len(emitted_raw_text)} chars), ending cleanly: {e}")
                return
            # Hard upstream rejections (BardErrorInfo) - retrying is futile,
            # except 1013 which is transient per upstream behavior
            if "Gemini upstream error" in str(e) and "[1013]" not in str(e):
                raise
            status = getattr(getattr(e, "response", None), "status_code", 0)
            if status in (400, 405) and not emitted_raw_text and refresh_bl_and_xsrf():
                log("Stream retrying with refreshed BL/XSRF...")
                last_err = e
                continue
            last_err = e
        if attempt < CONFIG["retry_attempts"] - 1:
            log(f"Stream retry {attempt+1}/{CONFIG['retry_attempts']}: {last_err}")
            time.sleep(CONFIG["retry_delay_sec"])
    raise last_err
