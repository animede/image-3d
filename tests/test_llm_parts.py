"""マルチモーダルLLMパーツ検出アダプタ `server/llm_parts.py` のテスト。

実LLMサーバは呼ばない。`urllib.request.urlopen` をmonkeypatchでモックし、
正常系(コードフェンス付きJSON)・不正JSON・タイムアウトを検証する。
"""
from __future__ import annotations

import json
import socket
from io import BytesIO

import numpy as np
import pytest

from server import llm_parts


def make_rgba(h=64, w=64) -> np.ndarray:
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 3] = 255
    img[:, :, :3] = [200, 50, 50]
    return img


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _chat_payload(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


# --------------------------------------------------------------------------
# is_available
# --------------------------------------------------------------------------
def test_is_available_false_for_empty_endpoint():
    assert llm_parts.is_available("") is False
    assert llm_parts.is_available(None) is False


def test_is_available_true_for_configured_endpoint():
    assert llm_parts.is_available("http://localhost:1234") is True


# --------------------------------------------------------------------------
# detect_parts: 正常系(コードフェンス付きJSON)
# --------------------------------------------------------------------------
def test_detect_parts_parses_fenced_json(monkeypatch):
    fenced = (
        "```json\n"
        '[{"name": "頭", "bbox": [0.1, 0.1, 0.5, 0.5]}, '
        '{"name": "胴体", "bbox": [0.1, 0.4, 0.9, 0.95]}]'
        "\n```"
    )

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(_chat_payload(fenced))

    monkeypatch.setattr(llm_parts.urllib.request, "urlopen", fake_urlopen)

    result = llm_parts.detect_parts(make_rgba(), "http://localhost:1234", timeout=5.0)
    assert result is not None
    assert len(result) == 2
    assert result[0]["name"] == "頭"
    assert result[0]["bbox"] == [0.1, 0.1, 0.5, 0.5]
    assert result[1]["name"] == "胴体"


def test_detect_parts_parses_plain_json_without_fence(monkeypatch):
    plain = '[{"name": "耳(右)", "bbox": [0.0, 0.0, 0.3, 0.3]}]'

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(_chat_payload(plain))

    monkeypatch.setattr(llm_parts.urllib.request, "urlopen", fake_urlopen)

    result = llm_parts.detect_parts(make_rgba(), "http://localhost:1234", timeout=5.0)
    assert result == [{"name": "耳(右)", "bbox": [0.0, 0.0, 0.3, 0.3]}]


def test_detect_parts_extracts_json_array_from_surrounding_prose(monkeypatch):
    prose = (
        "はい、検出結果は以下の通りです:\n"
        '[{"name": "頭", "bbox": [0.2, 0.1, 0.6, 0.5]}]\n'
        "以上です。"
    )

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(_chat_payload(prose))

    monkeypatch.setattr(llm_parts.urllib.request, "urlopen", fake_urlopen)

    result = llm_parts.detect_parts(make_rgba(), "http://localhost:1234", timeout=5.0)
    assert result == [{"name": "頭", "bbox": [0.2, 0.1, 0.6, 0.5]}]


# --------------------------------------------------------------------------
# detect_parts: 不正系 -> None
# --------------------------------------------------------------------------
def test_detect_parts_returns_none_for_unparseable_content(monkeypatch):
    def fake_urlopen(req, timeout=None):
        return _FakeResponse(_chat_payload("すみません、うまく認識できませんでした。"))

    monkeypatch.setattr(llm_parts.urllib.request, "urlopen", fake_urlopen)

    result = llm_parts.detect_parts(make_rgba(), "http://localhost:1234", timeout=5.0)
    assert result is None


def test_detect_parts_returns_none_for_invalid_bbox_ordering(monkeypatch):
    bad = '[{"name": "頭", "bbox": [0.5, 0.1, 0.1, 0.5]}]'  # x0 > x1

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(_chat_payload(bad))

    monkeypatch.setattr(llm_parts.urllib.request, "urlopen", fake_urlopen)

    result = llm_parts.detect_parts(make_rgba(), "http://localhost:1234", timeout=5.0)
    assert result is None


def test_detect_parts_returns_none_for_out_of_range_bbox(monkeypatch):
    bad = '[{"name": "頭", "bbox": [0.1, 0.1, 1.5, 0.5]}]'  # x1 > 1.0

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(_chat_payload(bad))

    monkeypatch.setattr(llm_parts.urllib.request, "urlopen", fake_urlopen)

    result = llm_parts.detect_parts(make_rgba(), "http://localhost:1234", timeout=5.0)
    assert result is None


def test_detect_parts_returns_none_for_missing_name(monkeypatch):
    bad = '[{"bbox": [0.1, 0.1, 0.5, 0.5]}]'

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(_chat_payload(bad))

    monkeypatch.setattr(llm_parts.urllib.request, "urlopen", fake_urlopen)

    result = llm_parts.detect_parts(make_rgba(), "http://localhost:1234", timeout=5.0)
    assert result is None


def test_detect_parts_truncates_to_max_parts(monkeypatch):
    many = [{"name": f"部位{i}", "bbox": [0.0, 0.0, 0.1, 0.1]} for i in range(20)]

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(_chat_payload(json.dumps(many)))

    monkeypatch.setattr(llm_parts.urllib.request, "urlopen", fake_urlopen)

    result = llm_parts.detect_parts(make_rgba(), "http://localhost:1234", timeout=5.0)
    assert result is not None
    assert len(result) == 12


def test_detect_parts_empty_endpoint_returns_none():
    result = llm_parts.detect_parts(make_rgba(), "", timeout=5.0)
    assert result is None


# --------------------------------------------------------------------------
# detect_parts: タイムアウト・通信エラー -> None
# --------------------------------------------------------------------------
def test_detect_parts_returns_none_on_timeout(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise socket.timeout("timed out")

    monkeypatch.setattr(llm_parts.urllib.request, "urlopen", fake_urlopen)

    result = llm_parts.detect_parts(make_rgba(), "http://localhost:1234", timeout=0.01)
    assert result is None


def test_detect_parts_returns_none_on_connection_error(monkeypatch):
    import urllib.error

    def fake_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(llm_parts.urllib.request, "urlopen", fake_urlopen)

    result = llm_parts.detect_parts(make_rgba(), "http://localhost:1234", timeout=5.0)
    assert result is None
