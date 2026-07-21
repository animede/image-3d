"""マルチモーダルLLMによるパーツ検出アダプタ (SPEC.md §3.12 / FR-13 第3層誘導)。

ジオメトリ(肉厚+くびれ)・画像色領域(use_image)に続く第3の誘導層として、
ユーザー環境のOpenAI互換マルチモーダルLLMサーバ(例: llama.cpp + Gemma系
visionモデル)へ正面画像を投げ、部位名+bboxのリストを取得する。

このモジュールはアダプタ側(HTTPアクセス + PIL画像処理)であり、
`server/pattern/` の純粋モジュール方針(DEVELOPMENT_POLICY.md §3.5)には
含まれない。`server/pattern/parts.py` からは import されず、`server/main.py`
からのみ呼び出される想定。

`requests` はこのプロジェクトの依存に含まれないため、標準ライブラリの
`urllib.request` のみを使用する。
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 送信前に画像を縮小する最大辺(px)。LLMの入力トークン数・レイテンシ抑制のため。
_MAX_IMAGE_SIDE = 512

# パーツ検出プロンプト(日本語、正面画像1枚からJSON配列のみを要求する)。
_PROMPT = (
    "この画像に写っているぬいぐるみ/キャラクターの部位を検出してください。"
    "頭・耳・胴体・腕・脚・帽子・しっぽ等、縫製パーツの単位で分けてください。"
    "左右がある部位には(右)(左)を付けてください(画像内での左右基準)。"
    "各部位について、名前と画像内でのバウンディングボックスを"
    "0〜1に正規化した [x0, y0, x1, y1] (x0<x1, y0<y1) で答えてください。"
    "出力は説明文を含めず、次の形式のJSON配列のみを返してください:\n"
    '[{"name": "頭", "bbox": [0.1, 0.1, 0.5, 0.5]}, ...]'
)

_MAX_PARTS = 12


def is_available(endpoint: Optional[str]) -> bool:
    """endpointが設定されていれば利用可能とみなす(疎通確認はしない)。

    `/api/health` を重くしないため、実際にLLMサーバへ接続確認はしない
    (設定の有無のみを見る軽量チェック)。
    """
    return bool(endpoint)


def _resize_image_for_llm(image_rgba: np.ndarray, max_side: int = _MAX_IMAGE_SIDE):
    """画像を最大辺 `max_side` px以下に縮小したPIL Imageを返す。"""
    from PIL import Image

    img = Image.fromarray(np.asarray(image_rgba, dtype=np.uint8), mode="RGBA")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def _image_to_base64_png(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json_array(text: str) -> Optional[list]:
    """LLM応答テキストからJSON配列を抽出する。

    1. コードフェンス(```json ... ``` や ``` ... ```)を除去してから
       そのまま `json.loads` を試みる。
    2. 失敗した場合、テキスト中の最初の `[` から対応する `]` までを
       正規表現で緩く抽出して再度 `json.loads` を試みる。
    それでも失敗すれば None を返す。
    """
    if not text:
        return None

    stripped = text.strip()
    # コードフェンス除去(```json ... ``` / ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    candidate = fence_match.group(1).strip() if fence_match else stripped

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # フォールバック: 最初の '[' から最後の ']' までを抽出
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = stripped[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _validate_parts(raw_parts: list) -> Optional[list[dict]]:
    """LLMが返したパーツリストをバリデーションする。

    各要素は `{"name": str, "bbox": [x0, y0, x1, y1]}` の形式で、
    bboxは0〜1の範囲かつ x0<x1, y0<y1 であること。最大 `_MAX_PARTS` 件まで。
    1件でも不正な要素があれば None を返す(部分採用はしない)。
    """
    if not isinstance(raw_parts, list) or len(raw_parts) == 0:
        return None
    if len(raw_parts) > _MAX_PARTS:
        raw_parts = raw_parts[:_MAX_PARTS]

    validated: list[dict] = []
    for item in raw_parts:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        bbox = item.get("bbox")
        if not isinstance(name, str) or not name.strip():
            return None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            return None
        if not all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
            return None
        if not (x0 < x1 and y0 < y1):
            return None
        validated.append({"name": name.strip(), "bbox": [x0, y0, x1, y1]})

    if not validated:
        return None
    return validated


def detect_parts(
    image_rgba: np.ndarray, endpoint: str, timeout: float = 60.0
) -> Optional[list[dict]]:
    """マルチモーダルLLMサーバへ正面画像を投げ、パーツ(名前+bbox)を検出する。

    Args:
        image_rgba: (H, W, 4) uint8 ndarray(背景除去済み正面画像)。
        endpoint: OpenAI互換サーバのベースURL(例: "http://host:port")。
            末尾の `/v1/chat/completions` は本関数が付与する。
        timeout: HTTPリクエストのタイムアウト秒数。

    Returns:
        `[{"name": str, "bbox": [x0, y0, x1, y1]}, ...]`(正規化座標、
        最大12件)。通信エラー・タイムアウト・不正な応答の場合は None
        (呼び出し側は色誘導/ジオメトリ誘導へフォールバックする)。
    """
    if not endpoint:
        return None

    try:
        pil_image = _resize_image_for_llm(image_rgba)
        image_b64 = _image_to_base64_png(pil_image)

        payload = {
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        }

        url = endpoint.rstrip("/") + "/v1/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))

        content = body["choices"][0]["message"]["content"]
        raw_parts = _extract_json_array(content)
        if raw_parts is None:
            logger.warning("llm_parts: failed to parse JSON array from LLM response")
            return None

        validated = _validate_parts(raw_parts)
        if validated is None:
            logger.warning("llm_parts: LLM response failed validation")
            return None
        return validated

    except urllib.error.URLError:
        logger.exception("llm_parts: request to LLM endpoint failed (%s)", endpoint)
        return None
    except TimeoutError:
        logger.exception("llm_parts: request to LLM endpoint timed out (%s)", endpoint)
        return None
    except Exception:
        logger.exception("llm_parts: unexpected error while detecting parts (%s)", endpoint)
        return None
