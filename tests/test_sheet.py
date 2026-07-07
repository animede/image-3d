"""server/sheet.py の単体テスト (SPEC.md §3.8 / FR-9)。

合成RGBA画像(透明背景に離れた色付きシルエット3つ)を使い、rembgに依存せず
アルファチャンネル経由のマスク検出パスを決定的に検証する。
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from server import sheet


def make_three_panel_sheet(
    size=(900, 400), panel_w=200, panel_h=300, gap=100
) -> Image.Image:
    """透明背景に3つの離れた矩形シルエット(色付き、不透明)を配置した
    決定的なテスト用RGBAシート画像を生成する。
    """
    w, h = size
    arr = np.zeros((h, w, 4), dtype=np.uint8)  # 全透明

    y0 = (h - panel_h) // 2
    colors = [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)]
    x_starts = [gap, gap * 2 + panel_w, gap * 3 + panel_w * 2]

    for x0, color in zip(x_starts, colors):
        arr[y0 : y0 + panel_h, x0 : x0 + panel_w] = color

    return Image.fromarray(arr, "RGBA")


def test_split_sheet_detects_three_panels_in_left_to_right_order():
    image = make_three_panel_sheet()
    panels = sheet.split_sheet(image)

    assert len(panels) == 3

    # 各パネルは左端の色(赤, 緑, 青)を含んでいるはず -> 順序確認
    expected_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for panel, expected in zip(panels, expected_colors):
        panel_rgb = np.asarray(panel.convert("RGB"))
        panel_alpha = np.asarray(panel.convert("RGBA").getchannel("A"))
        opaque = panel_alpha > 10
        assert opaque.any()
        mean_color = panel_rgb[opaque].mean(axis=0)
        # 支配的なチャンネルが期待通りであることを確認(厳密一致ではなく傾向で検証)
        dominant_channel = int(np.argmax(mean_color))
        expected_channel = int(np.argmax(expected))
        assert dominant_channel == expected_channel


def test_split_sheet_filters_small_noise_components():
    """画像全体の1%未満の小さいノイズ成分は除去されること。"""
    image = make_three_panel_sheet()
    arr = np.asarray(image).copy()

    # 1x1のノイズピクセルを追加(1%よりずっと小さい)
    arr[5, 5] = [10, 10, 10, 255]
    noisy_image = Image.fromarray(arr, "RGBA")

    panels = sheet.split_sheet(noisy_image)
    assert len(panels) == 3


def test_split_sheet_merges_close_boxes():
    """間隔が画像幅の2%未満の近接ボックスはマージされ、1パネルとして扱われること。"""
    w, h = 900, 400
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    y0 = 50
    panel_h = 300
    # 画像幅の1%(9px)しか離れていない2つの矩形 -> マージされるはず
    gap_px = int(w * 0.01)
    arr[y0 : y0 + panel_h, 100:300] = [255, 0, 0, 255]
    arr[y0 : y0 + panel_h, 300 + gap_px : 500 + gap_px] = [0, 255, 0, 255]

    image = Image.fromarray(arr, "RGBA")
    panels = sheet.split_sheet(image)
    assert len(panels) == 1


def test_split_sheet_returns_full_image_when_no_foreground_detected():
    """前景マスクが空(連結成分なし)の場合は画像全体を1パネルとして返す。

    アルファ情報が一様(無し)な画像はrembgフォールバックへ進むため、ここでは
    アルファ経路を使わせず `_connected_component_boxes` が空を返すケースを
    直接検証する(rembgの実際の挙動は入力に依存し決定的でないため)。
    """
    mask = np.zeros((200, 200), dtype=bool)
    boxes = sheet._connected_component_boxes(mask)
    assert boxes == []

    image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    # アルファ・rembgの両方を迂回し、空マスクからのフォールバックのみを検証する。
    import server.sheet as sheet_module

    original_get_mask = sheet_module._get_foreground_mask
    sheet_module._get_foreground_mask = lambda img: mask
    try:
        panels = sheet.split_sheet(image)
    finally:
        sheet_module._get_foreground_mask = original_get_mask

    assert len(panels) == 1
    assert panels[0].size == (200, 200)


@pytest.mark.parametrize(
    "count,expected_first",
    [
        (1, "front"),
        (2, "front"),
        (3, "front"),
        (4, "front"),
    ],
)
def test_suggested_views_starts_with_front_heuristic(count, expected_first):
    views = sheet.suggested_views(count)
    assert len(views) == count
    assert views[0] == expected_first


def test_suggested_views_three_panels_is_front_left_back():
    assert sheet.suggested_views(3) == ["front", "left", "back"]
