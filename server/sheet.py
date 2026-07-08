"""キャラクターシート自動分割 (SPEC.md §3.8 / FR-9)。

1枚のシート画像(複数ビューが並んだ画像)から被写体パネルを自動検出する。

検出手順:
  1. 前景マスク取得:
     a. RGBAでアルファチャンネルに情報があればそれを使用。
     b. 無ければ rembg でマスクを取得(導入済み、preprocess.py参照)。
     c. それも不可なら四隅の背景色との色差で2値化する。
  2. マスクの連結成分解析 (scipy.ndimage.label)。
     - 面積が画像全体の1%未満の成分は除去する。
     - バウンディングボックスが重なる、または近接する(間隔が画像幅の2%未満)
       "かつ" 面積比(小/大)が閾値未満(=小さい断片が大きい本体に吸収される
       ケースとみなせる)場合のみマージする。同程度の大きさの成分同士
       (=別々のキャラクター)はマージしない。
  3. 残ったボックスを左→右(同じ列とみなせるものは上→下)にソートし、
     少しのパディング付きで元画像から切り出して返す(最大 MAX_PANELS 枚。
     検出された被写体が全て見えるよう、通常のシートでは実質上限に
     達しない値を設定している)。
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)

MIN_AREA_FRACTION = 0.01  # 画像全体の1%未満の連結成分は除去
MERGE_GAP_FRACTION = 0.02  # 間隔が画像幅の2%未満のボックスはマージ候補
MERGE_AREA_RATIO = 0.5  # 面積比(小/大)がこの値未満の場合のみマージする
# (同程度の大きさの成分は別々の被写体とみなし、隣接していてもマージしない)
MAX_PANELS = 20  # 暴走検出時の安全上限。通常のキャラクターシートでは到達しない想定
PADDING_FRACTION = 0.02  # 切り出し時のパディング(画像幅・高さに対する割合)

# パネル数に応じた suggested_view の割当順(左から)。
# SPEC.md §3.8: 「正面→側面→背面」の順を仮定した左からの並び順ヒューリスティクス。
_VIEW_ORDER_BY_COUNT: dict[int, list[str]] = {
    1: ["front"],
    2: ["front", "back"],
    3: ["front", "left", "back"],
    4: ["front", "left", "back", "right"],
    5: ["left", "front", "right", "back", "back"],
    6: ["left", "front", "right", "back", "left", "right"],
}


class BBox:
    """整数バウンディングボックス (x0, y0, x1, y1)、x1/y1は排他的境界。"""

    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, x0: int, y0: int, x1: int, y1: int) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def area(self) -> int:
        return max(0, self.x1 - self.x0) * max(0, self.y1 - self.y0)

    def merged(self, other: "BBox") -> "BBox":
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def gap_to(self, other: "BBox") -> float:
        """2つのボックス間の距離(重なっていれば負またはゼロ)。"""
        dx = max(self.x0 - other.x1, other.x0 - self.x1, 0)
        dy = max(self.y0 - other.y1, other.y0 - self.y1, 0)
        if dx == 0 and dy == 0:
            return 0.0
        return float(np.hypot(dx, dy))


def _mask_from_alpha(image: Image.Image) -> np.ndarray | None:
    """RGBAのアルファチャンネルに情報があればそれを前景マスクとして返す。"""
    if image.mode != "RGBA":
        return None
    alpha = np.asarray(image.getchannel("A"))
    if alpha.min() == alpha.max():
        # アルファが一様(情報なし)の場合は使えない
        return None
    return alpha > 10


def _mask_from_rembg(image: Image.Image) -> np.ndarray | None:
    """rembgで前景マスクを取得する。未導入・失敗時はNoneを返す。"""
    try:
        from rembg import remove
    except ImportError:
        return None
    try:
        rgb_image = image.convert("RGB")
        result = remove(rgb_image)
        if result.mode != "RGBA":
            result = result.convert("RGBA")
        alpha = np.asarray(result.getchannel("A"))
        return alpha > 10
    except Exception:
        logger.exception("rembg mask extraction failed; falling back to color diff.")
        return None


def _mask_from_color_diff(image: Image.Image) -> np.ndarray:
    """四隅の背景色との色差で前景マスクを2値化するフォールバック。"""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    h, w = rgb.shape[:2]

    corner_size = max(1, min(h, w) // 20)
    corners = [
        rgb[:corner_size, :corner_size],
        rgb[:corner_size, -corner_size:],
        rgb[-corner_size:, :corner_size],
        rgb[-corner_size:, -corner_size:],
    ]
    bg_color = np.mean([c.reshape(-1, 3).mean(axis=0) for c in corners], axis=0)

    diff = np.linalg.norm(rgb - bg_color, axis=2)
    threshold = max(20.0, diff.std() * 0.5 + diff.mean() * 0.25)
    return diff > threshold


def _get_foreground_mask(image: Image.Image) -> np.ndarray:
    """前景マスクをアルファ→rembg→色差の優先順で取得する。"""
    mask = _mask_from_alpha(image)
    if mask is not None:
        return mask
    mask = _mask_from_rembg(image)
    if mask is not None:
        return mask
    return _mask_from_color_diff(image)


def _connected_component_boxes(mask: np.ndarray) -> list[BBox]:
    """マスクの連結成分バウンディングボックスを面積フィルタ付きで返す。"""
    h, w = mask.shape
    total_area = h * w
    min_area = total_area * MIN_AREA_FRACTION

    labeled, num_features = ndimage.label(mask)
    boxes: list[BBox] = []
    slices = ndimage.find_objects(labeled)
    for idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        component_mask = labeled[sl] == idx
        area = int(component_mask.sum())
        if area < min_area:
            continue
        y_slice, x_slice = sl
        boxes.append(BBox(x_slice.start, y_slice.start, x_slice.stop, y_slice.stop))
    return boxes


def _should_merge(a: BBox, b: BBox, max_gap: float) -> bool:
    """2つのボックスをマージすべきか判定する。

    間隔が近い(または重なる)だけでなく、面積比(小/大)が
    MERGE_AREA_RATIO 未満の場合のみマージ対象とする。同程度の大きさの
    ボックス同士は別々の被写体(例: 隣接する2キャラクター)とみなし、
    どれだけ近くてもマージしない。
    """
    if a.gap_to(b) >= max_gap:
        return False
    smaller = min(a.area(), b.area())
    larger = max(a.area(), b.area())
    if larger == 0:
        return False
    return (smaller / larger) < MERGE_AREA_RATIO


def _merge_close_boxes(boxes: list[BBox], image_width: int) -> list[BBox]:
    """近接(間隔が画像幅の2%未満)し、かつ面積比が閾値未満のボックスをマージする。"""
    max_gap = image_width * MERGE_GAP_FRACTION
    merged = list(boxes)

    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if _should_merge(merged[i], merged[j], max_gap):
                    merged[i] = merged[i].merged(merged[j])
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def _sort_boxes(boxes: list[BBox], image_height: int) -> list[BBox]:
    """左→右(同列なら上→下)にソートする。

    「同列」は行方向のバウンディングボックス中心が画像高さの15%以内に
    収まるものとして緩やかにクラスタリングする(通常のキャラクターシートは
    横一列に並ぶことがほとんどのため、単純なy中心の近接判定で十分)。
    """
    row_tolerance = image_height * 0.15

    def center_y(b: BBox) -> float:
        return (b.y0 + b.y1) / 2.0

    # まずy中心でクラスタリングして行番号を割り振る
    sorted_by_y = sorted(boxes, key=center_y)
    rows: list[list[BBox]] = []
    for b in sorted_by_y:
        placed = False
        for row in rows:
            if abs(center_y(row[0]) - center_y(b)) < row_tolerance:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])

    rows.sort(key=lambda row: min(center_y(b) for b in row))

    result: list[BBox] = []
    for row in rows:
        row.sort(key=lambda b: b.x0)
        result.extend(row)
    return result


def _crop_with_padding(image: Image.Image, box: BBox) -> Image.Image:
    w, h = image.size
    pad_x = int(w * PADDING_FRACTION)
    pad_y = int(h * PADDING_FRACTION)
    x0 = max(0, box.x0 - pad_x)
    y0 = max(0, box.y0 - pad_y)
    x1 = min(w, box.x1 + pad_x)
    y1 = min(h, box.y1 + pad_y)
    return image.crop((x0, y0, x1, y1))


def suggested_views(count: int) -> list[str]:
    """パネル数に応じた suggested_view の初期推定リストを返す
    (左からの並び順ヒューリスティクス。SPEC.md §3.8)。
    """
    if count in _VIEW_ORDER_BY_COUNT:
        return _VIEW_ORDER_BY_COUNT[count]
    # 想定外の枚数はfront/left/back/rightを繰り返す
    base = ["front", "left", "back", "right"]
    return [base[i % len(base)] for i in range(count)]


def split_sheet(image: Image.Image) -> list[Image.Image]:
    """キャラクターシート画像から被写体パネルを自動検出して切り出す。

    Args:
        image: シート画像(RGB/RGBA)。

    Returns:
        検出されたパネル画像のリスト(左→右、同列なら上→下の順)。
        最大 MAX_PANELS 枚。検出できなければ画像全体を1パネルとして返す。
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    mask = _get_foreground_mask(image)
    boxes = _connected_component_boxes(mask)

    if not boxes:
        return [image.copy()]

    boxes = _merge_close_boxes(boxes, image.width)
    boxes = _sort_boxes(boxes, image.height)
    boxes = boxes[:MAX_PANELS]

    return [_crop_with_padding(image, box) for box in boxes]
