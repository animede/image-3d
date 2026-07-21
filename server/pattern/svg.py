"""実寸SVG型紙出力 (SPEC.md §3.12 / FR-13 の4b部分)。

`flatten.flatten_panel()` が返す平坦化済みパネル(2D頂点・境界ループ・
3D頂点・歪み指標)のリストから、mm実寸のSVG型紙を組み立てる。

含まれる要素:
    - パネルごとの縫い線(実線、境界ループ)+縫い代線(破線、外側オフセット)
    - シーム(隣接パネルの共有境界)ごとの合印(ノッチ、対応する2パネルの
      両側に同一記号)
    - パネル番号ラベル・布目線(縦の両矢印)
    - 凡例(モデル名・高さ・縫い代・シーム対応表)

パネル配置は単純なシェルフ法(左上から右方向に敷き詰め、行の最大高さを
超えたら次の行へ)による矩形パッキング。

縫い代オフセットは境界ポリゴンの頂点法線(隣接エッジの外向き法線の平均)
方向へのオフセット+短エッジ間引きによる簡易クリーンアップ(自己交差の
完全排除は保証しないが、実用レベルの型紙には十分)。

このモジュールは純粋モジュール(server/DEVELOPMENT_POLICY.md §3.5)。
依存は numpy / scipy / trimesh のみ(SVGはXML文字列を直接組み立てる。
lxml等のXMLライブラリはアプリ/テスト側でのみ使用し、ここでは使わない)。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

_EPS = 1e-9

# シーム(隣接パネル境界)対応の3D距離許容誤差(mm)。平坦化前の3D座標が
# 元メッシュの共有頂点由来のため、通常は厳密一致に近いが、浮動小数点誤差や
# パネル抽出の丸めを見込んで許容幅を持たせる。
_SEAM_MATCH_TOLERANCE_MM = 0.5

# シェルフパッキングのパネル間余白(mm)。
_PACK_MARGIN_MM = 15.0

_SVG_NS = "http://www.w3.org/2000/svg"


# --------------------------------------------------------------------------
# ユーティリティ
# --------------------------------------------------------------------------
def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_ccw(points: np.ndarray) -> np.ndarray:
    if _polygon_area(points) < 0:
        return points[::-1].copy()
    return points.copy()


def _ensure_ccw_with_mask(points: np.ndarray, mask: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """`_ensure_ccw`と同じ巻き方向補正を、対応する真偽値マスク
    (取付口フラグ等、`points`と同じ順序・長さの配列)にも適用して返す。"""
    if _polygon_area(points) < 0:
        reversed_mask = mask[::-1].copy() if mask is not None else None
        return points[::-1].copy(), reversed_mask
    return points.copy(), (mask.copy() if mask is not None else None)


def _offset_polygon(points: np.ndarray, distance: float, min_edge_len: float = 0.3) -> np.ndarray:
    """境界ポリゴンを外側(反時計回りを前提に法線=左向き90度回転の逆)へ
    `distance` だけオフセットした簡易オフセットポリゴンを返す。

    手法: 各頂点で隣接する2エッジの外向き法線(エッジ法線の平均、正規化)
    方向にオフセットする(マイタ近似)。短すぎるエッジは事前に間引いて
    ジグザグによる自己交差を軽減する(簡易クリーンアップ)。
    """
    pts = _ensure_ccw(points)
    pts = _dedup_short_edges(pts, min_edge_len)
    n = len(pts)
    if n < 3:
        return pts

    offset_pts = np.zeros_like(pts)
    for i in range(n):
        prev_pt = pts[(i - 1) % n]
        cur_pt = pts[i]
        next_pt = pts[(i + 1) % n]

        e1 = cur_pt - prev_pt
        e2 = next_pt - cur_pt
        n1 = _outward_normal(e1)
        n2 = _outward_normal(e2)
        normal = n1 + n2
        norm_len = np.linalg.norm(normal)
        if norm_len < _EPS:
            normal = n1
            norm_len = np.linalg.norm(n1)
        if norm_len < _EPS:
            offset_pts[i] = cur_pt
            continue
        normal = normal / norm_len

        # マイタ長の暴走を防ぐため、内積が小さい(鋭角)場合はクランプする
        cos_half = float(np.dot(normal, _outward_normal(e1)))
        miter_scale = 1.0 / max(cos_half, 0.5)
        miter_scale = min(miter_scale, 3.0)

        offset_pts[i] = cur_pt + normal * distance * miter_scale

    return offset_pts


def _outward_normal(edge: np.ndarray) -> np.ndarray:
    """CCWポリゴンの辺ベクトルから外向き法線(右手系で辺を時計回りに90度
    回転させた向き)を返す。"""
    length = np.linalg.norm(edge)
    if length < _EPS:
        return np.zeros(2)
    direction = edge / length
    return np.array([direction[1], -direction[0]])


def _dedup_short_edges(pts: np.ndarray, min_edge_len: float) -> np.ndarray:
    if len(pts) < 4:
        return pts
    kept = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - kept[-1]) >= min_edge_len:
            kept.append(p)
    if len(kept) >= 2 and np.linalg.norm(kept[0] - kept[-1]) < min_edge_len:
        kept.pop()
    if len(kept) < 3:
        return pts
    return np.array(kept)


def _polygon_bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


# --------------------------------------------------------------------------
# シェルフパッキング
# --------------------------------------------------------------------------
def _shelf_pack(
    boxes: list[tuple[float, float]], margin: float = _PACK_MARGIN_MM, max_width: float = 900.0
) -> tuple[list[tuple[float, float]], float, float]:
    """(width, height) のリストをシェルフ法でパッキングし、各アイテムの
    左下オフセット(x, y)のリストと全体のサイズ(width, height)を返す。
    """
    x_cursor = margin
    y_cursor = margin
    shelf_height = 0.0
    offsets: list[tuple[float, float]] = []
    total_width = margin
    total_height = margin

    for w, h in boxes:
        if x_cursor + w + margin > max_width and x_cursor > margin:
            # 次の行へ
            x_cursor = margin
            y_cursor += shelf_height + margin
            shelf_height = 0.0
        offsets.append((x_cursor, y_cursor))
        x_cursor += w + margin
        shelf_height = max(shelf_height, h)
        total_width = max(total_width, x_cursor)
        total_height = max(total_height, y_cursor + shelf_height + margin)

    return offsets, total_width, total_height


# --------------------------------------------------------------------------
# シーム(隣接パネル境界)対応の検出
# --------------------------------------------------------------------------
def _detect_seams(panels: list[dict]) -> list[dict]:
    """パネル間で3D座標が近い境界点同士を突き合わせ、シーム(隣接境界)を
    検出する。

    Returns:
        `{"seam_id": int, "panel_a": panel_id, "panel_b": panel_id,
          "points_a": (N,2) ndarray, "points_b": (N,2) ndarray}` のリスト。
        `points_a`/`points_b` は3D位置で対応付けられた同数の2D座標列
        (それぞれのパネルの `boundary_loop_2d` 上の座標)。
    """
    valid_panels = [p for p in panels if not p.get("flatten_failed")]
    seams = []
    seam_id = 0

    for a_idx in range(len(valid_panels)):
        for b_idx in range(a_idx + 1, len(valid_panels)):
            panel_a = valid_panels[a_idx]
            panel_b = valid_panels[b_idx]

            # パーツグループ化時、異なるパーツ間ではシームを検出しない。
            # パーツ分解の切断面(取付口)のリム頂点は隣接パーツ間で3D座標が
            # 一致するため、パーツを跨いだ偽シーム(縫い合わせ指示)が
            # 出てしまう。パーツ間の接合は取付口ラベルで表現する。
            part_a = panel_a.get("part_id")
            part_b = panel_b.get("part_id")
            if part_a is not None and part_b is not None and part_a != part_b:
                continue

            loop_a_idx = panel_a["boundary_loop_indices"]
            loop_b_idx = panel_b["boundary_loop_indices"]
            verts3d_a = panel_a["vertices_3d"][loop_a_idx]
            verts3d_b = panel_b["vertices_3d"][loop_b_idx]

            if len(verts3d_a) == 0 or len(verts3d_b) == 0:
                continue

            # 総当たりで近接点を対応付け(パネル境界は通常小規模なため許容)
            matches_a = []
            matches_b = []
            used_b = set()
            for ia, p3 in enumerate(verts3d_a):
                dists = np.linalg.norm(verts3d_b - p3, axis=1)
                jb = int(np.argmin(dists))
                if dists[jb] <= _SEAM_MATCH_TOLERANCE_MM and jb not in used_b:
                    matches_a.append(ia)
                    matches_b.append(jb)
                    used_b.add(jb)

            if len(matches_a) < 2:
                continue

            uv_a = panel_a["boundary_loop_2d"][matches_a]
            uv_b = panel_b["boundary_loop_2d"][matches_b]

            seams.append(
                {
                    "seam_id": seam_id,
                    "panel_a": panel_a["panel_id"],
                    "panel_b": panel_b["panel_id"],
                    "points_a": uv_a,
                    "points_b": uv_b,
                }
            )
            seam_id += 1

    return seams


def _select_notch_points(points: np.ndarray, n_notches: int) -> np.ndarray:
    """境界点列から等間隔に近いn_notches個の代表点(インデックス)を選ぶ。"""
    n = len(points)
    if n == 0:
        return np.array([], dtype=np.int64)
    if n <= n_notches:
        return np.arange(n)
    idx = np.linspace(0, n - 1, n_notches).astype(np.int64)
    return np.unique(idx)


def _notch_direction(loop_points: np.ndarray, idx: int) -> np.ndarray:
    """境界ループ上の点における外向き法線方向(合印チックの向き)を返す。"""
    n = len(loop_points)
    prev_pt = loop_points[(idx - 1) % n]
    next_pt = loop_points[(idx + 1) % n]
    edge = next_pt - prev_pt
    normal = _outward_normal(edge)
    norm_len = np.linalg.norm(normal)
    if norm_len < _EPS:
        return np.array([0.0, 1.0])
    return normal / norm_len


# --------------------------------------------------------------------------
# SVG要素組み立て
# --------------------------------------------------------------------------
def _points_to_path(points: np.ndarray, closed: bool = True) -> str:
    if len(points) == 0:
        return ""
    parts = [f"M {points[0][0]:.3f},{points[0][1]:.3f}"]
    for p in points[1:]:
        parts.append(f"L {p[0]:.3f},{p[1]:.3f}")
    if closed:
        parts.append("Z")
    return " ".join(parts)


def _grainline_svg(bbox: tuple[float, float, float, float]) -> str:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    margin = (y1 - y0) * 0.12
    top = y0 + margin
    bottom = y1 - margin
    if bottom <= top:
        top, bottom = y0, y1
    arrow = 3.0
    return (
        f'<g class="grainline" stroke="#1565c0" stroke-width="0.6" fill="none">'
        f'<line x1="{cx:.3f}" y1="{top:.3f}" x2="{cx:.3f}" y2="{bottom:.3f}" />'
        f'<path d="M {cx - arrow:.3f},{top + arrow:.3f} L {cx:.3f},{top:.3f} L {cx + arrow:.3f},{top + arrow:.3f}" />'
        f'<path d="M {cx - arrow:.3f},{bottom - arrow:.3f} L {cx:.3f},{bottom:.3f} L {cx + arrow:.3f},{bottom - arrow:.3f}" />'
        f"</g>"
    )


def _notch_svg(point: np.ndarray, direction: np.ndarray, seam_id: int, length: float = 4.0) -> str:
    p0 = point
    p1 = point + direction * length
    mid = point + direction * (length * 0.55)
    return (
        f'<g class="notch">'
        f'<line x1="{p0[0]:.3f}" y1="{p0[1]:.3f}" x2="{p1[0]:.3f}" y2="{p1[1]:.3f}" '
        f'stroke="#c62828" stroke-width="0.5" />'
        f'<text x="{mid[0]:.3f}" y="{mid[1]:.3f}" font-size="2.6" fill="#c62828" '
        f'text-anchor="middle">{seam_id}</text>'
        f"</g>"
    )


def _attachment_opening_runs(mask: np.ndarray) -> list[list[int]]:
    """境界ループ(循環リスト)上でTrueが連続する区間(取付口)をインデックス
    列のリストとして返す。ループ全体がTrueの場合は1本の全周区間として返す。
    """
    n = len(mask)
    if n == 0 or not np.any(mask):
        return []
    if np.all(mask):
        return [list(range(n))]

    runs: list[list[int]] = []
    current: list[int] = []
    # 開始点をFalseの直後(=Trueの立ち上がり)に揃えるため、最初のFalseの
    # インデックスを探して回転させる。
    first_false = int(np.argmax(~mask))
    order = [(first_false + i) % n for i in range(n)]
    for idx in order:
        if mask[idx]:
            current.append(idx)
        else:
            if current:
                runs.append(current)
                current = []
    if current:
        runs.append(current)
    return runs


def _attachment_opening_svg(loop_shifted: np.ndarray, run_indices: list[int], label: str) -> str:
    """取付口(切断面)の境界区間を太線+ラベルで描画する。"""
    pts = loop_shifted[run_indices]
    if len(pts) == 0:
        return ""
    path = _points_to_path(pts, closed=False)
    mid = pts[len(pts) // 2]
    return (
        f'<g class="attachment-opening">'
        f'<path d="{path}" fill="none" stroke="#e65100" stroke-width="1.4" stroke-linecap="round" />'
        f'<text x="{mid[0]:.3f}" y="{mid[1]:.3f}" font-size="2.8" fill="#e65100" '
        f'text-anchor="middle" dy="-1.5">{_xml_escape(label)}</text>'
        f"</g>"
    )


def _panel_stats_svg(distortion: Optional[dict], x: float, y: float) -> str:
    if not distortion:
        return ""
    over_10pct = distortion.get("edge_length_over_10pct_fraction", 0.0)
    color = "#c62828" if over_10pct > 0.2 else "#37474f"
    text = f"歪み±10%超: {over_10pct * 100:.0f}%"
    return (
        f'<text x="{x:.3f}" y="{y:.3f}" font-size="3" fill="{color}">{text}</text>'
    )


# --------------------------------------------------------------------------
# 公開API
# --------------------------------------------------------------------------
def build_pattern_svg(
    panels_2d: list[dict],
    seam_allowance_mm: float = 7.0,
    label_prefix: str = "P",
    model_name: str = "",
    model_height_mm: float = 0.0,
) -> str:
    """平坦化済みパネルのリストから実寸SVG型紙を組み立てる。

    Args:
        panels_2d: `flatten_panel()` の戻り値に `panel_id` を加えた辞書の
            リスト。`flatten_failed=True` のパネルはスキップし、SVGには
            含めない(呼び出し側が別途警告表示することを想定)。
            以下のキーを追加すると、パーツ単位のグループ化
            (SPEC.md §3.12「2段階構成 (4c)」)が有効になる(省略時は
            全パネルを単一グループとして扱う従来通りの出力):
                - `part_id` (int): パーツID。同じ`part_id`のパネルは
                  SVG上で「部位N」の見出し+凡例色でグループ化される。
                - `part_label` (str): 見出しに使う表示名(省略時は
                  "部位{part_id+1}")。
                - `part_color_hex` (str): 凡例の色見本("#rrggbb")。
                - `panel_no` (int): パーツ内のパネル番号(1始まり)。
                  ラベルは「部位N-P{panel_no}」形式になる(省略時は
                  `panel_id + 1`)。
                - `attachment_mask` ((B,) bool ndarray): `boundary_loop_2d`
                  と同じ順序・長さの真偽値配列。Trueの区間は蓋(取付口/
                  詰め口)として太線+ラベルで強調される。
        seam_allowance_mm: 縫い代幅(mm)。
        label_prefix: パネル番号ラベルの接頭辞("P" → "P1", "P2", ...)。
            パーツグループ化が有効な場合は「部位N-P1」形式になり、
            この引数は無視される。
        model_name: 凡例に表示するモデル名(任意)。
        model_height_mm: 凡例に表示するモデル高さ(mm、任意)。

    Returns:
        SVG文字列(viewBoxはmm単位、width/heightに `mm` 単位を明記)。
    """
    valid_panels = [p for p in panels_2d if not p.get("flatten_failed")]
    has_parts = any(p.get("part_id") is not None for p in valid_panels)

    seams = _detect_seams(valid_panels)
    # パネルごとのシームID一覧(凡例テーブル用)
    panel_seam_ids: dict[int, list[int]] = {}
    for seam in seams:
        panel_seam_ids.setdefault(seam["panel_a"], []).append(seam["seam_id"])
        panel_seam_ids.setdefault(seam["panel_b"], []).append(seam["seam_id"])

    # 各パネルの境界ループ(縫い線)+縫い代線を準備し、パッキング用の
    # バウンディングボックスサイズを求める。
    prepared = []
    for panel in valid_panels:
        loop_raw = np.asarray(panel["boundary_loop_2d"], dtype=np.float64)
        attachment_mask_raw = panel.get("attachment_mask")
        if attachment_mask_raw is not None:
            attachment_mask_raw = np.asarray(attachment_mask_raw, dtype=bool)
        loop, attachment_mask = _ensure_ccw_with_mask(loop_raw, attachment_mask_raw)

        offset_loop = _offset_polygon(loop, seam_allowance_mm)
        bbox_seam = _polygon_bbox(offset_loop)
        width = bbox_seam[2] - bbox_seam[0]
        height = bbox_seam[3] - bbox_seam[1]
        prepared.append(
            {
                "panel_id": panel["panel_id"],
                "panel_no": panel.get("panel_no"),
                "part_id": panel.get("part_id"),
                "part_label": panel.get("part_label"),
                "part_color_hex": panel.get("part_color_hex"),
                "loop": loop,
                "attachment_mask": attachment_mask,
                "offset_loop": offset_loop,
                "bbox_seam": bbox_seam,
                "width": width,
                "height": height,
                "distortion": panel.get("distortion"),
            }
        )

    boxes = [(p["width"], p["height"]) for p in prepared]
    offsets, total_width, total_height = _shelf_pack(boxes)

    body_parts: list[str] = []
    part_legend: dict[int, dict] = {}

    for panel, (ox, oy) in zip(prepared, offsets):
        bx0, by0, _, _ = panel["bbox_seam"]
        shift = np.array([ox - bx0, oy - by0])

        loop_shifted = panel["loop"] + shift
        offset_shifted = panel["offset_loop"] + shift
        bbox_shifted = (
            bx0 + shift[0],
            by0 + shift[1],
            panel["bbox_seam"][2] + shift[0],
            panel["bbox_seam"][3] + shift[1],
        )

        panel_id = panel["panel_id"]
        part_id = panel["part_id"]
        if has_parts and part_id is not None:
            part_label = panel.get("part_label") or f"部位{int(part_id) + 1}"
            panel_no = panel.get("panel_no") or (panel_id + 1)
            label = f"{part_label}-P{panel_no}"
            part_legend.setdefault(
                int(part_id), {"label": part_label, "color": panel.get("part_color_hex")}
            )
        else:
            label = f"{label_prefix}{panel_id + 1}"

        group_attrs = f'data-panel-id="{panel_id}"'
        if part_id is not None:
            group_attrs += f' data-part-id="{part_id}"'
        group_parts = [f'<g class="panel" {group_attrs}>']
        group_parts.append(
            f'<path d="{_points_to_path(offset_shifted)}" fill="none" '
            f'stroke="#455a64" stroke-width="0.4" stroke-dasharray="2,1.5" />'
        )
        group_parts.append(
            f'<path d="{_points_to_path(loop_shifted)}" fill="none" '
            f'stroke="#212121" stroke-width="0.5" />'
        )
        group_parts.append(_grainline_svg(_polygon_bbox(loop_shifted)))

        label_x = (bbox_shifted[0] + bbox_shifted[2]) / 2.0
        label_y = bbox_shifted[1] + 5.0
        group_parts.append(
            f'<text x="{label_x:.3f}" y="{label_y:.3f}" font-size="5" '
            f'font-weight="bold" text-anchor="middle" fill="#000000">{_xml_escape(label)}</text>'
        )
        group_parts.append(_panel_stats_svg(panel["distortion"], label_x, label_y + 4.5))

        seam_note = ",".join(f"S{sid}" for sid in sorted(set(panel_seam_ids.get(panel_id, []))))
        if seam_note:
            group_parts.append(
                f'<text x="{label_x:.3f}" y="{bbox_shifted[3] - 2.0:.3f}" font-size="2.6" '
                f'text-anchor="middle" fill="#546e7a">seam: {seam_note}</text>'
            )

        # 取付口(パーツの切断面/詰め口): 太線+ラベルで強調する
        attachment_mask = panel["attachment_mask"]
        if attachment_mask is not None and len(attachment_mask) == len(loop_shifted):
            runs = _attachment_opening_runs(attachment_mask)
            for run in runs:
                group_parts.append(_attachment_opening_svg(loop_shifted, run, "取付口"))

        group_parts.append("</g>")
        panel["_render_shift"] = shift
        panel["_loop_shifted"] = loop_shifted
        body_parts.append("".join(group_parts))

    # 合印(ノッチ): シームごとに両パネルの対応点へ、対応する平行移動を
    # 適用した座標へ描画する。
    shift_by_panel = {p["panel_id"]: p["_render_shift"] for p in prepared}
    loop_by_panel = {p["panel_id"]: p["_loop_shifted"] for p in prepared}

    for seam in seams:
        pa = seam["panel_a"]
        pb = seam["panel_b"]
        if pa not in shift_by_panel or pb not in shift_by_panel:
            continue
        n_notches = min(4, max(2, len(seam["points_a"]) // 6 + 2))
        idx_sel = _select_notch_points(seam["points_a"], n_notches)

        loop_a_shifted = loop_by_panel[pa]
        loop_b_shifted = loop_by_panel[pb]

        for i in idx_sel:
            pt_a = seam["points_a"][i] + shift_by_panel[pa]
            pt_b = seam["points_b"][i] + shift_by_panel[pb]

            dir_a = _notch_direction_from_shifted(loop_a_shifted, pt_a)
            dir_b = _notch_direction_from_shifted(loop_b_shifted, pt_b)

            body_parts.append(_notch_svg(pt_a, dir_a, seam["seam_id"]))
            body_parts.append(_notch_svg(pt_b, dir_b, seam["seam_id"]))

    # 凡例
    legend_x = _PACK_MARGIN_MM
    legend_y = total_height + 6.0
    legend_lines = [
        f"モデル: {model_name}" if model_name else "モデル: (未指定)",
        f"高さ: {model_height_mm:.1f} mm" if model_height_mm else "高さ: -",
        f"縫い代: {seam_allowance_mm:.1f} mm",
        f"パネル数: {len(valid_panels)} / シーム数: {len(seams)}",
    ]
    if len(valid_panels) < len(panels_2d):
        n_failed = len(panels_2d) - len(valid_panels)
        legend_lines.append(f"※平坦化失敗パネル: {n_failed}(型紙に含まれません)")

    legend_svg_parts = ['<g class="legend" font-size="3" fill="#212121">']
    for i, line in enumerate(legend_lines):
        legend_svg_parts.append(
            f'<text x="{legend_x:.3f}" y="{legend_y + i * 4.0:.3f}">{_xml_escape(line)}</text>'
        )
    legend_svg_parts.append("</g>")
    body_parts.append("".join(legend_svg_parts))

    legend_bottom_y = legend_y + len(legend_lines) * 4.0

    # パーツ凡例(部位名とビューアの色分けとの対応表、SPEC.md §3.12「2段階構成」)
    if has_parts and part_legend:
        part_legend_y = legend_bottom_y + 5.0
        part_legend_svg = ['<g class="part-legend" font-size="3" fill="#212121">']
        part_legend_svg.append(
            f'<text x="{legend_x:.3f}" y="{part_legend_y:.3f}" font-weight="bold">パーツ対応表:</text>'
        )
        for row, (part_id, info) in enumerate(sorted(part_legend.items())):
            row_y = part_legend_y + 4.0 + row * 4.0
            color = info.get("color") or "#9e9e9e"
            part_legend_svg.append(
                f'<rect x="{legend_x:.3f}" y="{row_y - 2.6:.3f}" width="3" height="3" fill="{color}" '
                f'stroke="#212121" stroke-width="0.2" />'
            )
            part_legend_svg.append(
                f'<text x="{legend_x + 5.0:.3f}" y="{row_y:.3f}">{_xml_escape(info["label"])}</text>'
            )
        part_legend_svg.append("</g>")
        body_parts.append("".join(part_legend_svg))
        legend_bottom_y = part_legend_y + 4.0 + len(part_legend) * 4.0

    total_height_with_legend = legend_bottom_y + 6.0

    svg = (
        f'<svg xmlns="{_SVG_NS}" width="{total_width:.3f}mm" height="{total_height_with_legend:.3f}mm" '
        f'viewBox="0 0 {total_width:.3f} {total_height_with_legend:.3f}">'
        f"{''.join(body_parts)}"
        f"</svg>"
    )
    return svg


def _notch_direction_from_shifted(loop_shifted: np.ndarray, point: np.ndarray) -> np.ndarray:
    """シフト後の境界ループ上で `point` に最も近い頂点の外向き法線を返す。"""
    dists = np.linalg.norm(loop_shifted - point, axis=1)
    idx = int(np.argmin(dists))
    return _notch_direction(loop_shifted, idx)


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
