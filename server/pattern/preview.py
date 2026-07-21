"""パネル色分けプレビューメッシュ生成 (SPEC.md §3.12 / FR-13)。

このモジュールは純粋モジュール(server/DEVELOPMENT_POLICY.md §3.5)。
依存は numpy / trimesh のみ。
"""
from __future__ import annotations

import numpy as np
import trimesh

# 隣接パネルが似た色にならないよう、色相を大きく飛ばして並べた固定パレット
# (20色)。パネルID/パーツIDを巡回インデックスとして使う。暗色(olive以降)は
# ダーク背景で視認しにくいため末尾に置き、パーツ数が多いときのみ使われる。
# 公開定数: アダプタ・SVG凡例がビューアの色分けと対応を取るために参照する。
PALETTE_HEX = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
    "#469990",  # teal
    "#9a6324",  # brown
    "#ffe119",  # yellow
    "#dcbeff",  # lavender
    "#aaffc3",  # mint
    "#ffd8b1",  # apricot
    "#fffac8",  # beige
    "#a9a9a9",  # grey
    "#808000",  # olive
    "#800000",  # maroon
    "#000075",  # navy
]

# 後方互換エイリアス(既存の内部参照用)
_PALETTE_HEX = PALETTE_HEX


def _hex_to_rgba(hex_color: str) -> list[int]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return [r, g, b, 255]


_PALETTE_RGBA = np.array([_hex_to_rgba(h) for h in _PALETTE_HEX], dtype=np.uint8)


def build_preview_mesh(mesh: trimesh.Trimesh, labels: np.ndarray) -> trimesh.Trimesh:
    """パネルIDごとに固定パレットで面カラーを塗ったプレビュー用メッシュを返す。

    Args:
        mesh: 対象メッシュ。
        labels: `segment_panels` が返す (F,) 配列。

    Returns:
        面カラー(FaceColor)を持つ新しい `trimesh.Trimesh`(コピー)。
        頂点数・面数・トポロジーは元メッシュと同一。
    """
    preview = mesh.copy()
    n_faces = len(preview.faces)
    if n_faces == 0:
        return preview

    labels = np.asarray(labels)
    face_colors = _PALETTE_RGBA[labels % len(_PALETTE_RGBA)]
    preview.visual = trimesh.visual.ColorVisuals(mesh=preview, face_colors=face_colors)
    return preview
