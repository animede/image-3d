"""3Dプリント向けメッシュ後処理 (SPEC.md §3.4 / FR-4)。

- 非多様体・浮遊小部品(全体積の1%未満の連結成分)の除去
- 穴埋め(watertight化の試行)
- 法線の統一
- 指定サイズへのスケーリング(目標高さmm)+ 床(z=0)への接地
- ポリゴン数の簡略化(fast-simplification)
- 統計情報(vertices/faces/watertight/bbox_mm/volume_cm3)の返却
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh


@dataclass
class MeshStats:
    vertices: int
    faces: int
    watertight: bool
    bbox_mm: list[float]
    volume_cm3: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "vertices": self.vertices,
            "faces": self.faces,
            "watertight": self.watertight,
            "bbox_mm": self.bbox_mm,
            "volume_cm3": self.volume_cm3,
        }


def _remove_small_components(mesh: trimesh.Trimesh, volume_fraction_threshold: float = 0.01) -> trimesh.Trimesh:
    """全体積の`volume_fraction_threshold`未満の連結成分(浮遊小部品)を除去する。"""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh

    volumes = []
    for c in components:
        try:
            v = abs(c.volume) if c.is_watertight else c.convex_hull.volume
        except Exception:
            v = 0.0
        volumes.append(v)

    total = sum(volumes) or 1.0
    kept = [c for c, v in zip(components, volumes) if (v / total) >= volume_fraction_threshold]

    if not kept:
        # すべて小さい場合は最大体積のものだけ残す
        biggest_idx = int(np.argmax(volumes))
        kept = [components[biggest_idx]]

    if len(kept) == 1:
        return kept[0]
    return trimesh.util.concatenate(kept)


def _fill_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """穴埋め(watertight化の試行)。失敗しても例外は投げない。"""
    try:
        mesh.fill_holes()
    except Exception:
        pass
    return mesh


def _unify_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def _scale_to_height(mesh: trimesh.Trimesh, target_height_mm: float) -> trimesh.Trimesh:
    """Z軸方向のバウンディングボックス高さが target_height_mm になるようスケールし、
    床(z=0)に接地させる。
    """
    bounds = mesh.bounds
    extent = bounds[1] - bounds[0]
    height = extent[2]
    if height <= 0:
        height = 1e-6
    scale = target_height_mm / height
    mesh.apply_scale(scale)

    # 再計算して z=0 に接地
    bounds = mesh.bounds
    mesh.apply_translation([0.0, 0.0, -bounds[0][2]])

    # X/Yも中心を原点に寄せる(ビルドプレート中央に置きやすくする)
    bounds = mesh.bounds
    center_xy = (bounds[0][:2] + bounds[1][:2]) / 2.0
    mesh.apply_translation([-center_xy[0], -center_xy[1], 0.0])
    return mesh


def _simplify(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """面数が max_faces を超える場合、fast-simplification で簡略化する。"""
    if max_faces <= 0 or len(mesh.faces) <= max_faces:
        return mesh
    try:
        import fast_simplification

        target_count = max(max_faces, 4)
        new_vertices, new_faces = fast_simplification.simplify(
            mesh.vertices, mesh.faces, target_count=target_count
        )
        simplified = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=True)
        if len(simplified.faces) > 0:
            return simplified
    except Exception:
        pass
    return mesh


def process(
    mesh: trimesh.Trimesh,
    target_height_mm: float = 100.0,
    max_faces: int = 200_000,
) -> tuple[trimesh.Trimesh, MeshStats]:
    """メッシュ後処理パイプライン本体。

    Returns:
        (処理済みメッシュ, 統計情報)
    """
    mesh = mesh.copy()

    # 1. 浮遊小部品除去
    mesh = _remove_small_components(mesh)

    # 2. 穴埋め
    mesh = _fill_holes(mesh)

    # 3. 法線統一
    mesh = _unify_normals(mesh)

    # 4. スケーリング + 接地
    mesh = _scale_to_height(mesh, target_height_mm)

    # 5. 簡略化
    mesh = _simplify(mesh, max_faces)

    # 後処理後にもう一度穴埋め・法線統一(簡略化で崩れることがあるため)
    mesh = _fill_holes(mesh)
    mesh = _unify_normals(mesh)

    bounds = mesh.bounds
    bbox_mm = (bounds[1] - bounds[0]).tolist()
    try:
        volume_mm3 = float(mesh.volume) if mesh.is_volume else 0.0
    except Exception:
        volume_mm3 = 0.0
    volume_cm3 = volume_mm3 / 1000.0

    stats = MeshStats(
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.faces)),
        watertight=bool(mesh.is_watertight),
        bbox_mm=[float(x) for x in bbox_mm],
        volume_cm3=float(volume_cm3),
    )
    return mesh, stats
