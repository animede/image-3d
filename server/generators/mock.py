"""開発・テスト用のmockジェネレータ (SPEC.md §3.3)。

入力画像の内容には依らず、決定的なパラメトリックメッシュを返す。
`params.seed` によって形状のバリエーションが少し変わる。
モデル未ダウンロード環境でもアプリ全体が動作することを保証する
(NFR-5)ためのジェネレータ。
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import trimesh
from PIL import Image

from .base import GenerationParams, Generator


def _seed_to_int(seed) -> int:
    if seed is None:
        return 0
    if isinstance(seed, (int, float)):
        return int(seed)
    # 文字列等は安定したハッシュに変換
    return int(hashlib.sha256(str(seed).encode("utf-8")).hexdigest(), 16) % (2**32)


class MockGenerator(Generator):
    """フィギュアらしい非自明な形状(トーラス結び目の頭部 + カプセル胴体 + 球の腕)を
    決定的に生成するジェネレータ。
    """

    name = "mock"

    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        extra_views: Optional[dict[str, Image.Image]] = None,
    ) -> trimesh.Trimesh:
        # mockジェネレータはマルチビュー入力(extra_views)を無視し、
        # 常に画像内容に依らない決定的な形状を返す(SPEC.md §3.8)。
        rng = np.random.default_rng(_seed_to_int(params.seed))

        # seedにより少しだけ形状パラメータを揺らす(決定的: 同じseedなら同じ結果)
        knot_p = 2 + int(rng.integers(0, 2))  # 2 or 3
        knot_q = 3 + int(rng.integers(0, 3))  # 3,4,5
        radius_scale = 0.8 + float(rng.random()) * 0.4  # 0.8-1.2

        # --- 頭部: トーラス結び目(非自明な形状の代表) ---------------------
        head = trimesh.creation.torus(
            major_radius=1.0 * radius_scale,
            minor_radius=0.35 * radius_scale,
            major_sections=48,
            minor_sections=16,
        )
        # トーラス結び目のねじれを表現するため頂点を変形
        theta = np.arctan2(head.vertices[:, 1], head.vertices[:, 0])
        twist = 0.15 * np.sin(knot_p * theta) * np.cos(knot_q * theta)
        head.vertices[:, 2] += twist
        head.apply_scale(0.6)
        head.apply_translation([0, 0, 2.4])

        # --- 胴体: カプセル ------------------------------------------------
        body = trimesh.creation.capsule(radius=0.55, height=1.6, count=[24, 24])
        body.apply_translation([0, 0, 1.1])

        # --- 腕: 左右の小さいカプセル ---------------------------------------
        arm_l = trimesh.creation.capsule(radius=0.18, height=1.1, count=[12, 12])
        arm_l.apply_transform(
            trimesh.transformations.rotation_matrix(np.radians(85), [1, 0, 0])
        )
        arm_l.apply_translation([0.75, 0.15, 1.5])

        arm_r = arm_l.copy()
        arm_r.apply_translation([-1.5, 0, 0])

        # --- 脚: 左右の円柱 --------------------------------------------------
        leg_l = trimesh.creation.cylinder(radius=0.22, height=1.3, sections=24)
        leg_l.apply_translation([0.3, 0, 0.15])
        leg_r = leg_l.copy()
        leg_r.apply_translation([-0.6, 0, 0])

        # --- 台座: フィギュアらしい円盤ベース ---------------------------------
        base = trimesh.creation.cylinder(radius=1.3, height=0.15, sections=48)
        base.apply_translation([0, 0, -0.55])

        parts = [head, body, arm_l, arm_r, leg_l, leg_r, base]
        mesh = trimesh.util.concatenate(parts)
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
        return mesh
