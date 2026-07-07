"""Generator抽象基底クラス。

DEVELOPMENT_POLICY.md の方針通り、ジェネレータはプラガブルにする。
`hunyuan3d`(本番用)と `mock`(開発・テスト用)が本インターフェースを実装する。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import trimesh
from PIL import Image


@dataclass
class GenerationParams:
    """SPEC.md §3.3 / §5 のジョブパラメータ。"""

    steps: int = 30
    guidance_scale: float = 5.5
    octree_resolution: int = 384
    seed: Optional[int] = None
    remove_bg: bool = True
    target_height_mm: float = 100.0
    max_faces: int = 200_000
    color_mode: str = "none"
    n_colors: int = 4
    texture_mode: str = "none"
    extra: dict[str, Any] = field(default_factory=dict)


class Generator(ABC):
    """Image-to-3D ジェネレータの抽象基底。"""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        params: GenerationParams,
        extra_views: Optional[dict[str, Image.Image]] = None,
    ) -> trimesh.Trimesh:
        """画像から3Dメッシュ(trimesh.Trimesh)を生成する。

        Args:
            image: 正面(front)画像。単一ビュー生成時の唯一の入力。
            params: 生成パラメータ。
            extra_views: マルチビュー入力時の追加ビュー
                (キーは "back" / "left" / "right"、front自体は含まない)。
                SPEC.md §3.8 (FR-9)。Noneまたは空辞書の場合は単一ビュー生成
                (従来通り)を行う。マルチビュー非対応のジェネレータ
                (mock等)はこの引数を無視してよい。

        後処理(watertight化・スケーリング・簡略化)は呼び出し側(meshproc.process)
        が担当するため、ここでは生成のみを行う。
        """
        raise NotImplementedError
