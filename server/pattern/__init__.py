"""ぬいぐるみ型紙生成モジュール (SPEC.md §3.12 / FR-13)。

将来の独立リポジトリ化に備えた純粋モジュール。
`server/` 内の他モジュール(config, jobs, generators, colorproc等)を
一切importしない。入出力は `trimesh.Trimesh` / 標準型 + numpy のみ。
依存は numpy / scipy / trimesh に限定する(server/DEVELOPMENT_POLICY.md §3.5)。
"""
from .preprocess import prepare_mesh
from .segment import panel_stats, segment_panels
from .preview import build_preview_mesh
from .flatten import flatten_panel
from .svg import build_pattern_svg
from .parts import (
    cap_part,
    compute_local_thickness,
    decompose_parts,
    extract_image_regions,
    labels_from_bboxes,
    labels_from_seeds,
    part_stats,
    project_labels_to_faces,
    segment_part_panels,
)

__all__ = [
    "prepare_mesh",
    "segment_panels",
    "panel_stats",
    "build_preview_mesh",
    "flatten_panel",
    "build_pattern_svg",
    "decompose_parts",
    "cap_part",
    "part_stats",
    "compute_local_thickness",
    "segment_part_panels",
    "extract_image_regions",
    "labels_from_bboxes",
    "labels_from_seeds",
    "project_labels_to_faces",
]
