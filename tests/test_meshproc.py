"""meshproc.py の単体テスト (IMPLEMENTATION_PLAN.md タスク1-8 (a))。

意図的に穴を開けたメッシュ・浮遊小部品を含むメッシュを入力し、
watertight化・スケーリング・面数上限が正しく適用されることを検証する。
"""
import numpy as np
import pytest
import trimesh

from server import meshproc


def make_broken_box_with_debris():
    """穴あき(非watertight)のボックス + 浮遊する小さいデブリを持つメッシュ。"""
    box = trimesh.creation.box(extents=[10.0, 10.0, 20.0])
    faces = np.delete(box.faces, 0, axis=0)  # 1面を削除して穴を開ける
    broken = trimesh.Trimesh(vertices=box.vertices.copy(), faces=faces, process=False)
    assert not broken.is_watertight

    debris = trimesh.creation.box(extents=[0.05, 0.05, 0.05])
    debris.apply_translation([100, 100, 100])

    combined = trimesh.util.concatenate([broken, debris])
    return combined


def test_process_returns_watertight_mesh():
    mesh = make_broken_box_with_debris()
    processed, stats = meshproc.process(mesh, target_height_mm=100.0, max_faces=200_000)

    assert processed.is_watertight
    assert stats.watertight is True


def test_process_removes_floating_debris():
    mesh = make_broken_box_with_debris()
    processed, stats = meshproc.process(mesh, target_height_mm=100.0, max_faces=200_000)

    # デブリ除去後は単一の連結成分(ボックス由来の8頂点/12面)になるはず
    assert stats.vertices == 8
    assert stats.faces == 12


def test_process_scales_to_target_height():
    mesh = make_broken_box_with_debris()
    processed, stats = meshproc.process(mesh, target_height_mm=100.0, max_faces=200_000)

    height = stats.bbox_mm[2]
    assert height == pytest.approx(100.0, abs=1e-3)

    # 床(z=0)に接地していること
    assert processed.bounds[0][2] == pytest.approx(0.0, abs=1e-6)


def test_process_scales_to_different_height():
    mesh = make_broken_box_with_debris()
    _, stats = meshproc.process(mesh, target_height_mm=50.0, max_faces=200_000)
    assert stats.bbox_mm[2] == pytest.approx(50.0, abs=1e-3)


def test_process_applies_max_faces_limit():
    # 高解像度の球を使い、簡略化が効くことを確認する
    sphere = trimesh.creation.icosphere(subdivisions=5)  # 20480面程度
    assert len(sphere.faces) > 2000

    processed, stats = meshproc.process(sphere, target_height_mm=100.0, max_faces=2000)

    assert stats.faces <= 2000 * 1.2  # 簡略化アルゴリズムの誤差を許容
    assert processed.is_watertight


def test_process_stats_include_expected_keys():
    mesh = trimesh.creation.box(extents=[10, 10, 10])
    _, stats = meshproc.process(mesh, target_height_mm=100.0, max_faces=200_000)
    d = stats.to_dict()
    for key in ("vertices", "faces", "watertight", "bbox_mm", "volume_cm3"):
        assert key in d
    assert d["volume_cm3"] > 0
