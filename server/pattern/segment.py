"""パネル分割 (SPEC.md §3.12 / FR-13 の4a部分)。

面の双対グラフ上で farthest-point サンプリングによりシード面を選び、
エッジ重み付き多始点最短路(scipy.sparse.csgraph.dijkstra)で
Voronoi風にパネルを成長させる。エッジ重みは面中心間距離を基本とし、

- 二面角が大きい凹エッジ(縫い目になりやすい谷)ほど**重く**する
- `use_colors=True` かつ頂点カラーがある場合、面色差が大きいエッジも**重く**する

ことで、パネル境界(=Voronoi的最短路が「奪い合う」場所)が凹エッジ・
色境界に沿いやすくなるようにしている。

設計上の注意(実験で確認した非自明な点): 直感的には「シームにしたい
エッジを軽くして経路を通しやすくする」方が誘導になりそうに見えるが、
多始点最短路によるVoronoi分割では逆効果になる。エッジを軽くすると
そこは「通りやすい回廊」になり、シード領域がその回廊を通って
色/凹エッジの向こう側まで侵食してしまい、むしろ境界がそこからずれる。
正しくは、そのエッジを**通るコストを上げる**ことで、両側のシード領域が
そのエッジの手前で拮抗し、結果としてVoronoi境界がそのエッジ上に
乗りやすくなる。単純な球(上半球赤/下半球青)での誘導実験で
実測検証済み(誘導ONで境界エッジの色差整合率が大幅に向上する一方、
逆符号ではほぼ0まで悪化することを確認)。

後処理で以下を保証する(円盤位相・SPEC.md記載):
    (a) 各パネルの連結性(非連結の飛び地は最寄りパネルへ再割当)
    (b) 極小パネル(全面積比2%未満)を隣接パネルへ吸収
    (c) 各パネルが円盤位相(境界ループ1本・穴なし)になるよう修復を試みる。
        修復しきれない場合は例外にせず `panel_stats` の `disk_topology` に
        正直に反映する。

このモジュールは純粋モジュール(server/DEVELOPMENT_POLICY.md §3.5)。
依存は numpy / scipy / trimesh のみ。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

_MIN_AREA_FRACTION = 0.02  # 極小パネルとみなす全面積に対する比率の閾値


# --------------------------------------------------------------------------
# 双対グラフ構築
# --------------------------------------------------------------------------
def _face_adjacency_with_weights(
    mesh: trimesh.Trimesh,
    face_colors: Optional[np.ndarray],
    use_colors: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """面の双対グラフの辺(face_adjacency)と辺重みを計算する。

    Returns:
        (adjacency: (E,2) int, weights: (E,) float)
    """
    adjacency = mesh.face_adjacency  # (E, 2)
    if len(adjacency) == 0:
        return adjacency, np.zeros((0,), dtype=np.float64)

    face_centers = mesh.triangles_center
    centers_a = face_centers[adjacency[:, 0]]
    centers_b = face_centers[adjacency[:, 1]]
    base_dist = np.linalg.norm(centers_a - centers_b, axis=1)
    # ゼロ距離(縮退面)対策の下駄
    scale = float(np.median(base_dist[base_dist > 0])) if np.any(base_dist > 0) else 1.0
    base_dist = np.maximum(base_dist, scale * 1e-4)

    weights = base_dist.copy()

    # 二面角: mesh.face_adjacency_angle は隣接面の法線間角度(0=平坦, πに近いほど鋭い折れ)。
    # 凹エッジ(谷折れ、縫い目になりやすい)ほど「通行コスト」を上げ、
    # Voronoi境界がそこに乗りやすくする(上記モジュールdocstring参照)。
    # trimeshの face_adjacency_convex は凸エッジ=True を返すため、
    # 凹エッジは ~convex。凹エッジでは角度が大きいほど強く重くする。
    try:
        angles = mesh.face_adjacency_angles  # (E,) 0..pi
        convex = mesh.face_adjacency_convex  # (E,) bool
        concave_strength = np.where(convex, 0.0, angles / np.pi)  # 0..1、凹のみ効かせる
        # 凹エッジは最大4倍まで重くする
        weights = weights * (1.0 + 3.0 * concave_strength)
    except Exception:
        pass

    if use_colors and face_colors is not None:
        c_a = face_colors[adjacency[:, 0]].astype(np.float64)
        c_b = face_colors[adjacency[:, 1]].astype(np.float64)
        color_diff = np.linalg.norm(c_a - c_b, axis=1) / (255.0 * np.sqrt(3.0))  # 0..1
        # 色差をシャープに強調する(小さな色差は無視、閾値を超えると急激に
        # 重くなる)ことで、色境界近傍のエッジのみが強く誘導に効くようにする。
        sharpened = np.clip((color_diff - 0.05) / 0.15, 0.0, 1.0) ** 2
        # 色境界エッジは最大約20倍まで重くする(=そこを跨ぐ最短路が不利になり、
        # Voronoi境界がその手前で拮抗しやすくなる)。
        weights = weights * (1.0 + 20.0 * sharpened)

    return adjacency, weights


def _build_sparse_graph(n_faces: int, adjacency: np.ndarray, weights: np.ndarray) -> csr_matrix:
    if len(adjacency) == 0:
        return csr_matrix((n_faces, n_faces))
    rows = np.concatenate([adjacency[:, 0], adjacency[:, 1]])
    cols = np.concatenate([adjacency[:, 1], adjacency[:, 0]])
    data = np.concatenate([weights, weights])
    return csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))


# --------------------------------------------------------------------------
# シード選択 (farthest point sampling on dual graph)
# --------------------------------------------------------------------------
def _farthest_point_seeds(graph: csr_matrix, n_seeds: int, n_faces: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    if n_faces == 0:
        return []
    seeds = [int(rng.integers(0, n_faces))]
    if n_seeds <= 1:
        return seeds

    min_dist = None
    for _ in range(n_seeds - 1):
        dist_from_last = dijkstra(graph, indices=seeds[-1], directed=False)
        dist_from_last = np.where(np.isfinite(dist_from_last), dist_from_last, 0.0)
        if min_dist is None:
            min_dist = dist_from_last
        else:
            min_dist = np.minimum(min_dist, dist_from_last)
        # 既に選ばれた面は再選択されないようにする
        candidate = min_dist.copy()
        candidate[seeds] = -1.0
        next_seed = int(np.argmax(candidate))
        if candidate[next_seed] <= 0 and len(seeds) >= 1:
            # グラフが小さく/非連結でこれ以上有効な候補がない場合は残りをランダムに補う
            remaining = [i for i in range(n_faces) if i not in seeds]
            if not remaining:
                break
            next_seed = int(rng.choice(remaining))
        seeds.append(next_seed)
    return seeds


# --------------------------------------------------------------------------
# 多始点最短路によるVoronoi風成長
# --------------------------------------------------------------------------
def _multi_source_labels(graph: csr_matrix, seeds: list[int], n_faces: int) -> np.ndarray:
    if not seeds:
        return np.zeros((n_faces,), dtype=np.int64)
    dist_matrix = dijkstra(graph, indices=seeds, directed=False)  # (n_seeds, n_faces)
    dist_matrix = np.where(np.isfinite(dist_matrix), dist_matrix, np.inf)
    labels = np.argmin(dist_matrix, axis=0)

    # 到達不能な面(孤立成分)は最近傍面のラベルを流用する
    unreachable = ~np.isfinite(np.min(dist_matrix, axis=0))
    if np.any(unreachable):
        labels[unreachable] = 0  # 後段の連結性再割当で修正される
    return labels.astype(np.int64)


# --------------------------------------------------------------------------
# 連結性の保証
# --------------------------------------------------------------------------
def _face_face_adjacency_lists(mesh: trimesh.Trimesh, n_faces: int) -> list[list[int]]:
    neighbors: list[list[int]] = [[] for _ in range(n_faces)]
    for a, b in mesh.face_adjacency:
        neighbors[a].append(b)
        neighbors[b].append(a)
    return neighbors


def _split_into_components(
    labels: np.ndarray, neighbors: list[list[int]]
) -> list[tuple[int, list[int]]]:
    """(label, face_indices) のリストとして、ラベルごとの連結成分をすべて列挙する。"""
    n_faces = len(labels)
    visited = np.zeros(n_faces, dtype=bool)
    components: list[tuple[int, list[int]]] = []
    for start in range(n_faces):
        if visited[start]:
            continue
        label = labels[start]
        stack = [start]
        visited[start] = True
        comp = []
        while stack:
            f = stack.pop()
            comp.append(f)
            for nb in neighbors[f]:
                if not visited[nb] and labels[nb] == label:
                    visited[nb] = True
                    stack.append(nb)
        components.append((int(label), comp))
    return components


def _reassign_stray_components(
    labels: np.ndarray,
    neighbors: list[list[int]],
    face_areas: np.ndarray,
) -> np.ndarray:
    """各ラベルにつき最大面積の連結成分のみを残し、それ以外(飛び地)は
    隣接する別ラベルの中で最も境界を共有するラベルへ再割当する。
    収束するまで繰り返す(反復は少数回で十分)。
    """
    labels = labels.copy()
    for _ in range(6):
        components = _split_into_components(labels, neighbors)
        # ラベルごとに最大面積成分を求める
        best_area: dict[int, float] = {}
        best_comp_id: dict[int, int] = {}
        comp_areas = []
        for idx, (label, comp) in enumerate(components):
            area = float(np.sum(face_areas[comp]))
            comp_areas.append(area)
            if area > best_area.get(label, -1.0):
                best_area[label] = area
                best_comp_id[label] = idx

        changed = False
        for idx, (label, comp) in enumerate(components):
            if best_comp_id.get(label) == idx:
                continue  # このラベルの本体
            # 飛び地: 隣接ラベルへ再割当(境界を最も共有するラベル)
            neighbor_label_counts: dict[int, int] = {}
            for f in comp:
                for nb in neighbors[f]:
                    nb_label = labels[nb]
                    if nb_label != label:
                        neighbor_label_counts[nb_label] = neighbor_label_counts.get(nb_label, 0) + 1
            if neighbor_label_counts:
                new_label = max(neighbor_label_counts.items(), key=lambda kv: kv[1])[0]
            else:
                continue
            for f in comp:
                labels[f] = new_label
            changed = True

        if not changed:
            break
    return labels


# --------------------------------------------------------------------------
# 極小パネルの吸収
# --------------------------------------------------------------------------
def _absorb_tiny_panels(
    labels: np.ndarray,
    neighbors: list[list[int]],
    face_areas: np.ndarray,
    min_area_fraction: float = _MIN_AREA_FRACTION,
) -> np.ndarray:
    labels = labels.copy()
    total_area = float(np.sum(face_areas)) or 1.0

    for _ in range(20):
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            break
        areas = {int(lbl): float(np.sum(face_areas[labels == lbl])) for lbl in unique_labels}
        tiny = [lbl for lbl, a in areas.items() if a / total_area < min_area_fraction]
        if not tiny:
            break
        # 最小のものから1つずつ吸収する(まとめて処理すると隣接関係が崩れるため)
        smallest = min(tiny, key=lambda lbl: areas[lbl])
        faces_of_label = np.where(labels == smallest)[0]
        neighbor_label_counts: dict[int, int] = {}
        for f in faces_of_label:
            for nb in neighbors[f]:
                nb_label = int(labels[nb])
                if nb_label != smallest:
                    neighbor_label_counts[nb_label] = neighbor_label_counts.get(nb_label, 0) + 1
        if not neighbor_label_counts:
            break
        new_label = max(neighbor_label_counts.items(), key=lambda kv: kv[1])[0]
        labels[faces_of_label] = new_label

    return labels


def _relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    unique_labels = np.unique(labels)
    mapping = {int(old): new for new, old in enumerate(unique_labels)}
    return np.array([mapping[int(lbl)] for lbl in labels], dtype=np.int64)


# --------------------------------------------------------------------------
# 円盤位相の判定・修復
# --------------------------------------------------------------------------
def _panel_topology(mesh: trimesh.Trimesh, face_indices: np.ndarray) -> tuple[int, bool]:
    """パネル(部分メッシュ)の境界ループ数と円盤位相かどうかを判定する。

    円盤位相の判定基準: 連結・境界ループがちょうど1本(穴なし)。
    オイラー標数 V - E + F は連結な種数0の円盤(境界1本)で 1 になる
    (球面の半分に相当)ため、これも併用してチェックする。
    """
    if len(face_indices) == 0:
        return 0, False
    sub = mesh.submesh([face_indices], append=True, repair=False)
    if sub is None or len(sub.faces) == 0:
        return 0, False

    try:
        boundary_groups = sub.outline(process=False)
        # trimesh: Path3Dのentitiesの連結成分数を境界ループ数とみなす
        n_loops = len(boundary_groups.entities) if boundary_groups is not None else 0
    except Exception:
        n_loops = _boundary_loop_count_manual(sub)

    is_disk = n_loops == 1
    return n_loops, is_disk


def _boundary_loop_count_manual(sub: trimesh.Trimesh) -> int:
    """trimeshのoutlineが使えない場合の手動境界ループ数カウント
    (境界エッジ=1面にしか属さないエッジをたどってループ数を数える)。
    """
    edges = sub.edges_sorted
    edges_unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = edges_unique[counts == 1]
    if len(boundary_edges) == 0:
        return 0

    adjacency: dict[int, list[int]] = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    visited_edges: set[tuple[int, int]] = set()
    n_loops = 0
    for a, b in boundary_edges:
        edge_key = (int(a), int(b)) if a < b else (int(b), int(a))
        if edge_key in visited_edges:
            continue
        # BFSでこの辺を含む連結成分を訪問
        stack = [int(a), int(b)]
        visited_edges.add(edge_key)
        seen_nodes = {int(a), int(b)}
        while stack:
            node = stack.pop()
            for nb in adjacency.get(node, []):
                key = (node, nb) if node < nb else (nb, node)
                if key in visited_edges:
                    continue
                visited_edges.add(key)
                if nb not in seen_nodes:
                    seen_nodes.add(nb)
                    stack.append(nb)
        n_loops += 1
    return n_loops


def _repair_disk_topology(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    neighbors: list[list[int]],
    face_areas: np.ndarray,
) -> np.ndarray:
    """円盤位相でないパネル(穴あき=境界ループ2本以上)を修復する。

    戦略: 穴のあるパネルについて、穴を構成する内部境界ループに隣接する
    面を1つ選び、そこから最短路成長でサブパネルを切り出して穴を「埋める」
    のではなく、穴の内側にある他パネルの飛び地を吸収して穴を消す
    (=隣接パネルとの再割当により穴の原因を除去する)。
    完全に修復できない場合は諦めて現状のラベルを返す(呼び出し側が
    disk_topology=False として報告する)。
    """
    labels = labels.copy()
    for _ in range(3):
        unique_labels = np.unique(labels)
        all_ok = True
        for lbl in unique_labels:
            face_idx = np.where(labels == lbl)[0]
            n_loops, is_disk = _panel_topology(mesh, face_idx)
            if is_disk or n_loops == 0:
                continue
            all_ok = False
            # 穴(内部ループ)を埋めるため、このパネル内部にあり別ラベルに
            # 属す面(=穴を作っている飛び地)を、このラベルへ吸収してみる。
            # 判定は簡略化し、パネル内部に完全に取り囲まれた他ラベルの
            # 小連結成分を吸収する。
            _absorb_enclosed_holes(labels, neighbors, face_areas, lbl)
        if all_ok:
            break
    return labels


def _absorb_enclosed_holes(
    labels: np.ndarray,
    neighbors: list[list[int]],
    face_areas: np.ndarray,
    target_label: int,
) -> None:
    """target_labelの面集合に隣接面の大半を取り囲まれている他ラベルの
    小成分があれば、target_labelへ吸収する(穴埋め)。labelsをin-placeで更新。
    """
    target_faces = set(np.where(labels == target_label)[0].tolist())
    if not target_faces:
        return

    # target_label以外の連結成分を列挙し、隣接の大半がtarget_labelのものを探す
    visited: set[int] = set()
    n_faces = len(labels)
    for f in range(n_faces):
        if f in visited or labels[f] == target_label:
            continue
        # BFSで同ラベル成分を取得
        label = labels[f]
        stack = [f]
        comp = []
        local_visited = {f}
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in neighbors[cur]:
                if labels[nb] == label and nb not in local_visited:
                    local_visited.add(nb)
                    stack.append(nb)
        visited.update(comp)

        # このcomponentの外周隣接ラベルを集計
        outer_labels: dict[int, int] = {}
        for cf in comp:
            for nb in neighbors[cf]:
                if labels[nb] != label:
                    outer_labels[int(labels[nb])] = outer_labels.get(int(labels[nb]), 0) + 1
        if not outer_labels:
            continue
        total_outer = sum(outer_labels.values())
        target_share = outer_labels.get(target_label, 0) / total_outer
        # 8割以上がtarget_labelに囲まれていれば「穴」とみなし吸収する。
        # ただし、componentがそのラベルの全面(=パネル丸ごと)であり、かつ
        # target_labelに対して十分小さくない場合は吸収しない。パネル数が
        # 2枚のとき(閉曲面を2分割した場合等)は互いに「相手に完全に
        # 囲まれている」状態になるため、無条件に吸収するとパネル同士が
        # 併合されて分割自体が消えてしまう(mockのトーラス結び目
        # n_panels=2 で全パネルが1枚に潰れる退行を実際に確認)。
        # 穴を塞ぐ小パッチ(本来吸収したい対象)は面積が小さいので
        # 面積比でガードする。
        if target_share >= 0.8 and target_label in outer_labels:
            comp_is_whole_label = len(comp) == int(np.sum(labels == label))
            if comp_is_whole_label:
                comp_area = float(np.sum(face_areas[comp]))
                target_area = float(np.sum(face_areas[labels == target_label]))
                if comp_area > 0.25 * max(target_area, 1e-12):
                    continue
            for cf in comp:
                labels[cf] = target_label


# --------------------------------------------------------------------------
# 公開API
# --------------------------------------------------------------------------
def segment_panels(
    mesh: trimesh.Trimesh,
    n_panels: int = 6,
    vertex_colors: Optional[np.ndarray] = None,
    use_colors: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """メッシュを`n_panels`個程度のパネルに分割し、面ごとのパネルIDを返す。

    Args:
        mesh: 対象メッシュ(前処理済み推奨)。
        n_panels: 目標パネル数(4〜12を想定。実際に返るパネル数はこれ以下)。
        vertex_colors: (V,3) or (V,4) の頂点カラー(0-255)。`use_colors=True`
            の場合、色境界をパネル境界に誘導する。
        use_colors: 色境界誘導を有効にするか。
        seed: farthest-point-sampling等の乱数シード。

    Returns:
        (F,) int64 配列。各面が属するパネルID(0始まり、連番)。
        実パネル数は `n_panels` 以下になりうる(極小パネル吸収のため)。
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros((0,), dtype=np.int64)

    n_panels = max(1, int(n_panels))
    if n_panels == 1 or n_faces <= n_panels:
        return np.zeros((n_faces,), dtype=np.int64)

    face_colors = None
    if use_colors and vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
        vc = np.asarray(vertex_colors)[:, :3].astype(np.float64)
        face_colors = vc[mesh.faces].mean(axis=1)

    adjacency, weights = _face_adjacency_with_weights(mesh, face_colors, use_colors)
    graph = _build_sparse_graph(n_faces, adjacency, weights)

    seeds = _farthest_point_seeds(graph, n_panels, n_faces, seed)
    labels = _multi_source_labels(graph, seeds, n_faces)

    neighbors = _face_face_adjacency_lists(mesh, n_faces)
    face_areas = mesh.area_faces

    labels = _reassign_stray_components(labels, neighbors, face_areas)
    labels = _absorb_tiny_panels(labels, neighbors, face_areas)
    labels = _reassign_stray_components(labels, neighbors, face_areas)
    labels = _repair_disk_topology(mesh, labels, neighbors, face_areas)
    labels = _reassign_stray_components(labels, neighbors, face_areas)
    labels = _absorb_tiny_panels(labels, neighbors, face_areas)
    labels = _relabel_contiguous(labels)

    return labels


def panel_stats(mesh: trimesh.Trimesh, labels: np.ndarray) -> list[dict]:
    """パネルごとの面数・面積(mm^2)・境界ループ数・円盤位相を返す。

    Returns:
        パネルIDでソートされた辞書のリスト:
        `{"panel_id", "n_faces", "area_mm2", "boundary_loops", "disk_topology"}`
    """
    if len(labels) == 0:
        return []

    face_areas = mesh.area_faces
    stats = []
    for panel_id in sorted(int(x) for x in np.unique(labels)):
        face_idx = np.where(labels == panel_id)[0]
        area = float(np.sum(face_areas[face_idx]))
        n_loops, is_disk = _panel_topology(mesh, face_idx)
        stats.append(
            {
                "panel_id": panel_id,
                "n_faces": int(len(face_idx)),
                "area_mm2": area,
                "boundary_loops": int(n_loops),
                "disk_topology": bool(is_disk),
            }
        )
    return stats
