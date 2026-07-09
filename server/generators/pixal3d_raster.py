"""drtk (Meta製, MIT) を用いた nvdiffrast 互換ラスタライズシム。

`o_voxel.postprocess.to_glb` (third_party 経由で .venv-pixal3d に導入される
o_voxel パッケージ) は UV アトラスへのテクスチャベイクに `nvdiffrast.torch` の
以下3 APIのみを使用する (postprocess.py 223〜266行付近で確認済み):

  1. ``dr.RasterizeCudaContext()`` — ラスタライザコンテキストの生成
  2. ``dr.rasterize(ctx, uvs_rast, faces_chunk, resolution=[texture_size, texture_size])``
     — UV座標を [-1, 1] 化し z=0/w=1 を付けた「clip座標」での平面ラスタライズ
     (勾配・アンチエイリアシング・深度・パースペクティブは不使用)
  3. ``dr.interpolate(vertices, rast, faces)`` — rast の重心座標 + face ID による
     頂点属性補間

nvdiffrast は NVIDIA Source Code License (非商用研究用途限定) であるため、
本モジュールは同じ呼び出し規約・戻り値規約を `drtk` (MIT, Meta製,
https://github.com/facebookresearch/drtk) で再現し、
`server/generators/pixal3d.py` が `o_voxel.postprocess.dr` をこのモジュールに
差し替えることで、GLB化経路から nvdiffrast への依存を除去する。

## 座標系・規約の対応

nvdiffrast (`dr.rasterize`) の規約:
  - 入力 ``pos`` は clip 座標 (x, y, z, w)。NDC (x/w, y/w) が [-1, 1] の範囲で
    画像に写る。**行方向は OpenGL 式 (row 0 = 画像下端, y が上向き正)**。
  - 出力 ``rast`` は (B, H, W, 4) = (u, v, z/w, float(triangle_id + 1))。
    triangle_id は 0-based、未カバー画素は 4要素すべて 0 (face id 列も 0 =
    「空」)。**実機検証 (one-hot頂点属性を interpolate に通し、出力と
    u/v の相関を確認) で確定**: u は頂点0の重み、v は頂点1の重み
    (頂点2の重みが 1-u-v)。nvdiffrast公式ドキュメントの文言
    (「barycentrics」)だけでは頂点0/1/2のどれがu/vに対応するか曖昧なため、
    本シム実装では実測に基づくこの対応を採用する。
  - `dr.interpolate(attr, rast, tri)` は
    ``a0 + u * (a1 - a0) + v * (a2 - a0)`` ではなく、実際には
    ``(1-u-v)*a2 + u*a0 + v*a1`` (= 上記の頂点対応どおり) を計算する。
    ※ nvdiffrast公式ドキュメントの数式表記はu=頂点1,v=頂点2の対応を
    示唆するが、実機検証の結果はu=頂点0,v=頂点1であったため、本シムは
    実測を正とする。

drtk の規約:
  - `rasterize(v, vi, height, width)` の `v` はピクセル座標系
    (x, y, z)。画像左上が (-0.5, -0.5)、右下が (width-0.5, height-0.5)。
    **行方向は画像式 (row 0 = 画像上端, y が下向き正)** — nvdiffrastとは
    Y軸の向きが逆。出力は index_img (B, H, W) の triangle_id (0-based、
    未カバーは -1)。また、3頂点すべてのzが1e-8より大きくないと
    (near-plane相当のクリップで) 三角形が破棄される点に注意
    (実機検証で確認。z=0では常に空になる)。
  - `render(v, vi, index_img)` が depth_img と bary_img (B, 3, H, W) を返す。
    bary_img のチャンネル 0/1/2 はそれぞれ vi の頂点 0/1/2 の重心重みに対応
    (Meta実装のCUDAカーネル `render_kernel.cu` で確認:
    `bary = {1-bary_12.x-bary_12.y, bary_12.x, bary_12.y}` が vi の列順どおり)。
  - `interpolate(vert_attributes, vi, index_img, bary_img)` は
    ``sum_k bary_img[k] * vert_attributes[vi[..., k]]`` を計算する
    (= a0*bary0 + a1*bary1 + a2*bary2)。

本シムが吸収する差分:
  - **Y軸反転**: nvdiffrastは行0=下、drtkは行0=上。本シムはdrtk呼び出し前に
    NDC の y 成分を反転させ (`y' = -y`)、drtk呼び出し後は出力を上下反転
    (`flip(dims=[H軸])`) することで、シムのI/Oはnvdiffrast規約のまま維持する。
    (o_voxel.postprocess.to_glb はUV空間の正方形テクスチャに対して平面
    ラスタライズを行うだけなので、Y反転は「テクスチャの上下どちらを
    row0とするか」という規約差に過ぎず、シムの入口/出口でまとめて吸収すれば
    幾何学的な整合性は保たれる。)
  - **重心の頂点対応**: nvdiffrastの (u, v) = (頂点0の重み, 頂点1の重み)
    (実測確認、上記参照) は drtkの (bary_img[0], bary_img[1]) と同じ対応関係
    なので、頂点順の入れ替えは不要 (u=bary0, v=bary1 とすればよい)。
  - **z**: drtkは3頂点のzがすべて正であることを要求するため、本アプリの
    z=0/w=1直交投影の使用範囲では、全頂点で同一の正の定数 (1.0) を渡す
    (深度値そのものはo_voxel側で使用されないため、値は何でもよいが、
    全頂点同一であればパースペクティブ補正 (1/z線形補間) がスクリーン空間
    バリセントリックに影響しない)。
  - **face IDのオフセット**: nvdiffrastは1-based (0=空)、drtkは0-based
    (-1=空)。シム内で ``face_id_nv = index_img + 1`` (未カバーは0) に変換する。
  - **z/w列**: 本アプリの使用範囲では z=0/w=1 の直交投影のみなので、
    z/w列は常に0を返せば良い (o_voxel側もこの列を使用していない)。

## 想定外の使用方法

`o_voxel.postprocess.to_glb` が使う範囲 (chunked呼び出し、resolution指定、
z=0直交、grad_db不使用) のみをサポートする。勾配伝播・アンチエイリアシング・
パースペクティブ補正・`ranges`/バッチ次元>1 などは非対応。
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Union

import torch

logger = logging.getLogger(__name__)


class RasterizeCudaContext:
    """`nvdiffrast.torch.RasterizeCudaContext` 互換の no-op コンテキスト。

    drtkはコンテキストオブジェクトを必要としないため、呼び出し規約を保つための
    プレースホルダとして提供する。
    """

    def __init__(self, device: Optional[Union[str, "torch.device"]] = None) -> None:
        self.device = device


def _as_resolution(resolution: Sequence[int]) -> tuple[int, int]:
    height, width = int(resolution[0]), int(resolution[1])
    return height, width


def rasterize(
    glctx: RasterizeCudaContext,
    pos: torch.Tensor,
    tri: torch.Tensor,
    resolution: Sequence[int],
    ranges: Optional[torch.Tensor] = None,
    grad_db: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`nvdiffrast.torch.rasterize` 互換のラスタライズ (drtkバックエンド)。

    Args:
        glctx: `RasterizeCudaContext` (未使用、互換性のためのプレースホルダ)。
        pos: clip座標 (x, y, z, w)。形状 [1, V, 4] (instanced mode) のみ対応。
             本アプリの使用範囲はz=0/w=1の直交投影のみ。
        tri: 三角形の頂点インデックス。形状 [F, 3]、dtypeは整数。
        resolution: (height, width)。
        ranges: 非対応 (Noneのみ)。range modeは使用しない。
        grad_db: 無視される (勾配非対応)。

    Returns:
        nvdiffrast と同じ (rast, rast_db) のタプル。
        rast: [1, H, W, 4] = (u, v, z/w, float(triangle_id + 1))。
        rast_db: 常にゼロ埋めの空テンソル (勾配非対応、o_voxel側は未使用)。
    """
    if ranges is not None:
        raise NotImplementedError(
            "pixal3d_raster.rasterize: range mode (ranges引数) は未対応です。"
        )
    if pos.ndim != 3 or pos.shape[0] != 1:
        raise NotImplementedError(
            "pixal3d_raster.rasterize: instanced mode ([1, V, 4]) のみ対応です "
            f"(got shape {tuple(pos.shape)})。"
        )

    height, width = _as_resolution(resolution)
    device = pos.device

    xy = pos[0, :, :2]  # NDC x, y in [-1, 1] (画像座標ではない)
    x_ndc = xy[:, 0]
    y_ndc = xy[:, 1]

    # nvdiffrast: NDCのy軸は上向き正・row0=画像下端 (OpenGL式)。
    # drtk: ピクセル座標のy軸は下向き正・row0=画像上端 (画像式)。
    # ここでyを反転させることで、drtk側では「上下反転した画像」を作らせ、
    # 後段で出力を上下反転して戻し、シムの入出力規約はnvdiffrast式に統一する。
    y_ndc_flipped = -y_ndc

    # NDC [-1, 1] -> ピクセル座標 (-0.5 .. dim-0.5、ピクセル中心基準)
    px_x = (x_ndc + 1.0) * 0.5 * width - 0.5
    px_y = (y_ndc_flipped + 1.0) * 0.5 * height - 0.5
    # z: drtkのrasterizeカーネルは3頂点すべてのzが1e-8より大きいことを
    # 可視性判定の条件にしている (near-plane相当のクリップ)。本アプリの
    # 使用範囲 (z=0/w=1の直交投影、平面UVラスタライズ) では深度値そのものに
    # 意味はなく、全頂点で同一の正の定数 (1.0) を使えば
    # 1/z補間 (drtkはパースペクティブ補正のためdepthを1/z線形補間する) も
    # 全画素で1.0のまま一定になり、xy平面のバリセントリック計算
    # (screen-space、zに依存しない) には一切影響しない。
    z_cam = torch.ones_like(px_x)

    v_px = torch.stack([px_x, px_y, z_cam], dim=-1).unsqueeze(0).contiguous()  # [1, V, 3]

    import drtk

    vi = tri.to(torch.int32)
    index_img = drtk.rasterize(v_px, vi, height, width)  # [1, H, W] (0-based, -1=空)
    depth_img, bary_img = drtk.render(v_px, vi, index_img)  # bary_img: [1, 3, H, W]

    # 上下反転して戻す (drtk row0=上 -> nvdiffrast row0=下 規約に合わせる)
    index_img = torch.flip(index_img, dims=[1])
    bary_img = torch.flip(bary_img, dims=[2])

    mask = index_img >= 0
    # nvdiffrast: u = 頂点0の重み, v = 頂点1の重み (実測確認、モジュールdocstring参照)
    u = torch.where(mask, bary_img[:, 0], torch.zeros_like(bary_img[:, 0]))
    v = torch.where(mask, bary_img[:, 1], torch.zeros_like(bary_img[:, 1]))
    zw = torch.zeros_like(u)  # z=0/w=1固定の使用範囲では常に0
    face_id_nv = torch.where(
        mask, (index_img + 1).to(torch.float32), torch.zeros_like(u)
    )

    rast = torch.stack([u, v, zw, face_id_nv], dim=-1)  # [1, H, W, 4]
    rast_db = torch.zeros(
        (rast.shape[0], rast.shape[1], rast.shape[2], 0),
        device=device,
        dtype=rast.dtype,
    )
    return rast, rast_db


def interpolate(
    attr: torch.Tensor,
    rast: torch.Tensor,
    tri: torch.Tensor,
    rast_db: Optional[torch.Tensor] = None,
    diff_attrs: Optional[Union[str, List[int]]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`nvdiffrast.torch.interpolate` 互換の頂点属性補間 (drtkバックエンド)。

    Args:
        attr: 頂点属性。形状 [1, V, C]。
        rast: `rasterize` が返した [1, H, W, 4] = (u, v, z/w, face_id+1)。
        tri: 三角形の頂点インデックス。形状 [F, 3]。
        rast_db: 非対応 (勾配用、無視する)。
        diff_attrs: 非対応 (勾配用、無視する)。

    Returns:
        (out, out_db) のタプル。out: [1, H, W, C] (nvdiffrastと同じ
        ``u*a0 + v*a1 + (1-u-v)*a2`` を計算。モジュールdocstring参照)。
        out_db は常に空テンソル。
    """
    if attr.ndim != 3 or attr.shape[0] != 1:
        raise NotImplementedError(
            "pixal3d_raster.interpolate: attr は [1, V, C] 形状のみ対応です "
            f"(got shape {tuple(attr.shape)})。"
        )

    u = rast[..., 0]
    v = rast[..., 1]
    face_id_nv = rast[..., 3]
    mask = face_id_nv > 0
    # drtk.interpolate は index_img に int32 を要求し、未カバー画素は -1 とする規約
    # (drtk.rasterize の戻り値と同じ規約)。
    index_img = torch.where(
        mask,
        (face_id_nv - 1).to(torch.int32),
        torch.full_like(face_id_nv, -1, dtype=torch.int32),
    )

    # nvdiffrastの (u, v) = (頂点0の重み, 頂点1の重み) (実測確認、モジュール
    # docstring参照) を drtkの (bary_img[0], bary_img[1], bary_img[2]) 規約
    # (頂点0/1/2の重みの順) に変換する。
    bary2 = 1.0 - u - v
    bary_img = torch.stack([u, v, bary2], dim=1)  # [1, 3, H, W]
    # 未カバー画素はbary=0にしておく (drtk.interpolateは未定義値を返すため
    # 呼び出し側 (o_voxel) はrastのface_id列でマスクしてから使用する)。
    bary_img = torch.where(mask.unsqueeze(1), bary_img, torch.zeros_like(bary_img))

    import drtk

    vi = tri.to(torch.int32)
    out = drtk.interpolate(attr, vi, index_img, bary_img)  # [1, C, H, W]
    out = out.permute(0, 2, 3, 1).contiguous()  # [1, H, W, C] (nvdiffrast互換)

    out_db = torch.zeros(
        (out.shape[0], out.shape[1], out.shape[2], 0),
        device=out.device,
        dtype=out.dtype,
    )
    return out, out_db


def is_available() -> bool:
    """drtk (CUDA拡張ビルド込み) がimport可能かどうか。"""
    try:
        import drtk  # noqa: F401
    except ImportError:
        return False
    return True
