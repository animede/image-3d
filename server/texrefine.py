"""texgen のテクスチャを元画像の全解像度で上書きする (SPEC.md §3.9)。

texgen(Hunyuan3D-2 paint)のテクスチャは、参照画像がどれだけ精細でも
**512px の情報しか運べない**:

    参照 1024 → delight で 512 (dehighlight_utils.py:70)
             → マルチビュー生成 512 (multiview_utils.py:28 `view_size = 512`)
             → ×4 に単純拡大 (pipelines.py:226。超解像は上流でコメントアウト)
             → 2048 のアトラスへベイク

体全体を 512 に詰めるので顔は実質 150px 程度になり、毛並みやステッチのような
細部は原理的に残らない(実測: 帽子の赤が (175,50,70)→(216,1,20) と平坦化、
腕の毛は色こそ合うが質感が消える)。

一方で正面・背面の参照画像は元の解像度でそのまま手元にある。そこで
**カメラを正対して向いているテクセルだけ**を参照画像から直接サンプリングし直し、
texgen の結果へ上書きする。texgen は「参照が無い/浅い角度でしか見えない範囲」の
担保に回る。

視線に対し浅い角度の面を上書きしないのが肝で、そこはフチ画素を引き伸ばして
サンプリングしてしまい黒い筋の原因になる(colorproc の
`_LOW_CONFIDENCE_DOT_THRESHOLD` と同じ理由)。境目が出ないよう、法線の向きに
応じて texgen 側へなめらかに戻す。

もう一つの肝が**可視性**。直交投影は (x,z) しか見ないため、帽子の垂れや耳の
裏に隠れた面も「正面を向いてさえいれば」画像を拾ってしまい、そこを覆っている
帽子の赤や耳の黒が顔の縁に筋として写る(実測: 遮蔽された正面向き頂点の52%が
赤/黒を拾っていた)。表面サンプル自身からビューごとの深度バッファを作り、
最前面から `DEPTH_EPS_FRACTION` より奥のサンプルは参照から塗らず texgen に
任せる。これにより「最悪でも texgen、可視なら精細」という下限が保証される。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt

from . import colorproc

logger = logging.getLogger(__name__)

# 参照画像を全面的に信用する法線しきい値(視線との内積)。
# 0.50 ≒ 視線から60°以内。可視性テスト導入前は 0.75 だったが、遮蔽画素を
# 拾う心配が無くなったので広げた。この帯を狭くすると、texgen と半々に
# 混ざる帯域が広がり、texgen 側の目の輪郭が参照の目の下に「二重写し」で
# 残る(実ジョブ46b64850で確認)。
FULL_CONFIDENCE_DOT = 0.50

# 上書きを完全にやめる法線しきい値。0.20 ≒ 視線から78°。可視性テストと
# フチ除外(trusted mask)導入前は 0.35 だったが、両ガードが入った後の実測
# (c78dac23 の生アトラスで 0.35 と 0.20 を比較)では、0.20 の方が帽子の
# つば下の texgen ノイズ帯を参照で置き換えられ、側面の引き伸ばしは
# 増えなかった。ここから上へ向けて線形に重みを上げ、texgen との境目が
# 線にならないようにする。浅い角度では1画素が表面上で大きく引き伸ばされる
# (テクセル密度の問題)ので、下限自体は残す。
MIN_CONFIDENCE_DOT = 0.20

# 面あたりのサンプル数の目安(テクセル面積の何倍を撒くか)。
# ランダムなバリセントリックサンプリングなので取りこぼしが出るが、
# 残った穴は `HOLE_FILL_RADIUS_TEXELS` で埋める。少なすぎると遮蔽境界で
# テクセルごとの可視率の分散が大きくなり、ごま塩状のノイズになる。
SAMPLES_PER_TEXEL = 6.0

# 1面あたりのサンプル数の上限(極端に大きい面でメモリが跳ねるのを防ぐ)。
MAX_SAMPLES_PER_FACE = 4096

# メモリを一定に保つための面のチャンクサイズ。
FACE_CHUNK = 20000

# サンプルの取りこぼしと UV チャート外周(ガター)を埋める距離。
# ここを埋めておかないと、バイリニア補間がチャートの外から texgen の色を
# 拾って細い継ぎ目になる。
HOLE_FILL_RADIUS_TEXELS = 4

# --- 棄却領域の調和拡散(遮蔽された顔の白抜け対策) --------------------------
# 前髪のように顔へ覆いかぶさる形状では、その裏の面はどのビューからも
# 「遮蔽 or 浅い角度」で棄却され、texgen に戻る。アニメ顔では texgen が
# のっぺらぼう(白い平坦面)を返すため、目の周りに白い欠損として現れる
# (実ジョブ cb075cab)。棄却されたテクセルを、同じUVチャート内の転写済みの
# 色から調和拡散(ラプラス方程式)で埋める。UVチャートは表面の近傍関係を
# 保つので、これは「近くの肌・髪の色をなじませる」ことと等価
# (2ビュー時代に頂点カラーで実証済みの手法のテクセル版)。
#
# texgen に本物の描き込みがある場所(犬の帽子の垂れの裏など)を潰さないよう、
# **texgen が局所的に平坦(情報なし)なテクセルに限定**する。
REJECT_FILL_FLAT_WINDOW = 7  # texgen の平坦さを測る窓(テクセル)
REJECT_FILL_FLAT_STD = 6.0  # この局所標準偏差未満なら「情報なし」とみなす
# 領域境界のうち転写済みテクセルが占める最低割合。低い領域(どのビューからも
# 見えない広大な内側など)は拡散の根拠が乏しいので texgen のまま残す。
REJECT_FILL_MIN_BOUNDARY_COVERAGE = 0.25
# これを超える規模の連立方程式は解かない(異常なアトラスでの暴走防止)。
# アニメ髪は浅い角度の面が多く、実ジョブで73万テクセルが対象になった。
# 解法は共役勾配(下記)なのでこの規模でも数秒で解ける。
REJECT_FILL_MAX_UNKNOWNS = 2_000_000
# 棄却テクセルの点在を面に繋ぐ閉包の反復数。UV面積の大きい面はサンプル上限
# (MAX_SAMPLES_PER_FACE)に当たり、棄却領域が塩粒状に途切れることがある。
# 2反復=4テクセル幅までの隙間を繋ぐ(チャート間ガターはそれより広い前提)。
REJECT_FILL_CLOSING_ITERATIONS = 2

# --- 顔の特徴(目・鼻・口)は texgen に任せる -------------------------------
# 生成メッシュの目・鼻は参照画像と数%ずれる(実測: 目の中心で10〜25px/1024)。
# texgen はメッシュの法線・位置マップに条件付けて描くので**形状に揃っており**、
# 画像基準の直交投影だけだと、目のくぼみと描かれた目がずれて「浮いた」顔になる
# (リグ後の照明で顕著)。
#
# ワープで参照側を寄せる案は2通り試して**両方とも悪化**した:
#   - 密なオプティカルフロー(cv2 DIS): 参照の目の輪郭を texgen の目の輪郭に
#     合わせようとするが、texgen の特徴は512px品質で形・大きさが不正確な
#     ため、「縮んだ目・崩れた鼻」まで転写してしまう。
#   - 暗い塊の重心マッチ+ガウスRBF: 近接する特徴(目と耳の黒パッチ)の変位が
#     食い違い、その勾配が目を横切って形を歪めた(半目のように潰れる)。
#
# 「特徴領域は texgen に任せる」案も試したが、texgen の目は世代によって
# 小さく生気の無い点になり、質感の魅力を落とした。
#
# 採用したのは**特徴の剛体移動(切って貼る)**: 参照側の暗い塊を、対応する
# メッシュ側(合成ビュー)の位置へ**形を変えずに平行移動**した「補正済み参照」
# を作り、それを転写する。塊ごとに一定の変位なので変形が原理的に起きず、
# 参照のくっきりした目がそのままメッシュのくぼみ位置に載る。元の位置は
# 周囲の毛(最近傍の非特徴画素)で埋めて、二重写しを防ぐ。
FEATURE_MAX_LUMINANCE = 95.0  # 「暗い特徴」とみなす輝度上限
FEATURE_MAX_SATURATION = 70.0  # 同・彩度上限(有彩色の服などを除く)
FEATURE_MIN_AREA_PX = 60  # ノイズ除去: これ未満の塊は無視
FEATURE_MAX_AREA_FRACTION = 0.05  # 被写体比これ超の塊(大きな黒い服等)は無視
# 対応とみなす特徴間距離の上限(画像辺比)。正当なずれの実測は1〜2.5%(犬・
# 人型とも)。0.05 だと texgen が顔を描かなかったときに参照の目が4.6%先の
# 髪の影と誤対応した(目が髪へ移動し元位置が肌色で消える)ため、余裕を
# 持たせつつ 3.5% に絞る。
FEATURE_MATCH_MAX_FRACTION = 0.035
FEATURE_MIN_SHIFT_PX = 3.0  # これ未満のずれは移動しない(誤差の範囲)
# 対応とみなすための「同種の特徴らしさ」。texgen が顔を描かない場合(アニメ顔で
# 実際に発生)、参照の目に対応するメッシュ側の特徴が存在せず、貪欲マッチが
# 髪の影の細片と誤対応して**目を髪の中へ移動し、元の位置を肌色で消す**
# (実ジョブ cb075cab の左目白抜けの真因)。面積とアスペクト比が近い塊
# どうししか対応させない。
FEATURE_MATCH_MIN_AREA_RATIO = 0.35  # 面積比(小/大)の下限
FEATURE_MATCH_MAX_ASPECT_RATIO = 2.5  # アスペクト比どうしの比の上限
FEATURE_DILATE_FRACTION = 0.006  # 塊の周囲をこの半径(画像辺比)だけ一緒に運ぶ
FEATURE_FEATHER_FRACTION = 0.004  # 貼り付け境界をぼかす幅(画像辺比)

# --- 焼き込まれた照りの補正(黒締め) ----------------------------------------
# 参照画像は撮影(生成)時の照明ごと写っており、艶のある部位(プラスチックの
# 目・鼻)には反射の照りが焼き込まれている。テクスチャとして貼ると、ビューアの
# ライトがその上にさらに乗り、深い黒であるべき目が茶色いリングに見える
# (texgen 自身は delight 済みなので出にくい。全解像度再投影は元画素を
# そのまま使うため、この副作用だけが残る)。
# 「暗くて彩度が低い」テクセル(目・鼻・肉球の無彩色の照り)だけを黒へ寄せる。
# デニムのような暗い有彩色は彩度ガードで対象外。明るいグレーの服(輝度90超)も
# 対象外だが、木炭色など極端に暗い無彩色の衣装は締まりが強く出る点に注意。
SHADOW_DEEPEN_MAX_LUMINANCE = 90.0
SHADOW_DEEPEN_LUMINANCE_RAMP = 60.0
SHADOW_DEEPEN_MAX_SATURATION = 45.0
SHADOW_DEEPEN_SATURATION_RAMP = 30.0
SHADOW_DEEPEN_STRENGTH = 0.75

# 可視性判定の深度バッファは参照画像をこの係数で縮めた格子で持つ。
# サンプル密度はUVテクセル面積基準なので、画像解像度そのままでは
# 疎な面にピンホール(誤って「可視」になる穴)ができる。粗くするほど
# 密度は上がるが、深度の段差近傍で過剰に遮蔽扱いになる(texgenに戻る
# だけなので安全側)。
DEPTH_GRID_DIVISOR = 2

# 最前面からこの割合(ビュー方向の奥行き全長比)より奥のサンプルは
# 遮蔽とみなす。頂点単位の事前計測(2%)で帽子の垂れ・耳の裏の誤サンプル
# 1080頂点を妥当に検出できた値。
DEPTH_EPS_FRACTION = 0.02


@dataclass
class RefineStats:
    """精細化の結果。ログと job の警告に使う。"""

    applied: bool = False
    views: list[str] = field(default_factory=list)
    texture_size: tuple[int, int] = (0, 0)
    refined_texel_ratio: float = 0.0
    mean_blend_weight: float = 0.0
    occluded_sample_ratio: float = 0.0
    filled_texel_ratio: float = 0.0
    reason: Optional[str] = None


def _extract_texture_image(visual: Any) -> Optional[Image.Image]:
    material = getattr(visual, "material", None)
    if material is not None:
        for attr in ("image", "baseColorTexture"):
            image = getattr(material, attr, None)
            if image is not None:
                return image
    return getattr(visual, "image", None)


def _set_texture_image(visual: Any, image: Image.Image) -> bool:
    """テクスチャ画像を差し替える。SimpleMaterial / PBRMaterial の両方に対応。"""
    material = getattr(visual, "material", None)
    if material is not None:
        if getattr(material, "image", None) is not None:
            material.image = image
            return True
        if getattr(material, "baseColorTexture", None) is not None:
            material.baseColorTexture = image
            return True
    if getattr(visual, "image", None) is not None:
        visual.image = image
        return True
    return False


def _face_sample_counts(uv_texels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """各面のテクセル面積から、撒くサンプル数を決める。"""
    tri = uv_texels[faces]  # (F, 3, 2)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    counts = np.ceil(area * SAMPLES_PER_TEXEL).astype(np.int64)
    return np.clip(counts, 1, MAX_SAMPLES_PER_FACE)


def _barycentric(n: int, rng: np.random.Generator) -> np.ndarray:
    """三角形内に一様な重心座標を n 個返す (n, 3)。"""
    r1, r2 = rng.random(n), rng.random(n)
    su = np.sqrt(r1)
    return np.column_stack([1.0 - su, su * (1.0 - r2), su * r2]).astype(np.float32)


def _blend_weight(dot: np.ndarray) -> np.ndarray:
    """法線と視線の内積 -> 参照画像を信用する重み (0..1)。"""
    span = max(FULL_CONFIDENCE_DOT - MIN_CONFIDENCE_DOT, 1e-9)
    return np.clip((dot - MIN_CONFIDENCE_DOT) / span, 0.0, 1.0).astype(np.float32)


def _iter_surface_samples(
    vertices: np.ndarray,
    normals: np.ndarray,
    uv_texels: np.ndarray,
    faces: np.ndarray,
    counts_all: np.ndarray,
):
    """面のUVテクセル面積に比例したランダム表面サンプルをチャンク単位で返す。

    深度バッファ構築(パス1)と色サンプリング(パス2)で**同一のサンプル列**を
    使うため、呼び出しごとに同じシードで乱数列を作り直す(決定的)。
    """
    rng = np.random.default_rng(0)
    for start in range(0, len(faces), FACE_CHUNK):
        chunk = faces[start : start + FACE_CHUNK]
        counts = counts_all[start : start + FACE_CHUNK]
        total = int(counts.sum())
        if total == 0:
            continue

        face_idx = np.repeat(np.arange(len(chunk)), counts)
        bary = _barycentric(total, rng)[:, :, None]  # (S, 3, 1)

        tri_v = vertices[chunk[face_idx]]  # (S, 3, 3)
        tri_n = normals[chunk[face_idx]]
        tri_uv = uv_texels[chunk[face_idx]]

        points = (tri_v * bary).sum(axis=1)  # (S, 3)
        sample_normals = (tri_n * bary).sum(axis=1)
        norm = np.linalg.norm(sample_normals, axis=1, keepdims=True)
        sample_normals /= np.maximum(norm, 1e-9)
        sample_uv = (tri_uv * bary).sum(axis=1)  # (S, 2)
        yield points, sample_normals, sample_uv


class _SynthView:
    """参照画像と同じ解像度で「メッシュを texgen 色で描いた合成ビュー」を作る。

    最前面のサンプルの texgen 色を画素ごとに残す(単純なポイントラスタライザ)。
    フローワープの位置合わせ先として使う。
    """

    def __init__(self, image: Image.Image, view_normal: np.ndarray):
        self.w, self.h = image.size
        self.depth = np.full(self.h * self.w, np.inf, dtype=np.float64)
        self.color = np.zeros((self.h * self.w, 3), dtype=np.uint8)
        self.direction = -view_normal

    def record(self, points: np.ndarray, px: np.ndarray, py: np.ndarray, rgb: np.ndarray) -> None:
        d = points @ self.direction
        flat = py * self.w + px
        # 奥→手前の順に書けば、同一画素は最前面が最後に残る。チャンクを跨ぐ
        # 分は self.depth との比較で守る。
        order = np.argsort(-d)
        flat_o = flat[order]
        nearer = d[order] < self.depth[flat_o]
        idx = order[nearer]
        self.depth[flat_o[nearer]] = d[idx]
        self.color[flat_o[nearer]] = rgb[idx]

    def image(self) -> tuple[np.ndarray, np.ndarray]:
        """(色 (h,w,3) uint8, 被覆マスク (h,w)) を返す。未被覆は白。"""
        covered = np.isfinite(self.depth).reshape(self.h, self.w)
        rgb = self.color.reshape(self.h, self.w, 3).copy()
        rgb[~covered] = 255
        return rgb, covered


def _dark_feature_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """暗く彩度の低い塊(目・鼻・口・黒パッチ)のマスクを返す。

    小さすぎる塊(ノイズ)と大きすぎる塊(黒い服全体など、面積比
    `FEATURE_MAX_AREA_FRACTION` 超)は「特徴」とみなさない。
    """
    from scipy.ndimage import label

    a = rgb.astype(np.float32)
    dark = (a.mean(axis=2) < FEATURE_MAX_LUMINANCE) & (
        a.max(axis=2) - a.min(axis=2) < FEATURE_MAX_SATURATION
    ) & mask
    components, n = label(dark)
    if n == 0:
        return dark  # 全False
    areas = np.bincount(components.ravel())
    max_area = mask.sum() * FEATURE_MAX_AREA_FRACTION
    keep = (areas >= FEATURE_MIN_AREA_PX) & (areas <= max_area)
    keep[0] = False
    return keep[components]


def _feature_components(
    rgb: np.ndarray, mask: np.ndarray
) -> list[tuple[np.ndarray, float, float, float, float]]:
    """暗い特徴の連結成分ごとに (画素マスク, 重心x, 重心y, 面積, アスペクト比) を返す。"""
    from scipy.ndimage import label

    dark = _dark_feature_mask(rgb, mask)
    components, n = label(dark)
    out = []
    for i in range(1, n + 1):
        m = components == i
        ys, xs = np.where(m)
        width = float(xs.max() - xs.min() + 1)
        height = float(ys.max() - ys.min() + 1)
        aspect = max(width, height) / max(min(width, height), 1.0)
        out.append((m, float(xs.mean()), float(ys.mean()), float(len(ys)), aspect))
    return out


def _plausible_match(synth_feat, ref_feat) -> bool:
    """同種の特徴(目と目、口と口)らしい対応か。

    面積とアスペクト比が大きく違う対応(コンパクトな目 vs 細長い髪の影)は
    誤マッチとして棄却する。
    """
    _, _, _, area_s, aspect_s = synth_feat
    _, _, _, area_r, aspect_r = ref_feat
    area_ratio = min(area_s, area_r) / max(area_s, area_r)
    if area_ratio < FEATURE_MATCH_MIN_AREA_RATIO:
        return False
    aspect_ratio = max(aspect_s, aspect_r) / max(min(aspect_s, aspect_r), 1.0)
    return aspect_ratio <= FEATURE_MATCH_MAX_ASPECT_RATIO


def align_reference_features(
    synth_rgb: np.ndarray,
    synth_mask: np.ndarray,
    reference: Image.Image,
) -> tuple[Image.Image, int]:
    """参照画像の暗い特徴をメッシュ側の位置へ剛体移動した補正版を返す。

    合成ビュー(メッシュ側)と参照(画像側)の暗い塊の重心を距離昇順で一対一に
    対応付け、対応した塊を**形を変えずに**平行移動する。元の位置は周囲の
    非特徴画素(毛)で埋める。対応が無い塊は動かさない。

    Returns:
        (補正済み参照RGBA, 移動した特徴数)。
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

    ref = reference.convert("RGBA")
    w, h = ref.size
    if synth_rgb.shape[:2] != (h, w):
        raise ValueError("合成ビューと参照画像の解像度が一致していません。")
    ref_arr = np.asarray(ref).copy()
    ref_rgb = ref_arr[..., :3]

    synth_feats = _feature_components(synth_rgb, synth_mask)
    ref_feats = _feature_components(ref_rgb, ref_arr[..., 3] > 128)
    if not synth_feats or not ref_feats:
        return ref, 0

    limit = FEATURE_MATCH_MAX_FRACTION * max(w, h)
    pairs = []
    for i, sf in enumerate(synth_feats):
        for j, rf in enumerate(ref_feats):
            d = ((sf[1] - rf[1]) ** 2 + (sf[2] - rf[2]) ** 2) ** 0.5
            if FEATURE_MIN_SHIFT_PX <= d <= limit and _plausible_match(sf, rf):
                pairs.append((d, i, j))
    pairs.sort()
    used_s: set[int] = set()
    used_r: set[int] = set()
    moves = []  # (参照の塊マスク, dx, dy)
    for d, i, j in pairs:
        if i in used_s or j in used_r:
            continue
        used_s.add(i)
        used_r.add(j)
        _, sx, sy, _, _ = synth_feats[i]
        rmask, rx, ry, _, _ = ref_feats[j]
        moves.append((rmask, sx - rx, sy - ry))
    if not moves:
        return ref, 0

    radius = max(int(FEATURE_DILATE_FRACTION * max(w, h)), 2)
    sigma = max(FEATURE_FEATHER_FRACTION * max(w, h), 1.0)

    # 1) 動かす塊の元位置をすべて消す(最近傍の非特徴画素=毛で埋める)。
    #    先に全部消してから貼ることで、移動先が他の塊の元位置と重なっても
    #    消し残しが出ない。
    erase = np.zeros((h, w), bool)
    for rmask, _, _ in moves:
        erase |= binary_dilation(rmask, iterations=radius)
    _, (iy, ix) = distance_transform_edt(erase, return_distances=True, return_indices=True)
    filled_rgb = ref_rgb.copy()
    filled_rgb[erase] = ref_rgb[iy[erase], ix[erase]]
    out_rgb = filled_rgb.astype(np.float32)

    # 2) 塊をずらした位置へ、ふちをぼかして貼る。
    shifts = []
    for rmask, dx, dy in moves:
        patch = binary_dilation(rmask, iterations=radius)
        soft = gaussian_filter(patch.astype(np.float32), sigma=sigma)
        idx, idy = int(round(dx)), int(round(dy))
        shifted_soft = np.zeros_like(soft)
        src_y = slice(max(0, -idy), min(h, h - idy))
        src_x = slice(max(0, -idx), min(w, w - idx))
        dst_y = slice(max(0, idy), min(h, h + idy))
        dst_x = slice(max(0, idx), min(w, w + idx))
        shifted_soft[dst_y, dst_x] = soft[src_y, src_x]
        shifted_rgb = np.zeros_like(out_rgb)
        shifted_rgb[dst_y, dst_x] = ref_rgb[src_y, src_x]
        alpha = shifted_soft[..., None]
        out_rgb = out_rgb * (1.0 - alpha) + shifted_rgb * alpha
        shifts.append((dx * dx + dy * dy) ** 0.5)

    ref_arr[..., :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    logger.info(
        "Aligned %d dark features onto the mesh (shift mean %.1fpx / max %.1fpx).",
        len(moves),
        float(np.mean(shifts)),
        float(np.max(shifts)),
    )
    return Image.fromarray(ref_arr, "RGBA"), len(moves)


class _DepthBuffer:
    """1ビュー分の直交深度バッファ(可視性判定用)。

    表面サンプル自身を「その画素の最前面はどの奥行きか」の証言として使う。
    遮蔽物は裏を向いていても遮蔽するので、面の向きに関係なく全サンプルを
    登録する。格子は参照画像を `DEPTH_GRID_DIVISOR` で縮めた解像度
    (サンプル密度はUV面積基準のため、画像解像度そのままでは疎な面に
    ピンホールができる)。
    """

    def __init__(self, image: Image.Image, view_normal: np.ndarray, extents: np.ndarray):
        w, h = image.size
        self.gw = (w + DEPTH_GRID_DIVISOR - 1) // DEPTH_GRID_DIVISOR
        self.gh = (h + DEPTH_GRID_DIVISOR - 1) // DEPTH_GRID_DIVISOR
        self.depth = np.full(self.gh * self.gw, np.inf, dtype=np.float64)
        # 奥行き = 視線方向(カメラ→被写体)に沿った座標。カメラは view_normal 側に
        # あるので、視線方向は -view_normal。値が小さいほど手前。
        self.direction = -view_normal
        axis = int(np.argmax(np.abs(view_normal)))
        self.eps = float(extents[axis]) * DEPTH_EPS_FRACTION

    def _cells(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        return (py // DEPTH_GRID_DIVISOR) * self.gw + (px // DEPTH_GRID_DIVISOR)

    def record(self, points: np.ndarray, px: np.ndarray, py: np.ndarray) -> None:
        np.minimum.at(self.depth, self._cells(px, py), points @ self.direction)

    def finalize(self) -> None:
        """3x3の最小値フィルタでピンホール(疎な面の取りこぼし)を塞ぐ。

        深度の段差近傍では過剰に遮蔽扱いになりうるが、その場合は texgen に
        戻るだけなので安全側。
        """
        from scipy.ndimage import minimum_filter

        grid = self.depth.reshape(self.gh, self.gw)
        self.depth = minimum_filter(grid, size=3).reshape(-1)

    def visibility(self, points: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """可視度 (0..1)。最前面から eps までは 1、2*eps で 0 へ落とす。

        二値にすると、深度の段差(鼻の付け根・帽子のつばの縁)の周りで
        テクセルごとに可視/遮蔽が切り替わり、参照と texgen がまだらに
        混ざるごま塩ノイズになる(実ジョブc78dac23で確認)。
        """
        depth = points @ self.direction
        excess = depth - self.depth[self._cells(px, py)] - self.eps
        return np.clip(1.0 - excess / max(self.eps, 1e-9), 0.0, 1.0)


def _fill_rejected_regions(
    refined: np.ndarray,
    blend: np.ndarray,
    base: np.ndarray,
    *,
    rejected: np.ndarray,
) -> float:
    """棄却されたテクセル領域を周囲の転写済みの色で調和拡散して埋める。

    `refined` / `blend` を**その場で**書き換える。対象は次を全て満たす領域:

    1. サンプルは届いたが全て棄却された(遮蔽・浅い角度・フチ)
    2. texgen が局所的に平坦 = 情報を持たない(本物の描き込みは潰さない)
    3. 領域境界の一定割合以上が転写済み(拡散の根拠がある)

    Returns:
        埋めたテクセルの全体比。
    """
    if not rejected.any():
        return 0.0
    source = blend > 0.0  # 転写済み+ガター埋め(同一チャート近傍の複製)

    from scipy.ndimage import binary_closing, label, maximum_filter, uniform_filter
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import cg

    h, w = rejected.shape

    # texgen の局所的な平坦さ(グレースケール標準偏差)
    grey = base.mean(axis=2)
    mean = uniform_filter(grey, REJECT_FILL_FLAT_WINDOW)
    mean_sq = uniform_filter(grey * grey, REJECT_FILL_FLAT_WINDOW)
    local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    flat = local_std < REJECT_FILL_FLAT_STD

    # UV面積の大きい面はサンプル上限で棄却テクセルが塩粒状に途切れるため、
    # 閉包で面として繋いでから領域判定する。閉包で増えた分にも平坦条件と
    # 「まだ何も転写されていない」条件を課す(転写済み・描き込みは守る)。
    candidates = rejected & flat
    if not candidates.any():
        return 0.0
    candidates = (
        binary_closing(candidates, iterations=REJECT_FILL_CLOSING_ITERATIONS)
        & flat
        & (blend <= 0.0)
    )
    if not candidates.any():
        return 0.0

    # 領域ごとに「境界の何割が転写済みか」を見て採否を決める。
    # 境界テクセルの帰属は 3x3 最大値フィルタで近似する(2領域が同じ境界
    # テクセルに接する場合は大きいラベルが取るが、判定用途には十分)。
    regions, n_regions = label(candidates)
    ring_labels = maximum_filter(regions, size=3)
    ring = (regions == 0) & (ring_labels > 0)
    labels_on_ring = ring_labels[ring]
    total = np.bincount(labels_on_ring, minlength=n_regions + 1).astype(np.float64)
    covered_on_ring = np.bincount(
        labels_on_ring, weights=source[ring].astype(np.float64), minlength=n_regions + 1
    )
    keep = covered_on_ring / np.maximum(total, 1.0) >= REJECT_FILL_MIN_BOUNDARY_COVERAGE
    keep[0] = False
    fill_mask = keep[regions]
    n_unknown = int(fill_mask.sum())
    if n_unknown == 0:
        return 0.0
    if n_unknown > REJECT_FILL_MAX_UNKNOWNS:
        logger.warning(
            "Skipping rejected-region fill: %d texels exceed the solver cap (%d).",
            n_unknown,
            REJECT_FILL_MAX_UNKNOWNS,
        )
        return 0.0

    # 4近傍ラプラシアン。未知数=埋める対象、Dirichlet=転写済み、その他はNeumann。
    index = np.full((h, w), -1, dtype=np.int64)
    ys, xs = np.where(fill_mask)
    index[ys, xs] = np.arange(n_unknown)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    rhs = np.zeros((n_unknown, 3), dtype=np.float64)
    diag = np.zeros(n_unknown, dtype=np.float64)

    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        idx_self = index[ys[inside], xs[inside]]
        n_idx = index[ny[inside], nx[inside]]

        is_unknown = n_idx >= 0
        np.add.at(diag, idx_self[is_unknown], 1.0)
        rows.append(idx_self[is_unknown])
        cols.append(n_idx[is_unknown])
        vals.append(np.full(int(is_unknown.sum()), -1.0))

        is_source = (~is_unknown) & source[ny[inside], nx[inside]]
        np.add.at(diag, idx_self[is_source], 1.0)
        np.add.at(rhs, idx_self[is_source], refined[ny[inside][is_source], nx[inside][is_source]])

    # どの方向にも未知数・境界が無い孤立テクセルは texgen のまま固定する
    isolated = diag == 0.0
    diag[isolated] = 1.0
    rhs[isolated] = base[ys[isolated], xs[isolated]]

    matrix = coo_matrix(
        (
            np.concatenate([diag] + vals),
            (
                np.concatenate([np.arange(n_unknown)] + rows),
                np.concatenate([np.arange(n_unknown)] + cols),
            ),
        ),
        shape=(n_unknown, n_unknown),
    ).tocsr()

    # ラプラシアンは対称正定値なので共役勾配で解く(LU分解は73万未知数で
    # メモリも時間も跳ねる)。初期値に texgen の色を使うと、正解に近い場所
    # から始まるぶん速く、収束打ち切り時も「texgen より悪くならない」。
    solution = np.empty_like(rhs)
    for channel in range(3):
        x, info = cg(
            matrix,
            rhs[:, channel],
            x0=base[ys, xs, channel].astype(np.float64),
            rtol=1e-4,
            maxiter=1500,
        )
        if info != 0:
            logger.info("Harmonic fill CG stopped early (info=%d); using the partial solution.", info)
        solution[:, channel] = x

    refined[ys, xs] = np.clip(solution, 0.0, 255.0)
    blend[ys, xs] = 1.0
    ratio = n_unknown / float(h * w)
    logger.info(
        "Filled %.2f%% of texels (rejected but featureless in texgen) by harmonic diffusion.",
        ratio * 100,
    )
    return ratio


def deepen_neutral_shadows(texture: np.ndarray) -> np.ndarray:
    """暗く彩度の低いテクセルを黒へ寄せる(焼き込まれた照りの補正)。

    Args:
        texture: (h, w, 3) float32 の RGB。

    Returns:
        同形状の補正済み配列(入力は変更しない)。
    """
    luminance = texture.mean(axis=2)
    saturation = texture.max(axis=2) - texture.min(axis=2)
    dark = np.clip(
        (SHADOW_DEEPEN_MAX_LUMINANCE - luminance) / SHADOW_DEEPEN_LUMINANCE_RAMP, 0.0, 1.0
    )
    neutral = np.clip(
        (SHADOW_DEEPEN_MAX_SATURATION - saturation) / SHADOW_DEEPEN_SATURATION_RAMP, 0.0, 1.0
    )
    factor = 1.0 - SHADOW_DEEPEN_STRENGTH * dark * neutral
    return texture * factor[..., None]


def refine_texture_with_references(
    mesh: trimesh.Trimesh,
    references: dict[str, Image.Image],
) -> RefineStats:
    """texgen 済みメッシュのテクスチャを、参照画像の全解像度で上書きする。

    メッシュは **その場で** 書き換える(テクスチャ画像のみ差し替え、頂点・UVは不変)。

    Args:
        mesh: UV とテクスチャを持つ texgen の出力。Z-up・正面が -Y。
        references: ビュー名(`colorproc.VIEW_NAMES`)-> 背景除去済み参照画像。

    Returns:
        RefineStats。適用できなかった場合は `applied=False` と `reason`。
    """
    stats = RefineStats(views=sorted(references))

    views = [v for v in references if v in colorproc.VIEW_NAMES]
    if not views:
        stats.reason = "参照画像がありません。"
        return stats

    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    texture = _extract_texture_image(visual)
    if uv is None or texture is None:
        stats.reason = "メッシュにUVまたはテクスチャがありません。"
        return stats

    uv = np.asarray(uv, dtype=np.float64)
    if len(uv) != len(mesh.vertices):
        stats.reason = f"UV数({len(uv)})が頂点数({len(mesh.vertices)})と一致しません。"
        return stats

    base = np.asarray(texture.convert("RGB"), dtype=np.float32)  # (h, w, 3)
    h, w = base.shape[:2]
    stats.texture_size = (w, h)

    # UV(v=0が下端) -> テクセル座標(行0が上端)
    uv_texels = np.column_stack(
        [np.clip(uv[:, 0], 0.0, 1.0) * (w - 1), (1.0 - np.clip(uv[:, 1], 0.0, 1.0)) * (h - 1)]
    )

    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if normals.shape != vertices.shape:
        mesh_fixed = mesh.copy()
        mesh_fixed.fix_normals()
        normals = np.asarray(mesh_fixed.vertex_normals, dtype=np.float64)

    # ビューごとに、参照画像の画素と「フチを除いた信頼できる画素」を用意する。
    view_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, Image.Image]] = {}
    for view in views:
        image = references[view]
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
        trusted = colorproc._trusted_pixel_mask(alpha)
        view_data[view] = (rgb, trusted, np.asarray(colorproc._VIEW_NORMALS[view]), image)

    accum_rgb = np.zeros((h * w, 3), dtype=np.float32)
    accum_weight = np.zeros(h * w, dtype=np.float32)
    accum_count = np.zeros(h * w, dtype=np.float32)

    counts_all = _face_sample_counts(uv_texels, faces)

    # --- パス1: ビューごとの深度バッファ(可視性)と合成ビュー(位置合わせ) ----
    base_u8 = np.clip(base, 0, 255).astype(np.uint8)
    depth_buffers = {
        view: _DepthBuffer(view_data[view][3], view_data[view][2], mesh.extents)
        for view in views
    }
    synth_views = {
        view: _SynthView(view_data[view][3], view_data[view][2]) for view in views
    }
    for points, _, sample_uv in _iter_surface_samples(
        vertices, normals, uv_texels, faces, counts_all
    ):
        tx = np.clip(sample_uv[:, 0].astype(np.int64), 0, w - 1)
        ty = np.clip(sample_uv[:, 1].astype(np.int64), 0, h - 1)
        texgen_rgb = base_u8[ty, tx]
        for view in views:
            image = view_data[view][3]
            px, py = colorproc.project_points_to_pixels(mesh, image, points, view=view)
            depth_buffers[view].record(points, px, py)
            synth_views[view].record(points, px, py, texgen_rgb)
    for buffer in depth_buffers.values():
        buffer.finalize()

    # 参照の目・鼻・口をメッシュ側の位置へ剛体移動し、以降のサンプリングは
    # この補正済み参照から行う(位置ずれした顔パーツが「浮く」のを防ぐ)。
    for view in views:
        synth_rgb, synth_mask = synth_views[view].image()
        rgb, trusted, view_normal, image = view_data[view]
        try:
            corrected, moved = align_reference_features(synth_rgb, synth_mask, image)
        except Exception:
            logger.exception("Feature alignment failed for the %s view; using it as-is.", view)
            continue
        if moved:
            view_data[view] = (
                np.asarray(corrected.convert("RGB"), dtype=np.float32),
                trusted,
                view_normal,
                image,
            )

    # --- パス2: 可視かつ信頼できるサンプルだけを参照画像から転写する --------
    occluded_samples = 0
    assigned_samples = 0
    for points, sample_normals, sample_uv in _iter_surface_samples(
        vertices, normals, uv_texels, faces, counts_all
    ):
        tx = np.clip(sample_uv[:, 0].astype(np.int64), 0, w - 1)
        ty = np.clip(sample_uv[:, 1].astype(np.int64), 0, h - 1)
        flat = ty * w + tx

        # 各サンプルを、最も正対して見えるビューに割り当てる。
        dots = np.stack([sample_normals @ view_data[v][2] for v in views], axis=1)  # (S, V)
        best = np.argmax(dots, axis=1)
        best_dot = dots[np.arange(len(dots)), best]

        weight = _blend_weight(best_dot)
        # 遮蔽・フチ落ちしたサンプルも分母には入れる(境界のテクセルが
        # 自動的に texgen 側へ寄り、遮蔽の切り替わりが線にならない)。
        np.add.at(accum_count, flat, 1.0)

        for vi, view in enumerate(views):
            sel = (best == vi) & (weight > 0.0)
            if not sel.any():
                continue
            rgb, trusted, _, image = view_data[view]
            px, py = colorproc.project_points_to_pixels(mesh, image, points[sel], view=view)
            # 手前に別の面がある(=このビューから見えていない)サンプルは
            # 参照から塗らない。遮蔽を無視すると、帽子の垂れの裏の頭側面が
            # 帽子の赤を拾う等、覆っているパーツの色が筋として写る。
            visibility = depth_buffers[view].visibility(points[sel], px, py)
            assigned_samples += int(sel.sum())
            occluded_samples += int((visibility <= 0.0).sum())
            # フチ・透明画素に落ちたサンプルも信用しない(黒筋の原因)。
            ok = trusted[py, px] & (visibility > 0.0)
            if not ok.any():
                continue
            idx = np.flatnonzero(sel)[ok]
            wgt = weight[idx] * visibility[ok].astype(np.float32)
            np.add.at(accum_rgb, flat[idx], rgb[py[ok], px[ok]] * wgt[:, None])
            np.add.at(accum_weight, flat[idx], wgt)

    covered = accum_weight > 0.0
    if not covered.any():
        stats.reason = "参照画像から上書きできるテクセルがありませんでした。"
        return stats

    refined = np.zeros_like(accum_rgb)
    refined[covered] = accum_rgb[covered] / accum_weight[covered, None]
    # テクセルごとの混合率 = そのテクセルに落ちたサンプルの平均重み。
    # 正対する面ほど 1 に近づき、浅い角度では texgen 側が残る。
    blend = np.zeros(h * w, dtype=np.float32)
    counted = accum_count > 0
    blend[counted] = accum_weight[counted] / accum_count[counted]

    refined = refined.reshape(h, w, 3)
    blend = blend.reshape(h, w)
    covered_2d = covered.reshape(h, w)

    # 取りこぼしと UV チャート外周を、最近傍の上書き済みテクセルで埋める。
    # 埋めないとバイリニア補間がチャート外の texgen 色を拾って継ぎ目になる。
    # ただし「サンプルは届いたが遮蔽/フチで棄却された」テクセル(count>0,
    # weight=0)は意図して texgen に残した場所なので埋めない。
    if not covered_2d.all():
        never_sampled = accum_count.reshape(h, w) == 0
        distance, (iy, ix) = distance_transform_edt(
            ~covered_2d, return_distances=True, return_indices=True
        )
        fill = (distance > 0) & (distance <= HOLE_FILL_RADIUS_TEXELS) & never_sampled
        refined[fill] = refined[iy[fill], ix[fill]]
        blend[fill] = blend[iy[fill], ix[fill]]

    # 棄却領域(遮蔽・浅い角度)のうち texgen が情報を持たない場所を、
    # 周囲の転写済みの色から調和拡散で埋める(前髪の裏の白抜け対策)。
    # 拡散の根拠(アンカー)は blend>0 のテクセル。ガター埋めの複製も含むが、
    # それは同一チャート近傍(4テクセル以内)の色の複製なので根拠として十分。
    sampled_2d = accum_count.reshape(h, w) > 0
    stats.filled_texel_ratio = _fill_rejected_regions(
        refined,
        blend,
        base,
        rejected=sampled_2d & ~covered_2d,
    )

    # 混合率マップの孤立ノイズを除去する。遮蔽境界ではサンプルの当たり方の
    # 揺らぎでテクセル単位の外れ値(周囲は参照なのに1点だけtexgen、またはその逆)
    # が出て、ごま塩状に見える。色ではなく**混合率だけ**を均すので、参照側の
    # 毛並み等のディテールは鈍らない。ガター埋めの**後**に掛けること
    # (先に掛けるとチャート外周の blend=0 と混ざって縁が侵食される)。
    from scipy.ndimage import median_filter

    blend = median_filter(blend, size=3)

    result = base * (1.0 - blend[..., None]) + refined * blend[..., None]
    result = deepen_neutral_shadows(result)
    new_texture = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGB")

    if not _set_texture_image(visual, new_texture):
        stats.reason = "テクスチャ画像を差し替えられませんでした。"
        return stats

    stats.applied = True
    stats.refined_texel_ratio = float((blend > 0).mean())
    stats.mean_blend_weight = float(blend[blend > 0].mean())
    stats.occluded_sample_ratio = occluded_samples / max(assigned_samples, 1)
    logger.info(
        "Refined the texgen atlas from %s references at full resolution: "
        "%.1f%% of %dx%d texels touched (mean blend %.2f, %.1f%% of samples occluded).",
        "+".join(views),
        stats.refined_texel_ratio * 100,
        w,
        h,
        stats.mean_blend_weight,
        stats.occluded_sample_ratio * 100,
    )
    return stats
