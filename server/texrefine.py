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

# 側面参照があるときの下限。前後だけのときは真横まで届かせる必要があって
# 0.20(視線から78°)まで下げていたが、側面参照が入ると**どの面もどれかの
# ビューから正対して見える**ので、浅い角度の転写に頼る理由が無くなる。
# 浅い角度は1画素が表面上で大きく伸び、前後の取り合いが面ごとに揺れて
# まだら模様になる(実測: 頭の真横で前後の内積が平均±0.01=事実上コイン投げ)。
MIN_CONFIDENCE_DOT_WITH_SIDES = 0.40

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

# 埋めを繰り返す回数。1パスだと「埋めた結果が次のパスの足場になる」性質を
# 使えず、前後ビューの境目(側面)のように棄却領域が広い場所で大量に取り
# 残す(実測: 人型2048参照で47万テクセルが2パス目以降で初めて埋まる)。
# 進捗が止まれば早期終了する。
REJECT_FILL_PASSES = 4

# **孤島の例外**: 周囲がほぼ転写済みで囲まれた小さな棄却の塊は、texgen の
# 描き込みではなくサンプリング/可視性判定のゆらぎによる取りこぼしである。
# 平坦ゲート(texgenに情報があるか)を適用すると、texgen のノイズを「情報」と
# 誤認してゴマ塩が残る(実測: 側面の継ぎ目帯で2793個の1〜20px斑点)。
# サイズ上限を設けることで、texgen だけが見える広い領域(犬の帽子の垂れの裏)は
# 従来どおり保護される。
REJECT_FILL_ISLAND_MAX_PX = 400
REJECT_FILL_ISLAND_MIN_BOUNDARY = 0.9

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
# ノイズ除去: これ未満の塊は無視。1024px参照での実測値が基準。
# 面積は解像度の2乗で効くため、2048/4096参照ではそのまま使うと閾値が
# 相対的に緩くなり(同じ物理サイズの特徴でも画素数が4〜16倍になる)、
# 誤対応の原因になりうる小さな暗色斑点まで拾ってしまう。`_dark_feature_mask`
# で画像の一辺基準にスケールする。
FEATURE_MIN_AREA_PX = 60  # ノイズ除去: これ未満の塊は無視(1024px基準)
FEATURE_MIN_AREA_REFERENCE_PX = 1024
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

# 側面参照を足したとき、正面・背面が担当を譲る法線しきい値。
# `colorproc._PRIMARY_VIEW_MIN_DOT_WITH_SIDES` と同じ値・同じ理由。
# 側面参照はシルエットが一致していても**内部の特徴が数px ずれる**ことがあり
# (実測: 頭のシルエット不一致は11.4%と胴12.8%・脚19.2%より低いのに、頬や耳に
# 黒い塊が出た)、ずれが目に付くのは顔のような小さく細かい領域。逆に胴や脚の
# ような広く平坦な面ではずれは見えず、側面参照の効果だけが得られる。
# そこで**正面・背面が十分に正対して見えている面(dot>=0.5、視線から60°以内)は
# 側面参照に渡さない**。顔は正面のまま残り、真横を向く胴・脚の側面だけが
# 側面参照の担当になる。
PRIMARY_VIEW_MIN_DOT_WITH_SIDES = 0.5
_PRIMARY_VIEWS = ("front", "back")

# 側面参照を頭に使わない。頭は正面・背面が十分に観測しており(顔は正面向き、
# 後頭部は背面向き)、本当に側面しか見えないのは耳・こめかみという狭い領域
# だけ。一方で顔は細かいので、側面参照のわずかな位置ずれや拡散の混入が
# はっきり見える(実測: 頭のシルエット不一致は11.4%と胴12.8%・脚19.2%より
# 低い=位置は合っているのに、頬に黒い塊・髪に肌色の滲みが出た)。
# 利得がほぼ無くリスクだけがあるので、頭から上は前後に任せる。
#
# さらに頭では**拡散による穴埋めもしない**。頭の真横は前後どちらからも
# 正対して見えないので転写できず、そこを拡散で埋めると texgen と拡散と
# 転写の3すくみになって髪の色が斑に混ざる(実測: 直接34%/拡散24%/texgen43%)。
# 素の texgen の髪は滑らかで破綻が無いので、任せたほうが良い。
#
# 頭の付け根は**メッシュから測る**(固定の高さ比では体型差に耐えない)。
# Tポーズは腕で最大幅になり、肩から上で幅が急に落ちるので、その落ち際を
# 境目とする。実測: 人型で全高の81%(=首)、犬で59%(=首)と妥当に出る。
HEAD_WIDTH_FRACTION = 0.40
_HEAD_PROFILE_BINS = 64

# --- 頭部のチャート境界処理 (2026-07-30, TRELLIS.2ハイブリッドで導入) ----------
# o_voxel の to_glb が作るUVアトラスは微細チャートの集合で、顔だけで1000個超に
# 分断される (実測: trellis2ジョブで顔領域1419クラスタ)。チャートごとに参照の
# 投影・棄却が独立に決まるため、境界で色と混合率が不連続になり、目の周りの
# 矩形の継ぎ目・顔中央の縦線として見える。対策は2つ:
#   1. シーム調和: 「3Dでは隣接、UVでは別チャート」のテクセル対を突き合わせ、
#      両方転写済みなら色・混合率を平均する (法線一致ガード付き。薄いシェルの
#      表裏を誤って対にしない)。
#   2. 片側フェザー: 混合率を min(blend, gaussian(blend)) で**下げる方向だけ**
#      滑らかにする。上げる方向に滑らかにすると、転写できなかった(refinedが
#      黒い)テクセルへ blend が漏れて黒が滲むため、必ず min 合成にすること。
# どちらも頭部テクセル (head_texels, 側面参照があるときのみ) に限定し、
# 前後2面のみの従来ジョブ (犬など) の挙動は変えない。
HEAD_SEAM_HARMONIZE_UV_DIST = 2.0  # ガター(未サンプル域)からこのテクセル距離以内を境界とみなす
HEAD_SEAM_MIN_UV_DIST = 4.0  # UV距離がこれ以下の対は同一チャート内の単なる隣人なので除外
HEAD_SEAM_RADIUS_FACTOR = 2.0  # 3D対応付け半径 = 境界テクセルの典型間隔 × この係数
HEAD_SEAM_NORMAL_DOT = 0.5  # 法線の向きがこれ以上一致する対だけ平均する
HEAD_BLEND_FEATHER_SIGMA = 3.0  # 片側フェザーのガウスσ(テクセル)

# --- 元テクスチャの色調合わせ (match_base_colors, trellis2 経路) --------------
# 転写できたテクセル (アンカー) では「元色→参照色」の差分が取れる。この差分を
# **3D空間の近傍**から距離加重で補間し、転写できなかったテクセル (目のくぼみ・
# 鼻の脇など遮蔽棄却された凹み) の元色に足す。目のくぼみは周囲の頬から、
# 髪は隣の髪から補正を受ける。
# グローバルな変換 (アフィン/トーンカーブLUT) は不採用: 値→値の写像が一意で
# ない (実測: 元が明るい画素の対応先が白い服と黒い脚に割れ、アフィンは
# scale 0.33 へ発散、LUTは227階調の移動を要求) ため、必ず局所で行うこと。
# アトラス空間での平滑化も不可 (チャートが微細に分断され、隣のチャートは
# 3Dでは無関係な部位のため)。
BASE_MATCH_MIN_BLEND = 0.6  # アンカーに使う最小混合率
BASE_MATCH_MIN_PAIRS = 5_000  # アンカーがこれ未満なら無補正
BASE_MATCH_MAX_ANCHORS = 300_000  # KD木に載せるアンカー上限 (超過は無作為抽出)
BASE_MATCH_NEIGHBORS = 8  # 補間に使う近傍アンカー数
BASE_MATCH_RADIUS_FACTOR = 8.0  # 有効半径 = アンカー典型間隔 × この係数
BASE_MATCH_NORMAL_DOT = 0.3  # 法線がこれ以上一致するアンカーだけ使う (薄シェル表裏の混入防止)
BASE_MATCH_MAX_SHIFT = 96.0  # 補正量の上限 (チャンネル毎、階調)

# --- 隠れ面の色継承 (match_base_colors の追加パス, 2026-07-31) ----------------
# 服と体が別シェルのメッシュ(TRELLIS.2)では、服の下の体表面に生成器が
# 独自の色(実測: オレンジ)を塗る。静止時は不可視だが、リグでポーズを付けると
# 数mmの層ずれで覗き、「衣装の色が変わった」ように見える(rig-service側の
# ウェイト対策後も残る数mmは原理的に消せない)。
# 対策: **遮蔽の証拠があるテクセル**(まともな角度でビューに割り当てられたのに
# 深度バッファで遮られたもの=静止時に見えない面)の元色を、3D近傍で同じ向きの
# 転写済みアンカー(=それを覆っている表面)の**転写色そのもの**へ寄せる。
# 「服の裏は服の色」にしておけば、覗いても目立たない。
# 通常の局所色補正(差分の伝播)より広い半径で、色を直接ブレンドする点が違う。
# **「未転写」だけを条件にしてはならない**: 斜め棄却・内容ずれガード棄却の面は
# 静止時に**見えている**ので、塗ると脇のアクセント色がシャツ側面へ流れて
# マゼンタの汚れになる(実測: シャツ帯のマゼンタが51,785→93,916テクセルに倍増)。
# 頭部には適用しない(前髪下の額に髪色が付くと、下から覗いた時に破綻する)。
HIDDEN_INTERIOR_RADIUS_FACTOR = 40.0  # 半径 = アンカー典型間隔 × この係数
HIDDEN_INTERIOR_NORMAL_DOT = 0.3  # 覆っている面と同じ向きのアンカーだけ使う
HIDDEN_INTERIOR_BLEND = 0.85  # 継承の強さ (1.0で完全置換。元の陰影を少し残す)
# 「盲目領域」= 全ビューでの最大dotがこの値未満 = どの参照からも原理的に
# 覆えない面 (襟の胸元・肩の上面のような**上下向きの棚**。参照は4方向とも
# 水平視点のため)。ここに生成器の幻覚色(赤い胸元)が残ると、可視なのに
# 直しようがない。盲目領域も色継承の対象にする。周囲のアンカーは向きが
# 直交する(棚 vs 垂直面)ため、法線条件はこの緩い値を使う — 対面
# (dot≈-1, 薄シェルの裏)だけは引き続き排除する。
# 注意: 「ガードで棄却されただけの参照可視面」を対象にしてはならない
# (マゼンタ流出事故の教訓)。盲目領域は max_dot 条件で厳密に区別できる。
BLIND_ZONE_MAX_DOT = 0.15
BLIND_ZONE_NORMAL_DOT = -0.1

# --- シルエット不一致ガード(ポーズ違いの参照を安全に使う) ------------------
# 側面参照は、Tポーズのままだと腕がカメラ方向へ伸びて胴の側面を隠すため、
# 生成側で**腕を前へ出した**姿勢にせざるを得ない。すると腕だけメッシュ(Tポーズ)と
# 位置が食い違い、そのまま投影すると腕の色が背景や胴へ流れ込む。
# メッシュのシルエット(合成ビューの被覆)と参照のアルファを比べ、食い違う
# 領域からはサンプルしない。胴・脚は姿勢が一致するので通常どおり転写される。
# ポーズ違いに限らず、体型が食い違う参照全般への保護になる。
SILHOUETTE_DISAGREE_DILATE_FRACTION = 0.004  # 不一致域をこの半径(画像辺比)広げる
# 不一致がこの割合を超えるビューは、そもそも別物として警告する(捨てはしない:
# 一致部分は使えるため)。
SILHOUETTE_DISAGREE_WARN_FRACTION = 0.35

# --- 側面参照の内容ずれガード(手にシャツの色が乗る対策) ----------------------
# シルエット一致は「そこに何かが写っている」ことしか保証しない。Tポーズの手は
# 真横へ突き出ているため、側面ビューでは参照画像の**胴体や前に突き出した腕**の
# 位置に投影され、シルエットは一致するのに内容は別物(シャツの青緑・髪の紫)を
# 拾う(実測: 指先が側面ビューに正対するため強く転写され、指がシャツ色になる)。
# 対策: 側面ビューからの転写は、参照画素と合成ビュー(メッシュ+元テクスチャ)の
# 色差がこの閾値以内の場合に限る。「側面参照は精細化には使うが**色替えには
# 使わない**」という意味。閾値はチャンネル最大差で、texgen のくすみ
# (同系色で数十)は通し、別部位(肌vsシャツで150超)は弾く余裕を持たせる。
# 正面・背面には適用しない(texgen が顔を描かないことがあり、正当な大差がある)。
SIDE_VIEW_MAX_SYNTH_DELTA = 100.0

# --- 背景色の抜き残し(髪の隙間)の除外 ------------------------------------
# 背景除去は髪の房の**あいだの隙間**まで透過にできないことがあり、背景色
# (白など)が不透明のまま小さな島として残る (実測: 4面の参照とも白の小島が
# 70〜128個)。そのまま転写すると髪に白いポツが写る。
# 「背景色に近い色 × 小さな島 × シルエット(透明領域)の近く」の3条件で
# 信頼画素から外す。距離条件は、顔の奥にある目の白いハイライト(同じ白)を
# 巻き込まないためのもの (髪の隙間はシルエット外縁の近くに出る)。
# 背景色は透明画素のRGB中央値から推定する (rembg・色キーとも元の背景色を
# RGBに残したまま alpha=0 にするため)。透明画素が無い参照では何もしない。
BACKGROUND_GAP_TOLERANCE = 16  # 背景色とのチャンネル毎の許容差
BACKGROUND_GAP_MAX_AREA_FRACTION = 6e-4  # 島の最大面積 (画像画素数比。2048²で~2500px)
BACKGROUND_GAP_MAX_SILHOUETTE_DIST_FRACTION = 0.05  # シルエットからの最大距離 (長辺比)
# 形状条件: 髪の隙間は房のあいだの**細長い楔形**、目のハイライトは**円形**。
# 距離条件だけでは目のハイライトを巻き込む (実測: 前髪の透明な隙間が顔の
# すぐ近くまで来ており、左目のハイライトが距離条件を満たしてしまった)。
# 細長い (アスペクト比が高い) か、バウンディングボックスに対しスカスカ
# (充填率が低い=楔・ギザギザ) な島だけを落とす。
BACKGROUND_GAP_MIN_ASPECT = 2.0
BACKGROUND_GAP_MAX_SOLIDITY = 0.45  # bbox充填率がこれ以下なら楔形とみなす

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

# 参照画像の解像度上限。`_DepthBuffer`/`_SynthView` は参照と同じ画素数の
# バッファを持つため、メモリ使用量は参照解像度の2乗に比例する。ネイティブ
# アップロードが際限なく大きい場合に備え、実用上十分な精細さを保ちつつ
# ここで頭打ちにする。この上限を超える参照は事前に Lanczos で縮小してから
# 渡すこと(呼び出し側=jobs.pyの責務。texrefine自身はここでは縮小しない)。
MAX_REFERENCE_DIMENSION = 4096


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


def head_base_height(mesh: trimesh.Trimesh) -> float:
    """頭の付け根の高さ(Z)を返す。求まらなければメッシュ上端(=頭無し扱い)。

    Tポーズでは腕を含む横幅が最大になり、肩から上で急に細くなる。最大幅の段
    より上を走査し、幅が `HEAD_WIDTH_FRACTION` を下回った最初の段を境目とする。
    """
    vertices = np.asarray(mesh.vertices)
    z = vertices[:, 2]
    low, high = float(z.min()), float(z.max())
    if not np.isfinite(low) or high <= low:
        return high

    edges = np.linspace(low, high, _HEAD_PROFILE_BINS + 1)
    index = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, _HEAD_PROFILE_BINS - 1)
    widths = np.zeros(_HEAD_PROFILE_BINS)
    filled = np.flatnonzero(np.bincount(index, minlength=_HEAD_PROFILE_BINS))
    for i in filled:
        band = vertices[index == i, 0]
        widths[i] = float(band.max() - band.min())
    # 頂点が疎なメッシュでは空の段ができ、そこが幅0の切れ込みに見えて
    # 頭の付け根を誤検出する(colorproc._mesh_silhouette_profile と同じ罠)。
    if 2 <= len(filled) < _HEAD_PROFILE_BINS:
        widths = np.interp(np.arange(_HEAD_PROFILE_BINS), filled, widths[filled])

    widest = float(widths.max())
    if widest <= 0:
        return high
    for i in range(int(np.argmax(widths)), _HEAD_PROFILE_BINS):
        if widths[i] < widest * HEAD_WIDTH_FRACTION:
            return low + (i / _HEAD_PROFILE_BINS) * (high - low)
    return high


def _blend_weight(dot: np.ndarray, minimum=MIN_CONFIDENCE_DOT) -> np.ndarray:
    """法線と視線の内積 -> 参照画像を信用する重み (0..1)。

    `minimum` はスカラーまたはサンプルごとの配列 (ビュー×部位別の下限)。
    """
    span = np.maximum(FULL_CONFIDENCE_DOT - minimum, 1e-9)
    return np.clip((dot - minimum) / span, 0.0, 1.0).astype(np.float32)


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
        yield points, sample_normals, sample_uv, start + face_idx


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
    # 面積は解像度の2乗で効くので、1024px基準の閾値を辺長比の2乗でスケールする。
    scale = (max(rgb.shape[0], rgb.shape[1]) / FEATURE_MIN_AREA_REFERENCE_PX) ** 2
    min_area_px = FEATURE_MIN_AREA_PX * scale
    max_area = mask.sum() * FEATURE_MAX_AREA_FRACTION
    keep = (areas >= min_area_px) & (areas <= max_area)
    keep[0] = False
    return keep[components]


def silhouette_agreement_mask(
    synth_mask: np.ndarray, reference: Image.Image
) -> tuple[np.ndarray, float]:
    """メッシュと参照のシルエットが一致する画素のマスクと、不一致率を返す。

    腕を前へ出した側面参照のように**ポーズが違う**参照でも、一致する部分
    (胴・脚)だけを安全に使えるようにする。不一致域は少し広げて、境界付近の
    にじみも落とす。

    Args:
        synth_mask: メッシュを描いた合成ビューの被覆マスク (h, w)。
        reference: 参照画像(アルファ付き)。合成ビューと同じ解像度であること。

    Returns:
        (一致マスク, 不一致率)。不一致率は「どちらか一方にしかない画素」の割合。
    """
    from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes

    # 合成ビューは表面サンプルの点描きなので被覆マスクに細かい穴が空く。
    # 閉じてから穴埋めして、実体のあるシルエットに直してから比べる
    # (これを怠ると穴が全部「不一致」に数えられ、参照全体が捨てられる)。
    radius = max(int(SILHOUETTE_DISAGREE_DILATE_FRACTION * max(synth_mask.shape)), 1)
    solid = binary_fill_holes(binary_closing(synth_mask, iterations=radius))

    ref_mask = np.asarray(reference.convert("RGBA"))[..., 3] > 128
    disagree = solid ^ ref_mask
    union = solid | ref_mask
    ratio = float(disagree.sum()) / max(int(union.sum()), 1)
    return ~binary_dilation(disagree, iterations=radius), ratio


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

    def select_fill_mask() -> np.ndarray:
        """このパスで埋める対象テクセルを選ぶ。"""
        source = blend > 0.0  # 転写済み+ガター埋め(同一チャート近傍の複製)
        remaining = rejected & (blend <= 0.0)
        if not remaining.any():
            return np.zeros_like(remaining)

        def analyse(mask: np.ndarray):
            """閉包→連結成分ごとの (ラベル画像, 境界の転写済み率, サイズ) を返す。

            UV面積の大きい面はサンプル上限で棄却テクセルが塩粒状に途切れるため、
            閉包で面として繋いでから領域判定する。境界テクセルの帰属は 3x3
            最大値フィルタで近似する(2領域が同じ境界テクセルに接する場合は
            大きいラベルが取るが、判定用途には十分)。
            """
            closed = binary_closing(mask, iterations=REJECT_FILL_CLOSING_ITERATIONS)
            closed &= blend <= 0.0
            regions, n = label(closed)
            if n == 0:
                return regions, np.zeros(1), np.zeros(1), closed
            ring_labels = maximum_filter(regions, size=3)
            ring = (regions == 0) & (ring_labels > 0)
            on_ring = ring_labels[ring]
            total = np.bincount(on_ring, minlength=n + 1).astype(np.float64)
            covered = np.bincount(
                on_ring, weights=source[ring].astype(np.float64), minlength=n + 1
            )
            sizes = np.bincount(regions.ravel(), minlength=n + 1)
            return regions, covered / np.maximum(total, 1.0), sizes, closed

        # 孤島判定は**棄却テクセル全体**の連結成分で行う(平坦かどうかに関わらず、
        # 転写済みに囲まれた小さな塊はサンプリング/可視性のゆらぎによる
        # 取りこぼしなので、texgen のノイズごと埋めてよい)。
        regions_all, boundary_all, sizes_all, closed_all = analyse(remaining)
        island = (
            (sizes_all <= REJECT_FILL_ISLAND_MAX_PX)
            & (boundary_all >= REJECT_FILL_ISLAND_MIN_BOUNDARY)
        )
        island[0] = False

        # 広い領域は従来どおり「texgen が平坦」なテクセルだけを、平坦部分の
        # 連結成分で判定する(全棄却で括ると非平坦部と併合して境界率が下がり、
        # 本来埋まるはずの平坦域まで落ちる)。
        regions_flat, boundary_flat, _, closed_flat = analyse(remaining & flat)
        bulk = boundary_flat >= REJECT_FILL_MIN_BOUNDARY_COVERAGE
        bulk[0] = False

        return (closed_all & island[regions_all]) | (closed_flat & bulk[regions_flat] & flat)

    def solve_pass(fill_mask: np.ndarray) -> int:
        """1パス分の調和拡散を解いて `refined`/`blend` を更新し、埋めた数を返す。"""
        source = blend > 0.0
        n_unknown = int(fill_mask.sum())

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
        return n_unknown

    # 埋めた結果は次のパスの足場(Dirichlet境界)になるので、進捗が無くなるまで
    # 繰り返す。1パスだけだと前後ビューの境目のように棄却域が広い場所で
    # 大量に取り残す(実測: 人型2048参照で2パス目以降に47万テクセルが埋まる)。
    total_filled = 0
    for _ in range(REJECT_FILL_PASSES):
        fill_mask = select_fill_mask()
        n_unknown = int(fill_mask.sum())
        if n_unknown == 0:
            break
        if n_unknown > REJECT_FILL_MAX_UNKNOWNS:
            logger.warning(
                "Stopping rejected-region fill: %d texels exceed the solver cap (%d).",
                n_unknown,
                REJECT_FILL_MAX_UNKNOWNS,
            )
            break
        total_filled += solve_pass(fill_mask)

    if total_filled == 0:
        return 0.0
    ratio = total_filled / float(h * w)
    logger.info(
        "Filled %.2f%% of texels (rejected, no usable texgen content) by harmonic diffusion.",
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


def _background_gap_mask(rgb: np.ndarray, alpha: np.ndarray) -> Optional[np.ndarray]:
    """背景色の抜き残し(髪の隙間などの不透明な背景色の小島)のマスクを返す。

    定数ブロックのコメント参照。透明画素が無い(=背景色を推定できない)
    参照では None を返す。
    """
    from scipy import ndimage

    transparent = alpha == 0
    if not transparent.any():
        return None
    bg = np.median(rgb[transparent], axis=0)
    near_bg = (np.abs(rgb - bg[None, None, :]) <= BACKGROUND_GAP_TOLERANCE).all(axis=2)
    near_bg &= alpha > 0
    if not near_bg.any():
        return None

    h, w = alpha.shape
    max_area = BACKGROUND_GAP_MAX_AREA_FRACTION * (h * w)
    max_dist = BACKGROUND_GAP_MAX_SILHOUETTE_DIST_FRACTION * max(h, w)
    # 各画素からシルエット(透明領域)までの距離
    silhouette_dist = distance_transform_edt(~transparent)

    labels, n = ndimage.label(near_bg)
    if n == 0:
        return None
    index = range(1, n + 1)
    sizes = ndimage.sum(near_bg, labels, index)
    min_dist = ndimage.minimum(silhouette_dist, labels, index)
    slices = ndimage.find_objects(labels)
    box_h = np.array([s[0].stop - s[0].start for s in slices], dtype=np.float64)
    box_w = np.array([s[1].stop - s[1].start for s in slices], dtype=np.float64)
    aspect = np.maximum(box_h, box_w) / np.maximum(np.minimum(box_h, box_w), 1.0)
    solidity = sizes / np.maximum(box_h * box_w, 1.0)
    wedge_like = (aspect >= BACKGROUND_GAP_MIN_ASPECT) | (
        solidity <= BACKGROUND_GAP_MAX_SOLIDITY
    )
    drop = (sizes <= max_area) & (min_dist <= max_dist) & wedge_like
    if not drop.any():
        return None
    return drop[labels - 1] & near_bg


def _match_base_to_reference(
    base: np.ndarray,
    refined: np.ndarray,
    blend: np.ndarray,
    covered_2d: np.ndarray,
    sampled_2d: np.ndarray,
    positions: np.ndarray,
    normal_sums: np.ndarray,
    counts: np.ndarray,
) -> Optional[np.ndarray]:
    """元テクスチャの色調を、3D近傍の転写済みテクセルの差分で局所補正する。

    定数ブロックのコメント参照。戻り値は補正済み base (アンカー不足なら None)。
    """
    from scipy.spatial import cKDTree

    h, w = blend.shape
    counts_2d = counts.reshape(h, w)
    strong = covered_2d & (blend >= BASE_MATCH_MIN_BLEND) & (counts_2d > 0)
    n_anchors = int(strong.sum())
    if n_anchors < BASE_MATCH_MIN_PAIRS:
        return None
    targets_mask = sampled_2d & ~strong
    if not targets_mask.any():
        return None

    pos = positions.reshape(h, w, 3)
    nrm = normal_sums.reshape(h, w, 3)

    def _texel_data(mask):
        ys, xs = np.where(mask)
        c = counts_2d[ys, xs]
        p = pos[ys, xs] / c[:, None]
        n = nrm[ys, xs]
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        return ys, xs, p, n

    a_ys, a_xs, a_pos, a_nrm = _texel_data(strong)
    diff = refined[a_ys, a_xs] - base[a_ys, a_xs]
    if n_anchors > BASE_MATCH_MAX_ANCHORS:
        rng = np.random.default_rng(0)
        pick = rng.choice(n_anchors, size=BASE_MATCH_MAX_ANCHORS, replace=False)
        a_pos, a_nrm, diff = a_pos[pick], a_nrm[pick], diff[pick]

    tree = cKDTree(a_pos)
    sample = a_pos[:: max(1, len(a_pos) // 10_000)]
    nn_dist, _ = tree.query(sample, k=2)
    spacing = float(np.median(nn_dist[:, 1]))
    if spacing <= 0.0:
        return None
    radius = spacing * BASE_MATCH_RADIUS_FACTOR

    t_ys, t_xs, t_pos, t_nrm = _texel_data(targets_mask)
    dists, idx = tree.query(
        t_pos, k=BASE_MATCH_NEIGHBORS, distance_upper_bound=radius
    )
    valid = np.isfinite(dists)
    idx_safe = np.where(valid, idx, 0)
    # 薄いシェルの表裏 (3Dでは近いが逆向き) のアンカーは使わない
    dot = (a_nrm[idx_safe] * t_nrm[:, None, :]).sum(axis=2)
    valid &= dot >= BASE_MATCH_NORMAL_DOT
    weight = np.where(valid, 1.0 / np.maximum(dists, spacing * 0.25), 0.0)
    total = weight.sum(axis=1)
    has_support = total > 0.0
    if not has_support.any():
        return None
    correction = np.einsum("tk,tkc->tc", weight, diff[idx_safe])
    correction[has_support] /= total[has_support, None]
    correction[~has_support] = 0.0
    # 補正の届く範囲の**外縁だけ**柔らかく落とす (半径の70%までは全強度。
    # 距離0から線形に落とすと凹みの中心で補正が半減し、残像が出る)。
    nearest = np.where(valid, dists, np.inf).min(axis=1)
    feather = np.clip((radius - nearest) / (0.3 * radius), 0.0, 1.0)
    correction *= feather[:, None]
    correction = np.clip(correction, -BASE_MATCH_MAX_SHIFT, BASE_MATCH_MAX_SHIFT)

    matched = base.copy()
    matched[t_ys, t_xs] = np.clip(base[t_ys, t_xs] + correction, 0.0, 255.0)
    logger.info(
        "Locally matched the base texture tone to the references: %d of %d "
        "uncovered texels corrected from %d anchors (mean |shift| %.1f levels).",
        int(has_support.sum()),
        len(t_ys),
        len(a_pos),
        float(np.abs(correction[has_support]).mean()) if has_support.any() else 0.0,
    )
    return matched


def _paint_hidden_interiors(
    base: np.ndarray,
    refined: np.ndarray,
    blend: np.ndarray,
    covered_2d: np.ndarray,
    blocked_2d: np.ndarray,
    blind_2d: Optional[np.ndarray],
    positions: np.ndarray,
    normal_sums: np.ndarray,
    counts: np.ndarray,
    head_texels: Optional[np.ndarray],
    comp_2d: Optional[np.ndarray] = None,
) -> int:
    """遮蔽面・盲目領域へ、周囲の表面の転写色を継承させる。

    定数ブロック (HIDDEN_INTERIOR_* / BLIND_ZONE_*) のコメント参照。
    `comp_2d` (連結成分ラベル) があるときは**同じ成分のアンカーだけ**から
    継承する (シャツはシャツから、肌は肌から。襟下のシャツの棚に首の肌が
    流れ込むのを防ぐ)。base を**その場で**書き換え、継承したテクセル数を返す。
    """
    from scipy.spatial import cKDTree

    h, w = blend.shape
    counts_2d = counts.reshape(h, w)
    anchors_mask = covered_2d & (blend >= BASE_MATCH_MIN_BLEND) & (counts_2d > 0)
    # 対象: 転写されず、かつ (a)**遮蔽された割り当てサンプルを持つ** または
    # (b)**盲目領域** (全ビューで dot が低くどの参照からも覆えない) のみ。
    # 斜め棄却・ガード棄却だけの可視面を塗ってはならない (定数コメント参照)。
    targets_mask = blocked_2d.copy()
    if blind_2d is not None:
        targets_mask |= blind_2d
    targets_mask &= ~covered_2d
    if head_texels is not None:
        targets_mask &= ~head_texels
    blind_only = (
        blind_2d & ~blocked_2d if blind_2d is not None else np.zeros_like(blocked_2d)
    )
    if not anchors_mask.any() or not targets_mask.any():
        return 0

    pos = positions.reshape(h, w, 3)
    nrm = normal_sums.reshape(h, w, 3)

    def _texel_data(mask):
        ys, xs = np.where(mask)
        c = counts_2d[ys, xs]
        p = pos[ys, xs] / c[:, None]
        n = nrm[ys, xs]
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        return ys, xs, p, n

    a_ys, a_xs, a_pos, a_nrm = _texel_data(anchors_mask)
    if len(a_pos) > BASE_MATCH_MAX_ANCHORS:
        rng = np.random.default_rng(1)
        pick = rng.choice(len(a_pos), size=BASE_MATCH_MAX_ANCHORS, replace=False)
        a_pos, a_nrm = a_pos[pick], a_nrm[pick]
        a_ys, a_xs = a_ys[pick], a_xs[pick]
    a_rgb = refined[a_ys, a_xs]

    tree = cKDTree(a_pos)
    sample = a_pos[:: max(1, len(a_pos) // 10_000)]
    nn_dist, _ = tree.query(sample, k=2)
    positive = nn_dist[:, 1][nn_dist[:, 1] > 0]
    spacing = float(np.median(positive)) if positive.size else 0.0
    if spacing <= 0.0:
        return 0
    radius = spacing * HIDDEN_INTERIOR_RADIUS_FACTOR

    t_ys, t_xs, t_pos, t_nrm = _texel_data(targets_mask)
    dists, idx = tree.query(t_pos, k=BASE_MATCH_NEIGHBORS, distance_upper_bound=radius)
    valid = np.isfinite(dists)
    idx_safe = np.where(valid, idx, 0)
    dot = (a_nrm[idx_safe] * t_nrm[:, None, :]).sum(axis=2)
    # 盲目領域は周囲アンカーの向きが直交する(棚 vs 垂直面)ため緩い閾値、
    # 遮蔽面は覆っている層と平行なので従来の閾値 (定数コメント参照)。
    dot_threshold = np.where(
        blind_only[t_ys, t_xs], BLIND_ZONE_NORMAL_DOT, HIDDEN_INTERIOR_NORMAL_DOT
    )
    valid &= dot >= dot_threshold[:, None]
    if comp_2d is not None:
        a_comp = comp_2d[a_ys, a_xs]
        t_comp = comp_2d[t_ys, t_xs]
        both_known = (a_comp[idx_safe] >= 0) & (t_comp[:, None] >= 0)
        valid &= ~both_known | (a_comp[idx_safe] == t_comp[:, None])
    weight = np.where(valid, 1.0 / np.maximum(dists, spacing * 0.25), 0.0)
    total = weight.sum(axis=1)
    has_support = total > 0.0
    if not has_support.any():
        return 0
    inherited = np.einsum("tk,tkc->tc", weight, a_rgb[idx_safe])
    inherited[has_support] /= total[has_support, None]
    sel = has_support
    ys, xs = t_ys[sel], t_xs[sel]
    base[ys, xs] = (
        base[ys, xs] * (1.0 - HIDDEN_INTERIOR_BLEND)
        + inherited[sel] * HIDDEN_INTERIOR_BLEND
    )
    return int(sel.sum())


def _harmonize_chart_seams(
    refined: np.ndarray,
    blend: np.ndarray,
    positions: np.ndarray,
    normal_sums: np.ndarray,
    counts: np.ndarray,
    covered_2d: np.ndarray,
    region: np.ndarray,
) -> float:
    """チャート境界をまたいで、同じ3D位置に対応するテクセルの色・混合率を揃える。

    微細チャートのアトラス(o_voxel to_glb)では、3Dで隣接する面がUV上の離れた
    チャートに割れており、参照投影のわずかな位相差が境界の線になる。
    ガター(未サンプル域)に接する転写済みテクセルを3D位置でKD木対応付けし、
    対応グループの平均に置き換える。書き換えは `region`(頭部)かつ転写済み
    (covered)のテクセルのみ。

    Args:
        refined: (h, w, 3) 転写色。**その場で**書き換える。
        blend: (h, w) 混合率。**その場で**書き換える。
        positions: (h*w, 3) テクセルに落ちたサンプル3D座標の合計。
        normal_sums: (h*w, 3) 同・法線の合計。
        counts: (h*w,) 同・サンプル数。
        covered_2d: (h, w) 参照から転写できたテクセル。
        region: (h, w) 処理対象領域(頭部テクセル)。

    Returns:
        調和したテクセルの数(観測用)。
    """
    from scipy.spatial import cKDTree

    h, w = blend.shape
    counts_2d = counts.reshape(h, w)
    gutter = counts_2d == 0
    if not gutter.any():
        return 0.0
    distance = distance_transform_edt(~gutter)
    boundary = covered_2d & region & (distance <= HEAD_SEAM_HARMONIZE_UV_DIST)
    ys, xs = np.where(boundary)
    if len(ys) < 2:
        return 0.0

    flat = ys * w + xs
    pos = positions[flat] / counts[flat, None]
    nrm = normal_sums[flat]
    norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.maximum(norm, 1e-9)

    tree = cKDTree(pos)
    # 境界テクセルの典型3D間隔(正の最近傍距離の中央値)を対応付け半径の基準に
    # する。別チャートの対応点が**完全に同一座標**のこともある(最近傍距離0)ため、
    # 0は除いて測る。全対応が同一座標なら極小半径で同一点対だけを拾う。
    nn_dist, _ = tree.query(pos, k=2)
    positive = nn_dist[:, 1][nn_dist[:, 1] > 0.0]
    spacing = float(np.median(positive)) if positive.size else 0.0
    radius = spacing * HEAD_SEAM_RADIUS_FACTOR if spacing > 0.0 else 1e-9
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return 0.0

    # 同一チャート内の単なる隣人(UVでも近い)は除外し、薄いシェルの表裏を
    # 誤って対にしないよう法線の一致を要求する。
    uv_dist = np.hypot(
        ys[pairs[:, 0]].astype(np.float32) - ys[pairs[:, 1]].astype(np.float32),
        xs[pairs[:, 0]].astype(np.float32) - xs[pairs[:, 1]].astype(np.float32),
    )
    dot = (nrm[pairs[:, 0]] * nrm[pairs[:, 1]]).sum(axis=1)
    pairs = pairs[(uv_dist > HEAD_SEAM_MIN_UV_DIST) & (dot >= HEAD_SEAM_NORMAL_DOT)]
    if len(pairs) == 0:
        return 0.0

    rgb = refined[ys, xs]
    bl = blend[ys, xs]
    sum_rgb = rgb.copy()
    sum_blend = bl.copy()
    count = np.ones(len(ys), dtype=np.float32)
    i0, i1 = pairs[:, 0], pairs[:, 1]
    np.add.at(sum_rgb, i0, rgb[i1])
    np.add.at(sum_rgb, i1, rgb[i0])
    np.add.at(sum_blend, i0, bl[i1])
    np.add.at(sum_blend, i1, bl[i0])
    np.add.at(count, i0, 1.0)
    np.add.at(count, i1, 1.0)

    touched = count > 1.0
    refined[ys[touched], xs[touched]] = sum_rgb[touched] / count[touched, None]
    blend[ys[touched], xs[touched]] = sum_blend[touched] / count[touched]
    return float(touched.sum())


def refine_texture_with_references(
    mesh: trimesh.Trimesh,
    references: dict[str, Image.Image],
    *,
    match_base_colors: bool = False,
) -> RefineStats:
    """texgen 済みメッシュのテクスチャを、参照画像の全解像度で上書きする。

    メッシュは **その場で** 書き換える(テクスチャ画像のみ差し替え、頂点・UVは不変)。

    Args:
        mesh: UV とテクスチャを持つ texgen の出力。Z-up・正面が -Y。
        references: ビュー名(`colorproc.VIEW_NAMES`)-> 背景除去済み参照画像。
        match_base_colors: 元テクスチャの色調を、転写できたテクセルの
            「元色→参照色」対応から推定したチャンネル毎のアフィン変換で
            参照側に合わせる。TRELLIS.2 の自前テクスチャは参照より彩度が
            くすむため、遮蔽などで転写できず元テクスチャが残る領域
            (目のくぼみ・鼻の脇の凹み)が色調差の「箱」として見える。
            色調を揃えればこの境界は消える (trellis2 経路で有効化。
            texgen (delight済み) の経路は従来どおり無補正)。

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
        gaps = _background_gap_mask(rgb, alpha)
        if gaps is not None and gaps.any():
            trusted = trusted & ~gaps
            logger.info(
                "Ignoring %d background-coloured gap pixels on the %s reference "
                "(unpunched holes between hair strands).",
                int(gaps.sum()),
                view,
            )
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
    for points, _, sample_uv, _face_ids in _iter_surface_samples(
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
    # あわせて、メッシュとシルエットが食い違う領域(腕を前へ出した側面参照など)を
    # 信頼できる画素から外す。
    # 合成ビューは側面の内容ずれガード (SIDE_VIEW_MAX_SYNTH_DELTA) でも使うため
    # 保持しておく。
    synth_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for view in views:
        synth_rgb, synth_mask = synth_views[view].image()
        synth_maps[view] = (np.asarray(synth_rgb, dtype=np.float32), synth_mask)
        rgb, trusted, view_normal, image = view_data[view]

        agree, disagreement = silhouette_agreement_mask(synth_mask, image)
        if disagreement > SILHOUETTE_DISAGREE_WARN_FRACTION:
            logger.warning(
                "The %s reference disagrees with the mesh silhouette over %.0f%% of the "
                "subject (different pose or proportions?); using the matching parts only.",
                view,
                disagreement * 100,
            )
        else:
            logger.info(
                "Silhouette disagreement on the %s view: %.1f%%.", view, disagreement * 100
            )
        trusted = trusted & agree

        try:
            corrected, moved = align_reference_features(synth_rgb, synth_mask, image)
        except Exception:
            logger.exception("Feature alignment failed for the %s view; using it as-is.", view)
            view_data[view] = (rgb, trusted, view_normal, image)
            continue
        view_data[view] = (
            np.asarray(corrected.convert("RGB"), dtype=np.float32) if moved else rgb,
            trusted,
            view_normal,
            image,
        )

    # --- パス2: 可視かつ信頼できるサンプルだけを参照画像から転写する --------
    primary_columns = [i for i, v in enumerate(views) if v in _PRIMARY_VIEWS]
    has_sides = bool(primary_columns) and len(primary_columns) < len(views)
    min_dot = MIN_CONFIDENCE_DOT_WITH_SIDES if has_sides else MIN_CONFIDENCE_DOT
    head_base = head_base_height(mesh) if has_sides else None
    if head_base is not None:
        logger.info(
            "Side references stop at the head base (z=%.1f of %.1f); the head is left "
            "to the front/back references.",
            head_base,
            float(np.asarray(mesh.vertices)[:, 2].max()),
        )
    head_texels = np.zeros((h, w), dtype=bool) if has_sides else None
    # 頭部のシーム調和用: テクセルごとの3D位置・法線の平均を取るための累積
    # (has_sides のときのみ確保。50MB×2 程度)。
    accum_pos = np.zeros((h * w, 3), dtype=np.float32) if has_sides else None
    accum_nrm = np.zeros((h * w, 3), dtype=np.float32) if has_sides else None
    # 遮蔽の証拠 (深度バッファに遮られた割り当てサンプル数)。隠れ面の色継承用。
    accum_blocked = np.zeros(h * w, dtype=np.float32) if has_sides else None
    # 全ビューでの最大dot (盲目領域の判定用)。
    accum_maxdot = np.zeros(h * w, dtype=np.float32) if has_sides else None
    # テクセルごとの連結成分ラベル (色継承を「同じ成分から」に限定するため。
    # 襟下のシャツの棚に首の肌が継承される事故を防ぐ)。UVシームの頂点複製で
    # 成分が砕けるので、位置で溶接してから面隣接の成分を取る。
    texel_comp = None
    if has_sides:
        welded = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        welded.merge_vertices()
        import trimesh.graph as _tgraph

        face_comp = _tgraph.connected_component_labels(
            welded.face_adjacency, len(faces)
        ).astype(np.int32)
        texel_comp = np.full(h * w, -1, dtype=np.int32)
    occluded_samples = 0
    assigned_samples = 0
    for points, sample_normals, sample_uv, sample_faces in _iter_surface_samples(
        vertices, normals, uv_texels, faces, counts_all
    ):
        tx = np.clip(sample_uv[:, 0].astype(np.int64), 0, w - 1)
        ty = np.clip(sample_uv[:, 1].astype(np.int64), 0, h - 1)
        flat = ty * w + tx
        if accum_pos is not None:
            np.add.at(accum_pos, flat, points.astype(np.float32))
            np.add.at(accum_nrm, flat, sample_normals.astype(np.float32))
        if texel_comp is not None:
            texel_comp[flat] = face_comp[sample_faces]

        # 各サンプルを、最も正対して見えるビューに割り当てる。
        dots = np.stack([sample_normals @ view_data[v][2] for v in views], axis=1)  # (S, V)
        best = np.argmax(dots, axis=1)
        if accum_maxdot is not None:
            np.maximum.at(accum_maxdot, flat, dots.max(axis=1).astype(np.float32))

        # 側面参照があるときは、正面・背面が十分正対している面と、頭より上の
        # 面を側面に渡さない(顔の細部が側面参照のわずかなずれで壊れるのを防ぐ)。
        if has_sides:
            on_head = points[:, 2] >= head_base
            if on_head.any():
                head_tx = np.clip(sample_uv[on_head, 0].astype(np.int64), 0, w - 1)
                head_ty = np.clip(sample_uv[on_head, 1].astype(np.int64), 0, h - 1)
                head_texels[head_ty, head_tx] = True
            primary_dots = dots[:, primary_columns]
            primary_pick = np.asarray(primary_columns)[np.argmax(primary_dots, axis=1)]
            keep_primary = primary_dots.max(axis=1) >= PRIMARY_VIEW_MIN_DOT_WITH_SIDES
            keep_primary |= on_head
            best = np.where(keep_primary, primary_pick, best)

        best_dot = dots[np.arange(len(dots)), best]

        # dot下限はビュー×部位別 (MIN_CONFIDENCE_DOT_WITH_SIDES のコメント参照):
        # 側面参照は常に 0.40。正面・背面は、頭(前後の取り合いがコイン投げに
        # なる場所)だけ 0.40 で、頭より下は従来どおり 0.20 まで届かせる。
        # 0.40 を全ビューに掛けると、襟の上向き斜面・脚の内側の斜め帯が
        # どのビューにも覆われず、生成器の生の色(赤い胸元・ピンクの内腿)が
        # 残る (実測: 襟元帯の63%が生のまま)。
        if has_sides:
            is_primary = np.isin(best, primary_columns)
            minimum = np.where(
                is_primary & ~on_head, MIN_CONFIDENCE_DOT, MIN_CONFIDENCE_DOT_WITH_SIDES
            )
        else:
            minimum = min_dot
        weight = _blend_weight(best_dot, minimum)
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
            if accum_blocked is not None:
                # 遮蔽の証拠を記録 (隠れ面の色継承の対象判定に使う)
                blocked_local = np.flatnonzero(sel)[visibility <= 0.0]
                if len(blocked_local):
                    np.add.at(accum_blocked, flat[blocked_local], 1.0)
            # フチ・透明画素に落ちたサンプルも信用しない(黒筋の原因)。
            ok = trusted[py, px] & (visibility > 0.0)
            if view not in _PRIMARY_VIEWS:
                # 側面の内容ずれガード (定数ブロック参照): 参照画素が合成ビュー
                # (メッシュ+元テクスチャ)の色と大きく食い違う場所は、ポーズ違いで
                # 別の部位(胴のシャツ等)が写っているので転写しない。
                synth_rgb_map, synth_mask_map = synth_maps[view]
                delta = np.abs(synth_rgb_map[py, px] - rgb[py, px]).max(axis=1)
                ok &= synth_mask_map[py, px] & (delta <= SIDE_VIEW_MAX_SYNTH_DELTA)
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

    # 元テクスチャの色調を参照に合わせる (trellis2 経路のみ。定数ブロック参照)。
    # 以降の拡散シード・最終混合はすべて補正済みの base を使う。
    # 3D位置の累積 (accum_pos) が必要なため側面参照ありのジョブに限られる。
    if match_base_colors and accum_pos is not None:
        matched_base = _match_base_to_reference(
            base,
            refined,
            blend,
            covered_2d,
            accum_count.reshape(h, w) > 0,
            accum_pos,
            accum_nrm,
            accum_count,
        )
        if matched_base is not None:
            base = matched_base
        # 服の下などの遮蔽面には、覆っている表面の色を継承させる
        # (ポーズ時の覗きを目立たなくする。HIDDEN_INTERIOR_* 参照)。
        painted = _paint_hidden_interiors(
            base,
            refined,
            blend,
            covered_2d,
            accum_blocked.reshape(h, w) > 0,
            (accum_count.reshape(h, w) > 0)
            & (accum_maxdot.reshape(h, w) < BLIND_ZONE_MAX_DOT),
            accum_pos,
            accum_nrm,
            accum_count,
            head_texels,
            comp_2d=texel_comp.reshape(h, w) if texel_comp is not None else None,
        )
        if painted:
            logger.info(
                "Painted %d fully-occluded interior texels with the colour of "
                "the covering surface.",
                painted,
            )

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
    fillable = sampled_2d & ~covered_2d
    if head_texels is not None:
        # 頭は拡散で埋めず texgen に任せる(髪の色が斑に混ざるのを防ぐ)
        fillable &= ~head_texels
    stats.filled_texel_ratio = _fill_rejected_regions(
        refined,
        blend,
        base,
        rejected=fillable,
    )

    # 混合率マップの孤立ノイズを除去する。遮蔽境界ではサンプルの当たり方の
    # 揺らぎでテクセル単位の外れ値(周囲は参照なのに1点だけtexgen、またはその逆)
    # が出て、ごま塩状に見える。色ではなく**混合率だけ**を均すので、参照側の
    # 毛並み等のディテールは鈍らない。ガター埋めの**後**に掛けること
    # (先に掛けるとチャート外周の blend=0 と混ざって縁が侵食される)。
    from scipy.ndimage import median_filter

    blend = median_filter(blend, size=3)

    # 頭部のチャート境界処理 (定数ブロックのコメント参照):
    # シーム調和(3D対応するチャート境界テクセルの色・混合率を平均)のあと、
    # 混合率を下げる方向にだけフェザーして、参照/texgen の切り替わりの
    # 矩形の縁を柔らかくする。min 合成なので黒滲みは構造的に起きない。
    if head_texels is not None and head_texels.any():
        harmonized = _harmonize_chart_seams(
            refined,
            blend,
            accum_pos,
            accum_nrm,
            accum_count,
            covered_2d,
            head_texels,
        )
        from scipy.ndimage import gaussian_filter

        feathered = gaussian_filter(blend, sigma=HEAD_BLEND_FEATHER_SIGMA)
        blend = np.where(head_texels, np.minimum(blend, feathered), blend)
        logger.info(
            "Head chart-seam treatment: harmonized %d boundary texels, feathered "
            "the blend map (sigma %.1f).",
            int(harmonized),
            HEAD_BLEND_FEATHER_SIGMA,
        )

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
