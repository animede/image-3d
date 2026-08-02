# Image-3D

画像から3Dプリントデータ(STL / 3MF / GLB / OBJ)を生成するローカルWebアプリ。

Web UIの使い方は [`docs/USAGE.md`](docs/USAGE.md)、詳細仕様は
[`docs/SPEC.md`](docs/SPEC.md)、開発方針は
[`docs/DEVELOPMENT_POLICY.md`](docs/DEVELOPMENT_POLICY.md)、実装計画は
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) を参照。
自動リグ+VRM化サービス **rig-service は別リポジトリ**(`../rig-service/`)。
設計・実装計画はそちらの `docs/RIG_SERVICE_PLAN.md` を参照。

## 入力にできるもの

**Tポーズを要求するのはリグ化(rig-service)だけ**で、3D化そのものは被写体を
選ばない。

- **任意のポーズ・任意の体型**のキャラクター(座り・アクション・頭身の低い
  デフォルメ体型・動物型・多足・翼つきなど)
- **上半身だけ / バストアップ / 顔だけ**の絵
- **キャラクター以外の物体**(小物・乗り物・建物・食べ物など)

形状生成器 (TRELLIS.2 / Hunyuan3D-2) は単一画像からの汎用 image-to-3D であって、
人体の構造を前提にしていない。Tポーズが要るのは rig-service 側の事情で、
腕の張り出しから肩・股下・首の位置を**実測**してボーンを置くため
(全身・Tポーズ以外では関節位置を人体標準比で代用することになり精度が落ちる)。
つまり **Tポーズの全身立ち絵を入れたときだけ、3D化に加えてリグ化もできる**
という関係になる。

## 現在の状態(Phase 1〜3b)

- Image-to-3D 生成は **mockジェネレータ**(決定的なパラメトリックメッシュ)を使用。
- GPU / Hunyuan3D-2 は未導入でも、アップロード → 生成 → メッシュ後処理 →
  3Dビューア表示 → STL/3MF/GLB/OBJ ダウンロードの全パイプラインがE2Eで動作する。
- Phase 2 で `IMAGE3D_GENERATOR=hunyuan3d` により実モデルに切り替え可能(下記参照)。
- Phase 2.5 で4色カラープリンタ向け出力(`color_mode=color4`)に対応
  (下記「Phase 2.5: 4色カラープリント対応」参照)。
- Phase 3a でマルチビュー入力(正面+背面/左/右)に対応
  (下記「Phase 3a: マルチビュー入力」参照)。
- Phase 3b でパラメータプリセット(FR-11)とビューアのオーバーハングヒートマップ
  (FR-12)に対応(下記「Phase 3b: プリセット+オーバーハングヒートマップ」参照、
  フロントエンドのみの変更でサーバAPIは不変)。
- Phase 3c でテクスチャ生成(`texture_mode=paint`、FR-10)に対応(下記
  「Phase 3c: テクスチャ生成 (texgen)」参照)。custom_rasterizer CUDA拡張の
  ビルドが必要で、未導入環境では `/api/health` の `texgen_available=false` に
  応じてUI上で無効表示し、正面/背面投影方式(FR-8)にフォールバックする。
- Phase R4 で **自動リグ+VRM化サービス(rig-service、別リポジトリ)
  との連携**に対応(下記「Phase R4: リグ/VRM化サービス連携」参照)。
  `IMAGE3D_RIGSVC_URL` を設定すると完了ジョブに「リグ/VRM化」ボタンが出る。
  **Tポーズの立ち絵から生成したモデル**を送ると、21ボーンのリグ済みGLBと
  VRM 1.0 が得られる。

## セットアップ

### 前提

- Python 3.12
- (Phase 2用) NVIDIA GPU + CUDA 12.8 対応ドライバ

### VRAM最小要件(実測ベース)

RTX PRO 6000 Blackwell 96GB での実測ピーク(既定パラメータ:
`octree_resolution=384`, `max_faces=200000`, テクスチャ2048×2048)に基づく目安。

| 使用機能 | 実測ピーク | 最小要件 | 備考 |
|---|---|---|---|
| mockジェネレータのみ | — | GPU不要 | 開発・UI確認用 |
| 形状生成(単一ビュー/マルチビュー) | 約12GB | **16GB** | 単一ビュー・mvの両パイプライン常駐+生成中ピークを含む |
| +テクスチャ生成 (`texture_mode=paint`) | 約25GB | **32GB** | shape+paint(delight・multiview diffusion)常駐+生成中ピーク |

- `octree_resolution=512` や `max_faces` 増(高精細プリセット)ではピークが上記より
  増加する。VRAMが最小要件付近のGPUでは `octree_resolution=256` への引き下げを推奨。
- 生成ジョブは直列実行(NFR-2)のため、同時実行によるVRAM加算は発生しない。
  各ジョブ後に `torch.cuda.empty_cache()` で解放される(NFR-3で重みは常駐)。
- 他プロセスとGPUを共有する場合は、上記に加えてそのプロセスの使用量を確保すること。

### venv作成 + 依存インストール

Linux / macOS / WSL2:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

Windowsネイティブでは **Phase 1(mockジェネレータ / CPU)** の利用を想定する。
Hunyuan3D-2 / CUDA / texgen の実モデル生成は、依存関係やCUDA拡張ビルドの都合で
WSL2 Ubuntu または Linux 環境を推奨する。

`requirements.txt` は base 依存のみ(FastAPI / trimesh / fast-simplification 等)。
rembg・torch・hy3dgen 等の重い依存は `requirements-gpu.txt` に分離されており、
Phase 1(mockジェネレータ)では不要。未導入でもアプリ全体が動作する
(rembgは `server/preprocess.py` で遅延import + 自動スキップ)。

### フロントエンド(Three.js)

Three.js はビルド工程なしで `web/vendor/` にローカル配置済み
(`three.module.js` / `OrbitControls.js` / `GLTFLoader.js` / `BufferGeometryUtils.js`)。
追加のnpmインストールは不要。

## 起動

Linux / macOS / WSL2:

```bash
./run.sh
```

Windows PowerShell:

```powershell
.\run.ps1
```

PowerShellの実行ポリシーでブロックされる場合は、カレントプロセスのみ許可してから起動する:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1
```

デフォルトで `http://127.0.0.1:8000` で待ち受ける。ブラウザで開くとUIが表示される。

環境変数で上書き可能:

```bash
IMAGE3D_GENERATOR=mock IMAGE3D_HOST=127.0.0.1 IMAGE3D_PORT=8000 ./run.sh
```

Windows PowerShellでは `$env:` で指定する:

```powershell
$env:IMAGE3D_GENERATOR = "mock"
$env:IMAGE3D_HOST = "127.0.0.1"
$env:IMAGE3D_PORT = "8000"
.\run.ps1
```

主な環境変数(`server/config.py`):

| 変数 | デフォルト | 説明 |
|---|---|---|
| `IMAGE3D_GENERATOR` | `mock` | `mock` \| `hunyuan3d` \| `pixal3d` \| `trellis2` |
| `IMAGE3D_HOST` | `127.0.0.1` | バインドアドレス |
| `IMAGE3D_PORT` | `8000` | ポート |
| `IMAGE3D_MAX_UPLOAD_BYTES` | `20971520`(20MB) | アップロード上限 |
| `IMAGE3D_DEFAULT_TARGET_HEIGHT_MM` | `100` | 後処理のデフォルト目標高さ |
| `IMAGE3D_DEFAULT_MAX_FACES` | `200000` | 後処理のデフォルト面数上限 |

## API例

```bash
# ジョブ作成
curl -s -X POST http://127.0.0.1:8000/api/jobs \
  -F "image=@sample.png" \
  -F 'params={"target_height_mm":100,"seed":42}'
# => {"job_id": "..."}

# 状態確認(ポーリング)
curl -s http://127.0.0.1:8000/api/jobs/<job_id>

# ビューア用GLB取得
curl -s http://127.0.0.1:8000/api/jobs/<job_id>/model.glb -o model.glb

# STLダウンロード
curl -s "http://127.0.0.1:8000/api/jobs/<job_id>/download?format=stl" -o model.stl

# ジョブ一覧 / 削除 / ヘルスチェック
curl -s http://127.0.0.1:8000/api/jobs
curl -s -X DELETE http://127.0.0.1:8000/api/jobs/<job_id>
curl -s http://127.0.0.1:8000/api/health

# マルチビュージョブ作成(Phase 3a、FR-9。image_back/left/rightは任意)
curl -s -X POST http://127.0.0.1:8000/api/jobs \
  -F "image=@front.png" \
  -F "image_back=@back.png" \
  -F 'params={"seed":42}'

```

全エンドポイントは [`docs/SPEC.md`](docs/SPEC.md) §5 を参照。

## テスト

```bash
.venv/bin/pytest tests/ -v
```

- `tests/test_meshproc.py`: 意図的に穴を開けたメッシュ・浮遊小部品を含むメッシュに対する
  watertight化・スケーリング・面数上限の検証。
- `tests/test_api.py`: mockジェネレータでジョブのライフサイクル全体
  (作成→ポーリング→completed→GLB/STL/3MF/OBJ取得→削除)、および不正入力(非画像ファイル・
  巨大サイズ・不正JSON・不正パラメータ・不正なcolor_mode/n_colors)の4xx応答を検証。
  STL出力はtrimeshで再読込しwatertight・高さ(mm)を機械検証する。カラーモード
  (`color_mode=color4`)のE2Eテストも含む(stats.palette・3MFマルチオブジェクト・
  GLB頂点カラーの検証)。
- `tests/test_colorproc.py`: 合成4色ブロック画像+単純メッシュで、頂点カラー投影・
  k-means量子化(パレット数がn_colors以下)・色ごとの分割(面数合計が元メッシュと
  一致)・パレット統計(face_ratio合計≈1.0)を検証。
- `tests/test_api.py`(Phase 3a追加分): mockで `image` + `image_back` の2ビュー
  ジョブがcompletedし、`views` フィールドが正しく記録されることを検証。
- `tests/test_texture.py`(Phase 3c追加): GPU不要の純関数
  `texture.sample_vertex_colors_from_texture` を、合成UV平面メッシュ+
  既知の4色ブロックテクスチャでUV→ピクセル対応を検証。`texture.is_available()`
  がbool型を返すことも検証。
- `tests/test_api.py`(Phase 3c追加分): `texture_mode` の不正値が400になること、
  `/api/health` に `texgen_available` が含まれること、mock環境で
  `texture_mode=paint`(単体・`color_mode=color4`併用)を指定してもジョブが
  正常completedすること(paint失敗→フォールバック経路を`_run_paint`の
  モンキーパッチで検証。実際のpaint成功経路はGPU実機検証でカバー)。

## Phase 2: GPU導入手順(Hunyuan3D-2、実機検証済み)

RTX PRO 6000 Blackwell (sm_120) 上で動作確認済みの手順。

1. CUDA 12.8対応ドライバのマシンで、cu128ビルドのtorch/torchvisionを導入
   (Blackwellはcu128以降が必須。実機検証時のバージョン: torch 2.11.0+cu128 /
   torchvision 0.26.0+cu128):
   ```bash
   .venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
   ```
   確認:
   ```bash
   .venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
   ```

2. Hunyuan3D-2 (hy3dgen) をソースからcloneし、`--no-deps` でeditableインストール
   (setup.pyのinstall_requiresにはtexgen/デモ用途の重い依存(gradio, xatlas,
   pygltflib, ninja, pybind11等)が含まれ、shapeパイプラインのみの利用では
   不要なため、依存は個別に導入する):
   ```bash
   git clone https://github.com/Tencent/Hunyuan3D-2 third_party/Hunyuan3D-2
   .venv/bin/pip install -e third_party/Hunyuan3D-2 --no-deps
   .venv/bin/pip install -r requirements-gpu.txt
   ```
   `requirements-gpu.txt` には diffusers / transformers / einops / omegaconf /
   accelerate / opencv-python-headless / scikit-image / pymeshlab (shapeパイプ
   ラインのpostprocessorsが依存) と rembg / onnxruntime(CPU版)が含まれる。

   注意: rembg/hy3dgen系の依存解決により numpy が 2.x系に上がる
   (`requirements.txt` は `numpy<3.0` に緩和済み。trimesh / fast-simplification /
   meshproc は numpy 2.x でも問題なく動作することを確認済み)。

3. ジェネレータを切り替えて起動:
   ```bash
   IMAGE3D_GENERATOR=hunyuan3d ./run.sh
   ```
   初回生成リクエスト時にモデルがHuggingFaceの `tencent/Hunyuan3D-2` リポジトリ
   (`hunyuan3d-dit-v2-0` サブフォルダ、標準shapeモデル、約9.2GB)から
   `~/.cache/huggingface` にダウンロードされ、以降はプロセスに常駐する
   (`server/generators/hunyuan3d.py`、NFR-3)。

4. 実画像での実測結果(テスト画像: ぬいぐるみのフィギュア写真、640x960、
   `IMAGE3D_GENERATOR=hunyuan3d`、steps=30, octree_resolution=384、RTX PRO 6000
   Blackwell、他プロセスがVRAM約30GB使用中の状態で計測):
   - パイプラインロード時間(初回): 約15秒
   - 生成時間(ロード後、diffusion + volume decoding): 約13秒
   - ジョブ全体(前処理〜後処理〜completed): 約31秒(NFR-1の60秒以内を達成)
   - VRAMピーク: 約36.6GB(他プロセス分含む。Hunyuan3D-2自体の純増分は約7GB)
   - 生成メッシュ(後処理前): 386,134頂点 / 772,232面、non-watertight
   - 後処理後(meshproc、max_faces=200,000、target_height_mm=100):
     99,998頂点 / 200,000面、**watertight**、高さ 100.01mm

5. 環境変数(`server/config.py`、必要な場合のみ上書き):

   | 変数 | デフォルト | 説明 |
   |---|---|---|
   | `IMAGE3D_HY3DGEN_MODEL_PATH` | `tencent/Hunyuan3D-2` | HuggingFaceリポジトリID |
   | `IMAGE3D_HY3DGEN_SUBFOLDER` | `hunyuan3d-dit-v2-0` | 使用するshapeモデルのサブフォルダ(mini版に切替可) |
   | `IMAGE3D_HY3DGEN_MODELS_DIR` | (hy3dgen既定の`~/.cache/hy3dgen`) | hy3dgenのローカルモデルキャッシュ探索先 |

## Phase 2.5: 4色カラープリント対応 (FR-8)

Bambu Lab AMS、Prusa MMU等のマルチフィラメント方式カラー3Dプリンタ(最大4色)
向けの出力に対応する。テクスチャ生成AIは使わず、入力画像(背景除去後)を
メッシュ正面から直交投影して頂点カラーを取得し、k-meansで2〜4色に量子化する
簡易方式(`server/colorproc.py`)。正面画像は正面側の頂点にのみ投影し、追加ビューに
背面画像がある場合は背面側へ背面画像を投影する。背面画像が無い場合、背面側と
側面/上下の曖昧な頂点はベース色になる。

### 使い方

1. パラメータフォームの「カラーモード(4色プリンタ向け)」にチェックを入れる。
2. 「色数 (n_colors)」で2〜4を選択(デフォルト4)。
3. 生成後、モデル情報バーに量子化されたパレット(色チップ■+面数比率%)が表示される。
4. ビューアには頂点カラー付きモデルが表示される(GLBに`COLOR_0`属性として出力、
   three.jsのGLTFLoaderが自動で頂点カラー表示する)。
5. エクスポートの「3MF」ボタンでダウンロードすると、通常の単色3MFではなく
   **色ごとに分割された最大4オブジェクト**(名前 `color_1`〜`color_4`、
   表示色付き)を含む3MFが得られる。STL/OBJ/通常想定の単一3MFは従来通り
   形状のみ(色情報なし)。

APIパラメータ: `params` JSONに `color_mode`(`"none"` | `"color4"`)と
`n_colors`(2〜4、デフォルト4)を指定する。

```bash
curl -s -X POST http://127.0.0.1:8000/api/jobs \
  -F "image=@sample.png" \
  -F 'params={"color_mode":"color4","n_colors":4,"seed":42}'

# 3MF(カラーモード時は色ごとに分割されたマルチオブジェクト版)
curl -s "http://127.0.0.1:8000/api/jobs/<job_id>/download?format=3mf" -o model_color.3mf
```

ジョブ完了時の `stats.palette` にHEXカラーと面数比率が入る:

```json
"palette": [
  {"hex": "#090512", "face_ratio": 0.356},
  {"hex": "#f0e1cc", "face_ratio": 0.249},
  {"hex": "#b17f7a", "face_ratio": 0.230},
  {"hex": "#753444", "face_ratio": 0.166}
]
```

### スライサーでのフィラメント割当手順(概説)

1. 上記3MFファイル(`model_color.3mf`相当、ダウンロード時のファイル名は
   `<job_id>_color.3mf`)をBambu Studio / PrusaSlicerなど対応スライサーで開く。
2. 3MF内には最大4個のオブジェクト(`color_1`〜`color_4`)が別々のパーツとして
   読み込まれる。各オブジェクトはモデル情報バーのパレット表示・
   `stats.palette`のHEXに対応する色でエクスポートされている。
3. スライサーのオブジェクト/パーツ一覧から各 `color_N` を選択し、
   対応するAMS/MMUスロットのフィラメント色を割り当てる
   (パレットのHEXに近い色のフィラメントを選ぶと元画像の配色に近くなる)。
4. 通常のマルチカラー印刷設定(パージタワー・ウォッシングタワー等)で
   スライスする。

### 実機検証結果 (GPU, momo.png)

`IMAGE3D_GENERATOR=hunyuan3d`、`color_mode=color4`, `n_colors=4`, `seed=42`、
入力画像 `momo.png`(640x960、ぬいぐるみ写真)で検証:

- ジョブ完了時間: 約34秒
- `stats.palette`: 4色(黒系・生成りの毛色・肌色系・臙脂色の4クラスタ)、
  face_ratio合計 ≈ 1.0
- 3MFダウンロード → trimeshで再読込 → ジオメトリ数4(`color_1`〜`color_4`、
  面数合計200,000 = 単色出力時と同一)
- GLBに`COLOR_0`頂点カラー属性が含まれ、three.jsビューアで色表示を確認
- **左右ミラー検証**: 入力画像は非対称な特徴(右耳の黒い内側パネル)を持つため、
  生成メッシュの頂点カラーをメッシュ正面(-Y向き)から直交投影して可視化し、
  画像の右側にある黒いパネルがメッシュの+X側(画像を正面から見て右側)に
  正しく再現されることを確認した。`server/colorproc.py`の`_U_TO_X_SIGN=+1`
  (画像u=0が-X側、u=1が+X側)がこの実機検証で確定した値である。

## Phase 3a: マルチビュー入力 (FR-9)

複数ビュー画像(正面必須+背面/左側面/右側面の任意組合せ)から3Dモデルを生成できる。
複数ビュー時は Hunyuan3D-2 のマルチビューモデル `hunyuan3d-dit-v2-mv`
(リポジトリ `tencent/Hunyuan3D-2mv`。単一ビュー用の `tencent/Hunyuan3D-2` とは
別リポジトリである点に注意)を使用する。単一画像時は従来通り
`hunyuan3d-dit-v2-0` を使用する。両パイプラインは別インスタンスとして
共存常駐する(`server/generators/hunyuan3d.py`)。

### 使い方(マルチビュー生成)

1. 左ペイン「1. 画像アップロード」で正面画像をアップロードする(必須)。
2. 「追加ビュー(任意)」の背面/左側面/右側面の枠に、対応する画像を
   ドラッグ&ドロップまたはクリックしてアップロードする(いずれも省略可、
   個別に「×クリア」で解除可能)。
3. 追加ビューを1枚以上指定すると、進捗欄付近に「Nビュー(front/back/...)で
   生成」という表示が出る。
4. 「3Dモデルを生成」を押すと、複数ビュー時は自動的にマルチビューパイプライン
   (`hunyuan3d-dit-v2-mv`)で生成される。

APIでは `POST /api/jobs` の multipart フィールドとして `image`(正面、必須)に
加え `image_back` / `image_left` / `image_right`(任意)を送信する:

```bash
curl -s -X POST http://127.0.0.1:8000/api/jobs \
  -F "image=@front.png" \
  -F "image_back=@back.png" \
  -F 'params={"seed":42}'
```

ジョブ完了後、`GET /api/jobs/<job_id>` の応答に `views`(例:
`["front", "back"]`)が含まれ、実際にどのビューが使われたかを確認できる。
各追加ビューにも背景除去(`remove_bg`指定時)が適用される。カラー投影
(`color_mode=color4`時の頂点カラー、FR-8)は正面(front)画像を正面側に使い、
背面(back)画像があれば背面側にも使用する。

### 実機検証結果 (GPU, momo.png + 左右反転画像)

`IMAGE3D_GENERATOR=hunyuan3d`、front=`momo.png`(640x960)、
back=momo.pngの左右反転画像、`seed=42`、`color_mode=color4`, `n_colors=4`
の2ビュージョブで検証(ポート8021、8020の既存プロセスとは別プロセス):

- **mvモデル初回DLサイズ**: 約9.2GB
  (`tencent/Hunyuan3D-2mv`、subfolder `hunyuan3d-dit-v2-mv`、
  `~/.cache/huggingface` に保存。単一ビュー用モデルとは別リポジトリのため
  重複してDLされる)。
- **生成時間**:
  - 初回(モデルDL含む): ジョブ作成から完了まで約135秒
    (うちDL+ロード+生成が約120秒)。
  - 2回目以降(モデル常駐後): ジョブ作成から完了まで約22秒
    (NFR-1の60秒以内を達成)。
- **生成メッシュ統計**(後処理後、max_faces=200,000、target_height_mm=100):
  99,972頂点 / 200,000面、non-watertight、高さ 100.00mm、
  bbox (68.9 x 46.3 x 100.0) mm。
- **GLB**: 頂点カラー(`COLOR_0`)付きで出力、99,972頂点分のカラーを保持。
- **3MF**: 色ごとに分割された4オブジェクト、面数合計200,000
  (単色出力時と一致)。
- **STL**: 高さ100.00mm、trimeshで再読込可能。
- 同一サーバプロセス上で単一ビュー用パイプライン(`hunyuan3d-dit-v2-0`)と
  マルチビュー用パイプライン(`hunyuan3d-dit-v2-mv`)が共存常駐し、
  それぞれ単一ビュージョブ・複数ビュージョブを問題なく処理できることを確認した
  (VRAM: 両モデル常駐時で合計使用量 約46GB、他プロセス分約35.7GB含む)。

## Phase 3b: プリセット+オーバーハングヒートマップ (FR-11, FR-12)

フロントエンドのみの拡張(サーバAPI変更なし)。`web/index.html` / `web/app.js` /
`web/viewer.js` / `web/style.css` を変更。

### プリセット (FR-11)

「2. 生成パラメータ」フォーム最上部にプリセットセレクタを追加。選択すると
対応するパラメータがフォームに一括反映される。

| プリセット | target_height_mm | octree_resolution | max_faces | カラーモード |
|---|---|---|---|---|
| フィギュア | 100 | 384 | 200,000 | 変更なし |
| 小型フィギュア | 60 | 256 | 100,000 | 変更なし |
| ペンダント | 40 | 256 | 80,000 | OFFに強制 |
| 高精細 | 150 | 512 | 400,000 | 変更なし |

先頭の「カスタム」は何も反映しない初期値。プリセット反映後にユーザーが
`target_height_mm` / `octree_resolution` / `max_faces` / カラーモードの
いずれかを個別に変更すると、セレクタ表示は自動的に「カスタム」に戻る
(実装は `web/app.js` の `PRESETS` 定義と `change` イベントリスナー)。

### オーバーハングヒートマップ (FR-12)

3Dビューア上部の表示切替に「オーバーハング」ボタンを追加(既存の
シェーディング/ワイヤーフレームと排他)。クリックすると、表示中メッシュの
面法線から下向き傾斜角を算出し、頂点色として以下の配色でベイクした
`MeshBasicMaterial`(照明の影響を受けず頂点色をそのまま表示)に切り替える。

- 接地面付近(モデル高さの下端2%未満): 薄青(サポート不要)。
- 下向き傾斜角が閾値(既定45°)を超える面: 赤(超過度合いに応じて白→赤の
  グラデーション)。
- 閾値以下の面: 白〜薄グレー(傾斜が小さいほど白に近い)。

オーバーハングモード中のみ閾値スライダー(30°〜70°、1°刻み)を表示し、
変更するとその場でヒートマップを再計算する(サーバ通信なし、
`Viewer.setOverhangThreshold()`)。

傾斜角は、GLBロード時にワールド座標変換した面法線とワールド下方向
(`(0, -1, 0)`。ビューアは生成メッシュ(Z-up)をラッパーグループでX軸-90度回転し
Y-upとして表示しているため、シーン内では常にY軸が造形の高さ方向になる)との
なす角から求める。

シェーディング/ワイヤーフレームに戻すと、退避しておいた元のマテリアルと
頂点カラー属性(4色プリント時の `COLOR_0` 等)を復元する
(`Viewer._backupAndApplyOverhang()` / `_restoreOriginalMaterials()`)。
新しいモデルをロードするとオーバーハングモードは自動的に解除され、
表示はシェーディングに戻る。

### 動作確認 (mock、ポート8021)

`IMAGE3D_GENERATOR=mock` のサーバ(ポート8021、8020の既存プロセスとは別)で、
既存の完了済みジョブ(4色カラーのぬいぐるみ形状、99,972頂点/200,000面)を
ビューアにロードし、ブラウザJS経由で以下を確認した(新規ジョブは作成せず、
既存ジョブの参照のみで検証したためジョブ削除は不要だった)。

- プリセット4種それぞれで `target_height_mm` / `octree_resolution` /
  `max_faces` がフォームに反映されること、ペンダント選択時にカラーモードが
  OFFになること、反映後に個別フィールドを変更するとセレクタが「カスタム」に
  戻ることを確認。
- オーバーハングボタン押下で頂点カラー属性が書き換わり、閾値45°時に
  99,972頂点中 赤(オーバーハング)7,883・薄青(接地面)1,723・
  白〜グレー(安全)90,366(その他0)に分類されることを確認(数値は
  `geometry.getAttribute("color")` を直接読み出して集計)。
- 閾値スライダーを30°に下げると赤判定頂点が16,745に増加、70°に上げると
  ごく一部(股下など急傾斜面のみ)に減ることを確認(閾値と赤面積が単調に
  連動)。
- スクリーンショットで赤(オーバーハング: 腕の下側・肩・股下等)/
  白(安全)/薄青(接地面)の3配色が視認できることを確認。
- シェーディングボタンに戻すと、頂点カラーが元のパレット値
  (`[0.553, 0.345, 0.2]` 等)に完全復元され、マテリアルも元のカラー表示に
  戻ることを確認。

## Phase 3c: テクスチャ生成 (texgen, FR-10)

Hunyuan3D-2 の paint パイプライン(`hunyuan3d-paint-v2-0` + `hunyuan3d-delight-v2-0`)
を用いて、生成メッシュに全周テクスチャ(UV展開 + 2048x2048テクスチャ画像)を
焼き込む。`texture_mode=paint` を指定した場合のみ実行される(デフォルト `none`)。

### セットアップ(custom_rasterizer CUDA拡張のビルド)

texgenの内部レンダラ(`hy3dgen/texgen/differentiable_renderer`)は
`custom_rasterizer_kernel` というCUDA拡張(pybind11 + CUDA C++)を要求する。
この拡張は Hunyuan3D-2 リポジトリに同梱されているがビルド済みバイナリは
配布されないため、対象マシンでソースからビルドする必要がある。

**重要な既知の落とし穴**: システムの `nvcc`(`nvcc --version` で確認)と
torchのCUDAビルド(`python -c "import torch; print(torch.version.cuda)"`)の
**メジャーバージョンが一致しないとビルドに失敗する**。本プロジェクトの実機は
システムCUDAが13.0、torchがcu128(CUDA 12.8)ビルドという不一致環境だったが、
`/usr/local/cuda-12.8` に別途CUDA 12.8ツールチェーンが用意されていたため、
`CUDA_HOME`/`PATH` で明示的にそちらを指定してビルドすることで解決した
(torchバージョンチェックの回避等は不要だった)。

```bash
# 1. 追加のPython依存を導入(shapeパイプラインのみの --no-deps 導入では
#    含まれていないもの。requirements-gpu.txt参照)
.venv/bin/pip install xatlas pybind11 ninja pygltflib

# 2. torchのCUDAビルドと同じメジャーバージョンのCUDAツールチェーンを用意する。
#    無ければ https://developer.nvidia.com/cuda-12-8-0-download-archive 等から
#    該当バージョンのtoolkitのみ(ドライバは不要)を追加インストールする。
ls /usr/local/ | grep cuda   # 例: cuda-12.8 が既にあるか確認

# 3. custom_rasterizer をビルド・インストール
cd third_party/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
CUDA_HOME=/usr/local/cuda-12.8 PATH="/usr/local/cuda-12.8/bin:$PATH" \
  TORCH_CUDA_ARCH_LIST="12.0" \
  ../../../../../.venv/bin/pip install . --no-build-isolation --no-deps

# 4. 動作確認(torchを先にimportしないとlibc10.so等が解決できない点に注意)
.venv/bin/python -c "
import torch
import custom_rasterizer as cr
print('OK:', cr.rasterize)
"
```

`TORCH_CUDA_ARCH_LIST` は対象GPUのCompute Capabilityに合わせる
(RTX PRO 6000 Blackwell / sm_120 の場合は `"12.0"`)。

`differentiable_renderer/mesh_processor`(pybind11拡張、`mesh_processor.cpp`)は
**ビルド不要**: `mesh_render.py` は `from .mesh_processor import meshVerticeInpaint`
というパッケージ内相対importで読み込むため、同ディレクトリの純Python実装
(`mesh_processor.py`)がPythonのimport解決で優先され、コンパイル済み拡張が
無くても動作する。

**既知の追加修正(vendored コードのパッチ)**: 導入したdiffusersのバージョン
(0.39.0)では、ローカルの `custom_pipeline`(`hy3dgen/texgen/hunyuanpaint/`)を
`DiffusionPipeline.from_pretrained(..., custom_pipeline=...)` でロードする際に
`trust_remote_code=True` を明示しないと `ValueError` になる仕様変更が入っている。
`third_party/Hunyuan3D-2/hy3dgen/texgen/utils/multiview_utils.py` の
`DiffusionPipeline.from_pretrained(...)` 呼び出しに `trust_remote_code=True` を
追加するパッチを適用済み(このリポジトリに同梱された既知のコードを読み込む
だけなので安全)。third_party を作り直す場合は同様のパッチが必要になる。

### 可用性チェックとフォールバック(3c-3)

`server/texture.py` の `is_available()` が、依存import(`custom_rasterizer_kernel`,
`hy3dgen.texgen.Hunyuan3DPaintPipeline`)とGPU有無を実ロードせずに確認し、
`GET /api/health` の `texgen_available` に反映する。ビルド未実施・GPU無し環境
では `false` になり、UIの「テクスチャ生成(実験的)」チェックボックスが
無効化され「この環境では利用できません」と表示される(サーバAPI自体は
`texture_mode=paint` を引き続き受け付けるが、実行時にpaintが失敗した場合と
同様にgracefulにフォールバックする)。

paint実行が失敗した場合(モデル未DL・OOM・その他例外)もジョブは `failed` に
せず、`meta.json` の `warnings` に日本語メッセージを記録した上で、従来の
正面/背面投影方式(FR-8、`colorproc.project_multiview_colors`)による `color_mode=color4`
処理を続行する。ジョブJSONの `textured` フィールドで実際にpaintが成功したか
どうかを判定できる(`true`=テクスチャ付きGLB、`false`=フォールバック)。

### 使い方

1. `/api/health` で `texgen_available: true` であることを確認(UIでは
   チェックボックスが有効表示されていれば利用可能)。
2. パラメータフォームの「テクスチャ生成(実験的)」にチェックを入れて生成する。
3. 完了後、ビューア用GLB(`GET /api/jobs/<job_id>/model.glb`)にテクスチャ
   (2048x2048 PNG、`baseColorTexture`)付きのPBRマテリアルが焼き込まれる。
   STL/OBJ/通常3MFは従来通り形状のみ。
4. `color_mode=color4` と併用した場合、頂点カラーの取得元が
   「入力画像の正面投影」から「焼き込まれたテクスチャをUV経由でサンプリング」
   (`server/texture.py: sample_vertex_colors_from_texture`)に切り替わり、
   全周の実際の配色に基づいた4色3MFが生成される(側面・背面の色も反映される
   ため、FR-8単体運用時より配色精度が上がる)。

```bash
curl -s -X POST http://127.0.0.1:8000/api/jobs \
  -F "image=@sample.png" \
  -F 'params={"texture_mode":"paint","color_mode":"color4","n_colors":4,"seed":42}'
```

### 背面参照画像 (`image_back`) の利用

`texture_mode=paint` と同時に `image_back`(SPEC.md §3.8 / FR-9 の追加ビュー)を
指定すると、texgenのmultiview拡散に**背面の参照画像**を渡せる。指定しない場合、
texgenは正面画像1枚だけを参照して残り5ビュー(背面含む)をモデル自身が
推測生成するため、背面に正面の配色が回り込みやすい(下記「既知の制限」参照)。
`image_back` を渡すと `server/jobs.py: JobManager._run_paint` が
`TexturePipelineWrapper.paint(mesh, image, back_image=...)` を呼び出し、
背面ビュー生成時に実際の背面画像の配色(髪型・服の色等)が反映されるようになる。

```bash
curl -s -X POST http://127.0.0.1:8000/api/jobs \
  -F "image=@front.png" \
  -F "image_back=@back.png" \
  -F 'params={"texture_mode":"paint","seed":42}'
```

- `left`/`right` は今回のE2E検証では未対応(下記の実装メモ参照)。指定しても
  paintへは渡らない(back のみ反映される)。
- 内部実装: `server/texture.py` の `_patch_multiview_ref_camera_info` が、
  ロード済みパイプラインインスタンスの `multiview_model.__call__` を実行時に
  ラップし、`camera_info_ref`(参照画像インデックス)を画像枚数に応じて
  `[[0, 1, ...]]` に拡張する。vendored コード(`multiview_utils.py`)自体は
  書き換えない。ラップ対象の属性構造が想定と異なる場合は例外にせず警告ログを
  出し、従来動作(正面1枚のみ参照)にフォールバックする。

### 実機検証結果 (GPU, momo.png, ポート8021)

`IMAGE3D_GENERATOR=hunyuan3d`、`texture_mode=paint`、`color_mode=color4`、
`n_colors=4`、`seed=42`、入力画像 `momo.png` で検証(検証後ジョブは削除済み):

- custom_rasterizer のビルド: `/usr/local/cuda-12.8` を明示指定して**1回目の
  試行で成功**(torchバージョンチェック回避等のハック不要)。
- paintパイプライン初回実行: `hunyuan3d-paint-v2-0` + `hunyuan3d-delight-v2-0`
  のHuggingFaceからの初回ダウンロードを含めて完了(ジョブ全体で約104秒)。
- 2回目以降(モデル常駐後): ジョブ全体で約80〜100秒(shape生成 + 後処理 +
  paint)。
- 生成GLBの検証: glTF JSONを直接パースし、`materials[0].pbrMetallicRoughness
  .baseColorTexture` が存在し `images[0]` (PNG, 2048x2048) を参照、
  `meshes[0].primitives[0].attributes` に `TEXCOORD_0` が含まれることを確認。
  trimeshでの再読込でも `visual.kind == "texture"` かつ
  `material.baseColorTexture.size == (2048, 2048)` を確認した。
- 3MFダウンロードのジオメトリ数: 4(`color_1`〜`color_4`)。
- `job["textured"]` が `true`、`job["warnings"]` が空であることを確認。
- VRAMピーク: 約54.4GB(shapeパイプライン + paintパイプライン + delightモデル
  すべて常駐した状態。NFR上の96GB VRAM予算内)。
- `texture_mode=paint` かつ `color_mode=none` の組合せでも同様にテクスチャ付き
  GLBが生成されることを確認。
- 検証用サーバ(ポート8021)は検証後に停止し、テストジョブは全て削除した。

### 既知の制限(texgen固有)

- custom_rasterizer のビルドには、torchのCUDAビルドとメジャーバージョンが
  一致するCUDAツールチェーンが別途必要(システムのnvccと不一致な場合)。
  ビルド環境が用意できない場合は `texgen_available=false` となり自動的に
  フォールバックする(アプリ自体は壊れない)。
- paintパイプラインはCPU実行を想定していない(`Hunyuan3DTexGenConfig` が
  `device='cuda'` 固定)。GPU無し環境では `is_available()` が常に `false` を
  返す。
- paint処理は shape生成用パイプラインとは別にVRAMを消費する(delight +
  multiview拡散 + 内部レンダラ)。直列キュー(NFR-2)により同時実行は防がれる
  が、`target_height_mm`/`max_faces` を大きくした高解像度メッシュではVRAM
  使用量が増える点に注意。
- paint後のテクスチャは全周を6視点(正面/背面/左右/上/下相当)からの
  マルチビュー拡散結果をベイクする方式のため、細部の一貫性は入力画像の
  品質・被写体の複雑さに依存する。
- **背面参照画像 (`image_back`) 対応(2026-07)**: 従来、`texgen`は正面画像
  1枚だけを参照画像として使い、背面含む残り5ビューをマルチビュー拡散モデルが
  新規推測生成していたため、背面に正面の配色が回り込む問題があった
  (`third_party/Hunyuan3D-2/hy3dgen/texgen/utils/multiview_utils.py:80` の
  `camera_info_ref = [[0]]` 固定が原因)。`image_back` を指定すると
  `server/texture.py` の実行時パッチ(`_patch_multiview_ref_camera_info`)が
  `camera_info_ref` を参照画像枚数に応じて拡張し、背面ビュー生成時に実際の
  背面画像の配色が反映されるようになった(上記「背面参照画像」節参照)。
  `left`/`right` は3枚以上でのcamera_info_ref拡張が未検証のため今回は
  対象外(`server/jobs.py: JobManager._run_paint` はback画像のみをpaintへ渡す)。
- **目など小さく高コントラストな特徴のズレ・シームは既知の限界**:
  背面参照画像を渡しても、`Hunyuan3DPaintPipeline`
  (`third_party/Hunyuan3D-2/hy3dgen/texgen/pipelines.py`)がマルチビュー
  拡散モデルで生成する各ビューは独立生成のため完全には一致せず、
  それをメッシュへベイクする際にズレ・二重写り・輪郭のシームとして
  現れることがある。彫りの浅い(平坦に近い)ジオメトリのぬいぐるみ系被写体
  で特に目立ちやすい。`__call__(self, mesh, image)` にsteps/解像度等の
  品質調整パラメータは公開されておらず、アプリ側からのチューニング余地は
  無い。正面の色精度を優先したい場合は `texture_mode=none` にして
  従来の正面/背面投影(`colorproc.project_multiview_colors`)を使う方が
  ズレは出ないが、360°の質感は失われ側面が単色寄りになるトレードオフがある。
- `hy3dgen/texgen/utils/multiview_utils.py` に `trust_remote_code=True` を
  追加するパッチが必要(上記セットアップ参照)。third_party ディレクトリを
  再取得(git clone)した場合は再適用が必要。

## Phase R4: リグ/VRM化サービス連携 (rig-service)

生成した3Dモデルに**ヒューマノイドボーンを自動で付与**し、VRM 1.0 として
書き出す**別リポジトリのサービス** rig-service と連携できる。
image-3d は生成済みGLBをPOSTしてジョブURLを開くだけの疎結合で、
リグ結果は保持しない(コードの相互参照は無く、繋がりはHTTPのみ)。

### ワークフロー: Tポーズ画像で生成 → リグ化

1. **Tポーズの立ち絵を用意する**(腕を左右へ水平に広げた姿勢)。
   **背景は事前に抜いた透明PNGにするのが安全**。rembg は背景と同系色の細い部位
   (白い毛の腕など)を丸ごと切り落とすことがあり、実際にTポーズの片腕が消えた
   ことがある。透明な画像を渡せば rembg は自動でスキップされる。
   なお hunyuan3d は欠けた腕を補完しがちで問題が見えにくいが、Pixal3D は入力に
   忠実なので欠落がそのまま出る。
   これがリグ品質を決める最重要ポイント。rig-service は腕の張り出しから
   肩の高さを実測するため、腕を下ろした画像だと関節位置を推定できず
   人体標準比での代用になり精度が大きく落ちる(その旨が警告として返る)。
   頭身の低いデフォルメ体型・動物型キャラでも、Tポーズでさえあれば
   股下・肩・首・脚幅はメッシュから実測されるので問題ない。
2. 通常どおり image-3d で3Dモデルを生成する。
3. rig-service を起動する(別ターミナル)。rig-service は別リポジトリなので、
   未取得なら先に clone してセットアップする(手順はそちらの README 参照)。

   ```bash
   cd ../rig-service
   ./run.sh                       # http://127.0.0.1:8100
   ```

4. image-3d を `IMAGE3D_RIGSVC_URL` 付きで起動する。

   ```bash
   IMAGE3D_RIGSVC_URL=http://127.0.0.1:8100 ./run.sh
   ```

5. 完了したジョブを選び、エクスポート欄の **「リグ/VRM化」** ボタンを押す。
   別タブで rig-service が開き、リグ結果(ボーン数・ウェイト付与率・
   Tポーズ判定・警告)とダウンロードリンクが表示される。

**`IMAGE3D_RIGSVC_URL` が未設定ならボタンは表示されない**(`/api/health` の
`rigsvc_url` で判定)。rig-service を使わない運用に影響はない。

### API

```bash
# 完了ジョブのGLBを rig-service へ送る(paramsは任意、rig-serviceにそのまま渡る)
curl -s -X POST http://127.0.0.1:8000/api/jobs/<job_id>/rig \
  -F 'params={"height_m":1.6,"vrm_meta":{"authors":["yourname"]}}'
# => {"rig_job_id": "...", "url": "http://127.0.0.1:8100/?job=..."}
```

数十MBのGLBをブラウザに往復させないため、また rig-service 側にCORS設定を
強いないため、**サーバ経由で転送する**(ブラウザから直接POSTしない)。

| 状態 | 応答 |
|---|---|
| `IMAGE3D_RIGSVC_URL` 未設定 | 503 |
| ジョブが未完了 | 409 |
| rig-service に接続できない / エラー応答 | 502(image-3d 側は落ちない) |

### 出力

rig-service からは以下が得られる(詳細は rig-service の README):

- **リグ済みGLB** — glTF仕様準拠(Y-up・原点=足元・メートル系)、21ボーン。
  Godot 4 はそのままインポートでき、`SkeletonProfileHumanoid` の BoneMap が
  自動で埋まる(Godot 4.4.1 で検証済み)。
- **VRM 1.0** — 上記に `VRMC_vrm` 拡張(meta + humanoid 21ボーン)を足したもの。

## TRELLIS.2ジェネレータ (形状エンジンハイブリッド)

`IMAGE3D_GENERATOR=trellis2` で、形状生成を **TRELLIS.2-4B**
(microsoft/TRELLIS.2, MIT) に置き換えられる。アニメキャラで
hunyuan3d より形状忠実度が高い(指5本の分離・髪の房・眼窩の造形。
検証: `data/spikes/trellis2-hybrid-20260730/`、メモリ
trellis2-hybrid-spike-verdict)。

- **テクスチャは texgen (512px天井) を通らない**。TRELLIS.2 自身の
  テクスチャ付きメッシュ (`o_voxel.postprocess.to_glb(remesh=True)`,
  narrow-band Dual Contouring で実質閉曲面化) をそのままビューアGLBにし、
  `texture_refine=true` なら既存の texrefine で参照画像を全解像度反映する
  (`server/jobs.py` の `_run_pretextured_paint`)。
- 実測 (RTX PRO 6000): 生成 ~25s (VRAM ~3.3GB) + GLB化 ~4分 (VRAM ~4.5GB、
  うちGPU UV展開 ~226s)。
- 前提は pixal3d と同じ: 専用venv `.venv-pixal3d` + `third_party/TRELLIS.2`。
  `.claude/launch.json` の **image3d-server-trellis2** (port 8023) で起動する。
  hunyuan3d サーバ (8020) と同時起動できるので、ポートで使い分ける。
- 本物の xformers / nvdiffrast は不要:
  `server/generators/trellis2_shims/` の互換スタブ (xformers=SDPA代替,
  nvdiffrast=drtk転送) が import できない場合のみ自動で使われる。
- 生成は単一画像 (front) のみ。back/left/right を付けたジョブは
  texture_refine の参照としてのみ利用される。
  **複数画像から形状を生成することはできない**(後述「限界」参照)。
- TRELLIS.2 も画像条件付けに DINOv3 を使うため、UIに "Built with DINOv3"
  クレジットが表示される (DINOv3 License の付帯義務)。

### trellis2 経路のための texrefine の適応 (2026-07-30〜08-02)

texrefine は元々「texgen の顔は白紙(512px・のっぺらぼう)」という前提の
道具で、参照の直接転写が常に正解だった。TRELLIS.2 は**最初から位置の正しい
一貫したテクスチャを描く**ため、同じ扱いをすると逆に悪化する。
`_run_pretextured_paint` は以下を有効にして呼ぶ。

| 機能 | 何をするか | なぜ必要か |
|---|---|---|
| `head_tone_only` | 頭部は直接転写せず(blend=0)、転写できた頭部テクセルの「参照−元」差分を顔幅の半分の半径で平均した**低周波トーン**だけを乗せる | 数pxずれた参照を顔に直接重ねると頬・顎に斑点ノイズが出て素の顔より悪化する。構造は生成器、色味は参照、という分業にする |
| `match_base_colors` | 未転写テクセルの元色を、3D近傍の転写済みテクセルの「参照−元」差分で局所補正 | 生成器のテクスチャは参照より彩度がくすむ。遮蔽で転写できない凹み(目のくぼみ・鼻の脇)が色調差の「箱」として浮くのを防ぐ |
| 隠れ面の色継承 | **遮蔽の証拠がある**テクセル(まともな角度で割り当てられたのに深度で遮られた面)へ、覆っている表面の転写色を継承 | 服の下の体表面に生成器が幻覚色(実測: オレンジ)を塗るため、リグでポーズを付けると数mmの層ずれで覗く。「服の裏は服の色」にして目立たなくする |
| 盲目領域の継承 | 全ビューで dot<0.15 = 4方向の水平視点からは原理的に見えない上向き/下向きの棚も継承対象にする | 襟の胸元・肩の上面は参照が存在しないので、幻覚色が可視のまま残る |
| 側面の内容ずれガード | 側面ビューからの転写は、参照画素と合成ビューの色差が100以内のときのみ | Tポーズの手は側面ビューで参照の胴体(シャツ)に投影され、シルエットは一致するので既存ガードを通過してしまう。「側面は精細化に使うが色替えには使わない」 |
| 背景抜き残しの除外 | 背景色に近い・小さい・シルエット近傍・楔形の画素を信頼マスクから外す | 髪の房の隙間の背景(白)が不透明のまま残り、髪に白ポツとして転写される |
| 頭部のチャート境界処理 | 3D対応するチャート境界テクセルの色・混合率を平均し、混合率を下げる方向にだけフェザー | to_glb のアトラスは微細チャートの集合(顔だけで1400超)で、境界が矩形の継ぎ目として見える |

あわせて trellis2 ジェネレータ側で2点:

- **metallic/roughness テクスチャを破棄して完全拡散にする**。to_glb は
  TRELLIS.2 自身が考える目の位置に艶を焼くが、texrefine は baseColor しか
  書き換えないため艶だけがずれて「灰色の矩形」として浮く(環境マップの無い
  ビューアでは metallic 面は灰色に沈む)。アニメ調の参照は拡散色なので
  艶を落とす方が忠実。
- **UVチャート境界の複製頂点で法線を統一する**。to_glb のチャートはボクセル
  格子に沿って細かく分割され、頂点の44%が境界の複製で法線が中央値27.7°
  食い違う。そのままだと顔一面にシェーディングの段差が出る。

### 運用上のコツ

- **steps は 12 のまま**にする(UIは trellis2 選択時に自動で12にする)。
  hunyuan 向けの既定値30で生成すると前髪が乱れ、実例では「帽子のツバ+赤い
  バンド」を幾何ごと幻視した。
- **形状・テクスチャの幻覚はシード依存**。胸元の肌色まだら・靴の斑などは
  後処理では直せないので、**シードを2〜3個試して良い個体を選ぶ**のが最も
  費用対効果が高い(1回約5分)。
- 参照画像は4面の**衣装デザインと姿勢が一致**していること。側面生成時に
  デザインが変わると、その面だけ別の柄が転写される。

## リポジトリ構成

```
image-3d/
├── docs/                     # 仕様書・開発方針・実装計画
├── server/
│   ├── main.py               # FastAPIエントリポイント
│   ├── config.py             # 設定(環境変数)
│   ├── jobs.py               # ジョブ管理・直列実行キュー・永続化
│   ├── generators/
│   │   ├── base.py           # Generator抽象基底
│   │   ├── mock.py           # mockジェネレータ
│   │   └── hunyuan3d.py      # Hunyuan3D-2ラッパ(Phase 2、Phase 3aでmvパイプライン追加)
│   ├── preprocess.py         # 画像前処理(背景除去・リサイズ)
│   ├── meshproc.py           # メッシュ後処理
│   ├── colorproc.py          # 4色カラープリント対応(Phase 2.5、頂点カラー投影・量子化・分割)
│   └── texture.py            # テクスチャ生成 texgen 統合(Phase 3c、paint常駐ラッパ・頂点カラーサンプリング)
├── web/                      # 静的フロントエンド
├── tests/                    # pytest
├── data/jobs/                # 生成物(gitignore対象)
├── third_party/Hunyuan3D-2/  # hy3dgen本体(git clone、Phase 2、gitignore対象)
├── requirements.txt          # base依存
├── requirements-gpu.txt      # Phase 2用追加依存
├── run.sh
└── README.md
```

## 既知の制限

- mockジェネレータは画像内容を反映しない決定的な形状(seedでバリエーション)を返す。
  実際の画像に基づく生成にはPhase 2でのHunyuan3D-2導入が必要。mockジェネレータは
  マルチビュー入力(`extra_views`)を無視する(単一ビュー用の決定的形状を返す)。
- マルチビュー生成(FR-9)は `hunyuan3d-dit-v2-mv` モデル(約9.2GB、単一ビュー用
  モデルとは別リポジトリ `tencent/Hunyuan3D-2mv`)の追加ダウンロードが必要。
  厳密なマルチビュー幾何整合(正面・背面・側面の完全な形状一致)はモデル自体の
  性能に依存し、本アプリ側での補正は行わない(SPEC.md §7の制約通り)。
- テクスチャ生成AIによるカラー3Dプリント(Hunyuan3D-2 paint pipeline)は
  Phase 3cで `texture_mode=paint` として対応済み(上記「Phase 3c」参照)。
  ビルド・依存が利用できない環境では自動的に無効化され、Phase 2.5の
  入力画像の正面/背面投影+k-means量子化による簡易4色対応(`server/colorproc.py`)に
  フォールバックする。
- `hy3dgen` はPyPI未配布のため、`third_party/Hunyuan3D-2` をgit cloneしての
  editableインストール(`--no-deps`)が必要。
- Hunyuan3D-2の生成メッシュは非watertightで返る場合が通常であり、
  `meshproc.process()` の後処理(穴埋め・簡略化)により実用上のwatertight化を
  行う。まれに複雑な形状で後処理後もwatertight化に失敗する場合があり、
  その際は `stats.watertight=false` としてUIに明示される(SPEC.md FR-4)。
- カラーモード(FR-8)は背景除去済み画像の正面/背面投影で頂点カラーを決める。
  追加ビューに背面画像が無い場合、背面側と側面/上下の曖昧な頂点はベース色に
  なるため、実際の側面・背面の配色とは一致しない場合がある。
- 3MFの色ごとのサブメッシュ(`color_1`〜`color_4`)は単体ではwatertightと
  限らない(積層方式のマルチカラー印刷では通常問題にならない)。
- パレット量子化はRGB色空間での単純なk-means(scipy.cluster.vq.kmeans2)で
  あり、知覚色差(CIE Lab等)は考慮していない。

### trellis2 経路の限界 (実測で確認済み。再挑戦の前に読むこと)

- **複数画像から形状を生成することはできない**。TRELLIS.2 にはカメラ(視点)
  条件が無く、条件画像はすべて「同じ視点から見た姿」と解釈される。初代
  TRELLIS の学習不要マルチ画像手法(stochastic / multidiffusion)を移植して
  実測したが、多視点再構成ではなく**姿勢のブレンド**になった(4面を混ぜると
  腕が前方に突き出た形状になり、幅88→53→29mm・奥行き23→44→49mm と正面像と
  側面像の中間へ寄る)。Microsoft もマルチイメージ条件の重みを公開していない。
  検証: `data/spikes/trellis2-multiview-20260801/`。
- **色再現性**: 生成テクスチャは参照より彩度がくすむ。ただしこれは
  **γ(色空間)のバグではない**。コミュニティには「base_color がリニアなのに
  sRGB として書かれるので γ2.2 で誤差48%減」という報告があるが、TRELLIS.2
  自身のレンダラが `basecolor ** 2.2` でリニア化している(=格納値はsRGB)。
  実測でもγ補正はジョブ依存で逆転する(パステル系 MAE 44.7→36.5 改善、
  暗色系 44.7→60.4 悪化)。差の実体は「albedo(無照明) vs 参照(陰影付きの絵)」
  であり、**グローバルγは入れないこと**。局所トーン補正が正解。
- **参照が届かない帯が残る**。手足の側面は前後ビューから浅い角度でしか見えず、
  側面参照が使えない場合は転写されない。局所トーン補正で埋めるが、必要な
  補正量の約1/3しか運べず、脚一周で8階調程度の色差が残る(半径・近傍数・
  フェザー・アンカー閾値・目標式はすべて実測して効果なしと確認済み)。
  原理的に閉じるには補正場をメッシュ表面上の拡散問題として解く必要がある。
- **髪の張り出しが大きいキャラでは側面参照が効きにくい**。正面画像から
  推定される髪の奥行きが実際より厚くなり、側面シルエットが参照と 60〜67%
  食い違う(実測: メッシュの側面投影面積が参照の1.60倍。**腕のポーズ違いは
  投影面積の0.2%しかなく主因ではない**)。シルエット一致ガードにより自動的に
  一致部分だけが使われるので害は無いが、側面の精細化は期待できない。
- **足元(靴)など小さく複雑な部位の模様は潰れる**。生成テクスチャの段階で
  斑になっており、texrefine は参照が正対して写る面しか直せないため、前後
  ビューから浅い角度でしか見えない靴の甲・側面には補正が届かない。
  上記「参照が届かない帯」と同じ原因で、現状の後処理では改善できない。
- **顔の造形は seed 依存**。顎・目まわりの自然さはジョブごとに変わる。
  texrefine は `head_tone_only` で顔の**色**しか触らない(構造は生成器に任せる)
  設計なので、造形が崩れた場合に後処理で救う手段は無い。**seed を変えて
  引き直すのが唯一の対処**。
- **人型以外では「頭」の判定が意図しない位置に出ることがある**。参照画像で
  高精細化するとき、幅が急にくびれる段より上を頭とみなして色味補正のみに
  切り替える (`head_base_height`)。人型では首を正しく当てる(実測: 人型で
  全高の81%、犬で59%)が、くびれのある物体では別の位置に線が引かれうる。
  くびれが見つからなければ「頭無し」として扱われ、全面が通常の転写になる。
- **入力の背景除去の質がそのまま上限になる**。髪の房の隙間に背景(白)が
  不透明で残ると、それが髪の色として転写される。texrefine 側に除外ガードは
  入れてあるが(上表「背景抜き残しの除外」)、根治は入力画像側で行うこと。
- **プリント用メッシュは non-watertight** になりやすい(体積0)。ビューア・
  リグ用途には影響しない。

## ライセンス

このリポジトリ(`server/`・`web/`・`docs/`・`tests/` 等、本プロジェクトのオリジナル
コード)は [Polyform Small Business License 1.0.0](LICENSE) の下で提供されます。

要約(法的拘束力があるのは[LICENSE](LICENSE)本文のみです):

- **非商用利用**は誰でも自由に可能。
- **商用利用**も、利用者の所属組織が
  - 従業員・業務委託者を合わせて100人未満、かつ
  - 直近の課税年度の総収益が100万USD未満(1982〜1984年基準のCPIで物価調整)

  の「小規模事業者」に該当する場合は許可されます。上記条件を満たさない大企業
  による商用利用のみが制限されます。
- 個人利用・小規模団体の商用利用は上記の通り許可されるため、条件を除外(許可)
  しています。

**third_party/Hunyuan3D-2 は対象外**: このリポジトリには含まれず(`.gitignore`
対象)、利用者が別途 `git clone` して導入します。Tencentの
`TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT`
(`third_party/Hunyuan3D-2/LICENSE`)など、それぞれの配布元のライセンス条件に
従ってください(利用地域制限・利用者数に応じた追加許諾要件などが定められて
います)。

### 利用しているOSS

`requirements*.txt` に列挙されたPython依存パッケージ、および同梱の
フロントエンドライブラリは、それぞれ独自のOSSライセンス下にあります(本プロジェクト
自体のライセンスとは別)。主要なものは以下の通りです(ライセンス表記は各配布元の
情報に基づく参考情報であり、正確な条件は各プロジェクトの配布物・パッケージ情報を
必ず確認してください)。

**バックエンド (`requirements.txt`)**

| パッケージ | ライセンス |
|---|---|
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| python-multipart | Apache-2.0 |
| trimesh | MIT |
| SciPy | BSD-3-Clause |
| NetworkX | BSD-3-Clause |
| lxml | BSD-3-Clause |
| NumPy | BSD-3-Clause |
| Pillow | MIT-CMU (HPND系) |
| fast-simplification | MIT |
| pytest | MIT |
| HTTPX | BSD-3-Clause |

**GPU/Hunyuan3D-2連携 (`requirements-gpu.txt`)**

| パッケージ | ライセンス |
|---|---|
| rembg | MIT |
| onnxruntime | MIT |
| PyTorch / torchvision | BSD-3-Clause |
| huggingface_hub | Apache-2.0 |
| einops | MIT |
| OmegaConf | BSD-3-Clause |
| Transformers | Apache-2.0 |
| Diffusers | Apache-2.0 |
| Accelerate | Apache-2.0 |
| opencv-python-headless | MIT(同梱のOpenCV本体はApache-2.0) |
| scikit-image | BSD-3-Clause |
| **pymeshlab** | **GPL-3.0**(デュアルライセンス、商用ライセンスも別途提供)。本プロジェクト自身のコード(`server/`)からは呼び出しておらず、`third_party/Hunyuan3D-2`(hy3dgen)側の内部依存として使用される。GPLの条件に懸念がある場合は導入を見送ることも可能(その場合hy3dgen側の一部後処理機能が制限される可能性があります)。 |
| xatlas | MIT |
| pybind11 | BSD-3-Clause |
| Ninja | Apache-2.0 |
| pygltflib | MIT |

**フロントエンド (`web/vendor/`)**

| ライブラリ | ライセンス |
|---|---|
| Three.js (r160, `web/vendor/three/`) | MIT |

**別リポジトリのモデル(`third_party/Hunyuan3D-2`、本リポジトリには含まれない)**

| 対象 | ライセンス |
|---|---|
| Tencent Hunyuan3D-2 (hy3dgen) | TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT(独自ライセンス。地域制限・月間アクティブユーザー数100万人超での別途許諾要件あり) |
