# Image-3D

画像から3Dプリントデータ(STL / 3MF / GLB / OBJ)を生成するローカルWebアプリ。

Web UIの使い方は [`docs/USAGE.md`](docs/USAGE.md)、詳細仕様は
[`docs/SPEC.md`](docs/SPEC.md)、開発方針は
[`docs/DEVELOPMENT_POLICY.md`](docs/DEVELOPMENT_POLICY.md)、実装計画は
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) を参照。

バグ報告・機能要望は [Issues](https://github.com/animede/image-3d/issues)、
コントリビューションは [CONTRIBUTING.md](CONTRIBUTING.md) を参照してください。

## 現在の状態(Phase 1〜3b)

- Image-to-3D 生成は **mockジェネレータ**(決定的なパラメトリックメッシュ)を使用。
- GPU / Hunyuan3D-2 は未導入でも、アップロード → 生成 → メッシュ後処理 →
  3Dビューア表示 → STL/3MF/GLB/OBJ ダウンロードの全パイプラインがE2Eで動作する。
- Phase 2 で `IMAGE3D_GENERATOR=hunyuan3d` により実モデルに切り替え可能(下記参照)。
- Phase 2.5 で4色カラープリンタ向け出力(`color_mode=color4`)に対応
  (下記「Phase 2.5: 4色カラープリント対応」参照)。
- Phase 3a でマルチビュー入力(正面+背面/左/右)とキャラクターシート自動分割に対応
  (下記「Phase 3a: マルチビュー入力+キャラクターシート自動分割」参照)。
- Phase 3b でパラメータプリセット(FR-11)とビューアのオーバーハングヒートマップ
  (FR-12)に対応(下記「Phase 3b: プリセット+オーバーハングヒートマップ」参照、
  フロントエンドのみの変更でサーバAPIは不変)。
- Phase 3c でテクスチャ生成(`texture_mode=paint`、FR-10)に対応(下記
  「Phase 3c: テクスチャ生成 (texgen)」参照)。custom_rasterizer CUDA拡張の
  ビルドが必要で、未導入環境では `/api/health` の `texgen_available=false` に
  応じてUI上で無効表示し、正面/背面投影方式(FR-8)にフォールバックする。
- 3つ目のジェネレータとして **Pixal3D**(MITライセンス、PBRテクスチャ付き出力)を
  統合(下記「Pixal3Dジェネレータ」参照)。専用venv `.venv-pixal3d` +
  `IMAGE3D_GENERATOR=pixal3d` の明示指定で使用する。

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
| `IMAGE3D_GENERATOR` | `auto` | `auto` \| `mock` \| `hunyuan3d` \| `pixal3d`。`auto` はGPU+hy3dgenが利用可能なら `hunyuan3d`、なければ `mock` に自動解決(`pixal3d` は専用venvでの明示指定のみ。「Pixal3Dジェネレータ」の節を参照) |
| `IMAGE3D_HOST` | `127.0.0.1` | バインドアドレス |
| `IMAGE3D_PORT` | `8000` | ポート |
| `IMAGE3D_MAX_UPLOAD_BYTES` | `20971520`(20MB) | アップロード上限 |
| `IMAGE3D_DEFAULT_TARGET_HEIGHT_MM` | `100` | 後処理のデフォルト目標高さ |
| `IMAGE3D_DEFAULT_MAX_FACES` | `200000` | 後処理のデフォルト面数上限 |

### アップロード画像と無関係なテスト形状が生成されるとき

mockジェネレータで動作している(画像を反映しない開発用の固定形状を返す)。
UIヘッダ右上の「生成エンジン」バッジ、または `GET /api/health` の `generator` で
確認できる。mock時はUI上部に警告バナーも表示される。対処:

1. GPU導入手順(後述のPhase 2節)を完了させる。`IMAGE3D_GENERATOR` 未指定
   (= `auto`)なら、GPU+hy3dgenが使える環境では自動的に `hunyuan3d` が選ばれる。
2. autoでmockになってしまう場合は `IMAGE3D_GENERATOR=hunyuan3d ./run.sh` で
   明示起動し、起動ログのエラー(torch/hy3dgen未導入、CUDA不可等)を確認する。

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

# キャラクターシート分割(Phase 3a、ジョブを作らない同期API)
curl -s -X POST http://127.0.0.1:8000/api/sheet/split -F "image=@sheet.png"
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
- `tests/test_sheet.py`: 合成RGBAシート画像(透明背景に離れた色付きシルエット3つ)で
  パネル自動検出(面積フィルタ・近接マージ・左→右ソート)とsuggested_view推定を
  rembgに依存せず決定的に検証。
- `tests/test_api.py`(Phase 3a追加分): mockで `image` + `image_back` の2ビュー
  ジョブがcompletedし、`views` フィールドが正しく記録されること、および
  `POST /api/sheet/split` が合成シート画像から3パネルを検出することを検証。
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

## Phase 3a: マルチビュー入力+キャラクターシート自動分割 (FR-9)

複数ビュー画像(正面必須+背面/左側面/右側面の任意組合せ)から3Dモデルを生成できる。
複数ビュー時は Hunyuan3D-2 のマルチビューモデル `hunyuan3d-dit-v2-mv`
(リポジトリ `tencent/Hunyuan3D-2mv`。単一ビュー用の `tencent/Hunyuan3D-2` とは
別リポジトリである点に注意)を使用する。単一画像時は従来通り
`hunyuan3d-dit-v2-0` を使用する。両パイプラインは別インスタンスとして
共存常駐する(`server/generators/hunyuan3d.py`)。

また、1枚のキャラクターシート画像(複数ビューが並んだ画像)から被写体パネルを
自動検出し、各パネルをUI上で正面/背面/左/右のいずれかに割り当てて生成に
使用できる(`server/sheet.py`)。

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

### 使い方(キャラクターシート自動分割)

1. 左ペイン「キャラクターシート分割(任意)」の「シート画像を選んで分割」を
   押し、複数ビューが1枚に並んだシート画像を選択する。
2. `POST /api/sheet/split` が呼ばれ、検出されたパネルがサムネイル一覧として
   表示される。各パネルには割当セレクト(正面/背面/左/右/使わない)が付き、
   初期値は左からの並び順ヒューリスティクス(正面→側面→背面の順を仮定)で
   自動推定される。
3. 必要に応じて割当を修正し、「この割当を使用」を押すと、各パネル画像が
   対応するアップロード欄(正面画像・追加ビュー欄)に反映される。
4. 通常通り生成パラメータを設定して「3Dモデルを生成」を押す。

パネル自動検出のロジック(`server/sheet.py`):

1. 前景マスク取得: RGBA画像でアルファに情報があればそれを使用。無ければ
   rembgでマスクを取得。それも不可なら四隅の背景色との色差で2値化する。
2. マスクの連結成分解析(`scipy.ndimage.label`)。画像全体の1%未満の成分は
   除去し、間隔が画像幅の2%未満のバウンディングボックス同士はマージする。
3. 残ったボックスを左→右(同列なら上→下)にソートし、パディング付きで
   切り出す(最大6パネル)。

`/api/sheet/split` はジョブを作らない同期APIで、数秒で結果を返す:

```bash
curl -s -X POST http://127.0.0.1:8000/api/sheet/split -F "image=@sheet.png"
# => {"panels": [{"index": 0, "image_b64": "...", "suggested_view": "front"}, ...]}
```

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

### キャラクターシート分割の動作確認

合成RGBAシート画像(透明背景に離れた色付きシルエット3つ)と実際のUI操作
(ブラウザ経由でのcanvas生成シート画像)の両方で、3パネルの検出・
左→右の順序・suggested_view(front/left/back)の初期推定・パネル画像への
反映(正面画像プレビュー・追加ビュー欄への自動設定)を確認済み
(`tests/test_sheet.py`、`tests/test_api.py`)。

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
- **目など小さく高コントラストな特徴のズレ・シームは既知の限界**:
  正面画像は参照にのみ使われ、他の5ビューは `Hunyuan3DPaintPipeline`
  (`third_party/Hunyuan3D-2/hy3dgen/texgen/pipelines.py`)がマルチビュー
  拡散モデルで新規生成する。生成ビュー間で目の位置が完全には一致せず、
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

## Pixal3Dジェネレータ(MITライセンス、実機検証済み)

[Pixal3D](https://github.com/TencentARC/Pixal3D)(TencentARC、SIGGRAPH 2026、
TRELLIS.2基盤)を3つ目のジェネレータとして統合している。単一画像から
PBRテクスチャ付きの3Dメッシュを生成し、本アプリではテクスチャの
baseColorを頂点カラーとしてサンプリングして活用する(GLBは頂点カラー付きで
保存、`color_mode=color4` ではそのカラーから量子化・色分割3MFを出力)。

### ライセンス上の利点

Hunyuan3D-2 が独自のコミュニティライセンス(地域制限・MAU制限)なのに対し、
**Pixal3D はコードもモデル重みもMITライセンス**であり、ライセンス面での制約が
大幅に少ない。

GLB化のテクスチャベイク(`o_voxel.postprocess.to_glb`)には UV空間での平面
ラスタライズが必要で、upstream実装は nvdiffrast(NVIDIA Source Code License、
**非商用限定**)に依存している。本アプリはこの依存を除去するため、
`server/generators/pixal3d_raster.py` に **drtk(Meta製、MIT、
https://github.com/facebookresearch/drtk)** で同じ呼び出し規約
(`RasterizeCudaContext` / `rasterize` / `interpolate`)を再現したシムを実装し、
`to_glb` 呼び出し前に `o_voxel.postprocess.dr` をこのシムへ差し替える
(`IMAGE3D_PIXAL3D_RASTERIZER`、既定 `auto`。詳細は次節)。
これにより既定構成では nvdiffrast を一切使用せず、GLB化経路もMITライセンス
クリーンになる。nvdiffrastは drtk が利用できない環境向けの明示的フォールバック
としてのみ残している。

ただし、画像条件付けに使う **DINOv3 の重み**(`camenduru/dinov3-vitl16-pretrain-lvd1689m`、
Metaの `facebook/dinov3-vit7b16-pretrain-lvd1689m` からの蒸留モデル)は
Meta独自の **DINOv3 License** であり、完全なMITではない点に注意
(商用利用自体は許可されているが、attribution表示等の付帯義務がある。
詳細は「利用しているOSS」節の別表を参照)。以上より、Pixal3D構成は
「コード全体がMIT」ではあるが、**DINOv3重みを含めた構成全体としては
完全MITとは言えない**(重み・ライセンス面での完全クリーンではない)。

### 隔離venvのセットアップ

Pixal3D は Python 3.10 と独自の依存ピン(trimesh==4.10.1 等)を要求するため、
既存の `.venv`(Python 3.12)とは**完全に分離した専用venv `.venv-pixal3d`** を使う。
既存venvには1パッケージも追加しない。手順の詳細・注意点は
`requirements-pixal3d.txt` のコメントに集約してある。要約:

```bash
# 1. venv作成 + torch (Blackwell対応 cu128)
uv venv .venv-pixal3d --python 3.10
uv pip install --python .venv-pixal3d/bin/python torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# 2. pip依存
uv pip install --python .venv-pixal3d/bin/python -r requirements-pixal3d.txt

# 3. リポジトリclone(pipインストールせずsys.path経由でimportする)
git clone https://github.com/TencentARC/Pixal3D third_party/Pixal3D
git clone -b main --recursive https://github.com/microsoft/TRELLIS.2.git third_party/TRELLIS.2

# 4. CUDA拡張ビルド(CUDA 12.8ツールチェーンを明示。CUDACXXの明示が重要 —
#    cmakeがシステム既定の /usr/bin/nvcc (CUDA 13.0) を拾うとglibcヘッダ非互換で失敗する)
export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"

uv pip install --python .venv-pixal3d/bin/python --no-build-isolation third_party/TRELLIS.2/o-voxel

# ラスタライザ: drtk (MIT) を優先導入する。git経由のソースビルド (実測: sm_120で約3分40秒)。
uv pip install --python .venv-pixal3d/bin/python --no-build-isolation \
    "git+https://github.com/facebookresearch/drtk.git"
# nvdiffrast はdrtkが使えない場合のフォールバックのみ (非商用ライセンス、下記参照)。
# drtkのみ導入していれば省略可。
uv pip install --python .venv-pixal3d/bin/python --no-build-isolation \
    "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"

# 5. NATTEN(NAF特徴アップサンプラの必須依存。sm_120ソースビルド、実測9分強)
NATTEN_CUDA_ARCH="12.0" NATTEN_N_WORKERS=8 uv pip install \
    --python .venv-pixal3d/bin/python natten==0.21.0 --no-build-isolation
```

flash_attn は導入せず、**SDPAバックエンド**(`ATTN_BACKEND=sdpa` /
`SPARSE_ATTN_BACKEND=sdpa`)を使う(Blackwellでflash_attnのプリビルドが
torch ABI不一致のため)。モデル重みは初回生成時にHuggingFaceから自動DLされる
(TencentARC/Pixal3D 約23GB + DINOv3 約1.2GB + NAF重み約2.5MB)。

### 起動

```bash
env IMAGE3D_GENERATOR=pixal3d ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
    CUDA_HOME=/usr/local/cuda-12.8 \
    .venv-pixal3d/bin/uvicorn server.main:app --host 127.0.0.1 --port 8022
```

`.claude/launch.json` の `image3d-server-pixal3d`(ポート8022)にも同じ構成を
定義済み。`IMAGE3D_GENERATOR=auto` では解決されない(明示指定のみ)。

主な環境変数(`server/config.py`):

| 変数 | デフォルト | 説明 |
|---|---|---|
| `IMAGE3D_PIXAL3D_LOW_VRAM` | `true` | 低VRAMモード(モデルCPU常駐、ステージごとにGPUへ) |
| `IMAGE3D_PIXAL3D_RESOLUTION` | `1024` | パイプライン解像度(1024 / 1536) |
| `IMAGE3D_PIXAL3D_FOV` | `0.6` | カメラ水平FOV(ラジアン)。MoGe自動推定は不使用 |
| `IMAGE3D_PIXAL3D_TEXTURE_SIZE` | `2048` | GLB化時のテクスチャベイクサイズ |
| `IMAGE3D_PIXAL3D_MODEL_PATH` | `TencentARC/Pixal3D` | HFリポジトリID |
| `IMAGE3D_PIXAL3D_RASTERIZER` | `auto` | `auto` \| `drtk` \| `nvdiffrast`。GLB化のUVテクスチャベイクに使うラスタライザ。`auto` はdrtk (MIT) がimport可能ならdrtkを使い、無ければnvdiffrast (非商用ライセンス) にフォールバックしてログ出力する。`drtk`/`nvdiffrast` を明示指定した場合、それが利用不可だとエラーになる |

### ラスタライザ切替(drtk / nvdiffrast)

`o_voxel.postprocess.to_glb` はUVアトラスへのテクスチャベイクに
`RasterizeCudaContext` / `rasterize` / `interpolate` の3 APIを使う。upstream
実装はこれを nvdiffrast (NVIDIA Source Code License、非商用限定) で呼んでいるが、
本アプリは `server/generators/pixal3d_raster.py` に **drtk (Meta製、MIT)** で
同じ呼び出し規約・戻り値規約を再現したシムを実装し、`Pixal3DGenerator` が
`to_glb` 呼び出し直前に `o_voxel.postprocess.dr` をこのシムに差し替える
(`select_rasterizer_module` / `_inject_rasterizer`、`server/generators/pixal3d.py`)。

- 座標系の差(nvdiffrastはOpenGL式でrow0=画像下端、drtkはピクセル座標で
  row0=画像上端)はシム内でY軸反転により吸収し、シムの入出力はnvdiffrast
  互換のまま維持している。
- 重心座標の頂点対応(nvdiffrastの `u`/`v` がどの頂点の重みに対応するか)は
  実機でのA/B検証(one-hot頂点属性を`interpolate`に通し、出力との相関を確認)
  で確定させた(`u`=頂点0の重み、`v`=頂点1の重み)。
- 実機A/B検証(ランダムなUVメッシュ、数百面、nvdiffrastとdrtkシムの両方で
  ラスタライズし比較): face IDマップの一致率99.9%以上(不一致は三角形境界の
  1px差のみ)、一致した画素でのバリセントリック座標・interpolate結果は
  浮動小数点誤差程度(最大差 ~1e-6)で一致することを確認済み。
- drtk・nvdiffrastのどちらも未導入の場合は `RuntimeError`(明示的なエラー)
  になる。

### Hunyuan3D-2との比較実測値(RTX PRO 6000 Blackwell、momo.png、seed=42)

| 項目 | Hunyuan3D-2(形状のみ) | Hunyuan3D-2(+texgen) | Pixal3D(1024・低VRAM・steps=30) |
|---|---|---|---|
| 生成時間(モデル常駐後) | 22〜35秒 | 60〜90秒 | 約97秒(生成+GLB化)+後処理21秒 |
| 初回追加(モデルロード) | 数十秒 | 数十秒 | 約50〜60秒(23GB) |
| プロセスVRAMピーク | 約12GB | 約25GB | **約18.7GB** |
| テクスチャ/色 | なし(投影方式で色付け) | 全周テクスチャ | **全周PBRテクスチャ→頂点カラー** |
| watertight | true(体積計算可) | true | **false**(ボクセルリメッシュ由来、体積は0表示) |
| メッシュ統計(momo.png) | 約109cm³ / watertight | 同左 | 98,107頂点 / 200,000面 / bbox 67.4×48.3×100.0mm |
| パレット(color4) | 投影ベース | テクスチャサンプル | テクスチャサンプル(白70% / 紺21% / 灰5% / 赤4%) |
| ライセンス | 独自(地域・MAU制限) | 同左 | **MIT(重みまで)** ※DINOv3重みは別ライセンス、下記OSS一覧参照 |

- Pixal3D の生成時間は初回ジョブでさらに +2〜4分かかることがある
  (FlexGEMMカーネルautotune・ラスタライザのJITコンパイルが初回のみ走るため。
  2回目以降はディスクキャッシュで高速化)。
- 低VRAMモード(既定)はステージごとにモデルをGPUへ載せ替えるため遅いが
  ピークVRAMを抑える。96GB環境では `IMAGE3D_PIXAL3D_LOW_VRAM=false` +
  `IMAGE3D_PIXAL3D_RESOLUTION=1536` で品質・速度を上げられる(未計測)。

**drtkバックエンドでのE2E再検証(2026-07-09、ポート8022、
`IMAGE3D_PIXAL3D_RASTERIZER=drtk`明示指定、momo.png、seed=42、color_mode=color4)**:

| 項目 | nvdiffrastバックエンド(ジョブ58d8e1d0) | drtkバックエンド(ジョブb716c24a、本検証) |
|---|---|---|
| 総所要時間(モデルロード込み) | — | 121秒(モデルロード含む初回実行) |
| 頂点数 | 98,107 | 98,105 |
| 面数 | 200,000 | 200,000 |
| bbox (mm) | 67.37 × 48.30 × 100.02 | 67.38 × 48.35 × 100.03 |
| watertight | false | false |
| パレット(color4) | 白70% / 紺21% / 灰5% / 赤4% | 白系44% / 水色系28% / 紺系23% / 灰系5%(構成比) |

頂点数・面数・bboxはほぼ完全に一致しており、drtkベースのラスタライズが
nvdiffrast版と幾何学的に等価な結果を生成することを確認した(GLBの頂点カラーも
正常にサンプリング・保存されていることを確認済み)。パレットの正確な色相・
構成比はジョブ間で差があるが、これは拡散サンプリングそのものの実行間非決定性
(torch/CUDAのattentionカーネル等)によるもので、ラスタライザ差とは無関係
(頂点/面数/bboxのメッシュ形状自体は同一シードで再現しているため)。

### 制限・実装メモ

- **マルチビュー入力(FR-9)非対応**: `image_back` 等を指定するとジョブは
  明示的なエラーで失敗する。単一画像専用。
- **texture_mode=paint 非対応**: texgen は hy3dgen 依存のため `.venv-pixal3d`
  では利用不可(paint指定時は警告を記録して投影方式にフォールバック)。
  そもそも Pixal3D 自体が全周テクスチャを生成するため不要。
- **パラメータは steps / seed のみ接続**: guidance_scale / octree_resolution は
  Pixal3D のステージごとに調整済みの既定値と互換性が無いため無視される。
  steps はデフォルト30だが、Pixal3D 公式デフォルトは12(30でも動作するが
  サンプリングが遅くなる。速度優先なら steps=12 を指定)。
- **背景除去必須**: Pixal3D 公式の背景除去モデル(briaai/RMBG-2.0)はHFの
  ゲート付きリポジトリのためロードせず、本アプリの背景除去(rembg CPU)の
  結果を渡す。`remove_bg=false` でアルファ無し画像を送るとエラーになる。
- **watertightにならない**: ボクセルリメッシュ出力は閉じたソリッドではなく、
  体積は0と表示される。スライサーでの印刷は通常問題ないが、体積ベースの
  見積りはできない。確実なwatertightが必要なら hunyuan3d を使う。
- **UVシーム由来の頂点分断は自動修復**: `o_voxel` のGLB出力はUVアトラス境界で
  頂点が複製され数万個の連結成分に分断されているため、ジェネレータ内で
  頂点溶接(`merge_vertices`)してから後処理に渡す(これを外すと
  浮遊小部品除去が本体表面を削除してしまう。server/generators/pixal3d.py参照)。
- **座標系**: Pixal3D のGLB出力は 上=-Z / 正面=+Y(実測確認)。X軸まわり
  180°回転で本アプリの Z-up / 正面=-Y に変換している。
- **カメラFOVは固定値**: 公式の MoGe による自動FOV推定は導入していない
  (`IMAGE3D_PIXAL3D_FOV`、既定0.6rad ≈ 34°)。入力画像の遠近感が強い場合は
  調整の余地がある。

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
│   │   ├── hunyuan3d.py      # Hunyuan3D-2ラッパ(Phase 2、Phase 3aでmvパイプライン追加)
│   │   └── pixal3d.py        # Pixal3Dラッパ(MITライセンス、専用venv .venv-pixal3d で使用)
│   ├── preprocess.py         # 画像前処理(背景除去・リサイズ)
│   ├── meshproc.py           # メッシュ後処理
│   ├── colorproc.py          # 4色カラープリント対応(Phase 2.5、頂点カラー投影・量子化・分割)
│   ├── sheet.py              # キャラクターシート自動分割(Phase 3a、パネル検出)
│   └── texture.py            # テクスチャ生成 texgen 統合(Phase 3c、paint常駐ラッパ・頂点カラーサンプリング)
├── web/                      # 静的フロントエンド
├── tests/                    # pytest
├── data/jobs/                # 生成物(gitignore対象)
├── third_party/Hunyuan3D-2/  # hy3dgen本体(git clone、Phase 2、gitignore対象)
├── third_party/Pixal3D/      # Pixal3D本体(git clone、gitignore対象)
├── third_party/TRELLIS.2/    # o-voxelビルド用(git clone、gitignore対象)
├── requirements.txt          # base依存
├── requirements-gpu.txt      # Phase 2用追加依存
├── requirements-pixal3d.txt  # Pixal3D用隔離venv (.venv-pixal3d) の依存(セットアップ手順込み)
├── run.sh
└── README.md
```

## よくある質問(FAQ)

### Q. アップロードした画像と関係ないテスト形状(同じ形)ばかり生成される

mockジェネレータで動作しています。UIヘッダ右上の「生成エンジン」バッジが
`mock` になっていないか確認してください(mock時は画面上部に警告バナーも出ます)。
対処は「[アップロード画像と無関係なテスト形状が生成されるとき](#アップロード画像と無関係なテスト形状が生成されるとき)」を参照。
旧バージョンのアプリでは `IMAGE3D_GENERATOR=hunyuan3d` を明示指定してください
(`auto` は新バージョンのみ対応)。

### Q. GPUなしでも使えますか?

UIやAPIの動作確認はmockジェネレータでGPU不要で行えます。ただし実際の画像から
3Dを生成するにはNVIDIA GPUが必須です(形状生成のみ VRAM 16GB以上、テクスチャ
生成併用は 32GB以上。実測値は「VRAM最小要件」の表を参照)。CPUのみでの実生成は
サポートしていません。

### Q. 生成にどれくらい時間がかかりますか?

本README記載の実測環境(RTX PRO 6000)で、形状のみ約20〜40秒、テクスチャ生成
併用で約60〜90秒です。**初回だけ**はモデルのダウンロード(単一ビュー約9.2GB、
マルチビュー約9.2GB、テクスチャ用モデル数GB)とロード(十数秒〜)が加わるため、
数分〜数十分かかることがあります。2回目以降はモデルが常駐するため速くなります。

### Q. 初回生成時のモデルはどこに保存されますか? オフラインで使えますか?

HuggingFaceのキャッシュ(既定 `~/.cache/huggingface`)に保存されます。
一度ダウンロードすれば、以降の生成はインターネット接続なしで動作します。

### Q. 「watertight: NG」と表示されました。印刷できませんか?

多くの場合そのまま印刷できます。後処理で穴埋めを試みても閉じきらなかった
ことを示す表示で、最近のスライサー(Bambu Studio / PrusaSlicer 等)は読み込み時に
自動修復します。気になる場合は `octree_resolution` を1段下げる、`seed` を変えて
再生成する、スライサーの修復機能(またはWindowsの3D Builder等)を使う、の
いずれかで解消できることが多いです。

### Q. 4色の3MFをスライサーでどう使えばいいですか?

カラーモードで出力した3MFには `color_1`〜`color_4` の最大4オブジェクトが
入っています。Bambu Studio / PrusaSlicer で開き、オブジェクトごとに
AMS / MMU のフィラメント(スロット)を割り当ててスライスしてください。
アプリの「パレット」表示(色チップ+比率)が各オブジェクトの色の目安です。

### Q. 4色以外(2色・3色、あるいは5色以上)にできますか?

`n_colors` パラメータで2〜4色を指定できます。5色以上は対応していません
(4スロットのAMSを想定した仕様です)。

### Q. キャラクターシートのパネルがうまく検出されません

自動分割はパネル同士が離れていて背景とのコントラストがある構図を前提とした
簡易解析です。検出に失敗する場合は、シートを画像編集ソフトで切り分けて、
正面/背面/側面の各アップロード欄に個別に登録してください(生成品質は同じです)。

### Q. 「テクスチャ生成(実験的)」のチェックボックスが押せません

その環境でtexgen(CUDA拡張)が利用できないことを示します(`/api/health` の
`texgen_available` が `false`)。「Phase 3c」節のビルド手順(torchのCUDAバージョンと
一致するCUDAツールチェーンが必要)を実施してください。ビルドしなくても、
正面投影方式の4色カラー出力(FR-8)は利用できます。

### Q. 生成物のサイズ(高さ)を変えたい / 印刷に適したポリゴン数は?

「目標高さ(mm)」で出力サイズを指定できます(既定100mm、Z軸高さ基準で
スケーリング+接地済み)。面数は既定20万面で一般的なFDM印刷には十分です。
プリセット(フィギュア/小型フィギュア/ペンダント/高精細)を使うと、
高さ・解像度・面数をまとめて切り替えられます。

### Q. スライス(Gコード生成)やサポート材の生成はできますか?

できません。本アプリはプリント可能なメッシュデータ(STL/3MF)の生成までを担当し、
スライスはスライサー(Bambu Studio / PrusaSlicer / Cura 等)の役割です(SPEC.md §7)。
ビューアの「オーバーハング」表示で、サポートが必要になりそうな箇所(既定45°超)を
事前に確認できます。

### Q. 商用利用できますか?

本プロジェクトのコードは [Polyform Small Business License](LICENSE)(小規模事業者
まで商用可)ですが、**生成に使うHunyuan3D-2モデルはTencentのコミュニティ
ライセンスに別途従う必要があります**(利用地域・規模の制限あり)。
詳細は「ライセンス」節を参照してください。

### Q. `third_party/Hunyuan3D-2` を入れ直したらテクスチャ生成が壊れました

再clone時はvendoredパッチ(`hy3dgen/texgen/utils/multiview_utils.py` の
`trust_remote_code=True`)の再適用が必要です。「Phase 3c」節の手順を参照してください。

## 既知の制限

- mockジェネレータは画像内容を反映しない決定的な形状(seedでバリエーション)を返す。
  実際の画像に基づく生成にはPhase 2でのHunyuan3D-2導入が必要。mockジェネレータは
  マルチビュー入力(`extra_views`)を無視する(単一ビュー用の決定的形状を返す)。
- マルチビュー生成(FR-9)は `hunyuan3d-dit-v2-mv` モデル(約9.2GB、単一ビュー用
  モデルとは別リポジトリ `tencent/Hunyuan3D-2mv`)の追加ダウンロードが必要。
  厳密なマルチビュー幾何整合(正面・背面・側面の完全な形状一致)はモデル自体の
  性能に依存し、本アプリ側での補正は行わない(SPEC.md §7の制約通り)。
- キャラクターシート自動分割(`server/sheet.py`)は、パネル同士が明確に離れて
  いる・背景とのコントラストがある構図を前提とした簡易的な連結成分解析であり、
  パネルが複雑に重なる・背景と被写体の色が近いシートでは誤検出する場合がある
  (その場合はUI上で手動修正が必要)。
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

## ライセンス

このリポジトリ(`server/`・`web/`・`docs/`・`tests/` 等、本プロジェクトのオリジナル
コード)は [Polyform Small Business License 1.0.0](LICENSE) の下で提供されます。

要約(法的拘束力があるのは[LICENSE](LICENSE)本文のみです)。Hunyuan3D-2 /
Pixal3D(DINOv3)を有効にする場合の所定の謝辞・Notice文言は
[NOTICE](NOTICE) にまとめてあります(UIにも `IMAGE3D_GENERATOR=pixal3d` 時に
"Built with DINOv3" を表示します):

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

### Hunyuan3D-2構成での主な利用条件と、配布・サービス提供時の付帯義務

`third_party/Hunyuan3D-2/LICENSE`(2026-07時点の原文)に基づく要約。
法的拘束力があるのは原文のみであり、本節は法的助言ではありません。

**利用条件(使うだけの場合も適用)**:

- **地域**: EU・英国・韓国では利用不可(Territory外)。生成物(Output)を
  Territory外で使用・表示することも許諾されない。
- **規模**: 全製品・サービス合計の月間アクティブユーザーが**100万人**を超える
  事業者は、Tencentへの別途ライセンス申請が必要。

**本アプリ(Hunyuan3D-2構成)を第三者に配布、またはサービスとして提供する場合の
付帯義務**(同ライセンス §3):

1. 受領者にTencentライセンス本文の写しを提供すること(§3(a))
2. Hunyuan3D-2側のファイルを変更した場合は、変更した旨の告知を目立つ形で
   付すこと(§3(b))
3. ホステッドサービス以外の配布物には、所定の文言のNoticeテキストファイルを
   同梱すること(§3(d)。文言はライセンス原文参照)
4. サービス・製品の実際の提供者(法人名等)を明示し、**Tencentが提携・後援して
   いると誤認させる表示をしない**こと(§3(e))
5. 「Powered by Tencent Hunyuan」表示と技術紹介記事の公開は推奨(encouraged)で
   あり義務ではない(§3(c))

なお、Pixal3D構成の場合は上記のTencentライセンスは適用されません。
既定のラスタライザ構成(drtk、MIT)では nvdiffrast(非商用ライセンス)への
依存も無くなりましたが、画像条件付けに使う **DINOv3 の重み**
(Meta独自ライセンス、商用利用可・attribution等の付帯義務あり)が別途
制約になります(「Pixal3Dジェネレータ」節・OSS一覧参照)。

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

**別リポジトリのモデル(`third_party/`、本リポジトリには含まれない)**

| 対象 | ライセンス |
|---|---|
| Tencent Hunyuan3D-2 (hy3dgen) | TENCENT HUNYUAN 3D 2.0 COMMUNITY LICENSE AGREEMENT(独自ライセンス。地域制限・月間アクティブユーザー数100万人超での別途許諾要件あり) |
| TencentARC Pixal3D(コード+モデル重み) | MIT(コード・重みともにMIT。ただし画像条件付けに使うDINOv3重みは別ライセンス、下記参照) |
| Microsoft TRELLIS.2 / o-voxel | MIT |
| JeffreyXiang CuMesh(o-voxelのビルド時依存、GitHub直接取得) | MIT |
| JeffreyXiang FlexGEMM(o-voxelのビルド時依存、GitHub直接取得) | MIT |
| **Meta drtk**(既定のラスタライザ。`server/generators/pixal3d_raster.py` 経由で使用、実機検証済み) | **MIT** |
| NVlabs nvdiffrast(drtkが使えない場合のみのフォールバック。既定構成では未使用) | NVIDIA Source Code License(非商用研究用途。商用利用はNVIDIAの許諾が必要な点に注意) |
| NATTEN | MIT |
| valeoai NAF(NAFアップサンプラ重み、torch.hub経由) | Apache-2.0 |
| **Meta DINOv3 重み**(`camenduru/dinov3-vitl16-pretrain-lvd1689m`、Meta `facebook/dinov3-vit7b16-pretrain-lvd1689m` からの蒸留モデル。Pixal3Dの画像条件付けに使用) | **DINOv3 License**(Meta独自ライセンス、MITではない)。[ライセンス原文](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)の要約(法的拘束力は原文のみ): 商用利用は**許可**されており、企業規模・地域・MAU等による制限は無い。ただし (1) 再配布時はライセンス原文の写しを同梱すること、(2) 関連するWebサイト・UI・製品ドキュメント等に "Built with DINOv3" の表示を行うこと、(3) 研究成果を公開する場合はDINO Materialsの利用を謝辞に記載すること、(4) リバースエンジニアリング禁止、(5) 軍事・兵器・ITAR対象活動などへの利用禁止、が付帯義務として定められている。utils3d(upstream inference.pyのレンダリング/学習系専用ライブラリ)は本アプリの生成経路では使用しないため未導入。 |

**まとめ**: Pixal3D構成は「コード全体(Pixal3D本体・TRELLIS.2/o-voxel・CuMesh/
FlexGEMM・drtk・NATTEN)はMIT」であり、既定のラスタライザ構成
(`IMAGE3D_PIXAL3D_RASTERIZER=auto` かつdrtk導入済み)では nvdiffrast
(非商用ライセンス)を一切使用しない。しかし**画像条件付けに使うDINOv3の
重みがMeta独自ライセンス**であるため、**構成全体としては「完全MIT」とは
言えない**(商用利用自体は許可される緩やかなライセンスだが、attribution等の
付帯義務がある点に注意)。
