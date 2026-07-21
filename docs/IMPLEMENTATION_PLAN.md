# 実装計画: Image-3D

## フェーズ構成

### Phase 1: アプリ骨格 + mockジェネレータで全パイプライン完成(サブエージェント: sonnet)

**ゴール**: モデル無しで「画像アップロード → ジョブ実行 → 3Dビューア表示 → STL等ダウンロード」が
E2Eで動作し、pytestが通る。

| # | タスク | 成果物 |
|---|---|---|
| 1-1 | プロジェクト初期化(venv, requirements.txt, .gitignore, run.sh) | 実行可能な環境 |
| 1-2 | `server/config.py` + `server/generators/base.py` + `mock.py` | Generator抽象+mock実装(パラメトリックなトーラス結び目等、画像によらず決定的なメッシュ) |
| 1-3 | `server/meshproc.py`: クリーニング/watertight化/スケーリング/簡略化/統計 | 後処理モジュール+単体テスト |
| 1-4 | `server/preprocess.py`: 画像検証・リサイズ・背景除去(rembg、未導入時は自動スキップ) | 前処理モジュール |
| 1-5 | `server/jobs.py`: ジョブモデル・直列ワーカー・`data/jobs/`永続化 | ジョブ基盤 |
| 1-6 | `server/main.py`: 全APIエンドポイント(SPEC.md §5)+静的配信 | REST API |
| 1-7 | `web/`: UI(左: アップロード/パラメータ/履歴、右: Three.jsビューア) | フロントエンド一式(Three.jsはweb/vendor/にローカル配置) |
| 1-8 | `tests/`: meshproc単体テスト+APIライフサイクルテスト(mock) | pytest green |
| 1-9 | README.md(セットアップ・起動・GPU導入手順) | ドキュメント |

**検証基準(Phase 1完了条件)**:
- `pytest` 全通過
- サーバ起動→curlでジョブ作成→completed→ `model.glb` と `download?format=stl` が取得でき、
  STLがtrimeshで再読込可能かつwatertight
- ブラウザでビューア操作(回転/ズーム)とモデル情報表示が機能

### Phase 2: Hunyuan3D-2 実モデル統合(サブエージェント: sonnet、時間のかかるDLはバックグラウンド)

| # | タスク | 内容 |
|---|---|---|
| 2-1 | GPU環境構築 | venvへ torch cu128 / hy3dgen(Hunyuan3D-2リポジトリ)導入。`requirements-gpu.txt` 確定 |
| 2-2 | `generators/hunyuan3d.py` | 遅延ロード・常駐・`torch.cuda.empty_cache()`・パラメータ(steps/guidance/octree/seed)接続 |
| 2-3 | 実画像での生成検証 | キャラクター画像でSTL生成→watertight・寸法・見た目確認 |
| 2-4 | チューニング | 生成時間・VRAM計測、デフォルトパラメータ調整、README更新 |

**検証基準**: `IMAGE3D_GENERATOR=hunyuan3d` で実画像から60秒以内(目標)にメッシュ生成、
ビューア表示・STL出力が正常。

### Phase 2.5: 4色カラープリンタ対応(サブエージェント: sonnet)【追加】

SPEC.md §3.7 (FR-8) の実装。テクスチャ生成AIは使わず、入力画像の正面投影+k-means量子化方式。

| # | タスク | 内容 |
|---|---|---|
| 2.5-1 | `server/colorproc.py` | 画像→頂点カラー投影、k-means量子化(2〜4色)、色ごとのメッシュ分割、パレット統計 |
| 2.5-2 | jobs.py / main.py 拡張 | `color_mode` / `n_colors` パラメータ、カラーGLB・マルチオブジェクト3MF出力、`stats.palette` |
| 2.5-3 | フロントエンド拡張 | カラーモードUI、頂点カラー表示、パレットチップ表示 |
| 2.5-4 | テスト | colorproc単体+APIカラーモードE2E(mock)。3MFのオブジェクト数検証 |

**検証基準**: mockでカラーE2E(pytest)、実モデルで momo.png から4色3MF生成
→ 3MF内に最大4オブジェクト+表示色、ビューアでカラー表示確認。

### Phase 3: 品質向上(SPEC.md FR-9〜FR-12)— 3ステップ直列実行

**Phase 3a: マルチビュー入力+キャラクターシート自動分割(FR-9)** — sonnet
| # | タスク |
|---|---|
| 3a-1 | `server/sheet.py`: シート画像のパネル自動検出(rembg→アルファ連結成分→バウンディングボックス切出し) |
| 3a-2 | `generators/hunyuan3d.py` 拡張: 複数ビュー時 hunyuan3d-dit-v2-mv パイプラインに切替(辞書 {front/back/left/right} 入力)。モデルDL |
| 3a-3 | API: ジョブ作成でビューラベル付き複数画像受付+`POST /api/sheet/split`(シート→パネル画像群) |
| 3a-4 | UI: 複数ビューアップロード欄、シート分割ボタン+パネル割当UI |
| 3a-5 | テスト(mockで複数ビュー受付、sheet分割単体)+実モデルmv検証 |

**Phase 3b: プリセット+オーバーハングヒートマップ(FR-11, FR-12)** — sonnet
| # | タスク |
|---|---|
| 3b-1 | UI: プリセットセレクタ(4種)とパラメータ一括反映 |
| 3b-2 | viewer.js: オーバーハング表示モード(面傾斜角→頂点色、閾値スライダー、Three.jsのみで完結) |
| 3b-3 | E2E確認(mockで足りる) |

**Phase 3c: テクスチャ生成 texgen(FR-10)** — sonnet(失敗リスクあり・フォールバック必須)
| # | タスク |
|---|---|
| 3c-1 | custom_rasterizer 等CUDA拡張のビルド(torch cu128 vs システムnvcc 13.0の整合に注意。pip版CUDA 12.8ツールチェーン等) |
| 3c-2 | paint パイプライン統合(texture_mode=paint)、テクスチャ→頂点カラーサンプリング→FR-8の量子化・分割に接続 |
| 3c-3 | `/api/health` に texgen可否を含めUIで無効表示制御(フォールバック) |
| 3c-4 | 実モデル検証(momo.png でテクスチャ付きGLB+4色3MF) |

**検証基準**: 各ステップ完了時に pytest green + 実モデルE2E(3bはmockで可)。
3cはビルド不成立でも 3c-3 のフォールバックが機能すれば部分完了として扱う。

### Phase 4: ぬいぐるみ型紙生成(FR-13)— 2ステップ直列【追加】

`server/pattern/` は将来の独立リポジトリ化に備えた純粋モジュール
(入力trimesh→出力パネル/SVG、server内の他モジュールを一切importしない。
依存はnumpy/scipy/trimeshのみ)。SPEC.md §3.12参照。

**Phase 4a: パネル分割+3Dプレビュー** — sonnet
| # | タスク |
|---|---|
| 4a-1 | `server/pattern/preprocess.py`: 型紙用の平滑化(taubin/laplacian)+簡略化(1〜2万面) |
| 4a-2 | `server/pattern/segment.py`: 法線・曲率クラスタリングによるパネル分割(4〜12枚、円盤位相保証、色境界誘導オプション) |
| 4a-3 | アダプタ: `POST /api/jobs/{id}/pattern`(4aでは分割まで)+ `pattern_preview.glb`(パネル色分け)保存・配信 |
| 4a-4 | UI: 型紙パネル(パネル数等の設定+実行)、ビューアでのプレビュー表示 |
| 4a-5 | テスト(GPU不要): 球・カプセル等でパネル数・円盤位相・全面被覆を検証 |

**Phase 4b: 平坦化+実寸SVG型紙出力** — sonnet
| # | タスク |
|---|---|
| 4b-1 | `server/pattern/flatten.py`: LSCM初期解+ARAP反復の自前実装(scipy疎行列)。歪み指標算出 |
| 4b-2 | `server/pattern/svg.py`: 実寸SVG生成(縫い代オフセット・合印・パネル番号・布目線・A4タイル目安枠) |
| 4b-3 | アダプタ拡張: pattern.svg / pattern.json 保存・配信。UI: SVGダウンロード+パネル統計表示 |
| 4b-4 | テスト: 既知形状(円筒側面等)の展開結果の解析解比較、歪み閾値、SVGの妥当性(実寸・要素数) |
| 4b-5 | 実データ検証: momo.pngのジョブから型紙生成→SVG目視確認 |

**Phase 4c: 2段階化 — パーツ分解+パーツ単位型紙(SPEC §3.12「2段階構成」)** — sonnet
| # | タスク |
|---|---|
| 4c-1 | `server/pattern/parts.py`: 局所肉厚(内向きレイキャストSDF)+凹みベースのパーツ自動分解、切断面のキャップ、パーツ統計 |
| 4c-2 | アダプタ: `POST /api/jobs/{id}/pattern/parts`(パーツ分解+パーツ色分けプレビューGLB)→ 既存patternをパーツ単位実行に変更 |
| 4c-3 | UI: 2ステップ化(①パーツ分け(数の調整・再実行)→②型紙作成)。SVGのパーツグループ化・取付口ラベル |
| 4c-4 | テスト: 合成形状(球+円柱の腕脚を持つ雪だるま等)でパーツ数・分解位置・キャップ後の閉性を検証 |

**Phase 4d(将来)**: シーム直線化(縫い代オフセット自己交差の解消含む)、縫い合わせ長の厳密整合、詰め物膨張の逆補正、対話的パーツ切断編集、PDF出力。

**検証基準**: pytest green維持。momoメッシュで指定パネル数に分割・全パネル
円盤位相・SVGが実寸(高さ指定と整合)・ビューアでプレビュー表示。

## サブエージェント運用

- **Phase 1実装**: sonnetサブエージェント1体に本計画書・仕様書を参照させて一括実装。
- **Phase 2統合**: 環境構築(DL待ちが長い)はバックグラウンドのsonnetサブエージェント。
- 親エージェント(Fable)は各フェーズ完了時に検証基準を自ら再検証し、不合格なら差し戻す。

## マイルストーン

| M | 内容 | 判定 |
|---|---|---|
| M1 | Phase 1完了(mockでE2E動作+テストgreen) | curl/pytest/ブラウザ確認 |
| M2 | Phase 2完了(実モデルで生成成功) | 実画像→STL検証 |
