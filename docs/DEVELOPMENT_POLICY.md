# 開発方針: Image-3D

## 1. 基本方針

1. **動くものを最短で、ただし本番品質の骨格で。**
   生成AIモデル(Hunyuan3D-2)のセットアップは重い(依存関係・モデルDL数GB)。
   そこでジェネレータをプラガブルにし、`mock` ジェネレータで
   アップロード→生成→後処理→ビューア→エクスポートの全パイプラインを先に完成・検証する。
   実モデルは最後に差し込む。

2. **ビルドレス・フロントエンド。**
   Three.js は importmap + ESモジュール(ローカルにvendorしてFastAPIが配信)で使い、
   Node/webpack等のビルド工程を持たない。依存が少なく壊れにくい。

3. **3Dプリント適性を後処理で保証する。**
   生成AIの出力メッシュはそのままではプリントに不向きなことが多い
   (非watertight、浮遊ゴミ、過剰ポリゴン)。trimeshによる後処理パイプラインを
   独立モジュールとして実装し、単体テスト可能にする。

4. **GPUリソースは直列キューで守る。**
   生成は asyncio + 単一ワーカースレッドの直列キューで実行。
   モデルは初回ロード後プロセスに常駐(NFR-3)。

## 2. 技術選定と理由

| 領域 | 選定 | 理由 |
|---|---|---|
| Image-to-3D | **Hunyuan3D-2 (shape)** | オープンソース(hy3dgen)、単一画像から高品質メッシュ、96GB VRAMで余裕をもって動作。TripoSR(軽量だが品質低め)、TRELLIS(依存が重くBlackwell対応が不安定)と比較して品質/導入コストのバランスが最良 |
| PyTorch | **2.7+ / CUDA 12.8 (cu128)** | Blackwell (sm_120) 対応にはcu128ビルドが必須 |
| 背景除去 | rembg (onnxruntime) | 実績・導入容易 |
| メッシュ処理 | trimesh + fast-simplification | watertight化・簡略化・STL/3MF/GLB/OBJ出力を1ライブラリ系で完結 |
| Webフレームワーク | FastAPI + Uvicorn | 非同期ジョブ・multipart・静的配信が簡潔 |
| 3D表示 | Three.js (GLTFLoader + OrbitControls) | デファクト。GLBで受け渡し |
| パッケージ管理 | venv + pip (`requirements.txt` を段階分割) | base(アプリ)/ gpu(モデル)を分離し、mock動作を軽量に保つ |

## 3. リポジトリ構成

```
image-3d/
├── docs/                     # 仕様書・開発方針・実装計画
├── server/
│   ├── main.py               # FastAPIエントリポイント
│   ├── config.py             # 設定(環境変数 IMAGE3D_GENERATOR=mock|hunyuan3d 等)
│   ├── jobs.py               # ジョブ管理・直列実行キュー・永続化
│   ├── generators/
│   │   ├── base.py           # Generator抽象基底 (generate(image, params) -> trimesh.Trimesh)
│   │   ├── mock.py           # テスト用ジェネレータ
│   │   └── hunyuan3d.py      # Hunyuan3D-2ラッパ(遅延import・遅延ロード)
│   ├── preprocess.py         # 画像前処理(背景除去・リサイズ・正方形化)
│   └── meshproc.py           # メッシュ後処理(クリーニング・watertight・スケール・簡略化・統計)
├── web/                      # 静的フロントエンド(index.html, app.js, viewer.js, style.css, vendor/)
├── tests/                    # pytest(meshproc・API をmockジェネレータで検証)
├── data/jobs/                # 生成物(gitignore対象)
├── requirements.txt          # base依存
├── requirements-gpu.txt      # torch cu128 + hy3dgen系
├── run.sh                    # 起動スクリプト
└── README.md
```

## 4. コーディング規約

- Python: 型ヒント必須、`ruff` 準拠スタイル、モジュール間はデータクラス/辞書で疎結合。
- 例外はジョブ単位で捕捉し `status=failed` + `error` に格納。サーバは落とさない。
- フロントエンドUIテキストは日本語。コード・識別子・ログは英語。
- 設定は環境変数(`IMAGE3D_*`)で上書き可能にし、コードにハードコードしない。

## 5. テスト・検証方針

1. **単体テスト**: `meshproc.py`(壊れたメッシュ→watertight化・スケール・簡略化)を重点的に。
2. **APIテスト**: FastAPI TestClient + mockジェネレータでジョブのライフサイクル全体を検証。
3. **E2E手動検証**: mockでUI一式(アップロード→ビューア表示→STLダウンロード)を確認後、
   Hunyuan3D-2 で実画像により検証。
4. STL出力はtrimeshで再読込し、watertight・寸法を機械検証する。

## 6. リスクと対策

| リスク | 対策 |
|---|---|
| Blackwell GPUでのtorch/カスタムカーネル非互換 | cu128ビルド固定。hy3dgenはshapeパイプラインのみ使用(カスタムラスタライザ不要)。失敗時もmockでアプリは動作 |
| モデルDL(~数GB)の失敗・時間 | HuggingFaceキャッシュ利用、DLは初回のみ。セットアップ手順をREADMEに明記 |
| 生成メッシュが非watertight | 後処理で穴埋め試行+判定結果をUIに明示(ユーザーがスライサーで修復判断可能) |
| VRAM枯渇 | 直列キュー+生成後 `torch.cuda.empty_cache()` |
