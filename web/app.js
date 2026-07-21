// メインUIロジック (SPEC.md §6 画面仕様)
import { Viewer } from "/viewer.js";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const previewImage = document.getElementById("preview-image");
const paramsForm = document.getElementById("params-form");
const generateBtn = document.getElementById("generate-btn");

const progressStatus = document.getElementById("progress-status");
const progressBarFill = document.getElementById("progress-bar-fill");
const progressError = document.getElementById("progress-error");
const multiviewNote = document.getElementById("multiview-note");

const sheetFileInput = document.getElementById("sheet-file-input");
const sheetDropzone = document.getElementById("sheet-dropzone");
const sheetSplitBtn = document.getElementById("sheet-split-btn");
const sheetPanelsArea = document.getElementById("sheet-panels-area");
const sheetPanelsList = document.getElementById("sheet-panels-list");
const sheetApplyBtn = document.getElementById("sheet-apply-btn");

const VIEW_LABELS_JA = {
  front: "正面",
  back: "背面",
  left: "左側面",
  right: "右側面",
  none: "使わない",
};

const jobHistoryEl = document.getElementById("job-history");

const infoVertices = document.getElementById("info-vertices");
const infoFaces = document.getElementById("info-faces");
const infoBbox = document.getElementById("info-bbox");
const infoWatertight = document.getElementById("info-watertight");
const infoVolume = document.getElementById("info-volume");
const infoPaletteItem = document.getElementById("info-palette-item");
const infoPalette = document.getElementById("info-palette");

const colorModeCheckbox = document.getElementById("color-mode-checkbox");
const nColorsField = document.getElementById("n-colors-field");
const exportColorNote = document.getElementById("export-color-note");

const textureModeCheckbox = document.getElementById("texture-mode-checkbox");
const textureModeUnavailableNote = document.getElementById("texture-mode-unavailable-note");

const shadingBtn = document.getElementById("shading-btn");
const wireframeBtn = document.getElementById("wireframe-btn");
const overhangBtn = document.getElementById("overhang-btn");
const overhangControls = document.getElementById("overhang-controls");
const overhangThresholdSlider = document.getElementById("overhang-threshold");
const overhangThresholdValue = document.getElementById("overhang-threshold-value");
const viewerPlaceholder = document.getElementById("viewer-placeholder");
const viewerCanvas = document.getElementById("viewer-canvas");

const presetSelect = document.getElementById("preset-select");

const exportButtons = document.querySelectorAll(".export-btn");

const patternToggleBtn = document.getElementById("pattern-toggle-btn");
const patternViewSwitch = document.getElementById("pattern-view-switch");
const patternViewBtns = document.querySelectorAll(".pattern-view-btn");
const patternPanel = document.getElementById("pattern-panel");

// ステップ1: パーツ分解
const patternNPartsSelect = document.getElementById("pattern-n-parts");
const patternUseImageCheckbox = document.getElementById("pattern-use-image");
const patternUseLlmCheckbox = document.getElementById("pattern-use-llm");
const patternLlmUnavailableNote = document.getElementById("pattern-llm-unavailable-note");
const patternManualModeCheckbox = document.getElementById("pattern-manual-mode");
const patternManualSeedPanel = document.getElementById("pattern-manual-seed-panel");
const patternSeedList = document.getElementById("pattern-seed-list");
const patternSeedClearBtn = document.getElementById("pattern-seed-clear-btn");
const patternPropagateBackCheckbox = document.getElementById("pattern-propagate-back");
const patternPlanarBoundariesCheckbox = document.getElementById("pattern-planar-boundaries");
const patternPartsRunBtn = document.getElementById("pattern-parts-run-btn");
const patternPartsError = document.getElementById("pattern-parts-error");
const patternPartsResult = document.getElementById("pattern-parts-result");
const patternPartsGuidance = document.getElementById("pattern-parts-guidance");
const patternPartsCount = document.getElementById("pattern-parts-count");
const patternPartsManualInfo = document.getElementById("pattern-parts-manual-info");
const patternPartsList = document.getElementById("pattern-parts-list");

// ステップ2: パネル分割・SVG出力
const patternNPanelsInput = document.getElementById("pattern-n-panels");
const patternNPanelsValue = document.getElementById("pattern-n-panels-value");
const patternUseColorsCheckbox = document.getElementById("pattern-use-colors");
const patternSeamAllowanceInput = document.getElementById("pattern-seam-allowance");
const patternSeamAllowanceValue = document.getElementById("pattern-seam-allowance-value");
const patternRunBtn = document.getElementById("pattern-run-btn");
const patternError = document.getElementById("pattern-error");
const patternResult = document.getElementById("pattern-result");
const patternResultCount = document.getElementById("pattern-result-count");
const patternResultFlattenOk = document.getElementById("pattern-result-flatten-ok");
const patternPanelList = document.getElementById("pattern-panel-list");
const patternSvgDownloadBtn = document.getElementById("pattern-svg-download-btn");
const patternSvgPreview = document.getElementById("pattern-svg-preview");

// server/pattern/preview.py の PALETTE_HEX と同じ並び(隣接パネルが
// 似た色にならないよう色相を大きく飛ばした固定パレット、20色)。
const PATTERN_PALETTE_HEX = [
  "#e6194b", "#3cb44b", "#4363d8", "#f58231",
  "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
  "#fabed4", "#469990", "#9a6324", "#ffe119",
  "#dcbeff", "#aaffc3", "#ffd8b1", "#fffac8",
  "#a9a9a9", "#808000", "#800000", "#000075",
];

// 手動シードの名前入力欄の候補(datalistと同じ並び)
const PATTERN_SEED_NAME_SUGGESTIONS = [
  "頭", "胴体", "右腕", "左腕", "右脚", "左脚", "右耳", "左耳", "しっぽ",
];

const STATUS_LABELS = {
  queued: "待機中(キュー)",
  preprocessing: "画像前処理中...",
  generating: "3D生成中...",
  postprocessing: "メッシュ後処理中...",
  completed: "完了",
  failed: "失敗",
};

const STATUS_PROGRESS = {
  queued: 5,
  preprocessing: 25,
  generating: 60,
  postprocessing: 85,
  completed: 100,
  failed: 100,
};

let selectedFile = null;
let currentJobId = null;
let pollTimer = null;

// 型紙(Phase 4c): 現在ビューアに表示中のプレビュー種別
// ("model" | "parts" | "panels")
let patternCurrentView = "model";
// 手動シード誘導の状態管理。各要素 {id, x, y, z, name}
let manualSeeds = [];
let manualSeedNextId = 1;
// パーツ分解(ステップ1)が完了しているか(ステップ2の有効化に使う)
let patternPartsCompleted = false;
// LLM誘導が環境的に使えない(/api/health の llm_parts_available=false)場合、
// 手動モードのON/OFFに関わらずチェックボックスは常に無効のままにする。
let patternLlmForceDisabled = false;

// 追加ビュー(back/left/right)の選択中File(未選択はnull)
const extraViewFiles = { back: null, left: null, right: null };

// キャラクターシート分割結果(パネル配列、各要素 {index, image_b64, suggested_view, blob})
let sheetPanels = [];

const viewer = new Viewer(viewerCanvas);

// --- ファイル選択 / ドラッグ&ドロップ ---------------------------------------
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    handleFileSelect(e.dataTransfer.files[0]);
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) handleFileSelect(fileInput.files[0]);
});

function handleFileSelect(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImage.src = url;
  previewImage.hidden = false;
  updateMultiviewNote();
}

// --- 追加ビュー(背面/左/右)アップロード -------------------------------------
document.querySelectorAll(".extra-dropzone").forEach((zone) => {
  const view = zone.dataset.view;
  const input = zone.querySelector(".extra-view-input");

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      setExtraViewFile(view, e.dataTransfer.files[0]);
    }
  });
  input.addEventListener("change", () => {
    if (input.files.length > 0) setExtraViewFile(view, input.files[0]);
  });
});

document.querySelectorAll(".extra-view-clear").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    clearExtraViewFile(btn.dataset.view);
  });
});

function setExtraViewFile(view, file) {
  extraViewFiles[view] = file;
  const slot = document.querySelector(`.extra-view-slot[data-view="${view}"]`);
  const preview = slot.querySelector(".extra-view-preview");
  const label = slot.querySelector(".extra-view-label");
  const clearBtn = slot.querySelector(".extra-view-clear");

  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  label.hidden = true;
  clearBtn.hidden = false;
  updateMultiviewNote();
}

function clearExtraViewFile(view) {
  extraViewFiles[view] = null;
  const slot = document.querySelector(`.extra-view-slot[data-view="${view}"]`);
  const preview = slot.querySelector(".extra-view-preview");
  const label = slot.querySelector(".extra-view-label");
  const clearBtn = slot.querySelector(".extra-view-clear");
  const input = slot.querySelector(".extra-view-input");

  preview.hidden = true;
  preview.src = "";
  label.hidden = false;
  clearBtn.hidden = true;
  input.value = "";
  updateMultiviewNote();
}

function updateMultiviewNote() {
  const activeViews = ["front", ...Object.keys(extraViewFiles).filter((v) => extraViewFiles[v])];
  if (activeViews.length > 1) {
    multiviewNote.hidden = false;
    multiviewNote.textContent = `${activeViews.length}ビュー(${activeViews.join("/")})で生成`;
  } else {
    multiviewNote.hidden = true;
    multiviewNote.textContent = "";
  }
}

// --- キャラクターシート分割 ---------------------------------------------------
sheetDropzone.addEventListener("click", () => {
  if (!sheetSplitBtn.disabled) sheetFileInput.click();
});
sheetSplitBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  if (!sheetSplitBtn.disabled) sheetFileInput.click();
});
sheetDropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  if (!sheetSplitBtn.disabled) sheetDropzone.classList.add("dragover");
});
sheetDropzone.addEventListener("dragleave", () => sheetDropzone.classList.remove("dragover"));
sheetDropzone.addEventListener("drop", async (e) => {
  e.preventDefault();
  sheetDropzone.classList.remove("dragover");
  if (sheetSplitBtn.disabled || e.dataTransfer.files.length === 0) return;
  await handleSheetFile(e.dataTransfer.files[0]);
});
sheetFileInput.addEventListener("change", async () => {
  if (sheetFileInput.files.length === 0) return;
  await handleSheetFile(sheetFileInput.files[0]);
  sheetFileInput.value = "";
});

async function handleSheetFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("シート画像ファイルを選択してください。");
    return;
  }
  sheetSplitBtn.disabled = true;
  sheetSplitBtn.textContent = "分割中...";
  sheetDropzone.classList.add("processing");
  try {
    const formData = new FormData();
    formData.append("image", file);
    const res = await fetch("/api/sheet/split", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "シート分割に失敗しました。");
    }
    const data = await res.json();
    sheetPanels = data.panels;
    renderSheetPanels();
  } catch (err) {
    alert(err.message);
  } finally {
    sheetSplitBtn.disabled = false;
    sheetSplitBtn.textContent = "シート画像を選んで分割";
    sheetDropzone.classList.remove("processing");
  }
}

function renderSheetPanels() {
  sheetPanelsList.innerHTML = "";
  if (sheetPanels.length === 0) {
    sheetPanelsArea.hidden = true;
    return;
  }
  sheetPanelsArea.hidden = false;

  for (const panel of sheetPanels) {
    const item = document.createElement("div");
    item.className = "sheet-panel-item";

    const thumb = document.createElement("img");
    thumb.className = "sheet-panel-thumb";
    thumb.src = `data:image/png;base64,${panel.image_b64}`;

    const select = document.createElement("select");
    select.className = "sheet-panel-select";
    select.dataset.index = panel.index;
    for (const v of ["front", "back", "left", "right", "none"]) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = VIEW_LABELS_JA[v];
      if (v === panel.suggested_view) opt.selected = true;
      select.appendChild(opt);
    }

    item.appendChild(thumb);
    item.appendChild(select);
    sheetPanelsList.appendChild(item);
  }
}

sheetApplyBtn.addEventListener("click", () => {
  const selects = sheetPanelsList.querySelectorAll(".sheet-panel-select");
  const assignment = {}; // view -> panel
  selects.forEach((select) => {
    const view = select.value;
    if (view === "none") return;
    const idx = Number(select.dataset.index);
    const panel = sheetPanels.find((p) => p.index === idx);
    if (panel) assignment[view] = panel;
  });

  for (const [view, panel] of Object.entries(assignment)) {
    const raw = atob(panel.image_b64);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const blob = new Blob([bytes], { type: "image/png" });
    const file = new File([blob], `panel_${panel.index}_${view}.png`, { type: "image/png" });

    if (view === "front") {
      handleFileSelect(file);
    } else if (view in extraViewFiles) {
      setExtraViewFile(view, file);
    }
  }

  updateMultiviewNote();
});

// --- カラーモード切替 -------------------------------------------------------
colorModeCheckbox.addEventListener("change", () => {
  nColorsField.hidden = !colorModeCheckbox.checked;
});

// --- テクスチャ生成(実験的) (SPEC.md §3.9 / FR-10) --------------------------
// /api/health の texgen_available=false ならチェックボックスを無効化し、
// 「この環境では利用できません」を表示する(フォールバック、3c-3)。
// あわせて使用中ジェネレータをヘッダに表示し、mock時は警告バナーを出す。
// 型紙のLLM誘導 (llm_parts_available) が使えない場合はチェックボックスも
// 無効化する(SPEC.md §3.12 第3層誘導)。
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return;
    const data = await res.json();
    if (!data.texgen_available) {
      textureModeCheckbox.checked = false;
      textureModeCheckbox.disabled = true;
      textureModeUnavailableNote.hidden = false;
    }
    const badge = document.getElementById("generator-badge");
    badge.textContent = `生成エンジン: ${data.generator}`;
    badge.hidden = false;
    if (data.generator === "mock") {
      document.getElementById("mock-warning").hidden = false;
    }
    // Pixal3Dは画像条件付けにDINOv3を使用しており、DINOv3 Licenseの付帯義務
    // として "Built with DINOv3" の表示が求められている(README「ライセンス」節参照)。
    if (data.generator === "pixal3d") {
      document.getElementById("dinov3-credit").hidden = false;
    }
    if (!data.llm_parts_available) {
      patternLlmForceDisabled = true;
      patternUseLlmCheckbox.checked = false;
      patternUseLlmCheckbox.disabled = true;
      patternLlmUnavailableNote.hidden = false;
    }
  } catch (err) {
    console.error(err);
  }
}
checkHealth();

// --- プリセット (FR-11) -----------------------------------------------------
// SPEC.md §3.10: パラメータ一括設定。選択後も個別調整可能(個別変更で「カスタム」に戻る)。
const PRESETS = {
  figure: { target_height_mm: 100, octree_resolution: 384, max_faces: 200000 },
  small_figure: { target_height_mm: 60, octree_resolution: 256, max_faces: 100000 },
  pendant: { target_height_mm: 40, octree_resolution: 256, max_faces: 80000, color_mode: false },
  high_detail: { target_height_mm: 150, octree_resolution: 512, max_faces: 400000 },
};

// プリセット反映で発生するフォーム変更イベントを「ユーザー操作によるカスタム化」と
// 誤検知しないためのフラグ。
let applyingPreset = false;

presetSelect.addEventListener("change", () => {
  const preset = PRESETS[presetSelect.value];
  if (!preset) return; // "custom" 選択時は何もしない

  applyingPreset = true;
  paramsForm.elements["target_height_mm"].value = preset.target_height_mm;
  paramsForm.elements["octree_resolution"].value = String(preset.octree_resolution);
  paramsForm.elements["max_faces"].value = preset.max_faces;
  if (preset.color_mode === false) {
    colorModeCheckbox.checked = false;
    nColorsField.hidden = true;
  }
  applyingPreset = false;
});

// プリセット対象フィールドをユーザーが個別に変更したら「カスタム」表示に戻す。
["target_height_mm", "octree_resolution", "max_faces"].forEach((name) => {
  paramsForm.elements[name].addEventListener("change", () => {
    if (!applyingPreset) presetSelect.value = "custom";
  });
});
colorModeCheckbox.addEventListener("change", () => {
  if (!applyingPreset) presetSelect.value = "custom";
});

// --- 生成フォーム ---------------------------------------------------------
paramsForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) {
    alert("画像を選択してください。");
    return;
  }

  const formData = new FormData(paramsForm);
  const seedRaw = formData.get("seed");
  const params = {
    steps: Number(formData.get("steps")),
    guidance_scale: Number(formData.get("guidance_scale")),
    octree_resolution: Number(formData.get("octree_resolution")),
    seed: seedRaw ? Number(seedRaw) : null,
    remove_bg: formData.get("remove_bg") === "on",
    target_height_mm: Number(formData.get("target_height_mm")),
    max_faces: Number(formData.get("max_faces")),
    color_mode: formData.get("color_mode") === "on" ? "color4" : "none",
    n_colors: Number(formData.get("n_colors") || 4),
    texture_mode: formData.get("texture_mode") === "on" ? "paint" : "none",
  };

  const uploadData = new FormData();
  uploadData.append("image", selectedFile);
  uploadData.append("params", JSON.stringify(params));
  for (const [view, file] of Object.entries(extraViewFiles)) {
    if (file) uploadData.append(`image_${view}`, file);
  }

  generateBtn.disabled = true;
  progressError.hidden = true;
  setProgress("queued");

  try {
    const res = await fetch("/api/jobs", { method: "POST", body: uploadData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "ジョブ作成に失敗しました。");
    }
    const data = await res.json();
    currentJobId = data.job_id;
    startPolling(currentJobId);
    refreshJobHistory();
  } catch (err) {
    showError(err.message);
    generateBtn.disabled = false;
  }
});

function setProgress(status, errorMessage) {
  progressStatus.textContent = STATUS_LABELS[status] || status;
  const pct = STATUS_PROGRESS[status] ?? 0;
  progressBarFill.style.width = `${pct}%`;
  progressBarFill.classList.remove("completed", "failed");
  if (status === "completed") progressBarFill.classList.add("completed");
  if (status === "failed") progressBarFill.classList.add("failed");

  if (errorMessage) {
    progressError.hidden = false;
    progressError.textContent = errorMessage;
  } else {
    progressError.hidden = true;
  }
}

function showError(message) {
  progressError.hidden = false;
  progressError.textContent = message;
}

function startPolling(jobId) {
  stopPolling();
  pollTimer = setInterval(() => pollJob(jobId), 1000);
  pollJob(jobId);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function pollJob(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    if (!res.ok) return;
    const job = await res.json();
    setProgress(job.status, job.status === "failed" ? job.error : null);

    if (job.status === "completed") {
      stopPolling();
      generateBtn.disabled = false;
      await loadJobIntoViewer(job);
      refreshJobHistory();
    } else if (job.status === "failed") {
      stopPolling();
      generateBtn.disabled = false;
      refreshJobHistory();
    }
  } catch (err) {
    // ネットワーク一時エラーは無視して次のポーリングへ
    console.error(err);
  }
}

// --- ビューア + モデル情報 ---------------------------------------------------
async function loadJobIntoViewer(job) {
  currentJobId = job.job_id;
  try {
    await viewer.loadGLB(`/api/jobs/${job.job_id}/model.glb?t=${Date.now()}`);
    viewerPlaceholder.hidden = true;
  } catch (err) {
    console.error("Failed to load GLB", err);
  }

  // 新規モデルロード時はシェーディング表示に戻す(オーバーハングモードは
  // viewer側で自動解除されるため、ボタン表示側もそれに合わせる)。
  setViewModeButton(shadingBtn);
  overhangControls.hidden = true;

  updateModelInfo(job.stats);
  exportButtons.forEach((btn) => (btn.disabled = false));

  // 型紙(Phase 4a〜4c): 新しいジョブに切り替わったら状態をリセットする
  patternToggleBtn.disabled = false;
  patternPanel.hidden = true;
  patternViewSwitch.hidden = true;
  setPatternView("model");
  patternPartsCompleted = false;
  patternPartsResult.hidden = true;
  patternPartsError.hidden = true;
  patternResult.hidden = true;
  patternError.hidden = true;
  patternRunBtn.disabled = true;
  patternManualModeCheckbox.checked = false;
  setManualMode(false);
  clearManualSeeds();
}

function updateModelInfo(stats) {
  if (!stats) return;
  infoVertices.textContent = stats.vertices?.toLocaleString?.() ?? "-";
  infoFaces.textContent = stats.faces?.toLocaleString?.() ?? "-";
  const bbox = stats.bbox_mm || [0, 0, 0];
  infoBbox.textContent = bbox.map((v) => v.toFixed(1)).join(" x ");
  infoWatertight.textContent = stats.watertight ? "OK" : "NG";
  infoVolume.textContent = stats.volume_cm3 !== undefined ? stats.volume_cm3.toFixed(2) : "-";

  const palette = stats.palette || [];
  if (palette.length > 0) {
    infoPaletteItem.hidden = false;
    exportColorNote.hidden = false;
    infoPalette.innerHTML = "";
    for (const entry of palette) {
      const chip = document.createElement("span");
      chip.className = "palette-chip";
      chip.title = `${entry.hex} (${(entry.face_ratio * 100).toFixed(1)}%)`;

      const swatch = document.createElement("span");
      swatch.className = "palette-swatch";
      swatch.style.background = entry.hex;

      const label = document.createElement("span");
      label.className = "palette-ratio";
      label.textContent = `${(entry.face_ratio * 100).toFixed(0)}%`;

      chip.appendChild(swatch);
      chip.appendChild(label);
      infoPalette.appendChild(chip);
    }
  } else {
    infoPaletteItem.hidden = true;
    exportColorNote.hidden = true;
    infoPalette.innerHTML = "";
  }
}

// --- シェーディング / ワイヤーフレーム / オーバーハング切替 (FR-5, FR-12) -----
shadingBtn.addEventListener("click", () => {
  viewer.setOverhangMode(false);
  viewer.setWireframe(false);
  setViewModeButton(shadingBtn);
  overhangControls.hidden = true;
});
wireframeBtn.addEventListener("click", () => {
  viewer.setOverhangMode(false);
  viewer.setWireframe(true);
  setViewModeButton(wireframeBtn);
  overhangControls.hidden = true;
});
overhangBtn.addEventListener("click", () => {
  viewer.setOverhangThreshold(Number(overhangThresholdSlider.value));
  viewer.setOverhangMode(true);
  setViewModeButton(overhangBtn);
  overhangControls.hidden = false;
});

function setViewModeButton(activeBtn) {
  [shadingBtn, wireframeBtn, overhangBtn].forEach((btn) => btn.classList.remove("active"));
  activeBtn.classList.add("active");
}

overhangThresholdSlider.addEventListener("input", () => {
  const deg = Number(overhangThresholdSlider.value);
  overhangThresholdValue.textContent = String(deg);
  viewer.setOverhangThreshold(deg);
});

// --- エクスポート -----------------------------------------------------------
exportButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    if (!currentJobId) return;
    const format = btn.dataset.format;
    window.location.href = `/api/jobs/${currentJobId}/download?format=${format}`;
  });
});

// --- ぬいぐるみ型紙生成 (SPEC.md §3.12 / FR-13, Phase 4a〜4c) ----------------
//
// 2段階構成:
//   ステップ1 POST /api/jobs/{id}/pattern/parts — パーツ自動分解
//     (誘導方式は 手動シード > LLM > 画像色領域 > ジオメトリ の優先順位)
//   ステップ2 POST /api/jobs/{id}/pattern       — パーツ単位のパネル分割→
//     平坦化→SVG出力(ステップ1が未実行なら内部で自動実行される)

patternToggleBtn.addEventListener("click", () => {
  patternPanel.hidden = !patternPanel.hidden;
  patternViewSwitch.hidden = patternPanel.hidden;
});

// --- ビューア表示切替(モデル / パーツ分け / パネル分割) ---------------------
patternViewBtns.forEach((btn) => {
  btn.addEventListener("click", () => setPatternView(btn.dataset.view));
});

async function setPatternView(view) {
  patternCurrentView = view;
  patternViewBtns.forEach((btn) => btn.classList.toggle("active", btn.dataset.view === view));
  if (!currentJobId) return;

  let url;
  if (view === "parts") {
    url = `/api/jobs/${currentJobId}/pattern_parts_preview.glb?t=${Date.now()}`;
  } else if (view === "panels") {
    url = `/api/jobs/${currentJobId}/pattern_preview.glb?t=${Date.now()}`;
  } else {
    url = `/api/jobs/${currentJobId}/model.glb?t=${Date.now()}`;
  }

  try {
    await viewer.loadGLB(url);
    viewerPlaceholder.hidden = true;
  } catch (err) {
    console.error(`Failed to load ${view} preview GLB`, err);
  }

  // 手動シードのピッキングはモデル表示中のみ有効(パーツ/パネルプレビューは
  // 別メッシュのため座標系が一致しない可能性があるうえ、意味的にも不要)。
  if (patternManualModeCheckbox.checked && view === "model") {
    viewer.enableSeedPicking(onSeedPick);
  } else {
    viewer.disableSeedPicking();
  }
}

// --- ステップ1: パーツ数/画像/LLM誘導のUI -----------------------------------
patternManualModeCheckbox.addEventListener("change", () => {
  setManualMode(patternManualModeCheckbox.checked);
});

function setManualMode(enabled) {
  patternManualSeedPanel.hidden = !enabled;
  patternNPartsSelect.disabled = enabled;
  patternUseImageCheckbox.disabled = enabled;
  patternUseLlmCheckbox.disabled = enabled || patternLlmForceDisabled;

  if (enabled) {
    // 手動シード指定はビューア上のクリックで行うため、表示を強制的に
    // 「モデル」にしてシードピッキングを有効化する。
    setPatternView("model");
    viewer.enableSeedPicking(onSeedPick);
  } else {
    viewer.disableSeedPicking();
  }
  updatePartsRunBtnState();
}

// --- 手動シード管理 -----------------------------------------------------------
function onSeedPick(localPos) {
  const seed = {
    id: manualSeedNextId++,
    x: localPos.x,
    y: localPos.y,
    z: localPos.z,
    name: "",
  };
  manualSeeds.push(seed);
  renderManualSeeds();
}

// シードの色分け: 同じ名前(トリム後、空文字は個別扱い)ごとにグループ化し、
// グループの初出順でパレットを巡回させる。
function computeSeedColors() {
  const groupKeyOf = (s) => (s.name && s.name.trim() ? s.name.trim() : `__unnamed_${s.id}`);
  const order = [];
  for (const s of manualSeeds) {
    const key = groupKeyOf(s);
    if (!order.includes(key)) order.push(key);
  }
  const colorByKey = new Map();
  order.forEach((key, idx) => {
    colorByKey.set(key, PATTERN_PALETTE_HEX[idx % PATTERN_PALETTE_HEX.length]);
  });
  const result = new Map();
  for (const s of manualSeeds) {
    result.set(s.id, colorByKey.get(groupKeyOf(s)));
  }
  return result;
}

function renderManualSeeds() {
  const colors = computeSeedColors();

  patternSeedList.innerHTML = "";
  for (const seed of manualSeeds) {
    const li = document.createElement("li");
    li.className = "pattern-seed-item";

    const swatch = document.createElement("span");
    swatch.className = "pattern-panel-swatch";
    swatch.style.background = colors.get(seed.id);

    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "pattern-seed-name-input";
    nameInput.value = seed.name;
    nameInput.placeholder = "部位名(例: 頭)";
    nameInput.setAttribute("list", "pattern-seed-name-suggestions");
    nameInput.addEventListener("input", () => {
      seed.name = nameInput.value;
      renderManualSeeds();
    });

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "secondary-btn pattern-seed-remove-btn";
    removeBtn.textContent = "削除";
    removeBtn.addEventListener("click", () => {
      manualSeeds = manualSeeds.filter((s) => s.id !== seed.id);
      viewer.removeSeedMarker(seed.id);
      renderManualSeeds();
    });

    li.appendChild(swatch);
    li.appendChild(nameInput);
    li.appendChild(removeBtn);
    patternSeedList.appendChild(li);

    const colorInt = Number.parseInt(colors.get(seed.id).slice(1), 16);
    if (viewer._seedMarkers && viewer._seedMarkers.has(seed.id)) {
      viewer.setSeedMarkerColor(seed.id, colorInt);
    } else {
      viewer.addSeedMarker(seed.id, seed, colorInt);
    }
  }

  updatePartsRunBtnState();
}

patternSeedClearBtn.addEventListener("click", () => {
  clearManualSeeds();
});

function clearManualSeeds() {
  manualSeeds = [];
  patternSeedList.innerHTML = "";
  viewer.clearSeedMarkers();
  updatePartsRunBtnState();
  // パーツ分け実行後はビューアが「パーツ分け」表示に切り替わり、それに伴い
  // シードピッキングも無効化されている(setPatternView参照)。クリア後に
  // 再度クリックで指定し直せるよう、手動モード中は「モデル」表示へ戻して
  // ピッキングを復帰させる(戻さないと「クリアしたのに再指定できない」状態になる)。
  if (patternManualModeCheckbox.checked) {
    setPatternView("model");
  }
}

function updatePartsRunBtnState() {
  if (!patternManualModeCheckbox.checked) {
    patternPartsRunBtn.disabled = false;
    return;
  }
  const total = manualSeeds.length;
  // 名前未入力のシードは server/main.py 側で `part_{index+1}` という
  // (インデックス由来で必ずユニークな)名前が自動で振られ、1シード=1パーツ
  // として扱われる。ボタンの有効化判定もこれに合わせて未入力を許容する
  // (全シードへの命名必須にすると、命名前にクリックしただけでボタンが
  // 押せなくなり「常に禁止」に見えるバグになっていた)。
  const effectiveNames = manualSeeds.map((s, i) => s.name.trim() || `part_${i + 1}`);
  const uniqueNames = new Set(effectiveNames);
  patternPartsRunBtn.disabled = !(
    total >= 2 &&
    total <= 48 &&
    uniqueNames.size >= 2 &&
    uniqueNames.size <= 20
  );
}

// --- ステップ1実行: パーツ分けを実行 -----------------------------------------
patternPartsRunBtn.addEventListener("click", async () => {
  if (!currentJobId) return;

  patternPartsRunBtn.disabled = true;
  patternPartsError.hidden = true;
  patternPartsResult.hidden = true;

  try {
    const body = {};
    if (patternManualModeCheckbox.checked) {
      body.seeds = manualSeeds.map((s) => ({ x: s.x, y: s.y, z: s.z, name: s.name.trim() }));
      body.propagate_back = patternPropagateBackCheckbox.checked;
      body.planar_boundaries = patternPlanarBoundariesCheckbox.checked;
    } else {
      body.n_parts = Number(patternNPartsSelect.value);
      body.use_image = patternUseImageCheckbox.checked;
      body.use_llm = patternUseLlmCheckbox.checked;
    }

    const res = await fetch(`/api/jobs/${currentJobId}/pattern/parts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `パーツ分けに失敗しました(status=${res.status})`);
    }

    const data = await res.json();
    renderPatternPartsResult(data);

    await viewer.loadGLB(`/api/jobs/${currentJobId}/pattern_parts_preview.glb?t=${Date.now()}`);
    viewerPlaceholder.hidden = true;
    setPatternView("parts");
    viewer.disableSeedPicking();

    patternPartsCompleted = true;
    patternRunBtn.disabled = false;
  } catch (err) {
    console.error(err);
    patternPartsError.textContent = err.message || String(err);
    patternPartsError.hidden = false;
  } finally {
    updatePartsRunBtnState();
  }
});

const GUIDANCE_LABELS_JA = {
  manual: "手動シード",
  llm: "AI(LLM)検出",
  color: "画像の色境界",
  geometry: "ジオメトリのみ",
};

function renderPatternPartsResult(data) {
  patternPartsGuidance.textContent = GUIDANCE_LABELS_JA[data.guidance] || data.guidance;

  if (data.guidance === "manual") {
    patternPartsCount.textContent = `${data.n_parts_actual} (指定: ${data.n_parts_requested})`;
    const infoParts = [];
    infoParts.push(`背面仮想シード${data.n_virtual_seeds}個`);
    if (Array.isArray(data.planar_fit) && data.planar_fit.length > 0) {
      const applied = data.planar_fit.filter((p) => p.applied).length;
      infoParts.push(`境界平面化: ${applied}/${data.planar_fit.length}ペア適用`);
    }
    patternPartsManualInfo.textContent = infoParts.join(" / ");
    patternPartsManualInfo.hidden = false;
  } else {
    const auto = data.n_parts_requested === 0 ? "自動" : String(data.n_parts_requested);
    patternPartsCount.textContent = `${data.n_parts_actual} (要求: ${auto})`;
    patternPartsManualInfo.hidden = true;
  }

  patternPartsList.innerHTML = "";
  const totalFaces = data.parts.reduce((sum, p) => sum + p.n_faces, 0) || 1;
  data.parts.forEach((part) => {
    const li = document.createElement("li");

    const swatch = document.createElement("span");
    swatch.className = "pattern-panel-swatch";
    swatch.style.background = PATTERN_PALETTE_HEX[part.part_id % PATTERN_PALETTE_HEX.length];

    const label = document.createElement("span");
    const nameLabel = part.name ? `${part.name}` : `部位${part.part_id + 1}`;
    const faceRatio = ((part.n_faces / totalFaces) * 100).toFixed(0);
    label.textContent = `${nameLabel}: ${part.n_faces}面 (${faceRatio}%)`;

    li.appendChild(swatch);
    li.appendChild(label);
    patternPartsList.appendChild(li);
  });

  patternPartsResult.hidden = false;
}

// --- ステップ2: パネル分割・平坦化・SVG出力 ----------------------------------
patternNPanelsInput.addEventListener("input", () => {
  patternNPanelsValue.textContent = patternNPanelsInput.value;
});

patternSeamAllowanceInput.addEventListener("input", () => {
  patternSeamAllowanceValue.textContent = patternSeamAllowanceInput.value;
});

patternRunBtn.addEventListener("click", async () => {
  if (!currentJobId) return;

  patternRunBtn.disabled = true;
  patternError.hidden = true;
  patternResult.hidden = true;

  try {
    const res = await fetch(`/api/jobs/${currentJobId}/pattern`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        n_panels: Number(patternNPanelsInput.value),
        use_colors: patternUseColorsCheckbox.checked,
        seam_allowance_mm: Number(patternSeamAllowanceInput.value),
      }),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `型紙生成に失敗しました(status=${res.status})`);
    }

    const data = await res.json();
    renderPatternResult(data);

    await viewer.loadGLB(`/api/jobs/${currentJobId}/pattern_preview.glb?t=${Date.now()}`);
    viewerPlaceholder.hidden = true;
    setPatternView("panels");
  } catch (err) {
    console.error(err);
    patternError.textContent = err.message || String(err);
    patternError.hidden = false;
  } finally {
    patternRunBtn.disabled = false;
  }
});

function renderPatternResult(data) {
  patternResultCount.textContent = `${data.n_panels_total} (要求上限: ${data.n_panels_max_per_part}/パーツ)`;
  patternResultFlattenOk.textContent = `${data.n_panels_flattened} / ${data.n_panels_total}`;

  patternPanelList.innerHTML = "";
  data.parts.forEach((part) => {
    part.panels.forEach((panel) => {
      const li = document.createElement("li");
      if (!panel.disk_topology || panel.flatten_failed) li.classList.add("pattern-panel-warn");

      const swatch = document.createElement("span");
      swatch.className = "pattern-panel-swatch";
      swatch.style.background = part.color_hex;

      const label = document.createElement("span");
      const areaCm2 = (panel.area_mm2 / 100).toFixed(1);
      const topologyNote = panel.disk_topology ? "" : " ※穴あり";

      let distortionNote = "";
      if (panel.flatten_failed) {
        distortionNote = " ※平坦化失敗";
      } else if (panel.distortion) {
        const over10pct = panel.distortion.edge_length_over_10pct_fraction * 100;
        distortionNote = ` / 辺長歪み±10%超: ${over10pct.toFixed(0)}%`;
        if (over10pct > 20) li.classList.add("pattern-panel-warn");
      }

      label.textContent = `${part.part_label} #${panel.panel_no}: ${panel.n_faces}面 / ${areaCm2}cm²${topologyNote}${distortionNote}`;

      li.appendChild(swatch);
      li.appendChild(label);
      patternPanelList.appendChild(li);
    });
  });

  const svgUrl = `/api/jobs/${currentJobId}/pattern.svg?t=${Date.now()}`;
  patternSvgDownloadBtn.href = svgUrl;
  patternSvgPreview.src = svgUrl;

  patternResult.hidden = false;
}

// --- ジョブ履歴 -------------------------------------------------------------
async function refreshJobHistory() {
  try {
    const res = await fetch("/api/jobs");
    if (!res.ok) return;
    const jobs = await res.json();
    renderJobHistory(jobs);
  } catch (err) {
    console.error(err);
  }
}

function renderJobHistory(jobs) {
  jobHistoryEl.innerHTML = "";
  for (const job of jobs) {
    const li = document.createElement("li");
    if (job.job_id === currentJobId) li.classList.add("selected");

    const thumb = document.createElement("img");
    thumb.className = "job-thumb";
    thumb.src = `/api/jobs/${job.job_id}/input`;
    thumb.onerror = () => (thumb.style.visibility = "hidden");

    const meta = document.createElement("div");
    meta.className = "job-meta";
    const createdAt = new Date(job.created_at).toLocaleString("ja-JP");
    meta.innerHTML = `
      <span class="job-status ${job.status}">${STATUS_LABELS[job.status] || job.status}</span>
      <span class="job-time">${createdAt}</span>
    `;

    li.appendChild(thumb);
    li.appendChild(meta);
    li.addEventListener("click", async () => {
      currentJobId = job.job_id;
      document.querySelectorAll("#job-history li").forEach((el) => el.classList.remove("selected"));
      li.classList.add("selected");
      setProgress(job.status, job.status === "failed" ? job.error : null);
      if (job.status === "completed") {
        await loadJobIntoViewer(job);
      }
    });

    jobHistoryEl.appendChild(li);
  }
}

// 初期ロード
refreshJobHistory();
