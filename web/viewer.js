// Three.js 3Dビューア (SPEC.md §3.5 / FR-5)
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const BUILD_PLATE_MM = 220;

export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x14161a);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
    this.camera.position.set(180, 160, 220);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 40, 0);

    this._setupLights();
    this._setupGrid();

    this.currentModel = null;
    this.wireframe = false;
    this.loader = new GLTFLoader();

    // オーバーハングヒートマップ (FR-12) の状態
    this.overhangMode = false;
    this.overhangThresholdDeg = 45;
    // メッシュごとの元マテリアル・元頂点カラー属性を退避しておく (Map<mesh, {material, color}>)
    this._overhangBackup = new Map();

    // 手動シードピッキング (パーツ分けの手動シード誘導、SPEC.md §3.12) の状態
    this._seedPickingEnabled = false;
    this._onSeedPick = null;
    this._raycaster = new THREE.Raycaster();
    this._seedMarkers = new Map(); // Map<id, THREE.Mesh>
    this._seedMarkerGroup = new THREE.Group();
    this._pointerDownPos = null;

    // window の resize イベントだけでは、サイドパネルの開閉など
    // ウィンドウサイズ自体は変わらないレイアウト変化(型紙パネル表示等)で
    // ビューアコンテナの寸法だけが変わるケースを検知できず、レンダラーの
    // アスペクト比が古いまま残って表示が縦横比崩れ(縦が潰れる)になる。
    // ResizeObserver でコンテナ自身のサイズ変化を直接監視する。
    window.addEventListener("resize", () => this._onResize());
    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this.canvas.parentElement);
    this._onResize();
    this._animate();
  }

  _setupLights() {
    const hemi = new THREE.HemisphereLight(0xffffff, 0x2a2a2a, 1.1);
    this.scene.add(hemi);

    const dir = new THREE.DirectionalLight(0xffffff, 1.0);
    dir.position.set(150, 300, 200);
    this.scene.add(dir);

    const dir2 = new THREE.DirectionalLight(0xffffff, 0.4);
    dir2.position.set(-200, 100, -150);
    this.scene.add(dir2);
  }

  _setupGrid() {
    // グリッド床 + ビルドプレート枠 (220x220mm相当)
    const grid = new THREE.GridHelper(BUILD_PLATE_MM, 22, 0x555a66, 0x2a2d36);
    this.scene.add(grid);

    const plateGeom = new THREE.EdgesGeometry(
      new THREE.BoxGeometry(BUILD_PLATE_MM, 0.5, BUILD_PLATE_MM)
    );
    const plateMat = new THREE.LineBasicMaterial({ color: 0x5b8cff });
    const plateEdges = new THREE.LineSegments(plateGeom, plateMat);
    plateEdges.position.y = 0;
    this.scene.add(plateEdges);
  }

  _onResize() {
    const parent = this.canvas.parentElement;
    const width = parent.clientWidth;
    const height = parent.clientHeight;
    if (width <= 0 || height <= 0) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    // 型紙パネルの開閉などでビューア領域の縦横比が変わった際、カメラ距離を
    // 据え置いたままアスペクトだけ更新すると、キャラクターの上下(または
    // 左右)がフレーム外に切れて縦横比が崩れて見える。距離を現在の視線
    // 方向を保ったまま新アスペクトに合わせて再計算し、常にモデル全体が
    // 収まるようにする(ユーザーの回転操作は保持、ズームは再フィットされる)。
    if (this.currentModel) {
      this._refitDistanceToAspect(this.currentModel);
    }
    // スクロールバーの出現/消失など、この時点ではまだ確定していない
    // 追加のレイアウト変化が1フレーム遅れて反映されることがある
    // (`scrollbar-gutter: stable` で大半は防いでいるが保険として残す)。
    // 次フレームでサイズが変わっていれば、静かに再測定・再フィットする。
    requestAnimationFrame(() => {
      const w2 = parent.clientWidth;
      const h2 = parent.clientHeight;
      if (w2 > 0 && h2 > 0 && (w2 !== width || h2 !== height)) {
        this._onResize();
      }
    });
  }

  _refitDistanceToAspect(object3d) {
    const box = new THREE.Box3().setFromObject(object3d);
    const size = new THREE.Vector3();
    box.getSize(size);
    const center = new THREE.Vector3();
    box.getCenter(center);
    const radius = Math.max(size.length() / 2, 1e-3);

    const vFov = THREE.MathUtils.degToRad(this.camera.fov);
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * this.camera.aspect);
    const dist = Math.max(radius / Math.sin(vFov / 2), radius / Math.sin(hFov / 2)) * 1.15;

    let dir = this.camera.position.clone().sub(this.controls.target);
    if (dir.lengthSq() < 1e-9) dir.set(0.7, 0.6, 0.9);
    dir.normalize();

    this.camera.position.copy(center).addScaledVector(dir, dist);
    this.controls.target.copy(center);
    this.camera.near = Math.max(dist / 100, 0.01);
    this.camera.far = dist * 50;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  _animate = () => {
    requestAnimationFrame(this._animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  async loadGLB(url) {
    return new Promise((resolve, reject) => {
      this.loader.load(
        url,
        (gltf) => {
          this._replaceModel(gltf.scene);
          resolve(gltf);
        },
        undefined,
        (err) => reject(err)
      );
    });
  }

  _replaceModel(object3d) {
    if (this.currentModel) {
      this.scene.remove(this.currentModel);
      this.currentModel.traverse((child) => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          const materials = Array.isArray(child.material) ? child.material : [child.material];
          materials.forEach((m) => m.dispose());
        }
      });
    }

    this._overhangBackup.clear();
    this.overhangMode = false;
    this.clearSeedMarkers();

    // Three.jsのY-up座標系に合わせ、生成メッシュ(Z-up, 床=z0)をX軸-90度回転して
    // Y軸を高さ方向にする。
    const wrapper = new THREE.Group();
    wrapper.add(object3d);
    wrapper.rotation.x = -Math.PI / 2;
    // シードマーカーはラッパー内に置くことで、メッシュと同じZ-upローカル座標
    // (=サーバに渡す座標系)をそのまま position に使える。
    wrapper.add(this._seedMarkerGroup);

    this.currentModel = wrapper;
    this.setWireframe(this.wireframe);
    this.scene.add(wrapper);
    this._frameCamera(wrapper);
  }

  _frameCamera(object3d) {
    const box = new THREE.Box3().setFromObject(object3d);
    const size = new THREE.Vector3();
    box.getSize(size);
    const center = new THREE.Vector3();
    box.getCenter(center);

    const maxDim = Math.max(size.x, size.y, size.z, 1);
    const dist = maxDim * 2.0;

    this.controls.target.copy(center);
    this.camera.position.set(center.x + dist * 0.7, center.y + dist * 0.6, center.z + dist * 0.9);
    this.camera.near = maxDim / 100;
    this.camera.far = maxDim * 50;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  setWireframe(enabled) {
    this.wireframe = enabled;
    if (!this.currentModel) return;
    this.currentModel.traverse((child) => {
      if (child.isMesh && child.material) {
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach((m) => (m.wireframe = enabled));
      }
    });
  }

  clear() {
    if (this.currentModel) {
      this.scene.remove(this.currentModel);
      this.currentModel = null;
    }
    this._overhangBackup.clear();
    this.overhangMode = false;
    this._seedMarkers.clear();
  }

  // --- オーバーハングヒートマップ (FR-12) --------------------------------------
  //
  // 表示中ジオメトリの面法線から、造形方向(ビューアのシーンではY軸が上、
  // GLB読込直後のオブジェクトはZ-upだったものをラッパーで-90度X回転してY-upに
  // 揃えている)に対する下向き傾斜角を求め、頂点色にベイクする。
  // 閾値角を超える下向き面(オーバーハング = サポートが必要になりやすい)ほど
  // 赤く、閾値以下は白〜薄グレー、接地面付近(モデル高さの2%未満)は薄青で
  // 「サポート不要」として区別する。

  setOverhangThreshold(deg) {
    this.overhangThresholdDeg = deg;
    if (this.overhangMode) {
      this._applyOverhangColors();
    }
  }

  setOverhangMode(enabled) {
    if (enabled === this.overhangMode) return;
    this.overhangMode = enabled;
    if (!this.currentModel) return;

    if (enabled) {
      this._backupAndApplyOverhang();
    } else {
      this._restoreOriginalMaterials();
    }
  }

  _eachMesh(callback) {
    if (!this.currentModel) return;
    this.currentModel.traverse((child) => {
      if (child.isMesh && child.geometry) callback(child);
    });
  }

  _backupAndApplyOverhang() {
    // モデル全体のワールド空間でのY方向範囲(高さ)を求め、底面判定の基準にする。
    this.currentModel.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(this.currentModel);
    const minY = box.min.y;
    const heightY = Math.max(box.max.y - minY, 1e-6);

    this._eachMesh((mesh) => {
      if (!this._overhangBackup.has(mesh)) {
        const geom = mesh.geometry;
        const originalColorAttr = geom.getAttribute("color") || null;
        this._overhangBackup.set(mesh, {
          material: mesh.material,
          color: originalColorAttr ? originalColorAttr.clone() : null,
          hadVertexColors: Array.isArray(mesh.material)
            ? mesh.material.some((m) => m.vertexColors)
            : !!mesh.material?.vertexColors,
        });
      }

      // ヒートマップとしての視認性を優先し、照明の影響を受けず頂点色をそのまま
      // 表示する MeshBasicMaterial を使う(陰影で赤/白の判別が難しくなるのを防ぐ)。
      const overhangMat = new THREE.MeshBasicMaterial({
        vertexColors: true,
        wireframe: this.wireframe,
      });
      mesh.material = overhangMat;
    });

    this._applyOverhangColors({ minY, heightY });
  }

  _applyOverhangColors(bounds) {
    if (!bounds) {
      this.currentModel.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(this.currentModel);
      bounds = { minY: box.min.y, heightY: Math.max(box.max.y - box.min.y, 1e-6) };
    }
    const { minY, heightY } = bounds;
    const thresholdRad = (this.overhangThresholdDeg * Math.PI) / 180;
    const baseEpsilon = heightY * 0.02;

    const downWorld = new THREE.Vector3(0, -1, 0);
    const normalMatrix = new THREE.Matrix3();
    const worldNormal = new THREE.Vector3();
    const worldPos = new THREE.Vector3();

    this._eachMesh((mesh) => {
      const geom = mesh.geometry;
      if (!geom.getAttribute("normal")) geom.computeVertexNormals();

      const posAttr = geom.getAttribute("position");
      const normalAttr = geom.getAttribute("normal");
      const count = posAttr.count;

      let colorAttr = geom.getAttribute("color");
      if (!colorAttr || colorAttr.itemSize !== 3 || colorAttr.count !== count) {
        colorAttr = new THREE.BufferAttribute(new Float32Array(count * 3), 3);
        geom.setAttribute("color", colorAttr);
      }

      mesh.updateMatrixWorld(true);
      normalMatrix.getNormalMatrix(mesh.matrixWorld);

      for (let i = 0; i < count; i++) {
        worldPos.fromBufferAttribute(posAttr, i).applyMatrix4(mesh.matrixWorld);
        worldNormal.fromBufferAttribute(normalAttr, i).applyMatrix3(normalMatrix).normalize();

        let r, g, b;
        if (worldPos.y - minY < baseEpsilon) {
          // 接地面付近: サポート不要 (薄青)
          r = 0.5;
          g = 0.72;
          b = 1.0;
        } else {
          // 法線と「真下」方向の角度が小さいほど下向き(オーバーハング)
          const cosDown = worldNormal.dot(downWorld);
          const angleFromDown = Math.acos(THREE.MathUtils.clamp(cosDown, -1, 1));
          // 傾斜角(水平面からの下向き角度) = 90° - angleFromDown
          const downwardTilt = Math.PI / 2 - angleFromDown;

          if (downwardTilt > thresholdRad) {
            // オーバーハング: 超過度合いに応じて白→赤のグラデーション
            const maxTilt = Math.PI / 2;
            const t = THREE.MathUtils.clamp(
              (downwardTilt - thresholdRad) / Math.max(maxTilt - thresholdRad, 1e-6),
              0,
              1
            );
            r = 1.0;
            g = 0.85 * (1 - t);
            b = 0.85 * (1 - t);
          } else {
            // 安全: 白〜薄グレー(傾斜が小さいほど白に近い)
            const t = THREE.MathUtils.clamp(downwardTilt / Math.max(thresholdRad, 1e-6), 0, 1);
            const shade = 0.95 - 0.15 * t;
            r = shade;
            g = shade;
            b = shade;
          }
        }

        colorAttr.setXYZ(i, r, g, b);
      }
      colorAttr.needsUpdate = true;
    });
  }

  _restoreOriginalMaterials() {
    this._eachMesh((mesh) => {
      const backup = this._overhangBackup.get(mesh);
      if (!backup) return;

      const geom = mesh.geometry;
      if (backup.color) {
        geom.setAttribute("color", backup.color);
      } else {
        geom.deleteAttribute("color");
      }

      if (mesh.material && mesh.material !== backup.material) {
        mesh.material.dispose();
      }
      mesh.material = backup.material;
    });
    this._overhangBackup.clear();
  }

  // --- 手動シードピッキング (パーツ分けの手動シード誘導、SPEC.md §3.12) -------
  //
  // モデルはZ-up(GLB読込直後)をラッパーGroupで-90度X回転してY-up表示している
  // (`_replaceModel` 参照)。ピック交点はワールド座標→ラッパーのローカル座標
  // (=元のZ-up・mm単位のメッシュ座標系)へ`worldToLocal`で逆変換して呼び出し側へ
  // 渡す(サーバ側のprepared meshと同一座標系。`prepare_mesh`は平行移動・
  // スケールを行わないため、model.glbの座標をそのまま使ってよい)。

  enableSeedPicking(onPick) {
    this._onSeedPick = onPick;
    if (this._seedPickingEnabled) return;
    this._seedPickingEnabled = true;
    this.canvas.addEventListener("pointerdown", this._onSeedPointerDown);
    this.canvas.addEventListener("pointerup", this._onSeedPointerUp);
  }

  disableSeedPicking() {
    if (!this._seedPickingEnabled) return;
    this._seedPickingEnabled = false;
    this._onSeedPick = null;
    this.canvas.removeEventListener("pointerdown", this._onSeedPointerDown);
    this.canvas.removeEventListener("pointerup", this._onSeedPointerUp);
  }

  _onSeedPointerDown = (event) => {
    this._pointerDownPos = { x: event.clientX, y: event.clientY, t: performance.now() };
  };

  _onSeedPointerUp = (event) => {
    if (!this._pointerDownPos) return;
    const dx = event.clientX - this._pointerDownPos.x;
    const dy = event.clientY - this._pointerDownPos.y;
    const dt = performance.now() - this._pointerDownPos.t;
    this._pointerDownPos = null;
    // OrbitControlsのドラッグ回転とクリックを区別する。実際のマウス/
    // トラックパッド操作では静止クリックのつもりでも数px動くことが多く、
    // 距離だけで5px閾値を取ると実機で「クリックしても反応しない」体感に
    // なる(2026-07-16のユーザー報告で判明)。距離が緩め(12px)の閾値内、
    // または素早い操作(300ms未満、ドラッグ回転は通常もっと長く続く)なら
    // クリックとして扱う。
    if (Math.hypot(dx, dy) >= 12 && dt >= 300) return;
    this._pickSeedAt(event);
  };

  _pickSeedAt(event) {
    if (!this.currentModel || !this._onSeedPick) return;

    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1
    );
    this._raycaster.setFromCamera(ndc, this.camera);

    const targets = [];
    this.currentModel.traverse((child) => {
      if (child.isMesh && child.geometry) targets.push(child);
    });
    const hits = this._raycaster.intersectObjects(targets, false);
    if (hits.length === 0) return;

    const hit = hits[0];
    // ワールド座標 → ラッパーGroupのローカル座標(元のZ-up・mm座標系)へ変換。
    this.currentModel.updateMatrixWorld(true);
    const localPos = this.currentModel.worldToLocal(hit.point.clone());

    this._onSeedPick({ x: localPos.x, y: localPos.y, z: localPos.z });
  }

  addSeedMarker(id, localPos, color = 0xffffff) {
    this.removeSeedMarker(id);
    if (!this.currentModel) return;

    // マーカー半径はモデルサイズの目安(バウンディングボックス最大辺の1.5%程度、
    // 最小2mm)にする。
    this.currentModel.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(this.currentModel);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z, 1);
    const radius = Math.max(maxDim * 0.015, 2);

    const geom = new THREE.SphereGeometry(radius, 16, 12);
    const mat = new THREE.MeshBasicMaterial({ color });
    const marker = new THREE.Mesh(geom, mat);
    marker.position.set(localPos.x, localPos.y, localPos.z);
    marker.raycast = () => {}; // Raycast対象から除外(シード自身がピックの邪魔をしないように)

    this._seedMarkerGroup.add(marker);
    this._seedMarkers.set(id, marker);
  }

  removeSeedMarker(id) {
    const marker = this._seedMarkers.get(id);
    if (!marker) return;
    this._seedMarkerGroup.remove(marker);
    marker.geometry.dispose();
    marker.material.dispose();
    this._seedMarkers.delete(id);
  }

  // マーカーの色だけを変更する(部位名の変更で同名グループの色が変わったとき、
  // ジオメトリを作り直さずに反映するため)。
  setSeedMarkerColor(id, color) {
    const marker = this._seedMarkers.get(id);
    if (!marker) return;
    marker.material.color.set(color);
  }

  clearSeedMarkers() {
    for (const id of Array.from(this._seedMarkers.keys())) {
      this.removeSeedMarker(id);
    }
  }
}
