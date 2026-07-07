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

    window.addEventListener("resize", () => this._onResize());
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
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
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

    // Three.jsのY-up座標系に合わせ、生成メッシュ(Z-up, 床=z0)をX軸-90度回転して
    // Y軸を高さ方向にする。
    const wrapper = new THREE.Group();
    wrapper.add(object3d);
    wrapper.rotation.x = -Math.PI / 2;

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
}
