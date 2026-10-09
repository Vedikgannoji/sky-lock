import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { GLTFSpecGlossExtension } from '../GLTFSpecGlossExtension.js';
import { OrbitState, setupOrbitLines, createOrbitLine } from '../orbit.js';

// ============================================================
// CONSTANTS & CONFIGURATION
// ============================================================

const EARTH_RADIUS = 10.0;
const EARTH_ROTATION_SPEED = 0.08; // rad/s
const SATELLITE_SCALE = 2.0;

// Math helpers
const _position = new THREE.Vector3();
const _targetQuat = new THREE.Quaternion();
const _s1BeaconPos = new THREE.Vector3();
const _s2BeaconPos = new THREE.Vector3();

// State variables
let isPaused = false;
let simulationSpeed = 1.0;
let isReady = false;
let hasLineOfSight = false;
let focusTarget = null; // null for Earth, or satellite Object3D
let externalClockMode = true; // external clock mode: setTime(t) drives orbits; internal clock never advances

// Visualization toggles
let showOrbitLines = true;
let showTrackingBeam = true;

// Three.js Core Objects
let scene, mainCamera, renderer, controls;
let earthMesh = null;
let sat1Obj = null;
let sat2Obj = null;
let beacon1Mesh = null;
let beacon2Mesh = null;
let orbit1 = null;
let orbit2 = null;
let orbitLines = [];

// Optical Tracking Beam Line
let trackingBeam = null;
let trackingBeamGeo = null;

// ============================================================
// SCENE SETUP
// ============================================================

function initScene() {
  const container = document.getElementById('webgl-container');
  const width = window.innerWidth;
  const height = window.innerHeight;

  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x020408);

  // Background stars / particle field
  createStarfield();

  // Main Camera
  mainCamera = new THREE.PerspectiveCamera(55, width / height, 0.1, 10000);
  mainCamera.position.set(34, 22, 40);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = false;
  container.appendChild(renderer.domElement);

  // OrbitControls
  controls = new OrbitControls(mainCamera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.minDistance = 12;
  controls.maxDistance = 250;
  controls.target.set(0, 0, 0);

  // Lighting
  const sunLight = new THREE.DirectionalLight(0xffffff, 3.2);
  sunLight.position.set(30, 15, 20);
  scene.add(sunLight);

  const fillLight = new THREE.DirectionalLight(0x60a5fa, 0.6);
  fillLight.position.set(-20, -10, -20);
  scene.add(fillLight);

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.35);
  scene.add(ambientLight);

  // Initial Orbits: S-1 (radius 20, inc 25°, phase 0°), S-2 (radius 26, inc 65°, phase 45°)
  orbit1 = new OrbitState(20, 0.3, 25, 0);
  orbit2 = new OrbitState(26, 0.2, 65, 45);
  orbitLines = setupOrbitLines(scene);

  // Optical Tracking Beam Line
  setupTrackingBeam();

  // Resize handler
  window.addEventListener('resize', onWindowResize);
}

function createStarfield() {
  const starCount = 1200;
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(starCount * 3);
  const colors = new Float32Array(starCount * 3);

  for (let i = 0; i < starCount; i++) {
    const r = 250 + Math.random() * 200;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);

    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    positions[i * 3 + 2] = r * Math.cos(phi);

    const brightness = 0.5 + Math.random() * 0.5;
    colors[i * 3] = brightness;
    colors[i * 3 + 1] = brightness * (0.9 + Math.random() * 0.1);
    colors[i * 3 + 2] = brightness * (0.95 + Math.random() * 0.05);
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 1.5,
    vertexColors: true,
    transparent: true,
    opacity: 0.85
  });

  const starfield = new THREE.Points(geometry, material);
  scene.add(starfield);
}

function onWindowResize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  mainCamera.aspect = width / height;
  mainCamera.updateProjectionMatrix();
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(width, height);
}

// ============================================================
// ASSET LOADING
// ============================================================

function normalizeModel(model, targetSize, name) {
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const scale = targetSize / maxDim;
  model.scale.set(scale, scale, scale);

  const center = box.getCenter(new THREE.Vector3());
  model.position.sub(center.multiplyScalar(scale));

  const wrapper = new THREE.Group();
  wrapper.name = name;
  wrapper.add(model);
  return wrapper;
}

async function loadGlbAssets() {
  const loader = new GLTFLoader();
  loader.register((parser) => new GLTFSpecGlossExtension(parser));

  const cleanBase = window.location.pathname.substring(0, window.location.pathname.lastIndexOf('/') + 1);

  const loadOne = (url, name) => {
    return new Promise((resolve) => {
      loader.load(
        url,
        (gltf) => {
          console.log(`Loaded ${name}`);
          resolve(gltf);
        },
        undefined,
        (err) => {
          console.warn(`Could not load ${url}, falling back to procedural mesh:`, err);
          resolve(null);
        }
      );
    });
  };

  // Fix 1: Load 'satellite.glb' for both S-1 and S-2 (symmetric models)
  const [earthGltf, sat1Gltf, sat2Gltf] = await Promise.all([
    loadOne(`${cleanBase}assets/earth.glb`, 'Earth'),
    loadOne(`${cleanBase}assets/satellite.glb`, 'S-1'),
    loadOne(`${cleanBase}assets/satellite.glb`, 'S-2')
  ]);

  // 1. Earth (ensure maximum sharpness and anisotropic texture filtering)
  if (earthGltf && earthGltf.scene) {
    const maxAnisotropy = renderer ? renderer.capabilities.getMaxAnisotropy() : 1;
    earthGltf.scene.traverse((child) => {
      if (child.isMesh && child.material) {
        const mat = child.material;
        if (mat.map) {
          mat.map.anisotropy = maxAnisotropy;
          mat.map.minFilter = THREE.LinearMipmapLinearFilter;
          mat.map.magFilter = THREE.LinearFilter;
          mat.map.generateMipmaps = true;
          mat.map.needsUpdate = true;
        }
      }
    });
    earthMesh = normalizeModel(earthGltf.scene, EARTH_RADIUS * 2, 'Earth');
  } else {
    const geo = new THREE.SphereGeometry(EARTH_RADIUS, 64, 64);
    const mat = new THREE.MeshStandardMaterial({
      color: 0x1d4ed8,
      roughness: 0.7,
      metalness: 0.1
    });
    earthMesh = new THREE.Mesh(geo, mat);
    earthMesh.name = 'Earth';
  }
  earthMesh.position.set(0, 0, 0);
  scene.add(earthMesh);

  // 2. Satellite 1 (S-1)
  if (sat1Gltf && sat1Gltf.scene) {
    sat1Obj = normalizeModel(sat1Gltf.scene, SATELLITE_SCALE * 2, 'S-1');
  } else {
    sat1Obj = createFallbackSatellite(0x38bdf8, 'S-1');
  }
  scene.add(sat1Obj);

  // 3. Satellite 2 (S-2)
  if (sat2Gltf && sat2Gltf.scene) {
    sat2Obj = normalizeModel(sat2Gltf.scene, SATELLITE_SCALE * 2, 'S-2');
  } else {
    sat2Obj = createFallbackSatellite(0xf59e0b, 'S-2');
  }
  scene.add(sat2Obj);

  // Fix 2: Attach identical optical beacons to both S-1 and S-2
  beacon1Mesh = attachBeacon(sat1Obj, 'S-1 Beacon');
  beacon2Mesh = attachBeacon(sat2Obj, 'S-2 Beacon');

  // Immediately initialize satellites at their orbital coordinates
  if (orbit1 && sat1Obj) {
    orbit1.getPosition(_position);
    sat1Obj.position.copy(_position);
    orbit1.getOrientation(_targetQuat);
    sat1Obj.quaternion.copy(_targetQuat);
    sat1Obj.updateMatrixWorld(true);
  }
  if (orbit2 && sat2Obj) {
    orbit2.getPosition(_position);
    sat2Obj.position.copy(_position);
    orbit2.getOrientation(_targetQuat);
    sat2Obj.quaternion.copy(_targetQuat);
    sat2Obj.updateMatrixWorld(true);
  }

  // Update tracking state and beam immediately upon load
  updateTrackingState();

  // Hide loading overlay
  const overlay = document.getElementById('loading-overlay');
  if (overlay) {
    overlay.style.opacity = '0';
    setTimeout(() => overlay.remove(), 400);
  }

  isReady = true;
  console.log('✅ SkyLock 3D Space Scene initialized successfully');
}

function createFallbackSatellite(colorHex, name) {
  const group = new THREE.Group();
  group.name = name;

  // Main bus
  const busGeo = new THREE.BoxGeometry(1.2, 0.8, 0.8);
  const busMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.8, roughness: 0.3 });
  const busMesh = new THREE.Mesh(busGeo, busMat);
  group.add(busMesh);

  // Solar panels
  const panelGeo = new THREE.BoxGeometry(2.4, 0.04, 0.7);
  const panelMat = new THREE.MeshStandardMaterial({ color: colorHex, metalness: 0.9, roughness: 0.2 });
  const panelL = new THREE.Mesh(panelGeo, panelMat);
  panelL.position.set(1.8, 0, 0);
  group.add(panelL);

  const panelR = new THREE.Mesh(panelGeo, panelMat);
  panelR.position.set(-1.8, 0, 0);
  group.add(panelR);

  return group;
}

function attachBeacon(satObj, name) {
  const beaconGroup = new THREE.Group();
  beaconGroup.name = name || 'OpticalBeacon';

  const beaconLight = new THREE.PointLight(0x38bdf8, 1.2, 25, 2);
  beaconLight.position.set(0, 0.5, 0);
  beaconGroup.add(beaconLight);

  const beaconGeo = new THREE.SphereGeometry(0.12, 16, 16);
  const beaconMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
  const beaconMesh = new THREE.Mesh(beaconGeo, beaconMat);
  beaconMesh.position.set(0, 0.5, 0);
  beaconGroup.add(beaconMesh);

  satObj.add(beaconGroup);
  return beaconMesh;
}

// ============================================================
// OPTICAL TRACKING BEAM
// ============================================================

function setupTrackingBeam() {
  const beamMat = new THREE.LineBasicMaterial({
    color: 0x00ffff,
    transparent: true,
    opacity: 0.95,
    linewidth: 2.0
  });
  const positions = new Float32Array(6);
  trackingBeamGeo = new THREE.BufferGeometry();
  trackingBeamGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  trackingBeam = new THREE.Line(trackingBeamGeo, beamMat);
  trackingBeam.name = 'OpticalTrackingBeam';
  trackingBeam.visible = false;
  trackingBeam.frustumCulled = false;
  scene.add(trackingBeam);
}

// ============================================================
// GEOMETRIC OCCLUSION & TRACKING LOGIC
// ============================================================

function checkLineOfSight(p1, p2, radius = EARTH_RADIUS) {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const dz = p2.z - p1.z;
  const segLenSq = dx * dx + dy * dy + dz * dz;
  if (segLenSq < 1e-10) return p1.length() >= radius;

  // Closest point on segment to origin (0, 0, 0)
  const t = -(p1.x * dx + p1.y * dy + p1.z * dz) / segLenSq;
  const tClamped = Math.max(0, Math.min(1, t));

  const cx = p1.x + tClamped * dx;
  const cy = p1.y + tClamped * dy;
  const cz = p1.z + tClamped * dz;

  return (cx * cx + cy * cy + cz * cz) >= (radius * radius);
}

function updateTrackingState() {
  if (!beacon1Mesh || !beacon2Mesh) return;

  beacon1Mesh.getWorldPosition(_s1BeaconPos);
  beacon2Mesh.getWorldPosition(_s2BeaconPos);

  // Geometric line-of-sight test directly between S-1 and S-2 beacons
  hasLineOfSight = checkLineOfSight(_s1BeaconPos, _s2BeaconPos, EARTH_RADIUS);

  // Tracking beam is ON only when:
  // 1. Direct line-of-sight between beacons is NOT blocked by Earth
  // 2. User has enabled tracking beam
  if (trackingBeam && trackingBeamGeo) {
    if (hasLineOfSight && showTrackingBeam) {
      const posAttr = trackingBeamGeo.attributes.position;
      const arr = posAttr.array;
      arr[0] = _s1BeaconPos.x;
      arr[1] = _s1BeaconPos.y;
      arr[2] = _s1BeaconPos.z;
      arr[3] = _s2BeaconPos.x;
      arr[4] = _s2BeaconPos.y;
      arr[5] = _s2BeaconPos.z;
      posAttr.needsUpdate = true;
      trackingBeamGeo.computeBoundingSphere();
      trackingBeam.visible = true;
    } else {
      trackingBeam.visible = false;
    }
  }
}

// ============================================================
// ANIMATION & RENDER LOOP
// ============================================================

let lastTime = performance.now();
let frameCount = 0;
let lastFpsTime = performance.now();
let currentFps = 60;

function animate(now) {
  requestAnimationFrame(animate);

  const dt = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;

  // FPS calculation
  frameCount++;
  if (now - lastFpsTime >= 500) {
    currentFps = Math.round((frameCount * 1000) / (now - lastFpsTime));
    frameCount = 0;
    lastFpsTime = now;
  }

  if (!externalClockMode) {
    const effectiveDt = isPaused ? 0 : dt * simulationSpeed;

    // 1. Earth rotation
    if (earthMesh && effectiveDt > 0) {
      earthMesh.rotation.y += EARTH_ROTATION_SPEED * effectiveDt;
    }

    // 2. Update Satellites along Orbits
    if (sat1Obj && orbit1 && effectiveDt > 0) {
      orbit1.update(effectiveDt);
      orbit1.getPosition(_position);
      sat1Obj.position.copy(_position);
      orbit1.getOrientation(_targetQuat);
      sat1Obj.quaternion.copy(_targetQuat);
    }

    if (sat2Obj && orbit2 && effectiveDt > 0) {
      orbit2.update(effectiveDt);
      orbit2.getPosition(_position);
      sat2Obj.position.copy(_position);
      orbit2.getOrientation(_targetQuat);
      sat2Obj.quaternion.copy(_targetQuat);
    }

    if (sat1Obj) sat1Obj.updateMatrixWorld(true);
    if (sat2Obj) sat2Obj.updateMatrixWorld(true);

    // 3. Continuous per-frame LOS check & laser beam update
    updateTrackingState();
  }

  // 4. Update OrbitControls & Focus
  if (focusTarget) {
    controls.target.lerp(focusTarget.position, 0.08);
  }
  controls.update();

  // 5. Render Main 3D View
  renderer.render(scene, mainCamera);
}

// ============================================================
// PYTHON ↔ 3D JAVASCRIPT BRIDGE
// ============================================================

window.skylock3d = {
  isReady: () => isReady,

  setGimbalPose: () => {},
  setCameraFov: () => {},
  resetCamera: () => {},
  setShowCameraFov: () => {},
  setShowOpticalAxis: () => {},

  setShowOrbitLines: (show) => {
    showOrbitLines = Boolean(show);
    orbitLines.forEach((l) => { if (l) l.visible = showOrbitLines; });
  },

  setShowTrackingBeam: (show) => {
    showTrackingBeam = Boolean(show);
    updateTrackingState();
  },

  setSatelliteOrbit: (satId, radius, incDeg, speed, phaseDeg) => {
    const r = Math.max(12, Math.min(60, Number(radius)));
    const inc = Number(incDeg);
    const spd = Number(speed);
    const phase = Number(phaseDeg);

    if (satId === 's1' || satId === 1) {
      orbit1 = new OrbitState(r, spd, inc, phase);
      if (orbitLines[0]) {
        scene.remove(orbitLines[0]);
        orbitLines[0].geometry.dispose();
        orbitLines[0].material.dispose();
      }
      orbitLines[0] = createOrbitLine(r, inc, 0x6699cc);
      orbitLines[0].visible = showOrbitLines;
      scene.add(orbitLines[0]);
    } else if (satId === 's2' || satId === 2) {
      orbit2 = new OrbitState(r, spd, inc, phase);
      if (orbitLines[1]) {
        scene.remove(orbitLines[1]);
        orbitLines[1].geometry.dispose();
        orbitLines[1].material.dispose();
      }
      orbitLines[1] = createOrbitLine(r, inc, 0xcc7766);
      orbitLines[1].visible = showOrbitLines;
      scene.add(orbitLines[1]);
    }
  },

  resetView: () => {
    focusTarget = null;
    if (controls) controls.target.set(0, 0, 0);
    if (mainCamera) mainCamera.position.set(34, 22, 40);
  },

  focusSatellite: (satId) => {
    if (satId === 's1' || satId === 1) {
      if (sat1Obj) focusTarget = sat1Obj;
    } else if (satId === 's2' || satId === 2) {
      if (sat2Obj) focusTarget = sat2Obj;
    } else {
      focusTarget = null;
      if (controls) controls.target.set(0, 0, 0);
    }
  },

  updateState: (state) => {
    if (!state) return;
    if (state.showOrbitLines !== undefined) {
      showOrbitLines = Boolean(state.showOrbitLines);
      orbitLines.forEach((l) => { if (l) l.visible = showOrbitLines; });
    }
    if (state.showTrackingBeam !== undefined) {
      showTrackingBeam = Boolean(state.showTrackingBeam);
      updateTrackingState();
    }
    if (state.paused !== undefined) {
      isPaused = Boolean(state.paused);
    }
    if (state.speed !== undefined) {
      simulationSpeed = Number(state.speed);
    }
  },

  getFps: () => currentFps,

  setPaused: (paused) => {
    isPaused = Boolean(paused);
  },

  setExternalClock: (enabled) => {
    externalClockMode = Boolean(enabled);
  },

  setTime: (timeSec) => {
    const t = Number(timeSec);
    if (orbit1 && sat1Obj) {
      orbit1.setTime(t);
      orbit1.getPosition(_position);
      sat1Obj.position.copy(_position);
      orbit1.getOrientation(_targetQuat);
      sat1Obj.quaternion.copy(_targetQuat);
    }
    if (orbit2 && sat2Obj) {
      orbit2.setTime(t);
      orbit2.getPosition(_position);
      sat2Obj.position.copy(_position);
      orbit2.getOrientation(_targetQuat);
      sat2Obj.quaternion.copy(_targetQuat);
    }
    if (earthMesh) {
      earthMesh.rotation.y = EARTH_ROTATION_SPEED * t;
    }
    if (sat1Obj) sat1Obj.updateMatrixWorld(true);
    if (sat2Obj) sat2Obj.updateMatrixWorld(true);
    updateTrackingState();
  },

  setSimulationSpeed: (speed) => {
    simulationSpeed = Number(speed);
  },

  getState: () => {
    const s1Pos = sat1Obj ? { x: sat1Obj.position.x, y: sat1Obj.position.y, z: sat1Obj.position.z } : null;
    const s2Pos = sat2Obj ? { x: sat2Obj.position.x, y: sat2Obj.position.y, z: sat2Obj.position.z } : null;
    return {
      ready: isReady,
      lineOfSight: hasLineOfSight,
      beamActive: hasLineOfSight && showTrackingBeam,
      fps: currentFps,
      paused: isPaused,
      externalClock: externalClockMode,
      speed: simulationSpeed,
      satellite1: s1Pos,
      satellite2: s2Pos
    };
  }
};

// ============================================================
// BOOTSTRAP
// ============================================================

initScene();
loadGlbAssets().catch((err) => console.error('Asset load error:', err));
requestAnimationFrame(animate);
