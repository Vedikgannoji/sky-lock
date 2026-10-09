import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { GLTFSpecGlossExtension } from '../GLTFSpecGlossExtension.js';
import { OrbitState } from '../orbit.js';

// ============================================================
// CONSTANTS & CONFIGURATION
// ============================================================

const EARTH_RADIUS = 10.0;
const EARTH_ROTATION_SPEED = 0.08; // rad/s
const SATELLITE_SCALE = 0.3; // max dimension normalized to 0.6 => ~8.5% of frame at dist 25, FOV 16°

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
let mountSatId = 's1'; // 's1' or 's2'
let externalClockMode = true; // external clock mode: setTime(t) drives orbits; internal clock never advances

// Gimbal & FOV state (default 16° horizontal x 12° vertical at 4:3 aspect)
let currentFovHDeg = 16.0;
let currentFovVDeg = 12.0;
let currentTime = 0.0;

// Commanded Camera Aim (Forward and Up unit vectors in world coordinates)
const aimForward = new THREE.Vector3(1, 0, 0);
const aimUp = new THREE.Vector3(0, 1, 0);

// Viewport layout (4:3 letterboxed)
let viewportX = 0;
let viewportY = 0;
let viewportWidth = 0;
let viewportHeight = 0;

// Three.js Core Objects
let scene, camera, renderer;
let earthMesh = null;
let sat1Obj = null;
let sat2Obj = null;
let beacon1Sprite = null;
let beacon2Sprite = null;
let orbit1 = null;
let orbit2 = null;

// Optical Tracking Beam Line
let trackingBeam = null;
let trackingBeamGeo = null;

// ============================================================
// TEXTURE GENERATION
// ============================================================

function createBeaconTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');

  // Multi-stop radial gradient for rich additive optical glow
  const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0.0, 'rgba(255, 255, 255, 1.0)');
  grad.addColorStop(0.15, 'rgba(125, 211, 252, 0.95)');
  grad.addColorStop(0.35, 'rgba(56, 189, 248, 0.6)');
  grad.addColorStop(0.65, 'rgba(14, 165, 233, 0.2)');
  grad.addColorStop(1.0, 'rgba(0, 0, 0, 0.0)');

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 64, 64);

  const texture = new THREE.CanvasTexture(canvas);
  texture.generateMipmaps = false;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  return texture;
}

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

  // Background stars
  createStarfield();

  // Perspective camera (vertical FOV derived from 16° horizontal at 4:3 aspect)
  computeVerticalFov();
  camera = new THREE.PerspectiveCamera(currentFovVDeg, 4 / 3, 0.1, 2000);
  scene.add(camera);

  // Renderer matching 3D Space tab settings
  renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: 'high-performance',
    preserveDrawingBuffer: true
  });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = false;
  renderer.autoClear = false;
  container.appendChild(renderer.domElement);

  // Initial letterboxed viewport calculation
  updateViewport();

  // Lighting (identical to 3D Space tab)
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

  // Optical Tracking Beam Line
  setupTrackingBeam();

  // Resize handler
  window.addEventListener('resize', onWindowResize);
}

function computeVerticalFov() {
  // THREE camera.fov is vertical: derive it from horizontal FOV at aspect 4:3
  // tan(fov_v / 2) = (3 / 4) * tan(fov_h / 2)
  const fovHRad = THREE.MathUtils.degToRad(currentFovHDeg);
  const fovVRad = 2.0 * Math.atan(0.75 * Math.tan(fovHRad / 2.0));
  currentFovVDeg = THREE.MathUtils.radToDeg(fovVRad);
}

function updateViewport() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  const targetAspect = 4 / 3;

  if (width / height > targetAspect) {
    viewportHeight = height;
    viewportWidth = Math.round(height * targetAspect);
    viewportX = Math.round((width - viewportWidth) / 2);
    viewportY = 0;
  } else {
    viewportWidth = width;
    viewportHeight = Math.round(width / targetAspect);
    viewportX = 0;
    viewportY = Math.round((height - viewportHeight) / 2);
  }

  if (renderer) {
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
  }

  computeVerticalFov();
  if (camera) {
    camera.fov = currentFovVDeg;
    camera.aspect = targetAspect;
    camera.updateProjectionMatrix();
  }
}

function onWindowResize() {
  updateViewport();
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

  const [earthGltf, sat1Gltf, sat2Gltf] = await Promise.all([
    loadOne(`${cleanBase}assets/earth.glb`, 'Earth'),
    loadOne(`${cleanBase}assets/satellite.glb`, 'S-1'),
    loadOne(`${cleanBase}assets/satellite.glb`, 'S-2')
  ]);

  const maxAnisotropy = renderer ? renderer.capabilities.getMaxAnisotropy() : 1;

  // 1. Earth
  if (earthGltf && earthGltf.scene) {
    earthGltf.scene.traverse((child) => {
      if (child.isMesh && child.material) {
        child.renderOrder = 0;
        const mat = child.material;
        mat.depthTest = true;
        mat.depthWrite = true;
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
      metalness: 0.1,
      depthTest: true,
      depthWrite: true
    });
    earthMesh = new THREE.Mesh(geo, mat);
    earthMesh.name = 'Earth';
    earthMesh.renderOrder = 0;
  }
  earthMesh.position.set(0, 0, 0);
  scene.add(earthMesh);

  // 2. Satellite 1 (S-1)
  if (sat1Gltf && sat1Gltf.scene) {
    sat1Gltf.scene.traverse((child) => {
      if (child.isMesh && child.material) {
        child.renderOrder = 0;
        child.material.depthTest = true;
        child.material.depthWrite = true;
      }
    });
    sat1Obj = normalizeModel(sat1Gltf.scene, SATELLITE_SCALE * 2, 'S-1');
  } else {
    sat1Obj = createFallbackSatellite(0x38bdf8, 'S-1');
  }
  scene.add(sat1Obj);

  // 3. Satellite 2 (S-2)
  if (sat2Gltf && sat2Gltf.scene) {
    sat2Gltf.scene.traverse((child) => {
      if (child.isMesh && child.material) {
        child.renderOrder = 0;
        child.material.depthTest = true;
        child.material.depthWrite = true;
      }
    });
    sat2Obj = normalizeModel(sat2Gltf.scene, SATELLITE_SCALE * 2, 'S-2');
  } else {
    sat2Obj = createFallbackSatellite(0xf59e0b, 'S-2');
  }
  scene.add(sat2Obj);

  // Attach optical beacon sprites (additive, depth-tested so Earth occludes them)
  const beaconTexture = createBeaconTexture();
  beacon1Sprite = attachBeacon(sat1Obj, 'S-1 Beacon', beaconTexture);
  beacon2Sprite = attachBeacon(sat2Obj, 'S-2 Beacon', beaconTexture);

  // Immediately initialize satellites at orbital coordinates
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

  // Update visibility according to initial mount
  updateMountVisibility();
  updateCameraTransform();

  // Hide loading overlay
  const overlay = document.getElementById('loading-overlay');
  if (overlay) {
    overlay.style.opacity = '0';
    setTimeout(() => overlay.remove(), 400);
  }

  isReady = true;
  console.log('✅ SkyLock Gimbal POV View initialized successfully');
}

function createFallbackSatellite(colorHex, name) {
  const group = new THREE.Group();
  group.name = name;

  const busGeo = new THREE.BoxGeometry(1.2, 0.8, 0.8);
  const busMat = new THREE.MeshStandardMaterial({
    color: 0x94a3b8,
    metalness: 0.8,
    roughness: 0.3,
    depthTest: true,
    depthWrite: true
  });
  const busMesh = new THREE.Mesh(busGeo, busMat);
  busMesh.renderOrder = 0;
  group.add(busMesh);

  const panelGeo = new THREE.BoxGeometry(2.4, 0.04, 0.7);
  const panelMat = new THREE.MeshStandardMaterial({
    color: colorHex,
    metalness: 0.9,
    roughness: 0.2,
    depthTest: true,
    depthWrite: true
  });
  const panelL = new THREE.Mesh(panelGeo, panelMat);
  panelL.position.set(1.8, 0, 0);
  panelL.renderOrder = 0;
  group.add(panelL);

  const panelR = new THREE.Mesh(panelGeo, panelMat);
  panelR.position.set(-1.8, 0, 0);
  panelR.renderOrder = 0;
  group.add(panelR);

  return group;
}

function attachBeacon(satObj, name, texture) {
  const beaconGroup = new THREE.Group();
  beaconGroup.name = name || 'OpticalBeacon';

  const beaconLight = new THREE.PointLight(0x38bdf8, 2.5, 40, 1.5);
  beaconLight.position.set(0, 0, 0);
  beaconGroup.add(beaconLight);

  // Additive glow sprite with depth testing against Earth and satellites
  const spriteMat = new THREE.SpriteMaterial({
    map: texture,
    color: 0x38bdf8,
    blending: THREE.AdditiveBlending,
    depthTest: true,
    depthWrite: false,
    transparent: true
  });
  const sprite = new THREE.Sprite(spriteMat);
  sprite.renderOrder = 10;
  sprite.position.set(0, 0, 0);
  sprite.scale.set(0.2, 0.2, 1.0);
  beaconGroup.add(sprite);

  satObj.add(beaconGroup);
  return sprite;
}

function setupTrackingBeam() {
  const beamMat = new THREE.LineBasicMaterial({
    color: 0x00ffff,
    transparent: true,
    opacity: 0.9,
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

function checkLineOfSight(p1, p2, radius = EARTH_RADIUS) {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const dz = p2.z - p1.z;
  const segLenSq = dx * dx + dy * dy + dz * dz;
  if (segLenSq < 1e-10) return p1.length() >= radius;

  const t = -(p1.x * dx + p1.y * dy + p1.z * dz) / segLenSq;
  const tClamped = Math.max(0, Math.min(1, t));

  const cx = p1.x + tClamped * dx;
  const cy = p1.y + tClamped * dy;
  const cz = p1.z + tClamped * dz;

  return (cx * cx + cy * cy + cz * cz) >= (radius * radius);
}

function updateMountVisibility() {
  // Mount satellite must not appear in its own view
  const isS1Mount = mountSatId === 's1';
  if (sat1Obj) sat1Obj.visible = !isS1Mount;
  if (sat2Obj) sat2Obj.visible = isS1Mount;
}

// ============================================================
// CAMERA POSITIONING & ORIENTATION (setAim)
// ============================================================

function setAimVectors(fx, fy, fz, ux, uy, uz) {
  aimForward.set(Number(fx), Number(fy), Number(fz));
  if (aimForward.lengthSq() < 1e-8) {
    aimForward.set(1, 0, 0);
  } else {
    aimForward.normalize();
  }

  aimUp.set(Number(ux), Number(uy), Number(uz));
  if (aimUp.lengthSq() < 1e-8) {
    aimUp.set(0, 1, 0);
  } else {
    aimUp.normalize();
  }

  updateCameraTransform();
}

function updateCameraTransform() {
  const mountObj = (mountSatId === 's1') ? sat1Obj : sat2Obj;
  if (!mountObj || !camera) return;

  // camera.position = mount satellite position + 0.45 x forward
  camera.position.copy(mountObj.position).addScaledVector(aimForward, 0.45);

  // Exact orientation given by (forward, up):
  // In Three.js camera frame:
  // - looking down camera local -Z axis: camZ = -forward
  // - local +Y axis: orthoUp
  // - local +X axis: right = forward x up
  const right = new THREE.Vector3().crossVectors(aimForward, aimUp);
  if (right.lengthSq() < 1e-8) {
    right.set(0, 0, 1);
  } else {
    right.normalize();
  }

  const orthoUp = new THREE.Vector3().crossVectors(right, aimForward).normalize();
  const camZ = aimForward.clone().negate();

  const rotMat = new THREE.Matrix4().makeBasis(right, orthoUp, camZ);
  camera.quaternion.setFromRotationMatrix(rotMat);
  camera.updateMatrixWorld(true);
}

function updateBeaconSizes() {
  // Ensure beacon glow sprite has at least ~6 px on-screen size at any distance
  if (!camera) return;
  const targetSprite = (mountSatId === 's1') ? beacon2Sprite : beacon1Sprite;
  const targetObj = (mountSatId === 's1') ? sat2Obj : sat1Obj;

  if (targetSprite && targetObj && camera) {
    const dist = camera.position.distanceTo(targetObj.position);
    const vh = Math.max(1, viewportHeight);
    const fovVRad = THREE.MathUtils.degToRad(camera.fov);
    // On-screen pixels P = (worldSize / (2 * dist * tan(fov_v / 2))) * vh
    // To ensure P >= 7.0 px (at least ~6 px at any distance):
    const minWorldSize = (7.5 / vh) * (2.0 * dist * Math.tan(fovVRad / 2.0));
    const worldSize = Math.max(0.12, minWorldSize);
    targetSprite.scale.set(worldSize, worldSize, 1.0);
  }
}

// ============================================================
// ANIMATION & RENDER LOOP
// ============================================================

let lastTime = performance.now();

function animate(now) {
  requestAnimationFrame(animate);

  const dt = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;

  // External-clock mode: internal clock never advances.
  // setTime(t) drives orbits and Earth rotation.
  if (!externalClockMode) {
    const effectiveDt = isPaused ? 0 : dt * simulationSpeed;
    if (effectiveDt > 0) {
      currentTime += effectiveDt;
      if (earthMesh) {
        earthMesh.rotation.y += EARTH_ROTATION_SPEED * effectiveDt;
      }
      if (sat1Obj && orbit1) {
        orbit1.update(effectiveDt);
        orbit1.getPosition(_position);
        sat1Obj.position.copy(_position);
        orbit1.getOrientation(_targetQuat);
        sat1Obj.quaternion.copy(_targetQuat);
      }
      if (sat2Obj && orbit2) {
        orbit2.update(effectiveDt);
        orbit2.getPosition(_position);
        sat2Obj.position.copy(_position);
        orbit2.getOrientation(_targetQuat);
        sat2Obj.quaternion.copy(_targetQuat);
      }
    }
  }

  if (sat1Obj) sat1Obj.updateMatrixWorld(true);
  if (sat2Obj) sat2Obj.updateMatrixWorld(true);

  // Update camera mount position and orientation
  updateCameraTransform();

  // Dynamic beacon sprite size scaling (>= 6 px at any distance)
  updateBeaconSizes();

  // Update Optical Tracking Beam and Line-of-Sight
  if (sat1Obj && sat2Obj) {
    sat1Obj.getWorldPosition(_s1BeaconPos);
    sat2Obj.getWorldPosition(_s2BeaconPos);
    hasLineOfSight = checkLineOfSight(_s1BeaconPos, _s2BeaconPos, EARTH_RADIUS);

    if (trackingBeam && trackingBeamGeo) {
      if (hasLineOfSight) {
        const arr = trackingBeamGeo.attributes.position.array;
        arr[0] = _s1BeaconPos.x;
        arr[1] = _s1BeaconPos.y;
        arr[2] = _s1BeaconPos.z;
        arr[3] = _s2BeaconPos.x;
        arr[4] = _s2BeaconPos.y;
        arr[5] = _s2BeaconPos.z;
        trackingBeamGeo.attributes.position.needsUpdate = true;
        trackingBeamGeo.computeBoundingSphere();
        trackingBeam.visible = true;
      } else {
        trackingBeam.visible = false;
      }
    }
  }

  // Render 4:3 letterboxed view:
  // Clear full window canvas, then scissor test to 4:3 viewport
  renderer.setScissorTest(false);
  renderer.setClearColor(0x020408, 1.0);
  renderer.clear();

  renderer.setScissorTest(true);
  renderer.setViewport(viewportX, viewportY, viewportWidth, viewportHeight);
  renderer.setScissor(viewportX, viewportY, viewportWidth, viewportHeight);
  renderer.render(scene, camera);
}

// ============================================================
// PYTHON <-> GIMBAL CAM JAVASCRIPT BRIDGE
// ============================================================

window.gimbalcam = {
  isReady: () => isReady,

  setAim: (fx, fy, fz, ux, uy, uz) => {
    setAimVectors(fx, fy, fz, ux, uy, uz);
  },

  setExternalClock: (enabled) => {
    externalClockMode = Boolean(enabled);
  },

  setFov: (fovHDeg) => {
    const fov = Number(fovHDeg);
    if (fov > 0 && fov < 180) {
      currentFovHDeg = fov;
      computeVerticalFov();
      if (camera) {
        camera.fov = currentFovVDeg;
        camera.aspect = 4 / 3;
        camera.updateProjectionMatrix();
      }
    }
  },

  setMount: (satId) => {
    const id = String(satId).toLowerCase().replace('-', '');
    mountSatId = (id === 's2' || id === 'sat2' || id === '2') ? 's2' : 's1';
    updateMountVisibility();
    updateCameraTransform();
  },

  setPaused: (paused) => {
    isPaused = Boolean(paused);
  },

  setSimulationSpeed: (speed) => {
    simulationSpeed = Number(speed);
  },

  setSatelliteOrbit: (satId, radius, incDeg, speed, phaseDeg) => {
    const r = Math.max(12, Math.min(60, Number(radius)));
    const inc = Number(incDeg);
    const spd = Number(speed);
    const phase = Number(phaseDeg);

    if (satId === 's1' || satId === 1) {
      orbit1 = new OrbitState(r, spd, inc, phase);
    } else if (satId === 's2' || satId === 2) {
      orbit2 = new OrbitState(r, spd, inc, phase);
    }
  },

  setTime: (timeSec) => {
    const t = Number(timeSec);
    currentTime = t;
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
    updateCameraTransform();
  },

  captureDataUrl: () => {
    if (!renderer || !renderer.domElement) return null;
    return renderer.domElement.toDataURL('image/png');
  },

  getState: () => {
    const s1Pos = sat1Obj ? { x: sat1Obj.position.x, y: sat1Obj.position.y, z: sat1Obj.position.z } : null;
    const s2Pos = sat2Obj ? { x: sat2Obj.position.x, y: sat2Obj.position.y, z: sat2Obj.position.z } : null;
    return {
      ready: isReady,
      mount: mountSatId,
      paused: isPaused,
      externalClock: externalClockMode,
      fovH: currentFovHDeg,
      fovV: currentFovVDeg,
      lineOfSight: hasLineOfSight,
      currentTime: currentTime,
      aimForward: { x: aimForward.x, y: aimForward.y, z: aimForward.z },
      aimUp: { x: aimUp.x, y: aimUp.y, z: aimUp.z },
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
