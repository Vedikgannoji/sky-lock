import * as THREE from 'three';
import { MIN_SATELLITE_DISTANCE, MAX_SATELLITE_DISTANCE, MAX_MANUAL_SATELLITES } from './orbit.js';

/**
 * UI Control Panel Module for Sky Lock Space Environment Simulation
 */

export const SIMULATION_SPEEDS = [1, 2, 4];
export let simulationSpeed = 1;
export let orbitLinesVisible = true;
export let isPaused = false;
export let islLinkEnabled = true;
export let satelliteMode = 'AUTOMATIC'; // 'AUTOMATIC' | 'MANUAL'

// Mini 3D preview variables
let previewRenderer = null;
let previewScene = null;
let previewCamera = null;
let previewModel = null;
let isPreviewInitialized = false;

/**
 * Initialize 3D mini-preview of the actual satellite GLB inside the control panel.
 */
export function initSatellitePreview(sourceSatelliteModel) {
  const previewCanvas = document.getElementById('satellite-preview-canvas');
  if (!previewCanvas || !sourceSatelliteModel) return;

  previewScene = new THREE.Scene();
  previewCamera = new THREE.PerspectiveCamera(38, 140 / 90, 0.1, 100);
  previewCamera.position.set(0, 0, 8.5);

  previewRenderer = new THREE.WebGLRenderer({
    canvas: previewCanvas,
    alpha: true,
    antialias: true
  });
  previewRenderer.setSize(140, 90);
  previewRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  // Studio lighting for mini preview
  const previewAmbient = new THREE.AmbientLight(0xffffff, 1.4);
  previewScene.add(previewAmbient);

  const previewDir = new THREE.DirectionalLight(0xffffff, 2.2);
  previewDir.position.set(4, 5, 5);
  previewScene.add(previewDir);

  const previewFill = new THREE.DirectionalLight(0x38bdf8, 1.0);
  previewFill.position.set(-4, -3, -3);
  previewScene.add(previewFill);

  // Clone preview model
  previewModel = sourceSatelliteModel.clone(true);
  previewModel.position.set(0, 0, 0);
  previewModel.scale.set(0.9, 0.9, 0.9);
  previewScene.add(previewModel);

  isPreviewInitialized = true;
}

/**
 * Animate the 3D preview gently inside the manual control panel.
 */
export function renderSatellitePreview() {
  if (!isPreviewInitialized || satelliteMode !== 'MANUAL' || !previewModel) return;
  previewModel.rotation.y += 0.015;
  previewRenderer.render(previewScene, previewCamera);
}

/**
 * Dynamically render the Live Satellite Status list with individual controls.
 */
export function renderSatelliteStatusList(
  satellites,
  { onSatelliteSpeedChange, onSatelliteTogglePause, onSatelliteRemove }
) {
  const listContainer = document.getElementById('satellites-list');
  if (!listContainer) return;

  listContainer.innerHTML = '';

  if (!satellites || satellites.length === 0) {
    const emptyMsg = document.createElement('div');
    emptyMsg.className = 'no-satellites-msg';
    emptyMsg.textContent = 'No active satellites';
    listContainer.appendChild(emptyMsg);
    return;
  }

  satellites.forEach((sat) => {
    const card = document.createElement('div');
    card.className = 'satellite-card';
    card.setAttribute('data-sat-id', sat.id);

    const isSatPaused = sat.paused;
    const statusText = isSatPaused ? 'PAUSED' : 'ACTIVE';
    const statusClass = isSatPaused ? 'paused' : 'active';
    const pauseBtnText = isSatPaused ? 'RESUME' : 'PAUSE';
    const pauseBtnClass = isSatPaused ? 'paused' : '';

    const isManual = sat.isManual !== undefined ? sat.isManual : (satelliteMode === 'MANUAL');

    card.innerHTML = `
      <div class="sat-card-header">
        <span class="sat-id-tag">${sat.id}</span>
        <span class="sat-status-badge ${statusClass}">
          <span class="status-dot">●</span> ${statusText}
        </span>
      </div>
      <div class="sat-control-row">
        <span class="sat-control-label">Speed</span>
        <div class="sat-speed-group">
          <button class="sat-speed-btn ${sat.individualSpeed === 0.5 ? 'active' : ''}" data-speed="0.5">0.5×</button>
          <button class="sat-speed-btn ${sat.individualSpeed === 1 ? 'active' : ''}" data-speed="1">1×</button>
          <button class="sat-speed-btn ${sat.individualSpeed === 2 ? 'active' : ''}" data-speed="2">2×</button>
        </div>
      </div>
      <div class="sat-actions-row">
        <button class="sat-action-btn pause-action ${pauseBtnClass}">${pauseBtnText}</button>
        ${isManual ? '<button class="sat-action-btn remove-action">REMOVE</button>' : ''}
      </div>
    `;

    // Speed buttons listener
    const speedBtns = card.querySelectorAll('.sat-speed-btn');
    speedBtns.forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const speed = parseFloat(btn.getAttribute('data-speed'));
        if (isNaN(speed)) return;
        sat.individualSpeed = speed;
        speedBtns.forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        if (typeof onSatelliteSpeedChange === 'function') {
          onSatelliteSpeedChange(sat, speed);
        }
      });
    });

    // Pause button listener
    const pauseBtn = card.querySelector('.pause-action');
    if (pauseBtn) {
      pauseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        sat.paused = !sat.paused;
        if (typeof onSatelliteTogglePause === 'function') {
          onSatelliteTogglePause(sat, sat.paused);
        }
      });
    }

    // Remove button listener (manual mode only)
    if (isManual) {
      const removeBtn = card.querySelector('.remove-action');
      if (removeBtn) {
        removeBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          if (typeof onSatelliteRemove === 'function') {
            onSatelliteRemove(sat);
          }
        });
      }
    }

    listContainer.appendChild(card);
  });
}

/**
 * Setup UI control panel, event handlers, hover tooltip, and drag-and-drop.
 */
export function setupUI({
  onSpeedChange,
  onToggleOrbitLines,
  onToggleISLLink,
  onTogglePause,
  onModeChange,
  onDropSatellite,
  getManualCount,
  getActiveSatellites,
  camera,
  scene,
  controls,
  ghostModelTemplate
}) {
  // 1. Global Speed Buttons
  const speedButtons = document.querySelectorAll('.speed-btn');
  speedButtons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const speed = parseFloat(btn.getAttribute('data-speed'));
      if (isNaN(speed)) return;

      simulationSpeed = speed;
      speedButtons.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');

      if (typeof onSpeedChange === 'function') {
        onSpeedChange(speed);
      }
    });
  });

  // 2. Orbit Lines Toggle
  const orbitToggleBtn = document.getElementById('orbit-lines-toggle');
  if (orbitToggleBtn) {
    orbitToggleBtn.addEventListener('click', () => {
      orbitLinesVisible = !orbitLinesVisible;

      if (orbitLinesVisible) {
        orbitToggleBtn.textContent = 'ON';
        orbitToggleBtn.classList.remove('off');
        orbitToggleBtn.classList.add('on');
      } else {
        orbitToggleBtn.textContent = 'OFF';
        orbitToggleBtn.classList.remove('on');
        orbitToggleBtn.classList.add('off');
      }

      if (typeof onToggleOrbitLines === 'function') {
        onToggleOrbitLines(orbitLinesVisible);
      }
    });
  }

  // 3. ISL Link Toggle
  const islToggleBtn = document.getElementById('isl-link-toggle');
  if (islToggleBtn) {
    islToggleBtn.addEventListener('click', () => {
      islLinkEnabled = !islLinkEnabled;

      if (islLinkEnabled) {
        islToggleBtn.textContent = 'ON';
        islToggleBtn.classList.remove('off');
        islToggleBtn.classList.add('on');
      } else {
        islToggleBtn.textContent = 'OFF';
        islToggleBtn.classList.remove('on');
        islToggleBtn.classList.add('off');
      }

      if (typeof onToggleISLLink === 'function') {
        onToggleISLLink(islLinkEnabled);
      }
    });
  }

  // 4. Global Pause / Resume Toggle
  const pauseToggleBtn = document.getElementById('pause-toggle');
  if (pauseToggleBtn) {
    pauseToggleBtn.addEventListener('click', () => {
      isPaused = !isPaused;

      if (isPaused) {
        pauseToggleBtn.textContent = 'RESUME';
        pauseToggleBtn.classList.add('paused');
      } else {
        pauseToggleBtn.textContent = 'PAUSE';
        pauseToggleBtn.classList.remove('paused');
      }

      if (typeof onTogglePause === 'function') {
        onTogglePause(isPaused);
      }
    });
  }

  // 4. Satellite Mode Selector
  const modeAutoBtn = document.getElementById('mode-auto');
  const modeManualBtn = document.getElementById('mode-manual');
  const manualSection = document.getElementById('manual-satellite-section');

  function setMode(mode) {
    satelliteMode = mode;

    if (mode === 'AUTOMATIC') {
      modeAutoBtn.classList.add('active');
      modeManualBtn.classList.remove('active');
      if (manualSection) manualSection.style.display = 'none';
    } else {
      modeManualBtn.classList.add('active');
      modeAutoBtn.classList.remove('active');
      if (manualSection) manualSection.style.display = 'block';
    }

    if (typeof onModeChange === 'function') {
      onModeChange(mode);
    }
  }

  if (modeAutoBtn) {
    modeAutoBtn.addEventListener('click', () => setMode('AUTOMATIC'));
  }
  if (modeManualBtn) {
    modeManualBtn.addEventListener('click', () => setMode('MANUAL'));
  }

  // 5. Hover Satellite -> Show ID Tooltip
  const tooltip = document.getElementById('satellite-tooltip');
  const hoverRaycaster = new THREE.Raycaster();
  const hoverMouse = new THREE.Vector2();

  window.addEventListener('pointermove', (e) => {
    if (isDragging) {
      if (tooltip) tooltip.style.display = 'none';
      return;
    }

    hoverMouse.x = (e.clientX / window.innerWidth) * 2 - 1;
    hoverMouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

    hoverRaycaster.setFromCamera(hoverMouse, camera);

    const activeList = typeof getActiveSatellites === 'function' ? getActiveSatellites() : [];
    const activeModels = activeList.map((s) => s.model).filter((m) => m && m.visible);

    const intersects = hoverRaycaster.intersectObjects(activeModels, true);

    let satId = null;

    if (intersects.length > 0) {
      let obj = intersects[0].object;
      while (obj) {
        if (obj.userData) {
          if (obj.userData.satelliteId) {
            satId = obj.userData.satelliteId;
            break;
          }
          if (obj.userData.satelliteState && obj.userData.satelliteState.id) {
            satId = obj.userData.satelliteState.id;
            break;
          }
        }
        obj = obj.parent;
      }
    }

    // Proximity fallback: check distance from ray to satellite position for smooth hover
    if (!satId) {
      const _satWorldPos = new THREE.Vector3();
      const _camToSat = new THREE.Vector3();
      const _camDir = new THREE.Vector3();
      camera.getWorldDirection(_camDir);

      let closestDist = Infinity;
      const HOVER_TOLERANCE = 2.5; // world units

      for (const sat of activeList) {
        if (!sat.model || !sat.model.visible) continue;
        sat.model.getWorldPosition(_satWorldPos);

        _camToSat.subVectors(_satWorldPos, camera.position);
        if (_camToSat.dot(_camDir) > 0) {
          const rayDist = hoverRaycaster.ray.distanceToPoint(_satWorldPos);
          if (rayDist < HOVER_TOLERANCE && rayDist < closestDist) {
            closestDist = rayDist;
            satId = sat.id;
          }
        }
      }
    }

    if (satId && tooltip) {
      tooltip.textContent = satId;
      tooltip.style.display = 'block';
      tooltip.style.left = `${e.clientX}px`;
      tooltip.style.top = `${e.clientY}px`;
      return;
    }

    if (tooltip) {
      tooltip.style.display = 'none';
    }
  });

  // 6. Drag-and-Drop Placement from Simulation Panel into 3D Scene
  const previewContainer = document.getElementById('satellite-preview-container');
  const dragRaycaster = new THREE.Raycaster();
  const dragMouse = new THREE.Vector2();
  const dropPoint = new THREE.Vector3();
  const planeNormal = new THREE.Vector3();
  const plane = new THREE.Plane();

  let isDragging = false;
  let ghostSatellite = null;

  function createGhostSatellite() {
    if (!ghostModelTemplate) return null;
    const ghost = ghostModelTemplate.clone(true);
    ghost.traverse((child) => {
      if (child.isMesh) {
        if (child.geometry) {
          child.geometry = child.geometry.clone();
        }
        if (child.material) {
          child.material = child.material.clone();
          child.material.transparent = true;
          child.material.opacity = 0.65;
          if (child.material.emissive) {
            child.material.emissive.setHex(0x00aaff);
          }
        }
      }
    });
    ghost.visible = false;
    scene.add(ghost);
    return ghost;
  }

  function cleanupGhostSatellite() {
    if (!ghostSatellite) return;

    scene.remove(ghostSatellite);

    ghostSatellite.traverse((child) => {
      if (child.isMesh) {
        if (child.geometry) {
          child.geometry.dispose();
        }
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach((m) => m.dispose());
          } else {
            child.material.dispose();
          }
        }
      }
    });

    ghostSatellite = null;
  }

  if (previewContainer) {
    previewContainer.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 && e.pointerType === 'mouse') return;

      const currentCount = typeof getManualCount === 'function' ? getManualCount() : 0;
      if (currentCount >= MAX_MANUAL_SATELLITES) {
        alert('Maximum 2 satellites allowed.');
        return;
      }

      isDragging = true;
      previewContainer.classList.add('dragging');

      if (controls) controls.enabled = false;

      if (!ghostSatellite) {
        ghostSatellite = createGhostSatellite();
      }
      if (ghostSatellite) {
        ghostSatellite.visible = true;
      }

      updateGhostPosition(e.clientX, e.clientY);
    });
  }

  function updateGhostPosition(clientX, clientY) {
    if (!isDragging || !ghostSatellite || !camera) return;

    dragMouse.x = (clientX / window.innerWidth) * 2 - 1;
    dragMouse.y = -(clientY / window.innerHeight) * 2 + 1;

    dragRaycaster.setFromCamera(dragMouse, camera);

    // Plane passing through Earth (0,0,0) facing camera
    camera.getWorldDirection(planeNormal).negate();
    plane.setFromNormalAndCoplanarPoint(planeNormal, new THREE.Vector3(0, 0, 0));

    const hit = dragRaycaster.ray.intersectPlane(plane, dropPoint);
    if (hit) {
      let dist = dropPoint.length();
      if (dist < MIN_SATELLITE_DISTANCE) {
        dropPoint.normalize().multiplyScalar(MIN_SATELLITE_DISTANCE);
      } else if (dist > MAX_SATELLITE_DISTANCE) {
        dropPoint.normalize().multiplyScalar(MAX_SATELLITE_DISTANCE);
      }

      ghostSatellite.position.copy(dropPoint);
      ghostSatellite.lookAt(0, 0, 0);
    }
  }

  window.addEventListener('pointermove', (e) => {
    if (isDragging) {
      updateGhostPosition(e.clientX, e.clientY);
    }
  });

  window.addEventListener('pointerup', (e) => {
    if (!isDragging) return;

    isDragging = false;
    if (previewContainer) previewContainer.classList.remove('dragging');

    if (controls) controls.enabled = true;

    cleanupGhostSatellite();

    const panel = document.getElementById('control-panel');
    let droppedInPanel = false;
    if (panel) {
      const rect = panel.getBoundingClientRect();
      if (
        e.clientX >= rect.left &&
        e.clientX <= rect.right &&
        e.clientY >= rect.top &&
        e.clientY <= rect.bottom
      ) {
        droppedInPanel = true;
      }
    }

    if (!droppedInPanel && typeof onDropSatellite === 'function') {
      onDropSatellite(dropPoint.clone());
    }
  });
}
