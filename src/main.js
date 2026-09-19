import * as THREE from 'three';
import { setupScene } from './sceneSetup.js';
import { loadAssets } from './loadAssets.js';
import {
  initializeOrbits,
  setupOrbitLines,
  createManualOrbit,
  MAX_MANUAL_SATELLITES
} from './orbit.js';
import {
  setupUI,
  initSatellitePreview,
  renderSatellitePreview,
  renderSatelliteStatusList,
  simulationSpeed,
  isPaused,
  orbitLinesVisible,
  islLinkEnabled,
  satelliteMode
} from './ui.js';
import {
  update as updateLinkLine,
  setLinkLineVisible,
  hasLineOfSight
} from './tracking/linkLine.js';
import {
  initCommsConsole,
  update as updateCommsConsole,
  setCommsConsoleVisible
} from './tracking/commsConsole.js';

// ============================================================
// CONFIGURATION CONSTANTS
// ============================================================

/**
 * Earth continuous rotation speed (radians/second at 1x simulation speed).
 */
export const EARTH_ROTATION_SPEED = 0.08;

// Internal reusable math objects
const _position = new THREE.Vector3();
const _targetQuat = new THREE.Quaternion();

// ============================================================
// FPS COUNTER
// ============================================================

let frameCount = 0;
let lastFpsUpdate = 0;
const fpsUpdateInterval = 500; // ms

function updateFpsCounter(currentTime) {
  frameCount++;

  if (currentTime - lastFpsUpdate >= fpsUpdateInterval) {
    const fps = Math.round(frameCount / ((currentTime - lastFpsUpdate) / 1000));
    document.getElementById('fps-counter').textContent = `FPS: ${fps}`;
    frameCount = 0;
    lastFpsUpdate = currentTime;
  }
}

// ============================================================
// MAIN APPLICATION
// ============================================================

async function main() {
  // 1. Setup Three.js scene, camera, renderer, orbit controls
  const { scene, camera, renderer, controls } = setupScene();

  // 2. Load all GLB assets (Earth with Sketchfab textures + 2 satellites)
  const { earth, satellite1, satellite2 } = await loadAssets(scene);

  // Satellite collections
  let autoSatellites = [];
  const manualSatellites = [];

  // Helper to attach satellite state directly to the 3D model hierarchy
  function attachStateToHierarchy(model, state) {
    model.userData.satelliteId = state.id;
    Object.defineProperty(model.userData, 'satelliteState', {
      value: state,
      enumerable: false,
      writable: true,
      configurable: true
    });

    model.traverse((child) => {
      child.userData.satelliteId = state.id;
      Object.defineProperty(child.userData, 'satelliteState', {
        value: state,
        enumerable: false,
        writable: true,
        configurable: true
      });
      if (child.isMesh) {
        child.castShadow = false;
      }
    });
  }

  // Build automatic satellite instances
  function buildAutomaticSatellites() {
    // Clear any previous automatic satellites
    autoSatellites.forEach((sat) => {
      scene.remove(sat.model);
      scene.remove(sat.orbitLine);
      if (sat.orbitLine.geometry) sat.orbitLine.geometry.dispose();
      if (sat.orbitLine.material) sat.orbitLine.material.dispose();
    });

    const lines = setupOrbitLines(scene);
    const orbits = initializeOrbits();

    const sat1Data = {
      id: 'S-1',
      model: satellite1,
      orbit: orbits[0],
      orbitLine: lines[0],
      individualSpeed: 1.0,
      paused: false,
      isManual: false
    };

    const sat2Data = {
      id: 'S-2',
      model: satellite2,
      orbit: orbits[1],
      orbitLine: lines[1],
      individualSpeed: 1.0,
      paused: false,
      isManual: false
    };

    // Apply stable initial positions and orientations immediately
    [sat1Data, sat2Data].forEach((sat) => {
      sat.orbit.getPosition(_position);
      sat.model.position.copy(_position);
      sat.orbit.getOrientation(_targetQuat);
      sat.model.quaternion.copy(_targetQuat);
      attachStateToHierarchy(sat.model, sat);
      sat.model.visible = satelliteMode === 'AUTOMATIC';
      sat.orbitLine.visible = satelliteMode === 'AUTOMATIC' && orbitLinesVisible;
      scene.add(sat.model);
    });

    autoSatellites = [sat1Data, sat2Data];
  }

  buildAutomaticSatellites();

  // Return currently active satellite list based on mode
  function getActiveSatellites() {
    return satelliteMode === 'AUTOMATIC' ? autoSatellites : manualSatellites;
  }

  // Refresh the UI Satellite Status cards
  function refreshStatusList() {
    renderSatelliteStatusList(getActiveSatellites(), {
      onSatelliteSpeedChange: (sat, speed) => {
        sat.individualSpeed = speed;
        console.log(`${sat.id} individual speed set to ${speed}×`);
      },
      onSatelliteTogglePause: (sat, paused) => {
        sat.paused = paused;
        console.log(`${sat.id} ${paused ? 'PAUSED' : 'RESUMED'}`);
        refreshStatusList();
      },
      onSatelliteRemove: (sat) => {
        if (!sat.isManual) return;
        console.log(`Removing ${sat.id}`);
        scene.remove(sat.model);
        scene.remove(sat.orbitLine);
        if (sat.orbitLine.geometry) sat.orbitLine.geometry.dispose();
        if (sat.orbitLine.material) sat.orbitLine.material.dispose();

        sat.model.traverse((child) => {
          if (child.isMesh) {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
              if (Array.isArray(child.material)) {
                child.material.forEach((m) => m.dispose());
              } else {
                child.material.dispose();
              }
            }
          }
        });

        const idx = manualSatellites.indexOf(sat);
        if (idx !== -1) manualSatellites.splice(idx, 1);
        refreshStatusList();
      }
    });
  }

  // Initialize 3D preview in the control panel
  initSatellitePreview(satellite1);

  // Handle manual drag-and-drop placement
  function handleDropSatellite(dropPosition) {
    if (manualSatellites.length >= MAX_MANUAL_SATELLITES) {
      alert('Maximum 2 satellites allowed.');
      return;
    }

    // Determine unique ID ('S-1' or 'S-2')
    const usedIds = manualSatellites.map((s) => s.id);
    const newId = !usedIds.includes('S-1') ? 'S-1' : 'S-2';
    const satIndex = newId === 'S-1' ? 0 : 1;

    // First manual satellite uses satellite1, second uses satellite2
    const sourceModel = satIndex === 0 ? satellite1 : satellite2;
    const newModel = sourceModel.clone(true);
    newModel.traverse((child) => {
      if (child.isMesh && child.material) {
        if (Array.isArray(child.material)) {
          child.material = child.material.map((m) => m.clone());
        } else {
          child.material = child.material.clone();
        }
      }
    });

    // Generate elliptical orbit passing through dropPosition
    const { orbitState, orbitLine, initialPosition } = createManualOrbit(dropPosition, satIndex);

    const satData = {
      id: newId,
      model: newModel,
      orbit: orbitState,
      orbitLine: orbitLine,
      individualSpeed: 1.0,
      paused: false,
      isManual: true
    };

    // Set position and orientation immediately at drop location
    newModel.position.copy(initialPosition);
    orbitState.getOrientation(_targetQuat);
    newModel.quaternion.copy(_targetQuat);

    attachStateToHierarchy(newModel, satData);

    newModel.visible = satelliteMode === 'MANUAL';
    orbitLine.visible = satelliteMode === 'MANUAL' && orbitLinesVisible;

    scene.add(newModel);
    scene.add(orbitLine);

    manualSatellites.push(satData);

    console.log(`Placed ${newId} in orbit at:`, initialPosition);
    refreshStatusList();
  }

  // Handle switching between AUTOMATIC and MANUAL mode
  function handleModeChange(mode) {
    if (mode === 'AUTOMATIC') {
      // Re-create standard 2-satellite automatic configuration if needed
      if (autoSatellites.length < 2) {
        buildAutomaticSatellites();
      }

      autoSatellites.forEach((sat) => {
        sat.model.visible = true;
        sat.orbitLine.visible = orbitLinesVisible;
      });

      manualSatellites.forEach((sat) => {
        sat.model.visible = false;
        sat.orbitLine.visible = false;
      });
    } else {
      // MANUAL MODE
      autoSatellites.forEach((sat) => {
        sat.model.visible = false;
        sat.orbitLine.visible = false;
      });

      manualSatellites.forEach((sat) => {
        sat.model.visible = true;
        sat.orbitLine.visible = orbitLinesVisible;
      });
    }

    refreshStatusList();
  }

  // 3. Setup UI Control Panel
  setupUI({
    onSpeedChange: (speed) => {
      console.log(`Global simulation speed: ${speed}×`);
    },
    onToggleOrbitLines: (visible) => {
      const activeList = getActiveSatellites();
      activeList.forEach((sat) => {
        sat.orbitLine.visible = visible;
      });
    },
    onToggleISLLink: (enabled) => {
      if (!enabled) {
        setLinkLineVisible(false);
        setCommsConsoleVisible(false);
      } else {
        setCommsConsoleVisible(true);
      }
    },
    onTogglePause: (paused) => {
      console.log(`Global simulation ${paused ? 'PAUSED' : 'RESUMED'}`);
    },
    onModeChange: handleModeChange,
    onDropSatellite: handleDropSatellite,
    getManualCount: () => manualSatellites.length,
    getActiveSatellites,
    camera,
    scene,
    controls,
    ghostModelTemplate: satellite1
  });

  // Initial live status rendering
  refreshStatusList();

  // Initialize inter-satellite comms console
  initCommsConsole();

  // 4. Animation loop
  let lastTime = performance.now();

  function animate() {
    requestAnimationFrame(animate);

    const currentTime = performance.now();
    const deltaTime = (currentTime - lastTime) / 1000;
    lastTime = currentTime;

    // Simulation delta: 0 if global paused, otherwise deltaTime * simulationSpeed
    const simulationDelta = isPaused ? 0 : deltaTime * simulationSpeed;

    // Update OrbitControls (smooth damping)
    controls.update();

    // Update active satellites
    const currentActive = getActiveSatellites();
    currentActive.forEach((sat) => {
      if (sat.paused) return; // Individual pause!

      const satDelta = simulationDelta * sat.individualSpeed;
      if (satDelta <= 0) return;

      sat.orbit.update(satDelta);
      sat.orbit.getPosition(_position);
      sat.model.position.copy(_position);

      // Stable, flip-free orientation along direction of travel
      sat.orbit.getOrientation(_targetQuat);
      sat.model.quaternion.slerp(_targetQuat, 0.25);
    });

    // Update inter-satellite link line and comms console telemetry (when enabled)
    if (islLinkEnabled) {
      const hasLOS = updateLinkLine(scene, currentActive, earth);
      updateCommsConsole(currentActive, deltaTime, hasLOS);
    }

    // Continuous slow Earth rotation
    if (earth) {
      earth.rotation.y += EARTH_ROTATION_SPEED * simulationDelta;
    }

    // Render mini 3D preview in control panel (when in Manual mode)
    renderSatellitePreview();

    // Real-time FPS display
    updateFpsCounter(currentTime);

    // Render 3D scene
    renderer.render(scene, camera);
  }

  animate();
  console.log(
    '🌍 Simulation running with stable orientation, individual controls, and live status.'
  );
}

// ============================================================
// BOOTSTRAP
// ============================================================

main().catch((error) => {
  console.error('Failed to start application:', error);
  document.body.innerHTML = `
    <div style="color: #f00; padding: 20px; font-family: monospace;">
      <h2>Error Loading Application</h2>
      <pre>${error.message}</pre>
    </div>
  `;
});
