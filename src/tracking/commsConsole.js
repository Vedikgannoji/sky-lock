import * as THREE from 'three';

/**
 * Inter-Satellite Comms Console
 * Monospace terminal logging real telemetry every 5 seconds with Earth occlusion handling.
 */

let logContainer = null;
let statusDot = null;
let consolePanel = null;

// Wall-clock delta time accumulator in seconds (independent of simulationSpeed/pause)
let timeAccumulator = 0;
const CADENCE_INTERVAL_SEC = 5.0;

let tickCount = 0;
let isCurrentlyObstructed = false;

// Earth radius conversion (1 Three.js unit ≈ 637.1 km)
const KM_PER_UNIT = 637.1;
const SPEED_OF_LIGHT_KMS = 299792;
const MAX_LOG_LINES = 20;

// Reusable math objects
const _posA = new THREE.Vector3();
const _posB = new THREE.Vector3();
const _tanA = new THREE.Vector3();
const _tanB = new THREE.Vector3();
const _velA = new THREE.Vector3();
const _velB = new THREE.Vector3();
const _deltaV = new THREE.Vector3();

/**
 * Format current local time as [HH:MM:SS]
 */
function getTimeString() {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
}

/**
 * Append a formatted line to the terminal log with auto-scroll and cap.
 */
function appendLog(html) {
  if (!logContainer) return;

  const line = document.createElement('div');
  line.className = 'comms-line';
  line.style.flexShrink = '0';
  line.style.fontSize = '11px';
  line.style.lineHeight = '18px';
  line.style.height = '18px';
  line.style.minHeight = '18px';
  line.innerHTML = html;
  logContainer.appendChild(line);

  // Cap at ~20 lines (strictly remove old DOM nodes from top, never restyle existing ones)
  while (logContainer.children.length > MAX_LOG_LINES) {
    logContainer.removeChild(logContainer.firstChild);
  }

  // Auto-scroll only updates scrollTop without altering container geometry
  logContainer.scrollTop = logContainer.scrollHeight;
}

/**
 * Initialize the comms console DOM connections and initial status.
 */
export function initCommsConsole() {
  consolePanel = document.getElementById('comms-console');
  logContainer = document.getElementById('comms-log-container');
  statusDot = document.getElementById('comms-status-dot');

  if (!logContainer) return;

  appendLog(
    `[${getTimeString()}] <span class="comms-peer">SYSTEM</span> | <span class="comms-type">[INIT]</span> | ISL terminal online | 5s telemetry cadence`
  );
}

/**
 * Show or hide the comms console panel.
 *
 * @param {boolean} visible
 */
export function setCommsConsoleVisible(visible) {
  if (!consolePanel) {
    consolePanel = document.getElementById('comms-console');
  }
  if (consolePanel) {
    consolePanel.style.display = visible ? 'flex' : 'none';
  }
}

/**
 * Clear the console log contents.
 */
export function clearConsole() {
  if (logContainer) {
    logContainer.innerHTML = '';
  }
}

/**
 * Update comms console telemetry logs called from main animation loop.
 * Accumulates real deltaTime in seconds and fires every 5 seconds.
 * Halts logging and shows a persistent occlusion line while line of sight is obstructed.
 *
 * @param {Array<Object>} activeSatellites
 * @param {number} realDeltaTime - Real wall-clock elapsed time in seconds (independent of simulationSpeed/pause)
 * @param {boolean} hasLOS - True if line of sight is unobstructed by Earth, false if occluded
 */
export function update(activeSatellites, realDeltaTime = 0, hasLOS = true) {
  if (!logContainer) {
    consolePanel = document.getElementById('comms-console');
    logContainer = document.getElementById('comms-log-container');
    statusDot = document.getElementById('comms-status-dot');
    if (!logContainer) return;
  }

  // Filter operational satellites (in scene, visible, not paused)
  const operational = (activeSatellites || []).filter(
    (s) => s && s.model && !s.paused && s.model.visible !== false
  );
  const totalInMode = (activeSatellites || []).filter(
    (s) => s && s.model && s.model.visible !== false
  );

  const timeStr = getTimeString();

  // 1. Earth Occlusion Check
  if (operational.length >= 2 && !hasLOS) {
    if (!isCurrentlyObstructed) {
      isCurrentlyObstructed = true;
      timeAccumulator = 0; // Reset accumulator so cadence restarts cleanly upon restoration

      if (statusDot) {
        statusDot.className = 'comms-status-indicator obstructed';
      }

      appendLog(
        `[${timeStr}] <span class="comms-peer">ISL-MONITOR</span> | <span class="comms-type">[OCCLUSION]</span> | link: <span class="comms-link-obstructed">LINK OBSTRUCTED — EARTH OCCLUSION</span>`
      );
    }
    // While obstructed, do NOT log new lines
    return;
  }

  // If link was obstructed and now line of sight has returned
  if (isCurrentlyObstructed && hasLOS && operational.length >= 2) {
    isCurrentlyObstructed = false;
    timeAccumulator = 0;

    if (statusDot) {
      statusDot.className = 'comms-status-indicator';
    }

    appendLog(
      `[${timeStr}] <span class="comms-peer">ISL-MONITOR</span> | <span class="comms-type">[REACQUIRED]</span> | link: <span class="comms-link-locked">LOS RESTORED — LINK ACTIVE</span>`
    );
    return;
  }

  // Accumulate real wall-clock elapsed time
  if (typeof realDeltaTime === 'number' && realDeltaTime > 0) {
    timeAccumulator += realDeltaTime;
  }

  // Only fire a new log line once exactly 5 seconds have accumulated
  if (timeAccumulator < CADENCE_INTERVAL_SEC) {
    return;
  }
  timeAccumulator = 0; // Reset accumulator after each log
  tickCount++;

  // Update status dot for non-occluded states
  if (statusDot) {
    if (operational.length >= 2) {
      statusDot.className = 'comms-status-indicator';
    } else {
      statusDot.className = 'comms-status-indicator offline';
    }
  }

  // Case A: 2+ active satellites in mutual line of sight
  if (operational.length >= 2) {
    const satA = operational[0];
    const satB = operational[1];

    satA.model.getWorldPosition(_posA);
    satB.model.getWorldPosition(_posB);

    // Compute real distance in km
    const distUnits = _posA.distanceTo(_posB);
    const distKm = Math.round(distUnits * KM_PER_UNIT);

    // Compute relative velocity (Δv) from orbital tangents and radii
    let dv = 3.2;
    if (
      satA.orbit &&
      satB.orbit &&
      typeof satA.orbit.getTangent === 'function' &&
      typeof satB.orbit.getTangent === 'function'
    ) {
      satA.orbit.getTangent(_tanA);
      satB.orbit.getTangent(_tanB);

      const rA = Math.max(10, _posA.length()) * KM_PER_UNIT;
      const rB = Math.max(10, _posB.length()) * KM_PER_UNIT;

      const speedA = Math.sqrt(398600 / rA) * (satA.individualSpeed || 1);
      const speedB = Math.sqrt(398600 / rB) * (satB.individualSpeed || 1);

      _velA.copy(_tanA).multiplyScalar(speedA);
      _velB.copy(_tanB).multiplyScalar(speedB);
      _deltaV.subVectors(_velA, _velB);
      dv = _deltaV.length();
    }

    const dvStr = (dv > 0 ? dv : 3.2).toFixed(1);
    const direction =
      tickCount % 2 === 0 ? `${satA.id} -> ${satB.id}` : `${satB.id} -> ${satA.id}`;

    // Vary message types each 5-second tick
    const messageTypes = ['POSITION_SYNC', 'RANGING', 'HANDSHAKE', 'DATA_PACKET'];
    const currentType = messageTypes[tickCount % messageTypes.length];

    let logHtml = '';

    switch (currentType) {
      case 'POSITION_SYNC': {
        const senderPos = tickCount % 2 === 0 ? _posA : _posB;
        const x = Math.round(senderPos.x * KM_PER_UNIT);
        const y = Math.round(senderPos.y * KM_PER_UNIT);
        const z = Math.round(senderPos.z * KM_PER_UNIT);
        logHtml = `[${timeStr}] <span class="comms-peer">${direction}</span> | <span class="comms-type">[POSITION_SYNC]</span> | dist: <span class="comms-dist">${distKm.toLocaleString()} km</span> | Δv: <span class="comms-dv">${dvStr} km/s</span> | link: <span class="comms-link-locked">LOCKED</span> | eph: [${x}, ${y}, ${z}] sync OK`;
        break;
      }

      case 'RANGING': {
        const tofMs = ((distKm / SPEED_OF_LIGHT_KMS) * 1000).toFixed(1);
        logHtml = `[${timeStr}] <span class="comms-peer">${direction}</span> | <span class="comms-type">[RANGING]</span> | dist: <span class="comms-dist">${distKm.toLocaleString()} km</span> | Δv: <span class="comms-dv">${dvStr} km/s</span> | link: <span class="comms-link-locked">LOCKED</span> | tof: ${tofMs} ms`;
        break;
      }

      case 'HANDSHAKE': {
        const freqs = ['23.4 GHz', '23.8 GHz', '24.1 GHz', '24.5 GHz'];
        const freq = freqs[tickCount % freqs.length];
        logHtml = `[${timeStr}] <span class="comms-peer">${direction}</span> | <span class="comms-type">[HANDSHAKE]</span> | dist: <span class="comms-dist">${distKm.toLocaleString()} km</span> | Δv: <span class="comms-dv">${dvStr} km/s</span> | link: <span class="comms-link-locked">LOCKED</span> | carrier: ${freq} SYNC`;
        break;
      }

      case 'DATA_PACKET':
      default: {
        const packetSizes = ['256kb', '512kb', '1024kb', '128kb'];
        const packet = packetSizes[tickCount % packetSizes.length];
        logHtml = `[${timeStr}] <span class="comms-peer">${direction}</span> | <span class="comms-type">[DATA_PACKET]</span> | dist: <span class="comms-dist">${distKm.toLocaleString()} km</span> | Δv: <span class="comms-dv">${dvStr} km/s</span> | link: <span class="comms-link-locked">LOCKED</span> | packet: ${packet} sent`;
        break;
      }
    }

    appendLog(logHtml);
    return;
  }

  // Case B: Satellites exist, but one or both are paused
  if (totalInMode.length >= 2) {
    const pausedSat = totalInMode.find((s) => s.paused);
    const pausedId = pausedSat ? pausedSat.id : 'PEER';
    appendLog(
      `[${timeStr}] <span class="comms-peer">ISL-MONITOR</span> | <span class="comms-type">[LINK_HOLD]</span> | link: <span class="comms-link-wait">HOLD (${pausedId} PAUSED)</span> | carrier standby`
    );
    return;
  }

  // Case C: Only 1 satellite deployed
  if (totalInMode.length === 1) {
    appendLog(
      `[${timeStr}] <span class="comms-peer">${totalInMode[0].id}</span> | <span class="comms-type">[RANGING]</span> | link: <span class="comms-link-wait">SEARCHING</span> | awaiting peer satellite`
    );
    return;
  }

  // Case D: No satellites deployed
  appendLog(
    `[${timeStr}] <span class="comms-peer">ISL-CORE</span> | <span class="comms-type">[STANDBY]</span> | link: <span class="comms-link-wait">IDLE</span> | awaiting satellite deployment`
  );
}
