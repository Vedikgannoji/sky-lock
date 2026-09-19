import * as THREE from 'three';
import { EARTH_RADIUS } from '../loadAssets.js';

/**
 * Inter-Satellite Link Line (ISL)
 * Renders a high-visibility, emissive 3D link beam with glowing endpoint markers
 * between active satellites when mutual line of sight is clear.
 */

// Module-level persistent Three.js objects (prevents per-frame GC allocations)
let rootGroup = null;
let coreBeam = null;
let glowBeam = null;
let centralLine = null;
let markerGroupA = null;
let markerGroupB = null;

let coreMaterial = null;
let glowMaterial = null;
let lineMaterial = null;
let markerMaterial = null;
let markerGlowMaterial = null;

const _posA = new THREE.Vector3();
const _posB = new THREE.Vector3();
const _dir = new THREE.Vector3();
const _mid = new THREE.Vector3();
const _quat = new THREE.Quaternion();
const _yAxis = new THREE.Vector3(0, 1, 0);
const _defaultEarthCenter = new THREE.Vector3(0, 0, 0);

/**
 * Pure function: Test whether the line segment between posA and posB has an unobstructed
 * line of sight (i.e. does NOT intersect Earth's sphere).
 *
 * @param {THREE.Vector3|{x: number, y: number, z: number}} posA - First satellite position
 * @param {THREE.Vector3|{x: number, y: number, z: number}} posB - Second satellite position
 * @param {THREE.Vector3|{x: number, y: number, z: number}} [earthCenter={x:0, y:0, z:0}] - Center of Earth
 * @param {number} [earthRadius=EARTH_RADIUS] - Radius of Earth's occlusion sphere
 * @returns {boolean} True if line of sight is clear (unobstructed), false if occluded by Earth.
 */
export function hasLineOfSight(
  posA,
  posB,
  earthCenter = _defaultEarthCenter,
  earthRadius = EARTH_RADIUS
) {
  if (!posA || !posB) return false;

  const dx = posB.x - posA.x;
  const dy = posB.y - posA.y;
  const dz = posB.z - posA.z;
  const lenSq = dx * dx + dy * dy + dz * dz;

  const ecX = earthCenter.x || 0;
  const ecY = earthCenter.y || 0;
  const ecZ = earthCenter.z || 0;

  // Satellites coincident: check if inside Earth
  if (lenSq === 0) {
    const dSq = (posA.x - ecX) ** 2 + (posA.y - ecY) ** 2 + (posA.z - ecZ) ** 2;
    return dSq > earthRadius * earthRadius;
  }

  // Vector from posA to earthCenter: AC
  const acX = ecX - posA.x;
  const acY = ecY - posA.y;
  const acZ = ecZ - posA.z;

  // Project AC onto AB: t = (AC · AB) / |AB|^2
  const dot = acX * dx + acY * dy + acZ * dz;
  const t = Math.max(0, Math.min(1, dot / lenSq));

  // Closest point on segment AB to earthCenter
  const closeX = posA.x + t * dx;
  const closeY = posA.y + t * dy;
  const closeZ = posA.z + t * dz;

  // Squared distance from closest point to sphere center
  const distSq = (closeX - ecX) ** 2 + (closeY - ecY) ** 2 + (closeZ - ecZ) ** 2;

  // Clear if strictly outside Earth's sphere
  return distSq > earthRadius * earthRadius;
}

/**
 * Initialize 3D beam and endpoint marker meshes.
 */
function initBeamObjects(scene) {
  rootGroup = new THREE.Group();
  rootGroup.renderOrder = 999;

  // 1. Inner Core Cylinder (Bright Cyan Beam)
  const coreGeo = new THREE.CylinderGeometry(0.12, 0.12, 1, 10, 1, true);
  coreMaterial = new THREE.MeshBasicMaterial({
    color: 0x00ffff,
    transparent: true,
    opacity: 0.85,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });
  coreBeam = new THREE.Mesh(coreGeo, coreMaterial);

  // 2. Outer Glow Cylinder (Volumetric aura around the beam)
  const glowGeo = new THREE.CylinderGeometry(0.38, 0.38, 1, 10, 1, true);
  glowMaterial = new THREE.MeshBasicMaterial({
    color: 0x0099ff,
    transparent: true,
    opacity: 0.35,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });
  glowBeam = new THREE.Mesh(glowGeo, glowMaterial);

  // 3. Central Line (Ensures crisp visibility at extreme distances)
  const lineGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, -0.5, 0),
    new THREE.Vector3(0, 0.5, 0)
  ]);
  lineMaterial = new THREE.LineBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });
  centralLine = new THREE.Line(lineGeo, lineMaterial);

  const beamAssembly = new THREE.Group();
  beamAssembly.name = 'beamAssembly';
  beamAssembly.add(coreBeam);
  beamAssembly.add(glowBeam);
  beamAssembly.add(centralLine);
  rootGroup.add(beamAssembly);

  // 4. Glowing Endpoint Markers
  markerMaterial = new THREE.MeshBasicMaterial({
    color: 0x00ffff,
    transparent: true,
    opacity: 0.95,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });

  markerGlowMaterial = new THREE.MeshBasicMaterial({
    color: 0x00bfff,
    transparent: true,
    opacity: 0.4,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });

  const innerSphereGeo = new THREE.SphereGeometry(0.4, 16, 16);
  const outerSphereGeo = new THREE.SphereGeometry(0.75, 16, 16);

  markerGroupA = new THREE.Group();
  markerGroupA.add(new THREE.Mesh(innerSphereGeo, markerMaterial));
  markerGroupA.add(new THREE.Mesh(outerSphereGeo, markerGlowMaterial));

  markerGroupB = new THREE.Group();
  markerGroupB.add(new THREE.Mesh(innerSphereGeo, markerMaterial));
  markerGroupB.add(new THREE.Mesh(outerSphereGeo, markerGlowMaterial));

  rootGroup.add(markerGroupA);
  rootGroup.add(markerGroupB);

  scene.add(rootGroup);
}

/**
 * Control overall visibility of the link line and markers.
 *
 * @param {boolean} visible
 */
export function setLinkLineVisible(visible) {
  if (rootGroup) {
    rootGroup.visible = visible;
  }
}

/**
 * Update the inter-satellite link line each frame.
 * Renders between active satellites only when mutual line of sight is clear.
 *
 * @param {THREE.Scene} scene
 * @param {Array<Object>} activeSatellites
 * @param {THREE.Object3D} [earth] - Optional Earth reference for center position
 * @returns {boolean} True if link is active and line-of-sight is clear; false if occluded or fewer than 2 active.
 */
export function update(scene, activeSatellites, earth = null) {
  if (!scene) return false;

  // Filter operational satellites (deployed, visible, not paused)
  const active = (activeSatellites || []).filter(
    (sat) => sat && sat.model && !sat.paused && sat.model.visible !== false
  );

  // Hide link when fewer than 2 active satellites exist
  if (active.length < 2) {
    setLinkLineVisible(false);
    return false;
  }

  // Get current world positions of the primary active satellite pair
  active[0].model.getWorldPosition(_posA);
  active[1].model.getWorldPosition(_posB);

  // Earth center coordinate
  const earthCenter = earth ? earth.position : _defaultEarthCenter;

  // 1. Line-of-sight check against Earth sphere
  const losClear = hasLineOfSight(_posA, _posB, earthCenter, EARTH_RADIUS);
  if (!losClear) {
    setLinkLineVisible(false);
    return false; // Earth occlusion: link is OBSTRUCTED
  }

  // 2. Lazy initialization of 3D objects
  if (!rootGroup) {
    initBeamObjects(scene);
  } else if (rootGroup.parent !== scene) {
    scene.add(rootGroup);
  }

  rootGroup.visible = true;

  // 3. Orient and scale beam between _posA and _posB
  const dist = _posA.distanceTo(_posB);
  _dir.subVectors(_posB, _posA);
  _mid.addVectors(_posA, _posB).multiplyScalar(0.5);

  const beamAssembly = rootGroup.getObjectByName('beamAssembly');
  if (beamAssembly && dist > 0.001) {
    beamAssembly.position.copy(_mid);
    _dir.normalize();
    _quat.setFromUnitVectors(_yAxis, _dir);
    beamAssembly.quaternion.copy(_quat);
    beamAssembly.scale.set(1, dist, 1);
  }

  // 4. Update endpoint glowing markers
  markerGroupA.position.copy(_posA);
  markerGroupB.position.copy(_posB);

  // 5. Pulsing opacity and glow breathing effect
  const time = performance.now() * 0.006;
  const pulse = Math.sin(time);

  if (coreMaterial) coreMaterial.opacity = 0.75 + 0.25 * pulse;
  if (glowMaterial) glowMaterial.opacity = 0.28 + 0.16 * pulse;
  if (lineMaterial) lineMaterial.opacity = 0.8 + 0.2 * pulse;
  if (markerMaterial) markerMaterial.opacity = 0.85 + 0.15 * pulse;
  if (markerGlowMaterial) markerGlowMaterial.opacity = 0.3 + 0.15 * pulse;

  return true;
}
