import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const hint = document.getElementById("viewport-hint");
const gpuStatus = document.getElementById("gpu-status");
const cameraStatus = document.getElementById("camera-status");
const wardrobeGrid = document.getElementById("wardrobe-grid");
const sizeBadge = document.getElementById("size-badge");
const toast = document.getElementById("toast");
const btnScan = document.getElementById("btn-scan");
const btnSnap = document.getElementById("btn-snap");
const btnClear = document.getElementById("btn-clear");
const scanLine = document.getElementById("scan-line");
const fitProfile = document.getElementById("fit-profile");
const fitProfileBody = document.getElementById("fit-profile-body");
const snapModal = document.getElementById("snap-modal");
const snapResult = document.getElementById("snap-result");
const snapCaption = document.getElementById("snap-caption");
const snapClose = document.getElementById("snap-close");

const L_SHOULDER = 11, R_SHOULDER = 12, L_HIP = 23, R_HIP = 24;
const L_KNEE = 25, R_KNEE = 26;

const state = {
  selectedGarment: null,
  measurements: null,
  stream: null,
  poseLandmarker: null,
  lastVideoTime: -1,
  garmentAssets: {},
};

function showToast(msg, ms = 3500) {
  toast.textContent = msg;
  toast.classList.remove("hidden");
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => toast.classList.add("hidden"), ms);
}

function setStatus(el, text, cls) {
  el.textContent = text;
  el.className = `status-pill status-${cls}`;
}

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    setStatus(gpuStatus, data.cuda ? `GPU: ${data.gpu}` : "GPU unavailable (CPU)", data.cuda ? "ok" : "warn");
  } catch {
    setStatus(gpuStatus, "Backend offline", "err");
  }
}

async function initPoseLandmarker() {
  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    state.poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
      baseOptions: { modelAssetPath: "/models/pose_landmarker_full.task", delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  } catch (err) {
    console.error("Pose landmarker init failed", err);
    showToast("Pose model failed to load — overlay/measure unavailable");
  }
}

async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
    video.srcObject = state.stream;
    await video.play();
    hint.classList.add("hidden");
    setStatus(cameraStatus, "Camera on", "ok");
    btnScan.disabled = false;
    resizeOverlay();
  } catch (err) {
    hint.textContent = `Camera error: ${err.message}. Allow camera access and reload.`;
    setStatus(cameraStatus, "Camera blocked", "err");
  }
}

function resizeOverlay() {
  overlay.width = video.videoWidth || 1280;
  overlay.height = video.videoHeight || 960;
}

video.addEventListener("loadedmetadata", resizeOverlay);
window.addEventListener("resize", resizeOverlay);

async function loadWardrobe() {
  try {
    const res = await fetch("/api/wardrobe");
    const data = await res.json();
    if (!data.items.length) {
      wardrobeGrid.innerHTML =
        '<p class="empty-msg">No garments yet — add images to <code>wardrobe/</code></p>';
      return;
    }
    wardrobeGrid.innerHTML = "";
    for (const item of data.items) {
      const card = document.createElement("div");
      card.className = "garment-card";
      card.dataset.id = item.id;
      card.innerHTML = `
        <img src="${item.image_url}" alt="${item.name}" loading="lazy" />
        <div class="g-name">${item.name}</div>
      `;
      card.addEventListener("click", () => selectGarment(item, card));
      wardrobeGrid.appendChild(card);
    }
  } catch {
    wardrobeGrid.innerHTML = '<p class="empty-msg">Failed to load wardrobe from backend</p>';
  }
}

/* ---------- garment asset: load + make background transparent ---------- */

function removeWhiteBackground(img) {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const cx = c.getContext("2d");
  cx.drawImage(img, 0, 0);
  const data = cx.getImageData(0, 0, c.width, c.height);
  const d = data.data;
  for (let i = 0; i < d.length; i += 4) {
    const bright = (d[i] + d[i + 1] + d[i + 2]) / 3;
    if (bright > 244) d[i + 3] = 0;
    else if (bright > 224) d[i + 3] = Math.round((255 * (244 - bright)) / 20);
  }
  cx.putImageData(data, 0, 0);
  return c;
}

function loadGarmentAsset(item) {
  if (state.garmentAssets[item.id]) return Promise.resolve(state.garmentAssets[item.id]);
  const src = item.cutout_url || item.image_url;
  const needsKeying = !item.cutout_url;
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = needsKeying ? removeWhiteBackground(img) : toCanvas(img);
      const asset = { canvas, w: canvas.width, h: canvas.height, category: item.category };
      state.garmentAssets[item.id] = asset;
      resolve(asset);
    };
    img.onerror = reject;
    img.src = src;
  });
}

function toCanvas(img) {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  c.getContext("2d").drawImage(img, 0, 0);
  return c;
}

/* ---------- garment warping ---------- */

const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });
const scale = (v, s) => ({ x: v.x * s, y: v.y * s });
const len = (v) => Math.hypot(v.x, v.y);
const norm = (v) => {
  const l = len(v) || 1;
  return { x: v.x / l, y: v.y / l };
};
const lerp = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });

/* Build left/right garment contour "knots" (center + half-width) from pose landmarks. */
function garmentContours(lm, category) {
  const ls = lm[L_SHOULDER], rs = lm[R_SHOULDER], lh = lm[L_HIP], rh = lm[R_HIP];
  if (!ls || !rs || !lh || !rh) return null;
  const lk = lm[L_KNEE], rk = lm[R_KNEE];

  const shoulderMid = mid(ls, rs);
  const hipMid = mid(lh, rh);
  const ex = norm(sub(rs, ls));
  const torsoVec = sub(hipMid, shoulderMid);
  const ey = norm(torsoVec);
  const shoulderHalf = len(sub(rs, ls)) / 2;
  const hipHalf = len(sub(rh, lh)) / 2;
  const torsoLen = len(torsoVec);
  if (shoulderHalf < 12 || torsoLen < 20) return null;

  const kneeMid = lk && rk ? mid(lk, rk) : null;
  const knots = [];

  if (category === "lower") {
    const waist = lerp(shoulderMid, hipMid, 0.72);
    const hem = kneeMid && ey.y > -0.2 ? kneeMid : add(hipMid, scale(ey, torsoLen * 0.75));
    knots.push({ c: add(waist, scale(ey, -shoulderHalf * 0.12)), hw: hipHalf * 1.02 });
    knots.push({ c: lerp(waist, hem, 0.5), hw: hipHalf * 0.95 });
    knots.push({ c: hem, hw: hipHalf * (kneeMid ? 0.72 : 0.85) });
  } else if (category === "dress") {
    const waist = lerp(shoulderMid, hipMid, 0.42);
    const hem = kneeMid && ey.y > -0.2 ? kneeMid : add(hipMid, scale(ey, torsoLen * 0.95));
    knots.push({ c: add(shoulderMid, scale(ey, -shoulderHalf * 0.28)), hw: shoulderHalf * 1.08 });
    knots.push({ c: waist, hw: Math.min(shoulderHalf, hipHalf) * 0.92 });
    knots.push({ c: hem, hw: hipHalf * 1.55 });
  } else {
    const hem = add(hipMid, scale(ey, shoulderHalf * 0.12));
    knots.push({ c: add(shoulderMid, scale(ey, -shoulderHalf * 0.22)), hw: shoulderHalf * 1.12 });
    knots.push({ c: mid(shoulderMid, hipMid), hw: (shoulderHalf + hipHalf) / 2 * 1.05 });
    knots.push({ c: hem, hw: hipHalf * 1.12 });
  }

  const leftPts = knots.map((k) => add(k.c, scale(ex, -k.hw)));
  const rightPts = knots.map((k) => add(k.c, scale(ex, k.hw)));
  return { leftPts, rightPts };
}

function sampleContour(pts, u) {
  const seg = u * (pts.length - 1);
  const i = Math.min(Math.floor(seg), pts.length - 2);
  const t = seg - i;
  return lerp(pts[i], pts[i + 1], t);
}

/* Warp the garment over the torso as thin vertical slices so it follows the body. */
function drawGarment(lm, asset) {
  const contours = garmentContours(lm, asset.category);
  if (!contours) return;
  const { leftPts, rightPts } = contours;

  const N = 36;
  const H = asset.h;
  overlayCtx.save();
  overlayCtx.globalAlpha = 0.96;
  overlayCtx.imageSmoothingEnabled = true;
  overlayCtx.imageSmoothingQuality = "high";
  for (let i = 0; i < N; i++) {
    const u0 = i / N;
    const u1 = (i + 1) / N;
    const tl = sampleContour(leftPts, u0);
    const tr = sampleContour(rightPts, u0);
    const bl = sampleContour(leftPts, u1);
    const sx0 = u0 * asset.w;
    const sx1 = u1 * asset.w;
    const a = scale(sub(tr, tl), 1 / (sx1 - sx0));
    const d = scale(sub(bl, tl), 1 / H);
    const e = { x: tl.x - a.x * sx0, y: tl.y - a.y * sx0 };
    overlayCtx.setTransform(a.x, a.y, d.x, d.y, e.x, e.y);
    overlayCtx.drawImage(asset.canvas, sx0, 0, sx1 - sx0, H, sx0, 0, sx1 - sx0, H);
  }
  overlayCtx.restore();
}

/* ---------- render loop ---------- */

function renderLoop() {
  requestAnimationFrame(renderLoop);
  if (!state.poseLandmarker || !video.videoWidth || video.readyState < 2) return;
  if (video.currentTime === state.lastVideoTime) return;
  state.lastVideoTime = video.currentTime;

  let result;
  try {
    result = state.poseLandmarker.detectForVideo(video, performance.now());
  } catch {
    return;
  }

  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  const item = state.selectedGarment;
  if (!item) return;
  const asset = state.garmentAssets[item.id];
  if (!asset) return;
  if (!result || !result.landmarks || !result.landmarks.length) return;

  const lm = result.landmarks[0].map((p) => ({ x: p.x * overlay.width, y: p.y * overlay.height }));
  drawGarment(lm, asset);
}

/* ---------- wardrobe selection / size ---------- */

function updateSizeBadge() {
  const item = state.selectedGarment;
  if (!item) {
    sizeBadge.classList.add("hidden");
    return;
  }
  if (!item.size_chart) {
    sizeBadge.textContent = "No size chart for this item";
    sizeBadge.classList.remove("hidden");
    return;
  }
  if (!state.measurements) {
    sizeBadge.textContent = "Tap “Measure me” for a size recommendation";
    sizeBadge.classList.remove("hidden");
    return;
  }
  const size = recommendSize(state.measurements, item.size_chart);
  sizeBadge.textContent = size ? `Recommended size: ${size}` : "Size recommendation unavailable";
  sizeBadge.classList.remove("hidden");
}

function selectGarment(item, card) {
  document.querySelectorAll(".garment-card.selected").forEach((c) => c.classList.remove("selected"));
  card.classList.add("selected");
  state.selectedGarment = item;
  btnSnap.disabled = false;
  btnClear.disabled = false;
  updateSizeBadge();
  showToast(`Wearing: ${item.name}`);
  loadGarmentAsset(item).catch(() => showToast("Could not load garment image"));
}

function clearDress() {
  state.selectedGarment = null;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  document.querySelectorAll(".garment-card.selected").forEach((c) => c.classList.remove("selected"));
  sizeBadge.classList.add("hidden");
  btnSnap.disabled = true;
  btnClear.disabled = true;
}

btnClear.addEventListener("click", clearDress);

/* ---------- fit profile ---------- */

function renderFitProfile(m) {
  const rows = [
    ["Height", `${m.height_cm} cm`],
    ["Shoulder width", `${m.shoulder_width_cm} cm`],
    ["Chest", `${m.chest_cm} cm`],
    ["Waist", `${m.waist_cm} cm`],
    ["Hips", `${m.hip_cm} cm`],
  ];
  const confClass =
    m.confidence === "high" ? "conf-high" : m.confidence === "medium" ? "conf-med" : "conf-low";
  fitProfileBody.innerHTML = `
    <div class="measure-grid">
      ${rows.map(([k, v]) => `<div class="measure-row"><span>${k}</span><strong>${v}</strong></div>`).join("")}
    </div>
    <div class="conf-line ${confClass}">Confidence: ${m.confidence}</div>
    ${m.notes.map((n) => `<p class="note">${n}</p>`).join("")}
  `;
  fitProfile.classList.remove("hidden");
}

async function runMeasure() {
  if (!state.poseLandmarker) return showToast("Pose model still loading — try again in a moment");
  if (!video.videoWidth) return showToast("Camera not ready yet");
  scanLine.classList.remove("hidden");
  try {
    const result = state.poseLandmarker.detectForVideo(video, performance.now());
    if (!result.worldLandmarks || !result.worldLandmarks.length) {
      return showToast("No person detected — stand facing the camera, full body in frame");
    }
    const world = result.worldLandmarks[0].map((p) => [p.x, p.y, p.z]);
    const lm2d = (result.landmarks[0] || []).map((p) => [p.x, p.y, p.visibility ?? 1.0]);

    const res = await fetch("/api/measure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ world_landmarks: world, landmarks: lm2d.length ? lm2d : null }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    state.measurements = await res.json();
    renderFitProfile(state.measurements);
    updateSizeBadge();
    showToast("Fit profile updated");
  } catch (err) {
    showToast(`Measurement failed: ${err.message}`);
  } finally {
    scanLine.classList.add("hidden");
  }
}

btnScan.addEventListener("click", runMeasure);

btnSnap.addEventListener("click", runSnap);

async function runSnap() {
  const item = state.selectedGarment;
  if (!item) return;
  if (!video.videoWidth) return showToast("Camera not ready yet");

  const frame = await captureFrame();
  if (!frame) return showToast("Could not capture frame");

  btnSnap.disabled = true;
  const original = btnSnap.textContent;
  btnSnap.textContent = "Generating…";
  showToast("Generating photorealistic try-on — this takes ~10–30s…", 30000);

  const form = new FormData();
  form.append("person", frame, "frame.jpg");
  form.append("garment_id", item.id);
  form.append("steps", "20");

  try {
    const res = await fetch("/api/tryon", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    snapResult.src = data.result_url;
    const size = state.measurements
      ? recommendSize(state.measurements, item.size_chart)
      : null;
    snapCaption.textContent = size
      ? `${item.name} — recommended size ${size}`
      : item.name;
    snapModal.classList.remove("hidden");
    showToast("AI try-on ready");
  } catch (err) {
    showToast(`Try-on failed: ${err.message}`, 6000);
  } finally {
    btnSnap.disabled = false;
    btnSnap.textContent = original;
    toast.classList.add("hidden");
  }
}

function captureFrame() {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  return new Promise((resolve) =>
    c.toBlob((b) => resolve(b), "image/jpeg", 0.92)
  );
}

snapClose.addEventListener("click", () => snapModal.classList.add("hidden"));
snapModal.addEventListener("click", (e) => {
  if (e.target === snapModal) snapModal.classList.add("hidden");
});

function recommendSize(measurements, sizeChart) {
  const target = measurements.chest_cm;
  let best = null;
  let bestDiff = Infinity;
  for (const row of sizeChart) {
    const midChest = (row.chest_min + row.chest_max) / 2;
    const diff = Math.abs(target - midChest);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = row.size;
    }
  }
  return best;
}

checkHealth();
loadWardrobe();
initPoseLandmarker();
startCamera();
renderLoop();
