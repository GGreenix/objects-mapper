"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const CLASS_COLORS = ["#fbbf24", "#38bdf8", "#c084fc", "#34d399", "#fb7185", "#a3e635"];
const NO_STREAM = "data:,";

let state = null;
let boot = null;
let fieldGeo = null;
let fieldVersion = -1;
let logSeq = 0;
let currentTab = "live";
let flip = store("flip") === "1";
let settingsDirty = false;
let confDragging = false;
let setupDismissed = null;
let hover = null;
let es = null;

// ------------------------------------------------------------------ helpers
function store(k, v) {
  try {
    if (v === undefined) return localStorage.getItem("fieldmapper." + k);
    localStorage.setItem("fieldmapper." + k, v);
  } catch { /* storage unavailable */ }
  return null;
}
const fmt = (n, d = 2) => (n === null || n === undefined || Number.isNaN(n) ? "–" : Number(n).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const mmss = (s) => { s = Math.max(0, Math.floor(s || 0)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
const getPath = (o, path) => path.split(".").reduce((a, k) => (a == null ? undefined : a[k]), o);
function setPath(o, path, v) {
  const keys = path.split(".");
  let cur = o;
  keys.slice(0, -1).forEach((k) => { cur = cur[k] = cur[k] || {}; });
  cur[keys[keys.length - 1]] = v;
}

async function api(path, opts = {}) {
  const init = { method: opts.method || (opts.body !== undefined ? "POST" : "GET"), headers: {} };
  if (opts.body !== undefined) {
    if (opts.raw) {
      init.body = opts.body;
      init.headers["Content-Type"] = "application/octet-stream";
    } else {
      init.body = JSON.stringify(opts.body);
      init.headers["Content-Type"] = "application/json";
    }
  }
  const r = await fetch(path, init);
  let data = null;
  try { data = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) throw new Error((data && (data.error || data.detail)) || `${r.status} ${r.statusText}`);
  return data;
}

function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), kind === "bad" ? 7000 : 3500);
}

async function act(name, body = {}) {
  try {
    return await api(`/api/action/${name}`, { body });
  } catch (e) {
    toast(e.message, "bad");
    return null;
  }
}

// --------------------------------------------------------------------- tabs
function showTab(name) {
  if (!["live", "calibrate", "settings"].includes(name)) name = "live";
  currentTab = name;
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("hidden", t.id !== `tab-${name}`));
  history.replaceState(null, "", `#${name}`);
  updateStreams(true);
  if (name === "settings" && !settingsDirty) loadSettings();
  if (name === "calibrate") prefillCalibration();
  if (name === "live") requestAnimationFrame(() => { layoutCanvas(); drawField(); });
  if (state) renderSetup(state);
}

function streamUrl() {
  return `/stream.mjpg?field=${+$("#ov-field").checked}&det=${+$("#ov-det").checked}&t=${Date.now()}`;
}
// Only one camera stream is open at a time (the one on the visible tab).
function updateStreams(force = false) {
  const live = $("#stream");
  const cal = $("#cal-stream");
  const want = currentTab === "live" ? live : currentTab === "calibrate" ? cal : null;
  for (const img of [live, cal]) {
    if (img === want) {
      if (force || !img.dataset.on) { img.src = streamUrl(); img.dataset.on = "1"; }
    } else if (img.dataset.on) {
      img.src = NO_STREAM;
      delete img.dataset.on;
    }
  }
}
for (const img of [$("#stream"), $("#cal-stream")]) {
  img.addEventListener("error", () => {
    if (!img.dataset.on) return;
    setTimeout(() => { if (img.dataset.on) img.src = streamUrl(); }, 1500);
  });
}

// --------------------------------------------------------------- live state
function connect() {
  if (es) es.close();
  es = new EventSource(`/api/events?log_after=${logSeq}`);
  es.onmessage = (ev) => {
    const st = JSON.parse(ev.data);
    if ($("#offline").classList.contains("hidden") === false) {
      $("#offline").classList.add("hidden");
      updateStreams(true);
    }
    onState(st);
  };
  es.onerror = () => {
    $("#offline").classList.remove("hidden");
    es.close();
    setTimeout(connect, 1500);
  };
}

function onState(st) {
  if (boot !== null && st.boot !== boot) { // app restarted: start the log afresh
    logSeq = 0;
    $("#log").textContent = "";
    boot = st.boot;
    connect();
    return;
  }
  boot = st.boot;
  state = st;
  renderChips(st);
  renderSetup(st);
  if (st.field.version !== fieldVersion) loadField(st.field.version);
  if (currentTab === "live") {
    renderLive(st);
    drawField();
  }
  renderCalibrate(st);
  appendLogs(st.logs);
}

function chipHtml(label, cls, tab, title) {
  return `<span class="chip ${cls}" ${tab ? `data-tab="${tab}"` : ""} title="${esc(title || "")}">${esc(label)}</span>`;
}
function renderChips(st) {
  const c = st.camera, cal = st.calibration, m = st.model, p = st.pose, nt = st.nt;
  const out = [];
  if (c.status === "ok") out.push(chipHtml(c.type === "video" ? `Replay · ${c.name}` : `Camera · ${fmt(c.fps, 0)} fps`, "ok"));
  else if (c.status === "error") out.push(chipHtml("Camera problem", "bad", "settings", c.error));
  else out.push(chipHtml("Camera starting", "warn pulse", "settings"));

  if (cal.ok) out.push(chipHtml("Calibrated", "ok", "calibrate", cal.rms_px != null ? `RMS ${fmt(cal.rms_px)} px` : ""));
  else out.push(chipHtml(cal.error === "not calibrated" ? "Not calibrated" : "Recalibrate", "bad", "calibrate", cal.error));

  if (m.loading) out.push(chipHtml("Loading model", "warn pulse", "settings"));
  else if (m.ok) out.push(chipHtml(`Model · ${m.device === "cpu" ? "CPU" : "GPU"}`, m.device === "cpu" ? "warn" : "ok", "settings",
    m.device === "cpu" ? "Running on CPU - install CUDA PyTorch for speed" : m.path));
  else out.push(chipHtml(m.error && !m.error.startsWith("no model file") ? "Model error" : "No model", "bad", "settings", m.error));

  if (st.calibrating.active) out.push(chipHtml("Calibrating", "warn", "calibrate"));
  else if (p.valid) out.push(chipHtml(`Pose · ${(st.used_tags || []).length || (p.last_solve?.tags.length ?? 0)} tags`, "ok"));
  else out.push(chipHtml("No camera pose", "warn", null, p.last_solve?.reason));

  if (!nt.enabled) out.push(chipHtml("NT off", "", "settings"));
  else if (nt.connected) out.push(chipHtml("Robot connected", "ok"));
  else out.push(chipHtml("Robot not connected", "warn", "settings", nt.team ? `Looking for team ${nt.team}` : "Set the team number"));
  $("#chips").innerHTML = out.join("");
}
$("#chips").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip[data-tab]");
  if (chip) showTab(chip.dataset.tab);
});

// ------------------------------------------------------------------- setup
function setupSteps(st) {
  const cam = st.camera, cal = st.calibration, m = st.model;
  return [
    {
      key: "team", title: "Team number", done: st.nt.team > 0 || !!st.nt.server || !st.nt.enabled,
      desc: "So the app can find your robot.",
      action: ["Set team number", () => { showTab("settings"); setTimeout(() => $("#set-team").focus(), 150); }],
    },
    {
      key: "camera", title: "Camera", done: cam.status === "ok",
      desc: cam.error ? `Problem: ${cam.error}` : "Plug in the USB camera.",
      action: ["Choose camera", () => { showTab("settings"); setTimeout(() => $("#find-cameras").click(), 150); }],
    },
    {
      key: "calib", title: "Calibrate", done: cal.ok,
      desc: cal.error && cal.error !== "not calibrated" ? cal.error : "Measure the lens with a printed checkerboard (about 3 minutes).",
      action: ["Calibrate", () => showTab("calibrate")],
    },
    {
      key: "model", title: "Detection model", done: m.ok,
      desc: m.loading ? "Loading..." : (m.error && !m.error.startsWith("no model file") ? m.error : "Add your trained YOLO model (.pt file)."),
      action: m.loading ? null : ["Add model", () => { showTab("settings"); setTimeout(() => $("#set-model").focus(), 150); }],
    },
    {
      key: "tags", title: "See the tags", done: st.pose.valid,
      desc: "Aim the camera so at least 2 AprilTags are visible.", action: null,
    },
  ];
}
function renderSetup(st) {
  const steps = setupSteps(st);
  const missing = steps.filter((s) => !s.done).map((s) => s.key).join(",");
  const hide = !missing || setupDismissed === missing || st.calibrating.active;
  $("#setup").classList.toggle("hidden", hide);
  if (hide) return;
  const first = steps.findIndex((s) => !s.done);
  const ol = $("#setup-steps");
  const sig = JSON.stringify(steps.map((s) => [s.done, s.desc, !!s.action])) + first;
  if (ol.dataset.sig === sig) return;
  ol.dataset.sig = sig;
  ol.innerHTML = "";
  steps.forEach((s, i) => {
    const li = document.createElement("li");
    li.className = s.done ? "done" : i === first ? "current" : "";
    li.innerHTML = `<span class="num">${s.done ? "✓" : i + 1}</span><div><div class="t">${esc(s.title)}</div><div class="d">${esc(s.desc)}</div></div>`;
    if (!s.done && s.action) {
      const b = document.createElement("button");
      b.className = `btn small ${i === first ? "primary" : ""}`;
      b.textContent = s.action[0];
      b.onclick = s.action[1];
      li.lastChild.appendChild(b);
    }
    ol.appendChild(li);
  });
  if (!$("#setup-dismiss")) {
    const d = document.createElement("button");
    d.id = "setup-dismiss";
    d.className = "btn small ghost";
    d.textContent = "Hide";
    d.onclick = () => { setupDismissed = setupSteps(state).filter((s) => !s.done).map((s) => s.key).join(","); renderSetup(state); };
    $(".setup-head").appendChild(d);
  }
}

// -------------------------------------------------------------------- live
function renderLive(st) {
  // pieces
  const rows = st.pieces.map((p) => {
    const color = CLASS_COLORS[p.class_id % CLASS_COLORS.length];
    return `<tr class="${p.seen ? "" : "coast"}"><td>${p.id}</td><td><span class="swatch-cls" style="background:${color}"></span>${esc(p.class)}</td>` +
      `<td class="num">${fmt(p.x)}</td><td class="num">${fmt(p.y)}</td><td class="num">${fmt(p.confidence)}</td></tr>`;
  });
  $("#pieces").innerHTML = rows.join("");
  $("#pieces-empty").classList.toggle("hidden", rows.length > 0);
  $("#pieces-empty").textContent = st.mapping ? "No pieces on the field right now." : "Not mapping yet - finish setup above.";
  $("#piece-count").textContent = st.pieces.length;

  // pose
  const p = st.pose;
  const pill = $("#pose-pill");
  pill.textContent = p.valid ? "Valid" : st.calibrating.active ? "Calibrating" : "Not found";
  pill.className = `pill ${p.valid ? "ok" : "warn"}`;
  const ls = p.last_solve;
  const has = p.x !== undefined;
  const kv = [
    ["Position x, y, z", has ? `${fmt(p.x)}, ${fmt(p.y)}, ${fmt(p.z)} m` : "–"],
    ["Roll, pitch, yaw", has ? `${fmt(p.roll_deg, 1)}°, ${fmt(p.pitch_deg, 1)}°, ${fmt(p.yaw_deg, 1)}°` : "–"],
    ["Last check", ls ? `${fmt(ls.age_s, 1)} s ago` : "–"],
    ["Tags, error", ls ? `${ls.tags.length || ls.seen.length}${ls.error_px != null ? `, ${fmt(ls.error_px)} px` : ""}` : "–"],
    ["Result", ls ? `<span class="${ls.accepted ? "" : "muted"}">${esc(ls.reason)}</span>` : "–", true],
    ["Next check", p.next_solve_in_s != null ? `in ${fmt(p.next_solve_in_s, 1)} s` : "–"],
  ];
  $("#pose").innerHTML = kv.map(([k, v, text]) => `<dt>${k}</dt><dd class="${text ? "text" : ""}">${text ? v : esc(v)}</dd>`).join("");
  $("#btn-relocalize").disabled = !st.mapping;

  // controls
  if (!confDragging) {
    $("#conf").value = st.model.confidence;
    $("#conf-val").textContent = fmt(st.model.confidence);
  }
  const rec = $("#btn-record");
  const isVideo = st.camera.type === "video";
  rec.disabled = isVideo || st.camera.status !== "ok";
  rec.title = isVideo ? "Recording is only available with a live camera" : "Save the raw camera feed to the recordings folder";
  rec.classList.toggle("recording", st.recording.active);
  rec.textContent = st.recording.active ? `■ Stop  ${mmss(st.recording.seconds)}` : "● Record";
  $("#btn-reset").disabled = !st.mapping;

  const t = st.timing || {};
  $("#perf").innerHTML = [
    ["Frame rate", `${fmt(st.camera.fps, 1)} fps`],
    ["Detection", t.detect_ms != null ? `${fmt(t.detect_ms, 1)} ms` : "–"],
    ["Whole frame", t.step_ms != null ? `${fmt(t.step_ms, 1)} ms` : "–"],
    ["Tag solve", t.localize_ms ? `${fmt(t.localize_ms, 0)} ms` : "–"],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

  // replay
  const bar = $("#replay-bar");
  bar.classList.toggle("hidden", !isVideo);
  if (isVideo) {
    $("#replay-toggle").textContent = st.camera.ended ? "Finished" : st.camera.paused ? "Resume" : "Pause";
    $("#replay-toggle").disabled = !!st.camera.ended;
    const pct = st.camera.duration_s ? (100 * st.camera.position_s) / st.camera.duration_s : 0;
    $("#replay-progress").style.width = `${Math.min(100, pct)}%`;
    $("#replay-label").textContent = `${st.camera.name} · ${mmss(st.camera.position_s)}${st.camera.duration_s ? " / " + mmss(st.camera.duration_s) : ""}`;
  }
}

$("#ov-field").addEventListener("change", () => updateStreams(true));
$("#ov-det").addEventListener("change", () => updateStreams(true));
$("#btn-relocalize").onclick = () => act("relocalize");
$("#btn-reset").onclick = () => act("reset_tracker");
$("#btn-record").onclick = async () => {
  if (state?.recording.active) {
    await act("record_stop");
    toast("Recording saved to the recordings folder", "ok");
  } else {
    const r = await act("record_start");
    if (r) toast(`Recording to recordings/${r.result}`, "ok");
  }
};
$("#replay-toggle").onclick = () => act(state?.camera.paused ? "resume" : "pause");
$("#replay-restart").onclick = () => act("restart_replay");
const conf = $("#conf");
conf.addEventListener("input", () => { confDragging = true; $("#conf-val").textContent = fmt(+conf.value); });
conf.addEventListener("change", async () => { await act("confidence", { value: +conf.value }); confDragging = false; });
$("#log-clear").onclick = () => { $("#log").textContent = ""; };

// ------------------------------------------------------------------- field
const canvas = $("#field");
const ctx = canvas.getContext("2d");
let view = null;

async function loadField(version) {
  fieldVersion = version;
  try {
    const f = await api("/api/field");
    fieldGeo = f.ok ? f : null;
  } catch { fieldGeo = null; }
  layoutCanvas();
  drawField();
}

function layoutCanvas() {
  const L = fieldGeo ? fieldGeo.length : 16.54;
  const W = fieldGeo ? fieldGeo.width : 8.07;
  const m = 1.0;
  const cssW = canvas.parentElement.clientWidth || 600;
  const s = cssW / (L + 2 * m);
  const cssH = (W + 2 * m) * s;
  const dpr = window.devicePixelRatio || 1;
  canvas.style.height = `${cssH}px`;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  view = { L, W, m, s, cssW, cssH, dpr };
  $("#flip").textContent = flip ? "View from blue" : "View from red";
}
function toPx(x, y) {
  if (flip) { x = view.L - x; y = view.W - y; }
  return [(x + view.m) * view.s, (view.W + view.m - y) * view.s];
}
function fromPx(px, py) {
  let x = px / view.s - view.m;
  let y = view.W + view.m - py / view.s;
  if (flip) { x = view.L - x; y = view.W - y; }
  return [x, y];
}
function line(x1, y1, x2, y2) {
  const a = toPx(x1, y1), b = toPx(x2, y2);
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
}

function drawField() {
  if (!view) layoutCanvas();
  const { L, W, s, cssW, cssH, dpr } = view;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  // carpet + grid
  const [ax, ay] = toPx(flip ? L : 0, flip ? 0 : W);
  ctx.fillStyle = "#15211a";
  ctx.fillRect(ax, ay, L * s, W * s);
  ctx.strokeStyle = "#1f3127"; ctx.lineWidth = 1;
  for (let x = 1; x < L; x++) line(x, 0, x, W);
  for (let y = 1; y < W; y++) line(0, y, L, y);
  ctx.strokeStyle = "#566273"; ctx.setLineDash([6, 6]); line(L / 2, 0, L / 2, W); ctx.setLineDash([]);
  ctx.strokeStyle = "#8b97a8"; ctx.lineWidth = 1.5; ctx.strokeRect(ax, ay, L * s, W * s);
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#3b82f6"; line(0, 0, 0, W);
  ctx.strokeStyle = "#ef4444"; line(L, 0, L, W);

  const st = state;
  if (!fieldGeo) {
    ctx.fillStyle = "#8b97a8"; ctx.font = "14px system-ui"; ctx.textAlign = "center";
    ctx.fillText(st?.field.error ? `Field layout error: ${st.field.error}` : "Loading field...", cssW / 2, cssH / 2);
    return;
  }

  // camera footprint
  if (st?.footprint?.length > 2) {
    ctx.beginPath();
    st.footprint.forEach(([x, y], i) => { const [px, py] = toPx(x, y); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
    ctx.closePath();
    ctx.fillStyle = "rgba(56,189,248,0.09)"; ctx.fill();
    ctx.strokeStyle = "rgba(56,189,248,0.45)"; ctx.lineWidth = 1; ctx.stroke();
  }

  // tags
  const used = new Set(st?.used_tags || []);
  ctx.font = "10px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  for (const t of fieldGeo.tags) {
    const nx = Math.cos(t.yaw), ny = Math.sin(t.yaw);
    const sx = -ny * 0.16, sy = nx * 0.16;
    ctx.strokeStyle = used.has(t.id) ? "#22c55e" : "#d6dde6";
    ctx.lineWidth = used.has(t.id) ? 4 : 2;
    line(t.x - sx, t.y - sy, t.x + sx, t.y + sy);
    if (used.has(t.id)) {
      const [cx, cy] = toPx(t.x, t.y);
      ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI * 2); ctx.strokeStyle = "rgba(34,197,94,.6)"; ctx.lineWidth = 1.5; ctx.stroke();
    }
    const [lx, ly] = toPx(t.x + nx * 0.42, t.y + ny * 0.42);
    ctx.fillStyle = "#7d8899"; ctx.fillText(String(t.id), lx, ly);
  }

  // camera
  const p = st?.pose;
  if (p && p.x !== undefined) {
    let [cx, cy] = toPx(p.x, p.y);
    cx = Math.min(Math.max(cx, 8), cssW - 8); cy = Math.min(Math.max(cy, 8), cssH - 8);
    const yaw = (p.yaw_deg * Math.PI) / 180 + (flip ? Math.PI : 0);
    const col = p.valid ? "#22c55e" : "#f59e0b";
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.5;
    const tx = cx + Math.cos(yaw) * 28, ty = cy - Math.sin(yaw) * 28;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(tx, ty); ctx.stroke();
    const ah = 8;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - ah * Math.cos(yaw - 0.45), ty + ah * Math.sin(yaw - 0.45));
    ctx.lineTo(tx - ah * Math.cos(yaw + 0.45), ty + ah * Math.sin(yaw + 0.45));
    ctx.closePath(); ctx.fill();
    ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.fill();
  }

  // raw detections
  ctx.fillStyle = "#6b7280";
  for (const [x, y] of st?.raw || []) {
    const [px, py] = toPx(x, y);
    ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2); ctx.fill();
  }

  // tracked pieces
  const r = Math.max(5, 0.09 * s);
  ctx.font = "600 11px system-ui"; ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  for (const pc of st?.pieces || []) {
    const [px, py] = toPx(pc.x, pc.y);
    const col = CLASS_COLORS[pc.class_id % CLASS_COLORS.length];
    ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2);
    if (pc.seen) { ctx.fillStyle = col; ctx.fill(); } else { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke(); }
    if (hover && hover.piece === pc.id) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(px, py, r + 3, 0, Math.PI * 2); ctx.stroke(); }
    ctx.fillStyle = col; ctx.fillText(String(pc.id), px + r + 2, py - r + 2);
  }
}

canvas.addEventListener("mousemove", (e) => {
  if (!view) return;
  const rect = canvas.getBoundingClientRect();
  const [x, y] = fromPx(e.clientX - rect.left, e.clientY - rect.top);
  let near = null, best = 0.4;
  for (const pc of state?.pieces || []) {
    const d = Math.hypot(pc.x - x, pc.y - y);
    if (d < best) { best = d; near = pc; }
  }
  hover = { x, y, piece: near?.id };
  $("#cursor").textContent = near ? `piece #${near.id}: x ${fmt(near.x)}  y ${fmt(near.y)} m` : `x ${fmt(x)}  y ${fmt(y)} m`;
  drawField();
});
canvas.addEventListener("mouseleave", () => { hover = null; $("#cursor").textContent = ""; drawField(); });
$("#flip").onclick = () => { flip = !flip; store("flip", flip ? "1" : "0"); layoutCanvas(); drawField(); };
window.addEventListener("resize", () => { layoutCanvas(); drawField(); });

// --------------------------------------------------------------------- log
function appendLogs(logs) {
  if (!logs || !logs.length) return;
  const box = $("#log");
  const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
  for (const r of logs) {
    if (r.seq <= logSeq) continue;
    logSeq = r.seq;
    const d = document.createElement("div");
    d.className = r.level;
    const ts = new Date(r.t * 1000).toLocaleTimeString([], { hour12: false });
    d.innerHTML = `<span class="ts">${ts}</span>${r.level !== "INFO" ? `<b>${esc(r.level)}</b> ` : ""}${esc(r.msg)}`;
    box.appendChild(d);
  }
  while (box.childNodes.length > 300) box.firstChild.remove();
  if (atBottom) box.scrollTop = box.scrollHeight;
}

// --------------------------------------------------------------- calibrate
async function prefillCalibration() {
  if (state?.calibrating.active || $("#cal-cols").value) return;
  try {
    const s = await api("/api/settings");
    const c = s.values.calibration;
    $("#cal-cols").value = c.board_cols;
    $("#cal-rows").value = c.board_rows;
    $("#cal-square").value = +(c.square_size_m * 1000).toFixed(2);
    $("#cal-min").value = c.min_frames;
    updateBoardLink();
  } catch (e) { toast(e.message, "bad"); }
}
function updateBoardLink() {
  $("#print-board").href = `/board?cols=${$("#cal-cols").value || 9}&rows=${$("#cal-rows").value || 6}&square_mm=${$("#cal-square").value || 25}`;
}
["#cal-cols", "#cal-rows", "#cal-square"].forEach((s) => $(s).addEventListener("input", updateBoardLink));

function renderCalibrate(st) {
  const c = st.calibrating;
  $("#cal-idle").classList.toggle("hidden", c.active);
  $("#cal-active").classList.toggle("hidden", !c.active);
  const pill = $("#cal-board");
  if (!c.active) {
    pill.textContent = st.calibration.ok ? `Calibrated · ${fmt(st.calibration.rms_px)} px` : "Not calibrated";
    pill.className = `pill ${st.calibration.ok ? "ok" : "bad"}`;
    return;
  }
  pill.textContent = c.board_seen ? (c.still ? "Board: hold still ✓" : "Board: seen") : "No board";
  pill.className = `pill ${c.board_seen ? (c.still ? "ok" : "warn") : "bad"}`;
  $("#cal-bar").style.width = `${Math.min(100, (100 * c.views) / c.min_views)}%`;
  $("#cal-count").textContent = `${c.views} / ${c.min_views} views`;
  $("#cal-msg").textContent = c.message;
  $("#cal-auto").textContent = `Auto-capture: ${c.auto ? "on" : "off"}`;
  $("#cal-auto").classList.toggle("primary", c.auto);
  const solve = $("#cal-solve");
  solve.disabled = c.solving || c.views < c.min_views;
  solve.textContent = c.solving ? "Calibrating..." : "Calibrate & save";
  const res = $("#cal-result");
  if (c.result) {
    const r = c.result;
    const good = r.quality === "excellent" || r.quality === "good";
    res.className = `result ${good ? "good" : r.quality === "poor" ? "poor" : ""}`;
    res.innerHTML = `<h4>${good ? "✓ " : ""}Calibration ${esc(r.quality)} - ${fmt(r.rms_px)} px error</h4>` +
      `<dl class="kv"><dt>Field of view</dt><dd>${fmt(r.hfov_deg, 1)}° × ${fmt(r.vfov_deg, 1)}°</dd>` +
      `<dt>Views used</dt><dd>${r.views_used}${r.views_dropped ? ` (${r.views_dropped} dropped)` : ""}</dd>` +
      `<dt>Resolution</dt><dd>${r.width} × ${r.height}</dd></dl>` +
      (good ? "<p class='hint'>Saved. Press Done to start mapping.</p>" :
        "<p class='hint'>Saved, but the error is high. Check the square size, hold the board still and flat, then Clear and try again.</p>");
  } else {
    res.className = "result hidden";
  }
}
$("#cal-start").onclick = async () => {
  try {
    await api("/api/calibration/start", { body: {
      cols: +$("#cal-cols").value, rows: +$("#cal-rows").value, square_mm: +$("#cal-square").value, min_views: +$("#cal-min").value,
    } });
  } catch (e) { toast(e.message, "bad"); }
};
const calCmd = (name, body) => api(`/api/calibration/${name}`, { body: body || {} }).catch((e) => toast(e.message, "bad"));
$("#cal-capture").onclick = () => calCmd("capture");
$("#cal-auto").onclick = () => calCmd("auto", { on: !state?.calibrating.auto });
$("#cal-undo").onclick = () => calCmd("undo");
$("#cal-clear").onclick = () => calCmd("clear");
$("#cal-solve").onclick = () => calCmd("solve");
$("#cal-done").onclick = async () => { await calCmd("stop"); showTab("live"); };
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || currentTab !== "calibrate" || !state?.calibrating.active) return;
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(document.activeElement?.tagName)) return;
  e.preventDefault();
  calCmd("capture");
});

// ---------------------------------------------------------------- settings
const form = $("#settings-form");

function fillSelect(sel, options, current, emptyLabel) {
  const opts = options.map((o) => (Array.isArray(o) ? o : [o, o]));
  if (current && !opts.some(([v]) => v === current)) opts.unshift([current, `${current} (missing)`]);
  sel.innerHTML = opts.length ? opts.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("")
    : `<option value="">${esc(emptyLabel)}</option>`;
  sel.value = current ?? "";
}

async function loadSettings() {
  let s;
  try { s = await api("/api/settings"); } catch (e) { toast(e.message, "bad"); return; }
  const v = s.values, o = s.options;
  fillSelect($("#set-res"), o.resolutions.map(([w, h]) => [`${w}x${h}`, `${w} × ${h}`]), `${v.camera.width}x${v.camera.height}`);
  fillSelect($("#set-layout"), o.layouts, v.field.layout);
  fillSelect($("#set-model"), o.models, v.detector.model_path, "No models yet - upload one");
  fillSelect($("#set-video"), o.recordings, v.source.video_path, "No recordings yet");
  for (const el of $$("[data-key]", form)) {
    const key = el.dataset.key, type = el.dataset.type;
    if (type === "resolution" || el.tagName === "SELECT" && ["set-layout", "set-model", "set-video", "set-res"].includes(el.id)) continue;
    const val = getPath(v, key);
    if (type === "radio") el.checked = val === el.value;
    else if (type === "bool") el.checked = !!val;
    else if (type === "host") el.checked = !["127.0.0.1", "localhost"].includes(val);
    else if (type === "list") el.value = (val || []).join(", ");
    else el.value = val ?? "";
  }
  updateSourceVisibility();
  settingsDirty = false;
  $("#settings-note").textContent = s.cli_overrides.length
    ? `Some settings were set on the command line for this run: ${s.cli_overrides.join(", ")}`
    : "";
}

function gatherSettings() {
  const out = {};
  for (const el of $$("[data-key]", form)) {
    const key = el.dataset.key, type = el.dataset.type;
    let val;
    switch (type) {
      case "radio": if (!el.checked) continue; val = el.value; break;
      case "bool": val = el.checked; break;
      case "host": val = el.checked ? "0.0.0.0" : "127.0.0.1"; break;
      case "int": val = parseInt(el.value, 10); if (Number.isNaN(val)) continue; break;
      case "float": val = parseFloat(el.value); if (Number.isNaN(val)) continue; break;
      case "list": val = el.value.split(",").map((x) => x.trim()).filter(Boolean); break;
      case "resolution": {
        const [w, h] = el.value.split("x").map(Number);
        if (w && h) { setPath(out, "camera.width", w); setPath(out, "camera.height", h); }
        continue;
      }
      default: val = el.value;
    }
    setPath(out, key, val);
  }
  return out;
}

function updateSourceVisibility() {
  const video = $('input[name="source"][value="video"]').checked;
  $("#src-camera").classList.toggle("hidden", video);
  $("#src-video").classList.toggle("hidden", !video);
}

form.addEventListener("input", () => { settingsDirty = true; $("#settings-note").textContent = "Unsaved changes"; });
form.addEventListener("change", (e) => { if (e.target.name === "source") updateSourceVisibility(); });
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("#settings-save");
  btn.disabled = true;
  try {
    await api("/api/settings", { body: gatherSettings() });
    toast("Saved - applying", "ok");
    settingsDirty = false;
    await loadSettings();
  } catch (err) {
    toast(err.message, "bad");
  } finally {
    btn.disabled = false;
  }
});
$("#settings-reset").onclick = () => loadSettings();

$("#find-cameras").onclick = async () => {
  const btn = $("#find-cameras");
  const list = $("#camera-list");
  btn.disabled = true;
  btn.textContent = "Searching...";
  list.innerHTML = "";
  try {
    const cams = await api("/api/cameras");
    if (!cams.length) {
      list.innerHTML = `<p class="hint">No cameras found. Check the USB cable, then try again.</p>`;
    }
    const current = +$("#set-cam-index").value;
    for (const c of cams) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = `cam-option ${c.index === current ? "selected" : ""}`;
      b.innerHTML = `${c.thumbnail ? `<img src="${c.thumbnail}" alt="">` : "<img alt=''>"}` +
        `<span>Camera ${c.index} · ${c.width}×${c.height}${c.active ? " (in use)" : ""}</span>`;
      b.onclick = () => {
        $("#set-cam-index").value = c.index;
        $$(".cam-option").forEach((x) => x.classList.toggle("selected", x === b));
        settingsDirty = true;
        $("#settings-note").textContent = "Unsaved changes - press Save & apply";
      };
      list.appendChild(b);
    }
  } catch (e) {
    toast(e.message, "bad");
  } finally {
    btn.disabled = false;
    btn.textContent = "Find cameras";
  }
};

$("#model-upload").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  toast(`Uploading ${file.name}...`);
  try {
    const r = await api(`/api/models/${encodeURIComponent(file.name)}`, { method: "PUT", body: file, raw: true });
    await api("/api/settings", { body: { detector: { model_path: r.path } } });
    toast(`Model ${file.name} uploaded - loading it now`, "ok");
    await loadSettings();
  } catch (err) {
    toast(err.message, "bad");
  } finally {
    e.target.value = "";
  }
});

// -------------------------------------------------------------------- init
$$(".tabs button").forEach((b) => (b.onclick = () => showTab(b.dataset.tab)));
window.addEventListener("hashchange", () => {
  const tab = location.hash.slice(1);
  if (tab && tab !== currentTab) showTab(tab);
});
showTab(location.hash.slice(1) || "live");
layoutCanvas();
drawField();
connect();
