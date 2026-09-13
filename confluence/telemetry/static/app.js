const DRUGS = [
  ["anti_pd1", "anti-PD1"],
  ["tgfb_inhibitor", "TGF-βi"],
  ["mct1", "MCT1"],
  ["hdac", "HDAC"],
  ["targeted_kinase", "kinase"],
];
const PROTEINS = [
  ["protein_anti_pd1", "αPD-1 Ab"],
  ["protein_tgfb_trap", "TGF-β trap"],
  ["protein_ifng", "IFN-γ"],
  ["protein_il2", "IL-2"],
];
const FUSIONS = [
  ["tki_imatinib_like", "imatinib-like"],
  ["tki_alk", "ALK TKI"],
];
const EFFECTORS = DRUGS.concat(PROTEINS).concat(FUSIONS);

const MAX_POINTS = 180;
const wsProto = location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${wsProto}://${location.host}/ws/sim`);

const $ = (id) => document.getElementById(id);
const sliderBox = $("sliders");
const bars = $("drug-bars");
const proteinBars = $("protein-bars");
const fusionBars = $("fusion-bars");
const manual = {};
let lastEmb = null;
let lastConn = {};
let mujocoImg = null;
let meshPhase = "loading"; // loading | ready | missing

function addSliderAndBar(id, label, barParent, protein) {
  manual[id] = 0;
  const wrap = document.createElement("label");
  wrap.className = "slider";
  wrap.innerHTML = `<span>${label}<b id="sv-${id}">0.00</b></span>
    <input type="range" min="0" max="1" step="0.01" value="0" data-drug="${id}" />`;
  sliderBox.appendChild(wrap);
  wrap.querySelector("input").addEventListener("input", (ev) => {
    manual[id] = Number(ev.target.value);
    $(`sv-${id}`).textContent = manual[id].toFixed(2);
    sendOverride();
  });
  const row = document.createElement("div");
  row.className = protein ? "bar protein" : "bar";
  row.innerHTML = `<span>${label}</span><div class="track"><div class="fill" id="fill-${id}"></div></div><span id="c-${id}">0.00</span>`;
  barParent.appendChild(row);
}

DRUGS.forEach(([id, label]) => addSliderAndBar(id, label, bars, false));
PROTEINS.forEach(([id, label]) => addSliderAndBar(id, label, proteinBars, true));
if (fusionBars) FUSIONS.forEach(([id, label]) => addSliderAndBar(id, label, fusionBars, false));

function send(cmd, extra = {}) {
  if (ws.readyState === 1) ws.send(JSON.stringify({ cmd, ...extra }));
}
function sendOverride() {
  send("override", { enabled: $("override").checked, U: { ...manual } });
}

function makeTrace(label, color, dash) {
  return { label, color, dash: dash || [], ys: [] };
}

const traces = {
  cancer: [
    makeTrace("obs burden", "#e06b5c"),
    makeTrace("latent burden", "#f0b7a4", [5, 4]),
    makeTrace("obs resist", "#d4a054"),
    makeTrace("obs lactate", "#3ecfc0"),
    makeTrace("obs TGF-β", "#7aa2d4"),
    makeTrace("immune ratio", "#8fbc8f"),
    makeTrace("host H", "#e8e0d4", [2, 3]),
    makeTrace("fusion AF", "#c084fc"),
    makeTrace("junction neoAg", "#f0b7a4", [3, 3]),
  ],
  mbon: [
    makeTrace("MBON mean", "#d4a054"),
    makeTrace("DA", "#e06b5c"),
  ],
  drugs: EFFECTORS.map(([id], i) => {
    const colors = ["#e06b5c", "#d4a054", "#3ecfc0", "#7aa2d4", "#c084fc", "#f0b7a4", "#8fbc8f", "#e8e0d4", "#7aa2d4"];
    return makeTrace(`C ${id}`, colors[i % colors.length]);
  }),
  emb: [
    makeTrace("action RMS", "#d4a054"),
    makeTrace("reward", "#3ecfc0"),
  ],
};
const ts = [];

$("legend-cancer").innerHTML = traces.cancer
  .map((tr) => `<span><i style="background:${tr.color}"></i>${tr.label}</span>`)
  .join("");

function drawChart(canvasId, group, yMaxHint) {
  const canvas = $(canvasId);
  if (!canvas) return;
  const parent = canvas.parentElement;
  const w = Math.max(200, parent.clientWidth - 8);
  const h = Number(canvas.getAttribute("height")) || 180;
  if (canvas.width !== w) canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0c0c0e";
  ctx.fillRect(0, 0, w, h);
  const pad = { l: 36, r: 8, t: 8, b: 20 };
  const iw = w - pad.l - pad.r;
  const ih = h - pad.t - pad.b;
  const ymax = Math.max(yMaxHint, ...group.flatMap((tr) => tr.ys), 0.1);
  ctx.strokeStyle = "#1b1b22";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (ih * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
  }
  ctx.fillStyle = "#7a7468";
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText(ymax.toFixed(2), 4, pad.t + 10);
  ctx.fillText("0", 4, pad.t + ih);
  if (ts.length > 1) {
    ctx.fillText(ts[0].toFixed(1), pad.l, h - 4);
    ctx.fillText(ts[ts.length - 1].toFixed(1), w - 40, h - 4);
  }
  group.forEach((tr) => {
    if (tr.ys.length < 2) return;
    ctx.strokeStyle = tr.color;
    ctx.lineWidth = 1.6;
    ctx.setLineDash(tr.dash.length ? tr.dash : []);
    ctx.beginPath();
    tr.ys.forEach((y, i) => {
      const x = pad.l + (iw * i) / Math.max(1, tr.ys.length - 1);
      const py = pad.t + ih * (1 - y / ymax);
      if (i === 0) ctx.moveTo(x, py);
      else ctx.lineTo(x, py);
    });
    ctx.stroke();
  });
  ctx.setLineDash([]);
}

function paintHeat(rates) {
  const canvas = $("kc-heat");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#0c0c0e";
  ctx.fillRect(0, 0, w, h);
  const n = rates.length || 1;
  const cw = w / n;
  rates.forEach((v, i) => {
    const t = Math.min(1, Math.max(0, v) * 1.4);
    ctx.fillStyle = `rgb(${Math.round(28 + 210 * t)},${Math.round(18 + 130 * t)},${Math.round(12 + 28 * t)})`;
    ctx.fillRect(i * cw, 0, cw + 0.5, h);
  });
}

function redraw() {
  drawChart("chart-cancer", traces.cancer, 1.2);
  drawChart("chart-mbon", traces.mbon, 0.4);
  drawChart("chart-drugs", traces.drugs, 0.4);
  if ($("chart-emb")) drawChart("chart-emb", traces.emb, 0.3);
  drawHero(lastEmb, lastConn);
}

function drawHero(emb, conn) {
  const canvas = $("hero");
  if (!canvas) return;
  const w = window.innerWidth;
  const h = window.innerHeight;
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#050506";
  ctx.fillRect(0, 0, w, h);

  if (mujocoImg && mujocoImg.complete && mujocoImg.naturalWidth) {
    const iw = mujocoImg.naturalWidth;
    const ih = mujocoImg.naturalHeight;
    const scale = Math.max(w / iw, h / ih);
    ctx.drawImage(mujocoImg, (w - iw * scale) / 2, (h - ih * scale) / 2, iw * scale, ih * scale);
    return;
  }

  // Never draw the CPG / bead fly as the product.
  ctx.fillStyle = "#0a0c10";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#d4a054";
  ctx.font = "600 13px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  if (meshPhase === "missing") {
    ctx.fillText("FLYBODY MESH REQUIRED", w / 2, h / 2 - 28);
    ctx.fillStyle = "#d8d0c4";
    ctx.font = "15px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText("Hero viewport streams TuragaLab/flybody fruitfly.xml (MuJoCo).", w / 2, h / 2);
    ctx.fillStyle = "#7a7468";
    ctx.font = "13px ui-monospace, ui-sans-serif, monospace";
    ctx.fillText("bash scripts/install_flybody.sh   &&   export MUJOCO_GL=osmesa", w / 2, h / 2 + 28);
    ctx.fillText("CPG / bead-fly stub is not shown here", w / 2, h / 2 + 50);
    return;
  }
  ctx.fillText("LOADING FRUITFLY.XML", w / 2, h / 2 - 12);
  ctx.fillStyle = "#7a7468";
  ctx.font = "13px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText("TuragaLab/flybody MuJoCo mesh — not a CPG stub", w / 2, h / 2 + 14);
}

function drawFly(emb) {
  lastEmb = emb;
  drawHero(emb, lastConn);
}

function applyTraining(meta) {
  if (!meta) return;
  const tr = meta.training || meta.metrics || {};
  if (meta.n_neurons != null && $("brain-meta")) {
    $("brain-meta").textContent = `brain: ${meta.brain_mode || "—"} · ${meta.n_neurons} neurons`;
  }
  if ($("tr-ep")) $("tr-ep").textContent = String(tr.episodes ?? 0);
  if ($("tr-rew")) $("tr-rew").textContent = tr.last_reward != null ? Number(tr.last_reward).toFixed(3) : "—";
  if ($("tr-da")) $("tr-da").textContent = tr.last_da != null ? Number(tr.last_da).toFixed(3) : "—";
  if ($("tr-bur")) $("tr-bur").textContent = tr.last_burden != null ? Number(tr.last_burden).toFixed(3) : "—";
  if ($("tr-tox")) $("tr-tox").textContent = tr.last_toxicity != null ? Number(tr.last_toxicity).toFixed(3) : "—";
  if ($("tr-best")) $("tr-best").textContent = tr.best_reward != null ? Number(tr.best_reward).toFixed(3) : "—";
  const active = tr.protein_active || (meta.metrics && meta.metrics.protein_active) || [];
  if ($("tr-proteins")) $("tr-proteins").textContent = `proteins: ${active.length ? active.join(", ") : "—"}`;
}

function applyFrame(frame) {
  if (!frame) return;
  if (frame.training || frame.brain_mode) applyTraining(frame);
  if (frame.type && frame.type !== "frame" && frame.type !== "brain_mode" && frame.type !== "train_result") {
    if (frame.type === "halted") {
      $("status-pill").textContent = "terminal";
      $("status-pill").className = "pill halt";
    }
    return;
  }
  const t = frame.t;
  $("t-days").textContent = t.toFixed(2);
  const L = frame.latent;
  const Y = frame.observed;
  if (!ts.length) {
    Object.values(traces).forEach((g) => g.forEach((tr) => { tr.ys = []; }));
  }
  traces.cancer.forEach((tr, i) => tr.ys.push([
    Y.tumor_burden, L.tumor_burden, Y.resistance_frequency,
    Y.lactate, Y.tgfb, Y.immune_competence_ratio, L.H,
    Y.fusion_allele_fraction, Y.junction_neoantigen,
  ][i]));
  const C = frame.connectome || {};
  lastConn = C;
  const mbon = C.mbon_rates || [];
  const mbonMean = mbon.length ? mbon.reduce((a, b) => a + b, 0) / mbon.length : 0;
  traces.mbon[0].ys.push(mbonMean);
  traces.mbon[1].ys.push(C.da || 0);
  traces.drugs.forEach((tr, i) => tr.ys.push((frame.drugs.C || {})[EFFECTORS[i][0]] || 0));
  const E = frame.embodiment || {};
  lastEmb = E;
  traces.emb[0].ys.push(E.action_rms || 0);
  traces.emb[1].ys.push(E.reward || 0);
  ts.push(t);
  if (ts.length > MAX_POINTS) {
    ts.shift();
    Object.values(traces).forEach((g) => g.forEach((tr) => tr.ys.shift()));
  }
  if (E.frame_jpeg) {
    meshPhase = "ready";
    if (!mujocoImg) {
      mujocoImg = new Image();
      mujocoImg.onload = () => drawHero(lastEmb, lastConn);
    }
    mujocoImg.src = `data:image/jpeg;base64,${E.frame_jpeg}`;
  } else {
    mujocoImg = null;
    if (E.backend && E.backend !== "flybody") meshPhase = "missing";
    else if (frame.type === "hello" || frame.type === "frame") meshPhase = "missing";
  }
  redraw();
  $("kc-sp").textContent = (C.kc_sparsity ?? 0).toFixed(3);
  $("da").textContent = (C.da ?? 0).toFixed(3);
  $("wnorm").textContent = (C.plasticity_norm ?? 0).toFixed(2);
  $("graph-src").textContent = C.source || "none";
  paintHeat(C.kc_rates || []);
  EFFECTORS.forEach(([id]) => {
    const u = (frame.drugs.U || {})[id] || 0;
    const c = (frame.drugs.C || {})[id] || 0;
    const fill = $(`fill-${id}`);
    if (fill) fill.style.width = `${Math.min(100, 100 * c)}%`;
    const cEl = $(`c-${id}`);
    if (cEl) cEl.textContent = c.toFixed(2);
    if (!$("override").checked) {
      const sl = document.querySelector(`input[data-drug="${id}"]`);
      if (sl) {
        sl.value = u;
        const sv = $(`sv-${id}`);
        if (sv) sv.textContent = Number(u).toFixed(2);
      }
    }
  });
  const proteinActive = (frame.drugs.protein && frame.drugs.protein.active) || [];
  if ($("protein-active")) {
    $("protein-active").textContent = `active: ${proteinActive.length ? proteinActive.join(", ") : "none"}`;
  }
  const fusionActive = (frame.drugs.fusion && frame.drugs.fusion.active) || [];
  if ($("fusion-active")) {
    $("fusion-active").textContent = `active: ${fusionActive.length ? fusionActive.join(", ") : "none"} · research TKI, not a clinical assay`;
  }
  if ($("fusion-class")) {
    $("fusion-class").textContent = `fusion class: ${L.fusion_display || Y.fusion_id || "—"}`;
  }
  if ($("hud-burden")) $("hud-burden").textContent = Number(L.tumor_burden).toFixed(2);
  if ($("hud-resist")) $("hud-resist").textContent = Number(L.resistance_frequency).toFixed(2);
  if ($("hud-da")) $("hud-da").textContent = Number(C.da || 0).toFixed(3);
  if ($("hud-proteins")) {
    $("hud-proteins").textContent = proteinActive.length
      ? proteinActive.map((p) => p.replace("protein_", "")).join(" · ")
      : "—";
  }
  if ($("hud-fusion")) {
    $("hud-fusion").textContent = Number(Y.fusion_allele_fraction ?? L.fusion_allele_fraction ?? 0).toFixed(3);
  }
  $("source").textContent = `source: ${frame.drugs.source} ${frame.drugs.notes || ""}`;
  if ($("emb-backend")) {
    $("emb-backend").textContent = `backend: ${E.backend || "—"} · ${E.task || ""} · dim ${E.action_dim || 0}  ${E.notes || ""}`;
    $("emb-t").textContent = (E.t_fly ?? 0).toFixed(3);
    $("emb-rms").textContent = (E.action_rms ?? 0).toFixed(3);
    $("emb-rew").textContent = (E.reward ?? 0).toFixed(3);
    $("emb-hdg").textContent = (E.heading ?? 0).toFixed(2);
  }
  $("warn").classList.toggle("hidden", !Y.host_toxicity_warning);
  if (frame.terminal) {
    $("status-pill").textContent = "terminal";
    $("status-pill").className = "pill halt";
  }
}

function clearTraces() {
  ts.length = 0;
  Object.values(traces).forEach((g) => g.forEach((tr) => { tr.ys = []; }));
  redraw();
}

function setModeClass(mode) {
  document.body.classList.remove("mode-cancer", "mode-embodiment", "mode-both");
  document.body.classList.add(`mode-${mode || "both"}`);
  document.querySelectorAll("#mode-film [data-mode]").forEach((btn) => {
    btn.classList.toggle("on", btn.getAttribute("data-mode") === mode);
  });
  if ($("mode")) $("mode").value = mode;
}

function toggleDrawer(open) {
  const d = $("drawer");
  if (!d) return;
  if (open === undefined) open = d.hasAttribute("hidden");
  if (open) d.removeAttribute("hidden");
  else d.setAttribute("hidden", "");
}

ws.addEventListener("message", (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === "hello") {
    if (msg.mode) {
      $("mode").value = msg.mode;
      setModeClass(msg.mode);
    }
    if (msg.controller && $("controller")) $("controller").value = msg.controller;
    if (msg.brain_mode && $("brain-mode")) $("brain-mode").value = msg.brain_mode;
    applyTraining(msg);
    applyFrame(msg.frame);
    return;
  }
  if (msg.type === "mode") {
    setModeClass(msg.mode);
    return;
  }
  if (msg.type === "brain_mode" || msg.type === "train_result") {
    if (msg.controller && $("controller")) $("controller").value = msg.controller;
    if (msg.brain_mode && $("brain-mode")) $("brain-mode").value = msg.brain_mode;
    applyTraining(msg);
    if (msg.type === "train_result") {
      $("status-pill").textContent = "trained";
      $("status-pill").className = "pill run";
    }
    applyFrame(msg);
    return;
  }
  applyFrame(msg);
});

$("btn-play").onclick = () => {
  send("play");
  $("status-pill").textContent = "live";
  $("status-pill").className = "pill run";
};
$("btn-pause").onclick = () => {
  send("pause");
  $("status-pill").textContent = "paused";
  $("status-pill").className = "pill idle";
};
$("btn-step").onclick = () => send("step");
$("btn-reset").onclick = () => {
  send("reset", { archetype: $("archetype").value, controller: $("controller").value });
  clearTraces();
  $("status-pill").textContent = "idle";
  $("status-pill").className = "pill idle";
};
$("archetype").onchange = () => {
  send("set_archetype", { archetype: $("archetype").value });
  clearTraces();
};
$("controller").onchange = () => {
  send("set_controller", { controller: $("controller").value });
};
if ($("brain-mode")) {
  $("brain-mode").onchange = () => {
    send("set_brain_mode", { mode: $("brain-mode").value });
    clearTraces();
  };
}
if ($("btn-train")) {
  $("btn-train").onclick = () => {
    $("status-pill").textContent = "training";
    $("status-pill").className = "pill run";
    send("train_episode", { days: 80 });
  };
}
if ($("btn-train-short")) {
  $("btn-train-short").onclick = () => {
    $("status-pill").textContent = "training";
    $("status-pill").className = "pill run";
    send("train_episode", { days: 20 });
  };
}
$("mode").onchange = () => {
  send("set_mode", { mode: $("mode").value });
  setModeClass($("mode").value);
};
document.querySelectorAll("#mode-film [data-mode]").forEach((btn) => {
  btn.onclick = () => {
    const mode = btn.getAttribute("data-mode");
    send("set_mode", { mode });
    setModeClass(mode);
  };
});
if ($("btn-drawer")) $("btn-drawer").onclick = () => toggleDrawer();
if ($("btn-drawer-close")) $("btn-drawer-close").onclick = () => toggleDrawer(false);
$("override").onchange = sendOverride;
$("dt").oninput = () => {
  $("dt-val").textContent = $("dt").value;
  send("set_speed", { dt: Number($("dt").value), hz: 12 });
};

if (new URLSearchParams(location.search).get("cinema") === "1") {
  document.body.classList.add("cinema-hide");
}

window.addEventListener("resize", redraw);
drawHero(null, {});
