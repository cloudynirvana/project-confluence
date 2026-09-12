const DRUGS = [
  ["anti_pd1", "anti-PD1"],
  ["tgfb_inhibitor", "TGF-βi"],
  ["mct1", "MCT1"],
  ["hdac", "HDAC"],
  ["targeted_kinase", "kinase"],
];

const MAX_POINTS = 180;
const wsProto = location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${wsProto}://${location.host}/ws/sim`);

const $ = (id) => document.getElementById(id);
const sliderBox = $("sliders");
const bars = $("drug-bars");
const manual = {};

DRUGS.forEach(([id, label]) => {
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
  row.className = "bar";
  row.innerHTML = `<span>${label}</span><div class="track"><div class="fill" id="fill-${id}"></div></div><span id="c-${id}">0.00</span>`;
  bars.appendChild(row);
});

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
  ],
  mbon: [
    makeTrace("MBON mean", "#d4a054"),
    makeTrace("DA", "#e06b5c"),
  ],
  drugs: DRUGS.map(([id], i) => {
    const colors = ["#e06b5c", "#d4a054", "#3ecfc0", "#7aa2d4", "#c084fc"];
    return makeTrace(`C ${id}`, colors[i]);
  }),
};
const ts = [];

$("legend-cancer").innerHTML = traces.cancer
  .map((tr) => `<span><i style="background:${tr.color}"></i>${tr.label}</span>`)
  .join("");

function pushTraces(group, t, values) {
  ts.push(t);
  group.forEach((tr, i) => tr.ys.push(values[i]));
  if (ts.length > MAX_POINTS) {
    ts.shift();
    Object.values(traces).forEach((g) => g.forEach((tr) => tr.ys.shift()));
  }
}

function drawChart(canvasId, group, yMaxHint) {
  const canvas = $(canvasId);
  const parent = canvas.parentElement;
  const w = Math.max(200, parent.clientWidth - 8);
  const h = Number(canvas.getAttribute("height")) || 180;
  if (canvas.width !== w) canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0c151d";
  ctx.fillRect(0, 0, w, h);
  const pad = { l: 36, r: 8, t: 8, b: 20 };
  const iw = w - pad.l - pad.r;
  const ih = h - pad.t - pad.b;
  const ymax = Math.max(yMaxHint, ...group.flatMap((tr) => tr.ys), 0.1);
  ctx.strokeStyle = "#1b2733";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (ih * i) / 4;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(w - pad.r, y);
    ctx.stroke();
  }
  ctx.fillStyle = "#8d8476";
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
    if (tr.dash.length) ctx.setLineDash(tr.dash);
    else ctx.setLineDash([]);
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
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#0c151d";
  ctx.fillRect(0, 0, w, h);
  const n = rates.length || 1;
  const cw = w / n;
  rates.forEach((v, i) => {
    const g = Math.round(40 + 200 * Math.min(1, v));
    ctx.fillStyle = `rgb(${Math.round(g * 0.7)}, ${Math.round(g * 0.42)}, 18)`;
    ctx.fillRect(i * cw, 0, cw + 0.5, h);
  });
}

function redraw() {
  drawChart("chart-cancer", traces.cancer, 1.2);
  drawChart("chart-mbon", traces.mbon, 0.4);
  drawChart("chart-drugs", traces.drugs, 0.4);
}

function applyFrame(frame) {
  if (!frame) return;
  if (frame.type && frame.type !== "frame") {
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
  // cancer group is independent of mbon/drugs length; push per-group
  traces.cancer.forEach((tr, i) => tr.ys.push([
    Y.tumor_burden, L.tumor_burden, Y.resistance_frequency,
    Y.lactate, Y.tgfb, Y.immune_competence_ratio, L.H,
  ][i]));
  const C = frame.connectome || {};
  const mbon = C.mbon_rates || [];
  const mbonMean = mbon.length ? mbon.reduce((a, b) => a + b, 0) / mbon.length : 0;
  traces.mbon[0].ys.push(mbonMean);
  traces.mbon[1].ys.push(C.da || 0);
  traces.drugs.forEach((tr, i) => tr.ys.push(frame.drugs.C[DRUGS[i][0]] || 0));
  ts.push(t);
  if (ts.length > MAX_POINTS) {
    ts.shift();
    Object.values(traces).forEach((g) => g.forEach((tr) => tr.ys.shift()));
  }
  redraw();
  $("kc-sp").textContent = (C.kc_sparsity ?? 0).toFixed(3);
  $("da").textContent = (C.da ?? 0).toFixed(3);
  $("wnorm").textContent = (C.plasticity_norm ?? 0).toFixed(2);
  $("graph-src").textContent = C.source || "none";
  paintHeat(C.kc_rates || []);
  DRUGS.forEach(([id]) => {
    const u = frame.drugs.U[id] || 0;
    const c = frame.drugs.C[id] || 0;
    $(`fill-${id}`).style.width = `${Math.min(100, 100 * c)}%`;
    $(`c-${id}`).textContent = c.toFixed(2);
    if (!$("override").checked) {
      const sl = document.querySelector(`input[data-drug="${id}"]`);
      if (sl) {
        sl.value = u;
        $(`sv-${id}`).textContent = Number(u).toFixed(2);
      }
    }
  });
  $("source").textContent = `source: ${frame.drugs.source} ${frame.drugs.notes || ""}`;
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

ws.addEventListener("message", (ev) => {
  const msg = JSON.parse(ev.data);
  if (msg.type === "hello") applyFrame(msg.frame);
  else applyFrame(msg);
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
$("override").onchange = sendOverride;
$("dt").oninput = () => {
  $("dt-val").textContent = $("dt").value;
  send("set_speed", { dt: Number($("dt").value), hz: 12 });
};

window.addEventListener("resize", redraw);
