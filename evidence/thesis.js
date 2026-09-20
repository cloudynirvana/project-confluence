async function loadClaims() {
  const res = await fetch("claims.json", { cache: "no-store" });
  if (!res.ok) throw new Error("Could not load claims.json");
  return res.json();
}
function el(html) {
  const d = document.createElement("div");
  d.innerHTML = html.trim();
  return d.firstElementChild;
}
function renderClaim(c) {
  const node = el(`<article class="claim" id="${c.claim_id}">
    <h3>${c.claim_id} · ${c.claim_type.replace(/_/g, " ")}</h3>
    <p>${c.claim}</p>
    <p class="src"><strong>Source.</strong> ${c.source}<br>
    <a href="${c.source_uri}">${c.source_uri}</a><br>
    <strong>Population / period.</strong> ${c.population} · ${c.time_period}<br>
    <strong>Class.</strong> ${c.evidence_class} · <strong>Limit.</strong> ${c.limitations}</p>
    <button class="audit" type="button" data-id="${c.claim_id}">Audit this claim</button>
    <div class="audit-out" hidden></div>
  </article>`);
  node.querySelector("button").addEventListener("click", () => audit(c, node));
  return node;
}
async function audit(claim, node) {
  const out = node.querySelector(".audit-out");
  const btn = node.querySelector("button");
  btn.disabled = true;
  out.hidden = false;
  out.className = "audit-out";
  out.textContent = "Requesting server-side audit…";
  try {
    const res = await fetch("/api/grok-review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        claim: claim.claim,
        claim_id: claim.claim_id,
        population: claim.population,
        time_period: claim.time_period,
        source_uri: claim.source_uri,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      out.textContent = data.error || "Audit request failed.";
      if (data.hint) out.textContent += " " + data.hint;
      return;
    }
    out.classList.add(data.classification || "insufficient");
    const sources = (data.strongest_sources || [])
      .map((s) => {
        const label = s.title || s.url || "source";
        return s.url ? `<li><a href="${s.url}">${label}</a> (${s.source_type || ""}, ${s.date || ""})</li>` : `<li>${label}</li>`;
      })
      .join("");
    const limits = (data.limitations || []).map((x) => `<li>${x}</li>`).join("");
    out.innerHTML = `<p><strong>Evidence audit · ${data.classification}</strong></p>
      <p>${data.evidence_summary || ""}</p>
      <p><strong>Population / context.</strong> ${data.population_context || ""}</p>
      <p><strong>Falsifier.</strong> ${data.falsifier || ""}</p>
      <p><strong>Parameterization warning.</strong> ${data.parameterization_warning || ""}</p>
      <p><strong>Clinical-interpretation warning.</strong> ${data.clinical_claim_warning || ""}</p>
      <ul>${sources}</ul><ul>${limits}</ul>
      <p class="meta">${data.disclaimer || "Grok output is an evidence-audit aid, not ground truth."}</p>`;
  } catch (err) {
    out.textContent = "Network error talking to /api/grok-review.";
  } finally {
    btn.disabled = false;
  }
}
async function auditCustom() {
  const text = document.getElementById("custom-claim").value.trim();
  if (!text) return;
  const mount = document.getElementById("custom-out");
  mount.innerHTML = "";
  const fake = { claim: text, claim_id: "CUSTOM", population: "", time_period: "", source_uri: "" };
  const wrap = el(`<div><button class="audit" type="button">working</button><div class="audit-out"></div></div>`);
  wrap.querySelector("button").hidden = true;
  mount.appendChild(wrap);
  await audit(fake, wrap);
}
document.getElementById("audit-custom")?.addEventListener("click", auditCustom);
loadClaims()
  .then((rows) => {
    const root = document.getElementById("ledger");
    root.innerHTML = "";
    rows.forEach((c) => root.appendChild(renderClaim(c)));
  })
  .catch((err) => {
    document.getElementById("ledger").textContent = err.message;
  });
