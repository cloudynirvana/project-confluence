/**
 * GET  /api/grok-review — readiness probe (soft-fails without XAI_API_KEY)
 * POST /api/grok-review — claim audit
 * Secret: process.env.XAI_API_KEY only. Never ship the key to the browser.
 */
const XAI_URL = "https://api.x.ai/v1/responses";
const MODEL = "grok-4.6";
const MAX_CLAIM = 2000;
const WINDOW_MS = 60000;
const MAX_PER_WINDOW = 8;
const hits = new Map();

const SYSTEM = `You are auditing a scientific claim in the CONFLUENCE thesis.
CONFLUENCE is a computational research framework, not a medical device.
Hard rules:
1. Knowledge is not evidence.
2. Evidence is not automatically a causal mechanism.
3. Causal mechanism is not automatically an identified parameter.
4. A model parameter must belong to a named frozen model.
5. OnCo confidence is not CONFLUENCE evidence strength.
6. OnCo Idea maturity is not probability of truth.
7. Legacy gene-to-parameter mappings are not identified parameters.
8. Predictions are not clinical outcomes.
9. Synthetic data are not clinical validation.
10. A visually successful simulation is not biological validation.
11. Never claim a cure from a computational result.
12. Report uncertainty and limitations explicitly.
13. Prefer primary/authoritative sources over secondary summaries.
14. State when a source is outdated or population-specific.
15. If sources disagree, report the disagreement rather than hiding it.
Search current authoritative literature where necessary.
Separate direct evidence, inference, hypothesis and assumption.
Never convert a literature association into an identified model parameter.
Do not invent citations. If you cannot verify, classification must be insufficient.
Return ONLY valid JSON with keys: claim, classification (supported|partially_supported|insufficient|contradicted), evidence_summary, strongest_sources[{title,url,source_type,date}], population_context, limitations[], falsifier, parameterization_warning, clinical_claim_warning.`;

function clientIp(req) {
  const xf = req.headers["x-forwarded-for"];
  if (typeof xf === "string" && xf.length) return xf.split(",")[0].trim();
  return req.socket && req.socket.remoteAddress ? req.socket.remoteAddress : "unknown";
}
function limited(ip) {
  const now = Date.now();
  const fresh = (hits.get(ip) || []).filter((t) => now - t < WINDOW_MS);
  if (fresh.length >= MAX_PER_WINDOW) { hits.set(ip, fresh); return true; }
  fresh.push(now); hits.set(ip, fresh); return false;
}
function extractText(data) {
  if (!data || typeof data !== "object") return "";
  if (typeof data.output_text === "string") return data.output_text;
  const out = data.output;
  if (!Array.isArray(out)) return "";
  const chunks = [];
  for (const item of out) {
    const content = item && item.content;
    if (!Array.isArray(content)) continue;
    for (const part of content) if (typeof part.text === "string") chunks.push(part.text);
  }
  return chunks.join("\n");
}
function parseAudit(text, claim) {
  const fallback = {
    claim,
    classification: "insufficient",
    evidence_summary: "The auditor could not parse a structured review.",
    strongest_sources: [],
    population_context: "unspecified",
    limitations: ["Structured JSON was not returned."],
    falsifier: "Re-run against primary sources.",
    parameterization_warning: "Do not treat this claim as an identified parameter.",
    clinical_claim_warning: "Not a clinical result.",
  };
  if (!text) return fallback;
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end <= start) return fallback;
  try {
    const obj = JSON.parse(text.slice(start, end + 1));
    const allowed = ["supported", "partially_supported", "insufficient", "contradicted"];
    return {
      claim: String(obj.claim || claim),
      classification: allowed.includes(obj.classification) ? obj.classification : "insufficient",
      evidence_summary: String(obj.evidence_summary || fallback.evidence_summary),
      strongest_sources: Array.isArray(obj.strongest_sources)
        ? obj.strongest_sources.slice(0, 6).map((s) => ({
            title: String(s.title || "Untitled"),
            url: String(s.url || ""),
            source_type: String(s.source_type || "unspecified"),
            date: String(s.date || "unspecified"),
          }))
        : [],
      population_context: String(obj.population_context || "unspecified"),
      limitations: Array.isArray(obj.limitations) ? obj.limitations.map(String).slice(0, 8) : fallback.limitations,
      falsifier: String(obj.falsifier || fallback.falsifier),
      parameterization_warning: String(obj.parameterization_warning || fallback.parameterization_warning),
      clinical_claim_warning: String(obj.clinical_claim_warning || fallback.clinical_claim_warning),
    };
  } catch (e) {
    return fallback;
  }
}

module.exports = async function handler(req, res) {
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  res.setHeader("X-Research-Status", "SIMULATION / RESEARCH");
  const offline = {
    ok: false,
    reason: "auditor_offline",
    message: "Evidence auditor offline (no API key) — citations still load from ledger",
  };
  if (req.method === "OPTIONS") { res.status(204).end(); return; }
  const key = process.env.XAI_API_KEY;
  if (req.method === "GET") {
    if (!key) { res.status(200).json(offline); return; }
    res.status(200).json({
      ok: true,
      reason: "auditor_ready",
      message: "Evidence auditor is configured. Citations still load from the ledger regardless.",
    });
    return;
  }
  if (req.method !== "POST") { res.status(405).json({ error: "POST only" }); return; }
  if (!key) {
    // Soft-fail: same JSON on 200 so the thesis page can show a calm banner.
    res.status(200).json(offline);
    return;
  }
  const ip = clientIp(req);
  if (limited(ip)) { res.status(429).json({ error: "Rate limit. Try again in a minute." }); return; }
  let body = req.body;
  if (typeof body === "string") {
    try { body = JSON.parse(body); } catch (e) { res.status(400).json({ error: "Invalid JSON" }); return; }
  }
  if (!body || typeof body !== "object") { res.status(400).json({ error: "JSON object required" }); return; }
  const claim = String(body.claim || "").trim();
  if (!claim) { res.status(400).json({ error: "claim is required" }); return; }
  if (claim.length > MAX_CLAIM) { res.status(400).json({ error: "claim too long" }); return; }
  const user = [
    "Claim to audit:\n" + claim,
    body.claim_id ? "Ledger id: " + body.claim_id : "",
    body.population ? "Stated population: " + body.population : "",
    body.time_period ? "Stated period: " + body.time_period : "",
    body.source_uri ? "Page source URI: " + body.source_uri : "",
    "Do not invent citations. If unverified, classification=insufficient.",
  ].filter(Boolean).join("\n");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 45000);
  try {
    const upstream = await fetch(XAI_URL, {
      method: "POST",
      headers: { Authorization: "Bearer " + key, "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        input: [
          { role: "system", content: SYSTEM },
          { role: "user", content: user },
        ],
        tools: [{ type: "web_search" }],
        store: false,
      }),
      signal: controller.signal,
    });
    const raw = await upstream.text();
    let data;
    try { data = JSON.parse(raw); } catch (e) { res.status(502).json({ error: "Upstream returned non-JSON" }); return; }
    if (!upstream.ok) { res.status(502).json({ error: "xAI request failed", status: upstream.status }); return; }
    const audit = parseAudit(extractText(data), claim);
    audit.ok = true;
    audit.auditor = "grok-4.6";
    audit.disclaimer = "Grok output is an evidence-audit aid. It is not a substitute for expert review or primary-source verification.";
    res.status(200).json(audit);
  } catch (err) {
    res.status(504).json({ error: err && err.name === "AbortError" ? "Auditor timed out" : "Auditor request failed" });
  } finally {
    clearTimeout(timer);
  }
};
