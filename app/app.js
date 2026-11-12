/* Dashboard: static page, no framework, no network. All content comes from window.DASHBOARD_DATA. */
(function () {
"use strict";
const D = window.DASHBOARD_DATA;
const META = D.meta;

/* ---------- small helpers ---------- */
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  if (attrs) for (const k in attrs) {
    const v = attrs[k];
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of kids.flat()) if (c != null && c !== false) e.append(c.nodeType ? c : document.createTextNode(String(c)));
  return e;
}
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const isNum = (v) => typeof v === "number" && isFinite(v);
const sum = (a) => a.reduce((s, v) => s + v, 0);
const mean = (a) => (a.length ? sum(a) / a.length : NaN);
function median(a) {
  if (!a.length) return NaN;
  const s = a.slice().sort((x, y) => x - y), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function metrics(d) {
  d = d.filter(isNum);
  const n = d.length;
  if (!n) return { n: 0 };
  const b = mean(d), md = median(d);
  const sd = n > 1 ? Math.sqrt(sum(d.map((x) => (x - b) ** 2)) / (n - 1)) : NaN;
  return { n, bias: b, sd, rmse: Math.sqrt(mean(d.map((x) => x * x))), median: md, mad: median(d.map((x) => Math.abs(x - md))) };
}
function wilson(k, n) {
  if (!n) return [NaN, NaN, NaN];
  const z = 1.96, p = k / n, den = 1 + z * z / n;
  const c = (p + z * z / (2 * n)) / den, hw = (z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den;
  return [p, c - hw, c + hw];
}
function f(v, nd = 2, signed = false) {
  if (!isNum(v)) return "–";
  const s = v.toFixed(nd);
  return signed && v > 0 ? "+" + s : s;
}
const pct = (v, nd = 1) => (isNum(v) ? (v * 100).toFixed(nd) + " %" : "–");
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function groupBy(arr, fn) {
  const m = new Map();
  for (const x of arr) { const k = fn(x); if (!m.has(k)) m.set(k, []); m.get(k).push(x); }
  return m;
}
const unique = (a) => [...new Set(a)];

/* ---------- data ---------- */
const T = D.trees.rows.map((r) => { const o = {}; D.trees.cols.forEach((c, i) => (o[c] = r[i])); return o; });
const STATS = new Map();
for (const e of D.stats) STATS.set(e.id + "|" + (e.subset == null ? "" : e.subset), e);
const S = (id, subset) => STATS.get(id + "|" + (subset == null ? "" : subset));
const statsWith = (prefix, subset) => D.stats.filter((e) => e.id.startsWith(prefix) && (subset === undefined || e.subset === subset));
const PZ = D.parzellen;
const SAMPLES = D.samples;
const IDC = ["confirmed", "probable", "ambiguous", "rejected", "unmatched"];
const DET = ["detected", "uncertain", "not detected"];
const speciesOrder = [...groupBy(T, (t) => t.sp)].sort((a, b) => b[1].length - a[1].length).map((x) => x[0]);

/* ---------- state: filters, theme, hash ---------- */
const F = { p: "all", idc: "all", sub: "all_matched" };
const HASH = {};
function readHash() {
  const out = {};
  (location.hash || "").replace(/^#/, "").split("&").forEach((kv) => { const [k, v] = kv.split("="); if (k) out[k] = decodeURIComponent(v || ""); });
  return out;
}
function writeHash() {
  const q = Object.entries(HASH).filter(([, v]) => v).map(([k, v]) => k + "=" + encodeURIComponent(v)).join("&");
  history.replaceState(null, "", "#" + q);
}
const filtersActive = () => F.p !== "all" || F.idc !== "all" || F.sub !== "all_matched";
function idcOk(c) { return F.idc === "all" || (F.idc === "confirmed+probable" ? c === "confirmed" || c === "probable" : c === F.idc); }
const isMatched = (det) => (F.sub === "detected_only" ? det === "detected" : det !== "not detected");
// Trees after the Parzelle and identity filters; the subset filter only where a matched tree is needed.
const treesF = () => T.filter((t) => (F.p === "all" || String(t.p) === F.p) && idcOk(t.idc));
const matchedF = () => treesF().filter((t) => isMatched(t.det));
const samplesF = () => SAMPLES.filter((s) => (F.p === "all" || String(s.parzelle) === F.p) && idcOk(s.identity_class) && isMatched(s.detection));
const parzF = () => PZ.filter((p) => F.p === "all" || String(p.parzelle) === F.p);

/* ---------- colours and plotting ---------- */
let C = {};
function loadColors() {
  C = { ink: css("--ink"), ink2: css("--ink-2"), grid: css("--grid"), line: css("--line"), surface: css("--surface"),
        c: [1, 2, 3, 4, 5, 6, 7, 8].map((i) => css("--c" + i)), neutral: css("--neutral"), lo: css("--seq-lo"), hi: css("--seq-hi") };
  C.uls = C.c[0]; C.ground = C.c[1]; C.vertex = C.c[2]; C.practice = C.c[3];
  C.idc = { confirmed: C.c[0], probable: C.c[2], ambiguous: C.c[3], rejected: C.c[7], unmatched: C.neutral };
  C.det = { detected: C.c[0], uncertain: C.c[3], "not detected": C.c[7] };
  C.species = {};
  speciesOrder.forEach((s, i) => (C.species[s] = i < 3 ? [C.c[5], C.c[6], C.c[4]][i] : C.neutral));
}
let chartCount = 0, tableCount = 0;
function merge(a, b) {
  const o = Array.isArray(a) ? a.slice() : Object.assign({}, a);
  for (const k in b) o[k] = b[k] && typeof b[k] === "object" && !Array.isArray(b[k]) && a && typeof a[k] === "object" ? merge(a[k], b[k]) : b[k];
  return o;
}
function baseLayout() {
  const ax = { gridcolor: C.grid, zerolinecolor: C.line, linecolor: C.line, tickcolor: C.line, automargin: true, ticks: "outside" };
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: C.ink, size: 12 },
    margin: { l: 55, r: 12, t: 8, b: 40 }, xaxis: Object.assign({}, ax), yaxis: Object.assign({}, ax),
    legend: { orientation: "h", y: -0.22, x: 0, font: { size: 11 } }, hoverlabel: { font: { size: 12 } }, autosize: true,
  };
}
function card(parent, title, cap) {
  const c = h("div", { class: "card" }, title && h("h3", { text: title }), cap && h("div", { class: "cap", text: cap }));
  parent.appendChild(c);
  return c;
}
function plot(parent, traces, layout, o = {}) {
  const div = h("div", { class: "chart " + (o.cls || "") });
  parent.appendChild(div);
  chartCount++;
  try {
    const p = Plotly.newPlot(div, traces, merge(baseLayout(), layout || {}), Object.assign({ displayModeBar: false, responsive: true, staticPlot: false }, o.config || {}));
    if (p && p.catch) p.catch((e) => __report("plot: " + (e && e.message || e)));
  } catch (e) { __report("plot: " + e.message); }
  return div;
}
const noData = (parent, msg) => parent.appendChild(h("p", { class: "empty", text: msg || "Not available in the data files for this site." }));
const seq = () => [[0, C.lo], [1, C.hi]];
const div3 = () => [[0, C.ground], [0.5, "#b9b8b2"], [1, C.uls]];
const lineShape = (x0, y0, x1, y1, dash) => ({ type: "line", x0, y0, x1, y1, line: { color: C.ink2, width: 1, dash: dash || "dot" } });

/* ---------- stat display ---------- */
function fmtVal(e) {
  if (!e || !isNum(e.value)) return "–";
  const v = e.value, u = e.unit || "";
  if (u === "share") return pct(v);
  if (u === "m") return f(v, 2) + " m";
  if (u === "cm") return f(v, 2) + " cm";
  if (u === "m3") return f(v, Math.abs(v) < 1 ? 3 : 1) + " m³";
  if (["trees", "instances", "Parzellen"].includes(u)) return Math.round(v) + " " + u;
  return f(v, Math.abs(v) < 10 ? 2 : 1) + (u ? " " + u : "");
}
function fmtCI(e) {
  if (!e || !isNum(e.ci_low) || !isNum(e.ci_high)) return "";
  const u = e.unit === "share" ? "%" : e.unit === "m3" ? "m³" : e.unit || "";
  const k = e.unit === "share" ? 100 : 1, d = e.unit === "share" ? 1 : 2;
  return `95 % CI ${f(e.ci_low * k, d)} to ${f(e.ci_high * k, d)} ${u}`.trim();
}
function tile(label, e, o = {}) {
  const t = h("div", { class: "tile" });
  t.appendChild(h("div", { class: "lab", text: label }));
  if (!e) { t.appendChild(h("div", { class: "val", text: "–" })); t.appendChild(h("div", { class: "st", text: "Not available in the data files." })); return t; }
  t.appendChild(h("div", { class: "val", text: o.value || fmtVal(e) }));
  const ci = fmtCI(e);
  if (ci && !o.noci) t.appendChild(h("div", { class: "ci", text: ci }));
  if (o.extra) t.appendChild(h("div", { class: "extra" }, o.extra.filter((x) => x.e || x.text).map((x) => h("span", null, x.label + " ", h("b", { text: x.text || fmtVal(x.e) })))));
  t.appendChild(h("div", { class: "st", text: o.statement || e.statement }));
  t.appendChild(h("div", { class: "nn", text: "n = " + (e.n != null ? e.n : "–") + (e.subset ? " · " + e.subset.replace(/_/g, " ") : "") }));
  if (filtersActive() && o.follows !== true) t.appendChild(h("div", { class: "nofilter", text: "does not follow the filters" }));
  return t;
}
const stmt = (id, subset) => { const e = S(id, subset); return e ? e.statement : ""; };
const prettyId = (s) => s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
const METHOD_NAMES = { best_local_maxima: "Canopy-model local maxima (best window)", segmentanytree_instance_tops: "SAT, instance tops", segmentanytree_base_matching: "SAT, stem to instance base" };

/* ---------- tables ---------- */
function mkTable(cols, rows, o = {}) {
  tableCount++;
  const wrap = h("div", { class: "tablewrap" + (o.scroll ? " scroll-y" : "") });
  const state = { key: o.sortKey || null, dir: o.sortDir || 1 };
  const table = h("table", { class: "dt" });
  wrap.appendChild(table);
  function draw() {
    table.innerHTML = "";
    const head = h("tr");
    cols.forEach((c) => {
      const th = h("th", { class: c.cls || "", title: "Click to sort" }, c.label, state.key === c.k ? h("span", { class: "arrow", text: state.dir > 0 ? " ▲" : " ▼" }) : "");
      th.addEventListener("click", () => { state.dir = state.key === c.k ? -state.dir : 1; state.key = c.k; draw(); });
      head.appendChild(th);
    });
    table.appendChild(h("thead", null, head));
    let rs = rows.slice();
    if (state.key) {
      const c = cols.find((x) => x.k === state.key);
      const val = (r) => (c.sort ? c.sort(r) : r[c.k]);
      rs.sort((a, b) => { const x = val(a), y = val(b); if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : x < y ? -1 : 0) * state.dir; });
    }
    const body = h("tbody");
    rs.forEach((r) => {
      const tr = h("tr", { class: o.onRow ? "clickable" + (o.selected && o.selected() === r ? " sel" : "") : "" });
      cols.forEach((c) => {
        const v = c.fmt ? c.fmt(r) : r[c.k];
        tr.appendChild(h("td", { class: c.cls || "" }, v == null || v === "" ? "–" : v));
      });
      if (o.onRow) tr.addEventListener("click", () => { o.onRow(r); });
      body.appendChild(tr);
    });
    table.appendChild(body);
  }
  draw();
  wrap.redraw = draw;
  return wrap;
}

/* ---------- CSV download ---------- */
function csvText(rows, cols) {
  const q = (v) => { if (v == null) return ""; const s = String(v); return /[",\n;]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  return [cols.join(",")].concat(rows.map((r) => cols.map((c) => q(r[c])).join(","))).join("\r\n");
}
function download(name, rows, cols) {
  try {
    const blob = new Blob(["﻿" + csvText(rows, cols)], { type: "text/csv;charset=utf-8" });
    const a = h("a", { href: URL.createObjectURL(blob), download: name });
    document.body.appendChild(a); a.click(); a.remove();
  } catch (e) { __report("download: " + e.message); }
}
function dlButton(label, name, getRows, cols) { return h("button", { class: "btn", type: "button", onclick: () => download(name, getRows(), cols) }, label); }
const TREE_COLS = ["id", "p", "sp", "x", "y", "dbh", "dbhU", "dbhq", "hU", "hsrc", "det", "idc", "sc", "offc", "vF", "vU", "rem", "cause"];
const TREE_COL_LABELS = null;

/* ======================================================================
   Tabs
   ====================================================================== */
const TABS = [
  ["overview", "Overview", renderOverview], ["data", "Data", renderData], ["match", "Match confidence", renderMatch],
  ["height", "Height: ULS vs Maßband vs Vertex", renderHeight], ["wins", "Where ULS wins", renderWins],
  ["samples", "Sample explorer", renderSamples], ["volume", "Wood volume (old)", renderVolume], ["volume_gri", "Wood volume (GRI)", renderVolumeGri],
["methods", "Methods and caveats", renderMethods],
];

/* ---------- shared chart builders ---------- */
function coverageChart(parent) {
  const rows = [
    ["Field BHD (all trees)", S("inventory.field_trees")], ["ULS height and BHD (matched)", null],
    ["Height by Maßband", S("inventory.massband")], ["Height by Vertex", S("inventory.vertex")], ["Any sample-tree height", S("inventory.sample_trees")],
  ];
  const det = S("detection.count_detected"), unc = S("detection.count_uncertain");
  rows[1][1] = det && unc ? { value: det.value + unc.value } : null;
  const r = rows.filter((x) => x[1]);
  plot(parent, [{ type: "bar", orientation: "h", y: r.map((x) => x[0]), x: r.map((x) => x[1].value),
    marker: { color: r.map((x) => (x[0].startsWith("ULS") ? C.uls : C.ground)) }, text: r.map((x) => String(Math.round(x[1].value))), textposition: "outside", cliponaxis: false,
    hovertemplate: "%{y}: %{x} trees<extra></extra>" }],
  { xaxis: { title: "Number of trees", rangemode: "tozero" }, yaxis: { autorange: "reversed" }, margin: { l: 150, r: 40 } }, { cls: "short" });
}
function classBarChart(parent) {
  const ts = treesF();
  const cnt = IDC.map((c) => ts.filter((t) => t.idc === c).length);
  plot(parent, [{ type: "bar", x: IDC, y: cnt, marker: { color: IDC.map((c) => C.idc[c]) }, text: cnt.map(String), textposition: "outside", cliponaxis: false,
    hovertemplate: "%{x}: %{y} trees<extra></extra>" }], { xaxis: { title: "Identity class of the match" }, yaxis: { title: "Number of field trees", rangemode: "tozero" } }, { cls: "short" });
}
// Sort key of a diameter-class label such as "< 200 mm", "300-400 mm", "> 500 mm", "no value".
function clsKey(c) { const m = String(c).match(/\d+/); if (!m) return 1e9; const v = +m[0]; return /^\s*</.test(c) ? v - 0.5 : /^\s*>/.test(c) ? v + 0.5 : v; }
function detectionByClassChart(parent) {
  const ts = treesF().filter((t) => t.cls);
  const classes = unique(T.map((t) => t.cls).filter(Boolean)).sort((a, b) => clsKey(a) - clsKey(b));
  const rate = D.detection_model.filter((r) => r.kind === "rate_by_class");
  const useTable = !filtersActive() && rate.length;
  const xs = [], ys = [], lo = [], hi = [], ns = [];
  classes.forEach((c) => {
    const g = ts.filter((t) => t.cls === c);
    let p, a, b, n = g.length;
    const tr = rate.find((r) => r.term === c);
    if (useTable && tr) { p = tr.value; a = tr.ci_low; b = tr.ci_high; n = tr.n; } else [p, a, b] = wilson(g.filter((t) => t.det === "detected").length, g.length);
    if (n) { xs.push(c + "<br>n=" + n); ys.push(p * 100); lo.push((p - a) * 100); hi.push((b - p) * 100); }
  });
  plot(parent, [{ type: "bar", x: xs, y: ys, marker: { color: C.uls }, error_y: { type: "data", symmetric: false, array: hi, arrayminus: lo, color: C.ink2, thickness: 1.2 },
    text: ys.map((v) => v.toFixed(0) + " %"), textposition: "inside", insidetextanchor: "end", textangle: 0, textfont: { color: "#fff", size: 10 }, hovertemplate: "%{x}: %{y:.1f} % detected<extra></extra>" }],
  { xaxis: { title: "Field BHD class (mm)", tickangle: 0, tickfont: { size: 10 } }, yaxis: { title: "Trees detected (%)", range: [0, 105] } }, { cls: "short" });
}

/* ======================================================================
   Overview
   ====================================================================== */
function renderOverview(root) {
  const lede = h("p", { class: "lede" }, "Question: what is drone laser scanning (ULS, with SegmentAnyTree, SAT) worth for this site? The tiles give the headline numbers; the five cards below follow the story from coverage to limits. Every number is computed in the analysis pipeline; each sentence under a number explains what it means.");
  root.appendChild(lede);
  const tiles = h("div", { class: "grid tiles" });
  root.appendChild(tiles);
  const hm = (id) => ({ e: S("height.massband." + id, "all_matched") });
  tiles.append(
    tile("Trees found by ULS (matched)", S("detection.rate", "all_matched"), { extra: [{ label: "detected only", e: S("detection.rate", "detected_only") }] }),
    tile("Share of field wood volume in found trees", S("volume.detection_share.detected"), { extra: [{ label: "uncertain", e: S("volume.detection_share.uncertain") }, { label: "not found", e: S("volume.detection_share.not_detected") }] }),
    tile("ULS height vs Maßband: RMSE", S("height.massband.rmse", "all_matched"), { extra: [{ label: "bias", e: hm("bias").e }, { label: "median", e: hm("median").e }, { label: "MAD", e: hm("mad").e }] }),
    tile("Vertex vs Maßband: RMSE (noise floor)", S("height.vertex_vs_massband.rmse"), { extra: [{ label: "bias", e: S("height.vertex_vs_massband.bias") }, { label: "median", e: S("height.vertex_vs_massband.median") }, { label: "MAD", e: S("height.vertex_vs_massband.mad") }] }),
    tile("ULS BHD minus field BHD (bias)", S("identity.dbh_all_field_trees.all_matched.bias", "all_matched"), { extra: [{ label: "RMSE", e: S("identity.dbh_all_field_trees.all_matched.rmse", "all_matched") }] }),
    (() => {
      const ts = treesF(), c = ts.filter((t) => t.idc === "confirmed").length;
      const e = { value: ts.length ? c / ts.length : NaN, unit: "share", n: ts.length, statement: `${c} of ${ts.length} field trees have a confirmed identity: the ULS instance is the same tree as the field tree by position evidence (match_identity table, counted in the browser; follows the filters).` };
      return tile("Matches with confirmed identity", e, { follows: true, noci: true });
    })(),
    tile("Stand volume, ULS", S("volume.total.uls.stand"), { extra: [{ label: "field (with modelled)", e: S("volume.total.field_with_modelled.stand") }, { label: "gap", e: S("volume.gap.total") }] }),
  );
  root.appendChild(h("h2", { text: "The story in five statements" }));
  const g = h("div", { class: "grid" });
  root.appendChild(g);
  const story = (n, title, stmts) => { const c = card(g, null, null); c.appendChild(h("div", { class: "story" }, h("span", { class: "num", text: n + ". " }), h("b", { text: title }))); stmts.filter(Boolean).forEach((s) => c.appendChild(h("p", { class: "cap", text: s }))); return c; };
  let c = story(1, "Coverage: ULS reaches nearly every tree", [stmt("detection.rate", "all_matched"), stmt("inventory.sample_trees")]);
  coverageChart(c);
  c = story(2, "Height: about as good as the Vertex", [stmt("error_budget.verdict", "sample_trees_with_both"), stmt("height.massband.rmse_without_largest_influence", "all_matched")]);
  heightMini(c);
  c = story(3, "Identity: is it the same tree?", [stmt("match.within_high_confidence_real"), stmt("match.within_high_confidence_control")]);
  classBarChart(c);
  c = story(4, "Volume: the gap is mostly missed trees", [stmt("volume.gap.total"), stmt("volume.gap.diameter_difference")]);
  gapChart(c);
  c = story(5, "Limits: small and merged trees", [stmt("detection_model.rate_0_200_mm", "detected_only"), stmt("causes.not_detected.merged_with_a_neighbour")]);
  detectionByClassChart(c);
  if (META.effort) {
    root.appendChild(h("h2", { text: "Effort per tree" }));
    const c2 = card(root, null, "Effort figures entered in the site configuration.");
    c2.appendChild(mkTable([{ k: "k", label: "Item" }, { k: "v", label: "Value" }], Object.entries(META.effort).map(([k, v]) => ({ k, v: typeof v === "object" ? JSON.stringify(v) : v }))));
  }
}
function heightMini(parent) {
  const ss = samplesF().filter((s) => isNum(s.height_massband_m) && isNum(s.height_uls_m));
  scatter11(parent, ss, "height_massband_m", "height_uls_m", "Maßband length (m)", "ULS height (m)", C.uls, "short");
}
function gapChart(parent) {
  const parts = [["Trees without ULS volume", "volume.gap.trees_missed"], ["BHD difference", "volume.gap.diameter_difference"], ["ULS-only trees", "volume.gap.uls_only_trees"], ["Total gap", "volume.gap.total"]]
    .map(([l, id]) => [l, S(id)]).filter((x) => x[1]);
  if (!parts.length) return noData(parent);
  plot(parent, [{ type: "bar", orientation: "h", y: parts.map((x) => x[0]), x: parts.map((x) => x[1].value), text: parts.map((x) => f(x[1].value, 1, true) + " m³"), textposition: "outside", cliponaxis: false, constraintext: "none",
    marker: { color: parts.map((x, i) => (i === parts.length - 1 ? C.ink2 : C.uls)) }, hovertemplate: "%{y}: %{x:.1f} m³<extra></extra>" }],
  { xaxis: { title: "Volume difference ULS minus field (m³)", zeroline: true, range: [Math.min(0, ...parts.map((x) => x[1].value)) * 1.9, Math.max(0, ...parts.map((x) => x[1].value)) * 2.4 + 15] }, yaxis: { autorange: "reversed", automargin: true }, margin: { l: 150, r: 20 } }, { cls: "short" });
}

/* 1:1 scatter of two height columns, with the largest difference labelled */
function scatter11(parent, rows, xk, yk, xl, yl, color, cls) {
  const r = rows.filter((s) => isNum(s[xk]) && isNum(s[yk]));
  if (!r.length) return noData(parent, "No sample trees match the current filters.");
  const worst = r.reduce((a, b) => (Math.abs(b[yk] - b[xk]) > Math.abs(a[yk] - a[xk]) ? b : a), r[0]);
  const rest = r.filter((s) => s !== worst);
  const lo = Math.min(...r.map((s) => Math.min(s[xk], s[yk]))) - 1, hi = Math.max(...r.map((s) => Math.max(s[xk], s[yk]))) + 1;
  const showWorst = Math.abs(worst[yk] - worst[xk]) > META.explain_height_diff_m;
  const tr = [
    { type: "scatter", mode: "lines", x: [lo, hi], y: [lo, hi], line: { color: C.ink2, width: 1, dash: "dot" }, name: "1:1 line", hoverinfo: "skip" },
    { type: "scatter", mode: "markers", x: (showWorst ? rest : r).map((s) => s[xk]), y: (showWorst ? rest : r).map((s) => s[yk]), name: "sample trees (n=" + r.length + ")",
      text: (showWorst ? rest : r).map((s) => s.tree_id), marker: { color, size: 9, line: { color: C.surface, width: 1.5 } }, hovertemplate: "%{text}<br>x %{x:.2f} m, y %{y:.2f} m<extra></extra>" },
  ];
  if (showWorst) tr.push({ type: "scatter", mode: "markers+text", x: [worst[xk]], y: [worst[yk]], text: [worst.tree_id + ": " + f(worst[yk] - worst[xk], 1, true) + " m"], textposition: "bottom right",
    name: "largest difference", marker: { color: C.c[7], size: 12, symbol: "diamond", line: { color: C.surface, width: 1.5 } }, textfont: { color: C.ink, size: 11 },
    hovertemplate: worst.tree_id + "<br>x %{x:.2f} m, y %{y:.2f} m<extra></extra>" });
  plot(parent, tr, { xaxis: { title: xl, range: [lo, hi] }, yaxis: { title: yl, range: [lo, hi] } }, { cls: cls || "" });
}
function blandAltman(parent, rows, xk, yk, xl, yl, color) {
  const r = rows.filter((s) => isNum(s[xk]) && isNum(s[yk])).map((s) => ({ id: s.tree_id, m: (s[xk] + s[yk]) / 2, d: s[yk] - s[xk] }));
  if (!r.length) return noData(parent, "No sample trees match the current filters.");
  const m = metrics(r.map((x) => x.d)), loa = [m.bias - 1.96 * m.sd, m.bias + 1.96 * m.sd];
  const worst = r.reduce((a, b) => (Math.abs(b.d) > Math.abs(a.d) ? b : a), r[0]);
  const showWorst = Math.abs(worst.d) > META.explain_height_diff_m;
  const rest = showWorst ? r.filter((x) => x !== worst) : r;
  const xmin = Math.min(...r.map((x) => x.m)) - 0.5, xmax = Math.max(...r.map((x) => x.m)) + 0.5;
  const tr = [
    { type: "scatter", mode: "markers", x: rest.map((x) => x.m), y: rest.map((x) => x.d), text: rest.map((x) => x.id), name: "sample trees (n=" + r.length + ")", marker: { color, size: 9, line: { color: C.surface, width: 1.5 } }, hovertemplate: "%{text}<br>mean %{x:.2f} m, difference %{y:.2f} m<extra></extra>" },
    { type: "scatter", mode: "lines", x: [xmin, xmax], y: [m.bias, m.bias], name: "bias " + f(m.bias, 2, true) + " m", line: { color: C.ink, width: 1.5 } },
    { type: "scatter", mode: "lines", x: [xmin, xmax], y: [m.median, m.median], name: "median " + f(m.median, 2, true) + " m", line: { color: C.ink2, width: 1.5, dash: "dash" } },
    { type: "scatter", mode: "lines", x: [xmin, xmax, xmax, xmin], y: [loa[0], loa[0], loa[1], loa[1]], name: "bias ± 1.96 SD", line: { color: C.ink2, width: 1, dash: "dot" } },
  ];
  if (showWorst) tr.push({ type: "scatter", mode: "markers+text", x: [worst.m], y: [worst.d], text: [worst.id], textposition: "bottom right", textfont: { color: C.ink, size: 11 }, name: "largest difference",
    marker: { color: C.c[7], size: 12, symbol: "diamond", line: { color: C.surface, width: 1.5 } }, hovertemplate: worst.id + "<br>difference %{y:.2f} m<extra></extra>" });
  plot(parent, tr, { xaxis: { title: "Mean of the two heights (m)" }, yaxis: { title: yl + " minus " + xl + " (m)" }, legend: { y: -0.3 } });
}

/* ======================================================================
   Data
   ====================================================================== */
function renderData(root) {
  root.appendChild(h("p", { class: "lede", text: "What was measured, by whom, and how clean the ground records are. Field crew: stem position, species and BHD for every tree; height only on felled sample trees. ULS: height and BHD for almost every tree." }));
  root.appendChild(h("h2", { text: "What was measured by whom" }));
  const g1 = h("div", { class: "grid wide" });
  root.appendChild(g1);
  const c = card(root, "Measurements and their coverage", "Counts come from the analysis pipeline (stats.json).");
  const rowsM = [
    ["Trees in the inventory (stem position, species)", "Field crew", "inventory.field_trees"],
    ["BHD plausible (within the accepted range)", "Field crew", "inventory.plausible_dbh"],
    ["Height, felled sample trees (tape or Vertex)", "Field crew", "inventory.sample_trees"],
    ["  of which Maßband length", "Field crew", "inventory.massband"],
    ["  of which Vertex length", "Field crew", "inventory.vertex"],
    ["  of which both methods", "Field crew", "inventory.both_methods"],
    ["Standing height written in the sheet", "Field crew", "inventory.standing_height"],
    ["ULS instances (single trees)", "ULS + SAT", "detection.lidar_instances"],
    ["Field trees found (detected)", "ULS + SAT", "detection.count_detected"],
    ["Field trees found but uncertain", "ULS + SAT", "detection.count_uncertain"],
    ["Field trees not found", "ULS + SAT", "detection.count_not_detected"],
    ["ULS instances without a field tree", "ULS + SAT", "detection.lidar_without_field_tree"],
  ].map(([what, who, id]) => { const e = S(id); return { what, who, n: e ? Math.round(e.value) : null, unit: e ? e.unit : "", st: e ? e.statement : "" }; });
  c.appendChild(mkTable([{ k: "what", label: "Measurement", cls: "l" }, { k: "who", label: "By", cls: "l" }, { k: "n", label: "Number" }, { k: "unit", label: "Unit", cls: "l" }, { k: "st", label: "What it says", cls: "wrap l" }], rowsM));
  const g2 = h("div", { class: "grid wide" });
  root.appendChild(g2);
  const cs = card(g2, "Trees per Parzelle and species", "Number of field trees (follows the Parzelle and identity filters).");
  const ts = treesF();
  const pls = unique(ts.map((t) => t.p)).sort((a, b) => a - b);
  const sp = speciesOrder.slice(0, 3).concat(["other species"]);
  const spOf = (t) => (speciesOrder.slice(0, 3).includes(t.sp) ? t.sp : "other species");
  plot(cs, sp.map((s) => ({ type: "bar", name: s, x: pls.map(String), y: pls.map((p) => ts.filter((t) => t.p === p && spOf(t) === s).length), marker: { color: s === "other species" ? C.neutral : C.species[s] } })),
    { barmode: "stack", xaxis: { title: "Parzelle", type: "category" }, yaxis: { title: "Number of trees" } });
  const cb = card(g2, "BHD distribution, field and ULS", "Breast-height diameter in 20 mm classes. Field BHD within the accepted range; ULS BHD only where a circle fit exists.");
  const m = matchedF();
  plot(cb, [
    { type: "histogram", x: ts.filter((t) => t.valid !== false).map((t) => t.dbh).filter(isNum), name: "Field BHD (plausible values)", xbins: { size: 20 }, marker: { color: C.ground }, opacity: 0.75 },
    { type: "histogram", x: m.filter((t) => t.dbhq === "ok").map((t) => t.dbhU).filter(isNum), name: "ULS BHD (ok fits)", xbins: { size: 20 }, marker: { color: C.uls }, opacity: 0.7 },
  ], { barmode: "overlay", xaxis: { title: "BHD (mm)" }, yaxis: { title: "Number of trees" } });
  root.appendChild(h("h2", { text: "Data-quality ledger" }));
  root.appendChild(h("p", { class: "lede", text: "One row per issue found in the ground records or in the matching. Use the filters to build a follow-up list for the field crew, then download it." }));
  const cl = card(root, null, null);
  const counts = [...groupBy(D.ledger, (r) => r.issue)].sort((a, b) => b[1].length - a[1].length);
  const ctl = h("div", { class: "row-controls" });
  const sel = h("select", null, h("option", { value: "", text: "All issue types (" + D.ledger.length + ")" }), counts.map(([k, v]) => h("option", { value: k, text: k.replace(/_/g, " ") + " (" + v.length + ")" })));
  const txt = h("input", { type: "search", placeholder: "Filter by tree id or Parzelle", "aria-label": "Filter ledger" });
  ctl.append(h("label", null, "Issue ", sel), h("label", null, "Search ", txt));
  cl.appendChild(ctl);
  const holder = h("div");
  cl.appendChild(holder);
  const dlh = h("div", { class: "dl-row" });
  const ledgerRows = () => D.ledger.filter((r) => (!sel.value || r.issue === sel.value) && (!txt.value || (r.tree_id + " " + r.parzelle).toLowerCase().includes(txt.value.toLowerCase())) && (F.p === "all" || String(r.parzelle) === F.p));
  function drawLedger() {
    holder.innerHTML = "";
    const rows = ledgerRows();
    holder.appendChild(h("p", { class: "cap", text: rows.length + " of " + D.ledger.length + " rows" }));
    holder.appendChild(mkTable([{ k: "tree_id", label: "Tree", cls: "l" }, { k: "parzelle", label: "Parzelle" }, { k: "issue", label: "Issue", cls: "l", fmt: (r) => r.issue.replace(/_/g, " ") }, { k: "detail", label: "Detail", cls: "wrap l" }], rows, { scroll: true }));
  }
  sel.addEventListener("change", drawLedger); txt.addEventListener("input", drawLedger);
  drawLedger();
  root.appendChild(h("h2", { text: "Downloads (built in the browser from the embedded data)" }));
  root.appendChild(downloadRow());
  cl.appendChild(h("p", { class: "legend-note", text: "The ledger download holds the rows currently shown." }));
}
function downloadRow() {
  const r = h("div", { class: "dl-row" });
  r.append(
    dlButton("Trees (current filters) CSV", "trees.csv", () => treesF(), TREE_COLS),
    dlButton("Sample trees CSV", "sample_trees.csv", () => samplesF(), ["tree_id", "parzelle", "species", "height_massband_m", "height_vertex_m", "height_uls_m", "diff_uls_minus_massband_m", "diff_vertex_minus_massband_m", "diff_uls_minus_vertex_m", "identity_class", "detection", "cause", "dbh_field_mm", "dbh_uls_mm", "remark", "explanation"]),
    dlButton("Ledger CSV", "ledger.csv", () => D.ledger.filter((x) => F.p === "all" || String(x.parzelle) === F.p), ["tree_id", "parzelle", "issue", "detail"]),
    dlButton("Volume per tree (current filters) CSV", "volume.csv", () => treesF(), ["id", "p", "sp", "dbh", "dbhU", "hU", "hsrc", "vF", "vU", "det", "idc", "rem"]),
  );
  return r;
}

/* ======================================================================
   Match confidence
   ====================================================================== */
function renderMatch(root) {
  root.appendChild(h("p", { class: "lede", text: "A height or BHD comparison is only valid if the field tree and the ULS instance are the same tree. Identity is judged from geometry only (distance after a per-Parzelle fit, uniqueness, neighbourhood, stem points); height and BHD never enter the score." }));
  const g = h("div", { class: "grid" });
  root.appendChild(g);
  let c = card(g, "Identity classes", "Number of field trees per class. Confirmed and probable matches are treated as the same tree.");
  classBarChart(c);
  c = card(g, "Distance from field stem to ULS instance base", "Raw distance and distance after the per-Parzelle shift and rotation fit; trees beyond the matching limit have no instance.");
  const ts = treesF();
  plot(c, [
    { type: "histogram", x: ts.map((t) => t.off).filter(isNum), name: "raw", xbins: { size: 0.1, start: 0 }, marker: { color: C.ground }, opacity: 0.75 },
    { type: "histogram", x: ts.map((t) => t.offc).filter(isNum), name: "after plot fit", xbins: { size: 0.1, start: 0 }, marker: { color: C.uls }, opacity: 0.7 },
  ], { barmode: "overlay", xaxis: { title: "Distance stem to instance base (m)" }, yaxis: { title: "Number of trees" } });
  c.appendChild(h("p", { class: "cap", text: [stmt("match.distance_median"), stmt("match.distance_p95")].join(" ") }));
  c = card(g, "Real matches against chance", "A match is only evidence of identity if it beats what random positions would give. Stems shifted by a fixed distance are the chance control.");
  const real = S("match.within_high_confidence_real"), ctl = S("match.within_high_confidence_control");
  if (real && ctl) plot(c, [{ type: "bar", x: ["Real stems", "Shifted stems (chance)"], y: [real.value * 100, ctl.value * 100], marker: { color: [C.uls, C.neutral] }, text: [pct(real.value), pct(ctl.value)], textposition: "outside", cliponaxis: false }],
    { yaxis: { title: "Trees with an instance base within the high-confidence distance (%)", range: [0, 105] } }, { cls: "short" });
  c.appendChild(h("p", { class: "cap", text: stmt("match.within_high_confidence_control") }));
  c.appendChild(h("p", { class: "legend-note", text: "False-accept rate and recall of the identity score (printed by the identity stage) are not stored in stats.json and are therefore not shown here; the chance control above is the available measure." }));
  c = card(g, "Accuracy for confirmed matches only", "Does the error shrink when only confirmed or probable matches are used? Bars show the RMSE with its 95 % bootstrap interval; n under each bar.");
  accuracyByIdentity(c);
  const wide = h("div", { class: "grid wide" });
  root.appendChild(wide);
  c = card(wide, "Per-Parzelle plot offsets", "Each circle is a Parzelle (map coordinates, m), coloured by the residual after the fit; the arrow is the shift applied to the field stems, drawn 80 times longer than real. A large residual is a warning (mirrored y, wrong corners).");
  offsetMap(c);
  c = card(wide, "Stem map of one Parzelle", "Field stems in the plot frame; colour = identity class, shape = detection, size = BHD.");
  stemMap(c);
  root.appendChild(h("h2", { text: "Offsets and orientation per Parzelle" }));
  const ct = card(root, null, "Best orientation is the stem layout that fits the ULS instances best; a mirrored y axis is set in the site configuration.");
  ct.appendChild(mkTable([
    { k: "parzelle", label: "Parzelle" }, { k: "n_pairs", label: "Pairs" }, { k: "dx_m", label: "dx (m)", fmt: (r) => f(r.dx_m, 2, true) }, { k: "dy_m", label: "dy (m)", fmt: (r) => f(r.dy_m, 2, true) },
    { k: "rotation_deg", label: "Rotation (°)", fmt: (r) => f(r.rotation_deg, 2, true) }, { k: "residual_rms_m", label: "Residual RMS (m)", fmt: (r) => f(r.residual_rms_m, 2) },
    { k: "best_orientation", label: "Best orientation", cls: "l" }, { k: "y_mirrored", label: "y mirrored in config", fmt: (r) => (r.y_mirrored ? "yes" : "no"), cls: "l" },
  ], parzF(), { scroll: true }));
  root.appendChild(h("p", { class: "legend-note", text: "Missing: sensitivity of the detection rate to the matching radius (1 to 4 m) is planned but not in the data files." }));
}
function accuracyByIdentity(parent) {
  const rows = (D.identity.accuracy || []).filter((r) => r.quantity === "height");
  if (!rows.length) return noData(parent);
  const subs = ["all_matched", "detected_only", "confirmed_or_probable", "confirmed_only"];
  const names = { massband: "ULS vs Maßband", vertex: "ULS vs Vertex", mean: "ULS vs mean of both" };
  const colors = { massband: C.ground, vertex: C.vertex, mean: C.uls };
  const tr = ["massband", "vertex", "mean"].map((gr) => {
    const rs = subs.map((s) => rows.find((r) => r.group === gr && r.subset === s));
    return { type: "bar", name: names[gr], x: subs.map((s) => s.replace(/_/g, " ").replace(" ", "<br>")), customdata: rs.map((r) => (r ? r.n : null)), y: rs.map((r) => (r ? r.rmse : null)), marker: { color: colors[gr] },
      error_y: { type: "data", symmetric: false, array: rs.map((r) => (r ? r.rmse_ci_high - r.rmse : 0)), arrayminus: rs.map((r) => (r ? r.rmse - r.rmse_ci_low : 0)), color: C.ink2, thickness: 1 },
      hovertemplate: "%{x}<br>RMSE %{y:.2f} m, n=%{customdata}<extra>" + names[gr] + "</extra>" };
  });
  plot(parent, tr, { barmode: "group", yaxis: { title: "Height RMSE on sample trees (m)", rangemode: "tozero" }, xaxis: { tickangle: 0 }, legend: { y: -0.3 } });
  parent.appendChild(mkTable([{ k: "s", label: "Subset", cls: "l" }, ...["massband", "vertex", "mean"].map((g) => ({ k: g, label: names[g] + ": n" }))], subs.map((s) => ({ s: s.replace(/_/g, " "), ...Object.fromEntries(["massband", "vertex", "mean"].map((g) => [g, (rows.find((r) => r.group === g && r.subset === s) || {}).n])) }))));
  parent.appendChild(h("p", { class: "cap", text: stmt("identity.height_massband.rmse_change_confirmed", "confirmed_only") }));
}
function offsetMap(parent) {
  const ps = PZ.filter((p) => isNum(p.mx) && isNum(p.dx_m));
  if (!ps.length) return noData(parent);
  const k = 80;
  const ann = ps.map((p) => {
    const a = (p.frame_deg || 0) * Math.PI / 180, ux = p.dx_m * Math.cos(a) - p.dy_m * Math.sin(a), uy = p.dx_m * Math.sin(a) + p.dy_m * Math.cos(a);
    return { x: p.mx + ux * k, y: p.my + uy * k, ax: p.mx, ay: p.my, xref: "x", yref: "y", axref: "x", ayref: "y", showarrow: true, arrowhead: 2, arrowsize: 1, arrowwidth: 1.6, arrowcolor: C.ink };
  });
  plot(parent, [{ type: "scatter", mode: "markers+text", x: ps.map((p) => p.mx), y: ps.map((p) => p.my), text: ps.map((p) => String(p.parzelle)), textposition: "top center", textfont: { size: 10, color: C.ink2 },
    marker: { size: 13, color: ps.map((p) => p.residual_rms_m), colorscale: seq(), colorbar: { title: { text: "Residual RMS (m)" }, thickness: 12, len: 0.8 }, line: { color: C.surface, width: 1 } },
    customdata: ps.map((p) => [p.dx_m, p.dy_m, p.rotation_deg, p.n_pairs]),
    hovertemplate: "Parzelle %{text}<br>shift dx %{customdata[0]:.2f} m, dy %{customdata[1]:.2f} m<br>rotation %{customdata[2]:.2f}°, pairs %{customdata[3]}<br>residual %{marker.color:.2f} m<extra></extra>" }],
  { annotations: ann, xaxis: { title: "East (m, relative to the south-west corner)" }, yaxis: { title: "North (m)", scaleanchor: "x" }, margin: { r: 70 } }, { cls: "tall" });
}
function stemMap(parent) {
  const plots = unique(T.map((t) => t.p)).sort((a, b) => a - b);
  const sel = h("select", { "aria-label": "Parzelle for the stem map" }, plots.map((p) => h("option", { value: p, text: "Parzelle " + p })));
  sel.value = F.p !== "all" ? F.p : HASH.stem || String(plots[0]);
  if (!plots.map(String).includes(sel.value)) sel.value = String(plots[0]);
  const ctl = h("div", { class: "row-controls" }, h("label", null, "Show ", sel));
  const info = h("span", { class: "badge" });
  ctl.appendChild(info);
  parent.appendChild(ctl);
  const holder = h("div");
  parent.appendChild(holder);
  function draw() {
    chartCount -= holder.querySelector(".chart") ? 1 : 0;
    holder.innerHTML = "";
    HASH.stem = sel.value; writeHash();
    const ts = T.filter((t) => String(t.p) === sel.value && isNum(t.x) && isNum(t.y));
    const pz = PZ.find((p) => String(p.parzelle) === sel.value);
    info.textContent = pz && pz.y_mirrored ? "y axis mirrored for this Parzelle" : "y axis as in the sheet";
    const symbol = { detected: "circle", uncertain: "diamond", "not detected": "x" };
    const tr = [];
    IDC.forEach((c) => DET.forEach((d) => {
      const g = ts.filter((t) => t.idc === c && t.det === d);
      if (!g.length) return;
      tr.push({ type: "scatter", mode: "markers", name: c + (d === "detected" ? "" : ", " + d), x: g.map((t) => t.x), y: g.map((t) => t.y), text: g.map((t) => t.id),
        marker: { color: C.idc[c], symbol: symbol[d], size: g.map((t) => Math.max(6, Math.min(22, (t.dbh || 250) / 25))), line: { color: C.surface, width: d === "not detected" ? 2 : 1 } },
        customdata: g.map((t) => [t.dbh, t.hU, t.sc]), hovertemplate: "%{text}<br>BHD %{customdata[0]} mm, ULS height %{customdata[1]:.1f} m<br>score %{customdata[2]:.2f}<extra></extra>" });
    }));
    plot(holder, tr, { xaxis: { title: "Local x (m)" }, yaxis: { title: "Local y (m)", scaleanchor: "x" }, legend: { y: -0.3, font: { size: 10 } } }, { cls: "tall" });
  }
  sel.addEventListener("change", draw);
  draw();
}

/* ======================================================================
   Height
   ====================================================================== */
const PAIRS = [
  { k: "um", x: "height_massband_m", y: "height_uls_m", xl: "Maßband length", yl: "ULS height", name: "ULS vs Maßband", col: () => C.uls, stat: "height.massband" },
  { k: "uv", x: "height_vertex_m", y: "height_uls_m", xl: "Vertex length", yl: "ULS height", name: "ULS vs Vertex", col: () => C.uls, stat: "height.vertex" },
  { k: "vm", x: "height_massband_m", y: "height_vertex_m", xl: "Maßband length", yl: "Vertex length", name: "Vertex vs Maßband", col: () => C.vertex, stat: "height.vertex_vs_massband" },
];
function renderHeight(root) {
  root.appendChild(h("p", { class: "lede", text: "Only the felled sample trees have a field height. The Maßband (tape) length is the reference. The Vertex difference to the Maßband is the noise floor of ground measurement: ULS can only be judged against it. Vertex and mean of both are evaluated separately, not as reference." }));
  const all = SAMPLES.filter((s) => isNum(s.height_massband_m) && isNum(s.height_uls_m));
  if (all.length) {
    const w = all.reduce((a, b) => (Math.abs(b.diff_uls_minus_massband_m) > Math.abs(a.diff_uls_minus_massband_m) ? b : a), all[0]);
    if (Math.abs(w.diff_uls_minus_massband_m) > META.explain_height_diff_m) {
      const m = S("height.massband.rmse", "all_matched"), md = S("height.massband.median", "all_matched"), mad = S("height.massband.mad", "all_matched"), loo = S("height.massband.rmse_without_largest_influence", "all_matched");
      const co = h("div", { class: "callout" });
      co.appendChild(h("b", { text: `One sample tree dominates the RMSE: ${w.tree_id}. ` }));
      co.appendChild(document.createTextNode(`ULS minus Maßband is ${f(w.diff_uls_minus_massband_m, 1, true)} m (Maßband ${f(w.height_massband_m, 1)} m, ULS ${f(w.height_uls_m, 1)} m, identity class ${w.identity_class}). `));
      if (m) co.appendChild(document.createTextNode(`RMSE ${fmtVal(m)}`));
      if (md && mad) co.appendChild(document.createTextNode(`, but median ${fmtVal(md)} and MAD ${fmtVal(mad)}`));
      if (loo) co.appendChild(document.createTextNode(`; without this tree the RMSE is ${fmtVal(loo)}`));
      co.appendChild(document.createTextNode(". The tree is kept in every statistic below and marked in every chart. "));
      if (w.explanation) co.appendChild(h("div", { class: "legend-note", text: w.explanation }));
      root.appendChild(co);
    }
  }
  const ss = samplesF();
  root.appendChild(h("h2", { text: "Three pairwise comparisons" }));
  root.appendChild(h("p", { class: "cap", text: "Left: 1:1 scatter. Right: Bland-Altman (difference against mean; solid line bias, dashed median, dotted bias ± 1.96 SD). Charts follow the filters." }));
  const g = h("div", { class: "grid two" });
  root.appendChild(g);
  PAIRS.forEach((p) => {
    const rows = ss.filter((s) => isNum(s[p.x]) && isNum(s[p.y]));
    const c1 = card(g, p.name + ": 1:1", p.yl + " against " + p.xl + ", n = " + rows.length);
    scatter11(c1, ss, p.x, p.y, p.xl + " (m)", p.yl + " (m)", p.col());
    const c2 = card(g, p.name + ": Bland-Altman", "Difference " + p.yl + " minus " + p.xl);
    blandAltman(c2, ss, p.x, p.y, p.xl, p.yl, p.col());
  });
  root.appendChild(h("h2", { text: "Accuracy table" }));
  root.appendChild(h("p", { class: "cap", text: "Bias, RMSE, median and MAD with 95 % bootstrap intervals, in m. Median and MAD are robust to the one large error. Left: all matched trees; right: detected trees only. Fixed values from the analysis; they do not follow the filters." }));
  const tg = h("div", { class: "grid two" });
  root.appendChild(tg);
  ["all_matched", "detected_only"].forEach((sub) => {
    const c = card(tg, sub === "all_matched" ? "All matched trees (detected and uncertain)" : "Detected trees only", null);
    const rows = PAIRS.map((p) => {
      const sb = p.k === "vm" ? null : sub, id = (m) => S(p.stat + "." + m, sb);
      const b = id("bias"), r = id("rmse");
      return { name: p.name, n: r ? r.n : null, bias: b, rmse: r, median: id("median"), mad: id("mad") };
    });
    const cell = (e) => (e ? `${f(e.value, 2)}${isNum(e.ci_low) ? " (" + f(e.ci_low, 2) + " to " + f(e.ci_high, 2) + ")" : ""}` : "–");
    c.appendChild(mkTable([{ k: "name", label: "Comparison", cls: "l" }, { k: "n", label: "n" }, { k: "bias", label: "Bias (m)", fmt: (r) => cell(r.bias), sort: (r) => r.bias && r.bias.value },
      { k: "rmse", label: "RMSE (m)", fmt: (r) => cell(r.rmse), sort: (r) => r.rmse && r.rmse.value }, { k: "median", label: "Median (m)", fmt: (r) => cell(r.median), sort: (r) => r.median && r.median.value },
      { k: "mad", label: "MAD (m)", fmt: (r) => cell(r.mad), sort: (r) => r.mad && r.mad.value }], rows));
    const loo = S("height.massband.rmse_without_largest_influence", sub), jk = S("height.massband.jackknife_se", sub);
    if (loo) c.appendChild(h("p", { class: "cap", text: loo.statement + (jk ? " Jackknife standard error of the bias: " + fmtVal(jk) + "." : "") }));
    if (sub === "all_matched") c.appendChild(h("p", { class: "legend-note", text: "Vertex vs Maßband has one row for both tables: it does not depend on whether ULS found the tree." }));
  });
  if (filtersActive()) {
    const c = card(root, "Same table for the current filters (computed in the browser)", "n is small: read with care. No intervals.");
    const rows = PAIRS.map((p) => { const rs = ss.filter((s) => isNum(s[p.x]) && isNum(s[p.y])).map((s) => s[p.y] - s[p.x]); const m = metrics(rs); return { name: p.name, n: m.n, bias: m.bias, rmse: m.rmse, median: m.median, mad: m.mad }; });
    c.appendChild(mkTable([{ k: "name", label: "Comparison", cls: "l" }, { k: "n", label: "n" }, ...["bias", "rmse", "median", "mad"].map((k) => ({ k, label: k.toUpperCase() + " (m)", fmt: (r) => f(r[k], 2) }))], rows));
  }
  root.appendChild(h("h2", { text: "Error budget: ULS against the ground noise floor" }));
  const eg = h("div", { class: "grid" });
  root.appendChild(eg);
  let c = card(eg, "RMSE against Maßband, ULS and Vertex", "Same 14 sample trees with Maßband, Vertex and ULS height.");
  const eu = S("error_budget.rmse_uls", "sample_trees_with_both"), ev = S("error_budget.rmse_vertex", "sample_trees_with_both");
  if (eu && ev) plot(c, [{ type: "bar", x: ["ULS", "Vertex"], y: [eu.value, ev.value], marker: { color: [C.uls, C.vertex] }, text: [fmtVal(eu), fmtVal(ev)], textposition: "outside", cliponaxis: false,
    error_y: { type: "data", symmetric: false, array: [eu.ci_high - eu.value, ev.ci_high - ev.value], arrayminus: [eu.value - eu.ci_low, ev.value - ev.ci_low], color: C.ink2, thickness: 1.2 } }],
  { yaxis: { title: "RMSE against Maßband (m)", rangemode: "tozero" } }, { cls: "short" });
  ["error_budget.rmse_difference", "error_budget.variance_ratio", "error_budget.share_inside_noise", "error_budget.verdict"].forEach((id) => { const e = S(id, "sample_trees_with_both"); if (e) c.appendChild(h("p", { class: "cap", text: e.statement })); });
  c = card(eg, "Leave-one-out: what one tree does to the RMSE", "RMSE against Maßband with all trees and without the single most influential tree.");
  const bars = ["all_matched", "detected_only"].map((s) => [S("height.massband.rmse", s), S("height.massband.rmse_without_largest_influence", s), s]);
  if (bars[0][0] && bars[0][1]) plot(c, [
    { type: "bar", name: "all trees", x: bars.map((b) => b[2].replace("_", " ")), y: bars.map((b) => b[0].value), marker: { color: C.ground }, text: bars.map((b) => f(b[0].value, 2) + " m"), textposition: "outside", cliponaxis: false },
    { type: "bar", name: "without the most influential tree", x: bars.map((b) => b[2].replace("_", " ")), y: bars.map((b) => b[1].value), marker: { color: C.uls }, text: bars.map((b) => f(b[1].value, 2) + " m"), textposition: "outside", cliponaxis: false },
  ], { barmode: "group", yaxis: { title: "RMSE against Maßband (m)", rangemode: "tozero" } }, { cls: "short" });
  c = card(eg, "Residuals against height and BHD", "ULS minus Maßband for each sample tree: no pattern can be claimed from so few trees.");
  const rs = ss.filter((s) => isNum(s.diff_uls_minus_massband_m));
  plot(c, [{ type: "scatter", mode: "markers", name: "against Maßband height", x: rs.map((s) => s.height_massband_m), y: rs.map((s) => s.diff_uls_minus_massband_m), text: rs.map((s) => s.tree_id), marker: { color: C.uls, size: 9, line: { color: C.surface, width: 1.5 } }, hovertemplate: "%{text}<br>%{x:.1f} m, %{y:.2f} m<extra></extra>" },
    { type: "scatter", mode: "markers", name: "against field BHD", xaxis: "x2", x: rs.map((s) => s.dbh_field_mm), y: rs.map((s) => s.diff_uls_minus_massband_m), text: rs.map((s) => s.tree_id), marker: { color: C.ground, size: 9, symbol: "square", line: { color: C.surface, width: 1.5 } }, hovertemplate: "%{text}<br>%{x:.0f} mm, %{y:.2f} m<extra></extra>" }],
  { xaxis: { title: "Maßband length (m)", domain: [0, 0.46] }, xaxis2: { title: "Field BHD (mm)", domain: [0.54, 1], gridcolor: C.grid, linecolor: C.line, automargin: true }, yaxis: { title: "ULS minus Maßband (m)" } }, { cls: "short" });
  c.appendChild(h("p", { class: "legend-note", text: "Neighbour density and crown class as predictors are not in the data files." }));
  root.appendChild(h("h2", { text: "Height definition variants" }));
  const cv = card(root, null, "Alternative ways to read the tree height from the ULS instance, compared with the Maßband. No variant is adopted unless the interval of the RMSE change is below 0; with this few trees that is unlikely.");
  if (D.height_variants.length) {
    cv.appendChild(mkTable([{ k: "variant", label: "Variant", cls: "l" }, { k: "n", label: "n" }, { k: "bias_m", label: "Bias (m)", fmt: (r) => f(r.bias_m, 2, true) }, { k: "rmse_m", label: "RMSE (m)", fmt: (r) => `${f(r.rmse_m, 2)} (${f(r.rmse_ci_low, 2)} to ${f(r.rmse_ci_high, 2)})` },
      { k: "median_m", label: "Median (m)", fmt: (r) => f(r.median_m, 2, true) }, { k: "mad_m", label: "MAD (m)", fmt: (r) => f(r.mad_m, 2) },
      { k: "loo", label: "Leave-one-out RMSE (m)", fmt: (r) => `${f(r.loo_rmse_min_m, 2)} to ${f(r.loo_rmse_max_m, 2)}` },
      { k: "rmse_change_vs_first", label: "RMSE change (m)", fmt: (r) => (r.rmse_change_vs_first ? `${f(r.rmse_change_vs_first, 2, true)} (${f(r.rmse_change_ci_low, 2, true)} to ${f(r.rmse_change_ci_high, 2, true)})` : "reference") },
      { k: "improvement_claimed", label: "Improvement claimed", fmt: (r) => (r.improvement_claimed ? "yes" : "no") }], D.height_variants));
    statsWith("height_variant.").forEach((e) => cv.appendChild(h("p", { class: "cap", text: e.statement })));
  } else noData(cv);
}

/* ======================================================================
   Where ULS wins
   ====================================================================== */
function renderWins(root) {
  root.appendChild(h("p", { class: "lede", text: "ULS measures every tree; the ground crew measured heights on only a few felled trees. This tab shows what that coverage buys, then, in its own part at the end, where ULS is weaker." }));
  const g = h("div", { class: "grid" });
  root.appendChild(g);
  let c = card(g, "Coverage: all trees against the sample trees", "Number of trees with each kind of measurement.");
  coverageChart(c);
  c.appendChild(h("p", { class: "cap", text: stmt("detection.rate", "all_matched") }));
  c = card(g, "Stand height distribution (ULS)", "Height of every matched tree. Not obtainable from the sample trees alone.");
  const ms = matchedF().filter((t) => isNum(t.hU));
  plot(c, [{ type: "histogram", x: ms.map((t) => t.hU), xbins: { size: 1 }, marker: { color: C.uls }, name: "ULS height (n=" + ms.length + ")" }], { xaxis: { title: "ULS tree height (m)" }, yaxis: { title: "Number of trees" } }, { cls: "short" });
  c.appendChild(h("p", { class: "cap", text: stmt("stand.mean_height", "all_matched") }));
  c = card(g, "Mean and top height per Parzelle", "Mean ULS height of all matched trees and top height (mean of the thickest trees per hectare) of each Parzelle.");
  const ps = parzF().filter((p) => isNum(p.mean_height_m)).sort((a, b) => a.top_height_m - b.top_height_m);
  plot(c, [{ type: "scatter", mode: "markers", name: "mean height", y: ps.map((p) => String(p.parzelle)), x: ps.map((p) => p.mean_height_m), marker: { color: C.uls, size: 8 } },
    { type: "scatter", mode: "markers", name: "top height", y: ps.map((p) => String(p.parzelle)), x: ps.map((p) => p.top_height_m), marker: { color: C.ground, size: 8, symbol: "diamond" } }],
  { xaxis: { title: "Height (m)" }, yaxis: { title: "Parzelle", type: "category", tickfont: { size: 9 } } }, { cls: "tall" });
  c.appendChild(h("p", { class: "cap", text: stmt("stand.top_height_median_over_plots", "all_matched") }));
  c = card(g, "Sampling value: stand mean from k sample trees", "How far a stand mean from k randomly drawn trees strays from the ULS census mean (95 % of draws).");
  const sv = D.volume.sampling || [];
  if (sv.length) {
    plot(c, [{ type: "scatter", mode: "lines", x: sv.map((r) => r.k_used).concat(sv.map((r) => r.k_used).reverse()), y: sv.map((r) => r.q975_m).concat(sv.map((r) => r.q025_m).reverse()), fill: "toself", fillcolor: C.c[0] + "40", line: { color: "rgba(0,0,0,0)" }, name: "95 % of draws", hoverinfo: "skip" },
      { type: "scatter", mode: "lines+markers", x: sv.map((r) => r.k_used), y: sv.map((r) => r.estimate_mean_m), name: "mean of draws", line: { color: C.uls, width: 2 }, marker: { size: 7 } },
      { type: "scatter", mode: "lines", x: [sv[0].k_used, sv[sv.length - 1].k_used], y: [sv[0].census_mean_m, sv[0].census_mean_m], name: "ULS census mean", line: { color: C.ink, width: 1.5, dash: "dash" } }],
    { xaxis: { title: "Number of sample trees drawn (k)", type: "log", tickvals: sv.map((r) => r.k_used), ticktext: sv.map((r) => String(r.k_used)) }, yaxis: { title: "Stand mean height (m)" } });
    c.appendChild(h("p", { class: "cap", text: stmt("sampling.k5.sd") + " " + stmt("sampling.sample_trees_mean_minus_census") }));
  } else noData(c);
  const wide = h("div", { class: "grid wide" });
  root.appendChild(wide);
  c = card(wide, "Height-diameter curves", "ULS height against field BHD (plausible BHD only) with the fitted curve per species (Naeslund form, used for the modelled heights). Maps below: one circle per Parzelle at its position.");
  heightDiameter(c);
  root.appendChild(h("h2", { text: "Maps per Parzelle" }));
  plotMaps(root.appendChild(h("div")));
  const g2 = h("div", { class: "grid" });
  root.appendChild(g2);
  c = card(g2, "Detection by BHD class", "Share of field trees detected, with 95 % Wilson interval.");
  detectionByClassChart(c);
  c.appendChild(h("p", { class: "cap", text: stmt("detection.rate", "detected_only") }));
  c = card(g2, "Detection model", "Logistic regression of detected against uncertain or not found, per standard deviation of each predictor (log-odds, 95 % bootstrap interval). Right of 0 = more likely detected.");
  const co = D.detection_model.filter((r) => r.kind === "coefficient");
  if (co.length) {
    plot(c, [{ type: "scatter", mode: "markers", x: co.map((r) => r.value), y: co.map((r) => r.term), marker: { color: C.uls, size: 10 }, error_x: { type: "data", symmetric: false, array: co.map((r) => r.ci_high - r.value), arrayminus: co.map((r) => r.value - r.ci_low), color: C.ink2, thickness: 1.5 },
      hovertemplate: "%{y}: %{x:.2f} log-odds per SD<extra></extra>" }], { xaxis: { title: "Effect on log-odds of detection (per SD)", zeroline: true }, yaxis: { autorange: "reversed" }, margin: { l: 90 }, shapes: [lineShape(0, -0.5, 0, co.length - 0.5, "solid")] }, { cls: "short" });
    c.appendChild(h("p", { class: "cap", text: stmt("detection_model.diameter") }));
    c.appendChild(h("p", { class: "legend-note", text: "BHD and relative size are strongly correlated, so their separate effects are poorly determined; read the rates per class." }));
  } else noData(c);
  c = card(g2, "Why trees were missed or uncertain", "Cause assigned by the matching analysis, field trees not detected or uncertain (follows the Parzelle and identity filters).");
  causesChart(c);
  c = card(g2, "SegmentAnyTree against a canopy-model baseline", "Share of field trees found by each method, and in the smallest fifth by BHD. Baseline: local maxima of the canopy height model.");
  baselineChart(c);
  c = card(g2, "Trees ULS sees that nobody recorded", "ULS instances without a field tree, by what the analysis reads from them.");
  unmatchedChart(c);
  c.appendChild(h("p", { class: "cap", text: stmt("detection.lidar_without_field_tree") }));
  root.appendChild(h("h2", { class: "part", text: "Where ULS is weaker" }));
  root.appendChild(h("p", { class: "lede", text: "ULS is not a replacement everywhere. These are the weak spots found in this site's data." }));
  weakPart(root);
}
function heightDiameter(parent) {
  const ms = matchedF().filter((t) => isNum(t.hU) && isNum(t.dbh) && t.valid !== false);
  const sps = speciesOrder.slice(0, 3);
  const tr = [];
  sps.forEach((s) => {
    const g = ms.filter((t) => t.sp === s);
    if (g.length) tr.push({ type: "scatter", mode: "markers", name: s + " (n=" + g.length + ")", x: g.map((t) => t.dbh), y: g.map((t) => t.hU), text: g.map((t) => t.id), marker: { color: C.species[s], size: 5, opacity: 0.55 }, hovertemplate: "%{text}<br>BHD %{x:.0f} mm, height %{y:.1f} m<extra></extra>" });
  });
  const other = ms.filter((t) => !sps.includes(t.sp));
  if (other.length) tr.push({ type: "scatter", mode: "markers", name: "other species (n=" + other.length + ")", x: other.map((t) => t.dbh), y: other.map((t) => t.hU), marker: { color: C.neutral, size: 5, opacity: 0.6 } });
  const curves = D.volume.height_curves || [];
  curves.forEach((cv) => {
    const xs = []; for (let d = 100; d <= 800; d += 10) xs.push(d);
    const name = cv.scope === "pooled" ? "curve, all species" : "curve " + cv.species;
    tr.push({ type: "scatter", mode: "lines", name, x: xs, y: xs.map((d) => 1.3 + Math.pow((d / 10) / (cv.a + cv.b * (d / 10)), 2)), line: { color: cv.scope === "pooled" ? C.ink : C.species[cv.species] || C.neutral, width: cv.scope === "pooled" ? 2 : 2, dash: cv.scope === "pooled" ? "dash" : "solid" } });
  });
  plot(parent, tr, { xaxis: { title: "Field BHD (mm)" }, yaxis: { title: "ULS tree height (m)" } }, { cls: "tall" });
  statsWith("volume.height_curve.", undefined).filter((e) => e.id.endsWith("loo_rmse")).forEach((e) => parent.appendChild(h("p", { class: "cap", text: e.statement })));
}
function plotMaps(parent) {
  const ps = parzF().filter((p) => isNum(p.mx));
  if (!ps.length) return noData(parent);
  const grid = h("div", { class: "grid" });
  parent.appendChild(grid);
  const one = (title, unit, vals, cs, mid) => {
    const c = card(grid, title, null);
    plot(c, [{ type: "scatter", mode: "markers+text", x: ps.map((p) => p.mx), y: ps.map((p) => p.my), text: ps.map((p) => String(p.parzelle)), textposition: "top center", textfont: { size: 9, color: C.ink2 },
      marker: { size: 12, color: vals, colorscale: cs, cmid: mid, colorbar: { title: { text: unit, font: { size: 11 } }, thickness: 10, len: 0.9 }, line: { color: C.surface, width: 1 } },
      hovertemplate: "Parzelle %{text}: %{marker.color:.2f} " + unit + "<extra></extra>" }],
    { xaxis: { title: "East (m)" }, yaxis: { title: "North (m)", scaleanchor: "x" }, showlegend: false, margin: { r: 70 } }, { cls: "tall" });
  };
  one("Detection rate", "% of trees", ps.map((p) => p.detection_rate * 100), seq());
  one("BHD bias, ULS minus field", "cm", ps.map((p) => p.dbh_bias_uls_minus_field_cm), div3(), 0);
  one("Mean ULS height", "m", ps.map((p) => p.mean_height_m), seq());
}
function causesChart(parent) {
  const ts = treesF().filter((t) => t.det !== "detected");
  const causes = unique(ts.map((t) => t.cause || "no cause recorded"));
  const order = causes.sort((a, b) => ts.filter((t) => (t.cause || "no cause recorded") === b).length - ts.filter((t) => (t.cause || "no cause recorded") === a).length);
  if (!ts.length) return noData(parent, "No missed or uncertain trees in the current selection.");
  plot(parent, ["not detected", "uncertain"].map((d) => ({ type: "bar", orientation: "h", name: d, y: order, x: order.map((c) => ts.filter((t) => t.det === d && (t.cause || "no cause recorded") === c).length), marker: { color: C.det[d] } })),
    { barmode: "stack", xaxis: { title: "Number of field trees" }, yaxis: { autorange: "reversed", automargin: true }, margin: { l: 220 }, legend: { y: -0.25 } }, { cls: "short" });
}
function baselineChart(parent) {
  const ms = ["segmentanytree_base_matching", "segmentanytree_instance_tops", "best_local_maxima"];
  const rows = ms.map((m) => ({ m, found: S("baseline." + m + ".found"), small: S("baseline." + m + ".found_smallest_fifth"), prec: S("baseline." + m + ".precision"), f1: S("baseline." + m + ".f1"), rm: S("baseline." + m + ".massband_rmse"), md: S("baseline." + m + ".massband_median") })).filter((r) => r.found);
  if (!rows.length) return noData(parent);
  const ms2 = rows.map((r) => METHOD_NAMES[r.m] || prettyId(r.m));
  plot(parent, [{ type: "bar", name: "all field trees", x: ms2, y: rows.map((r) => r.found.value * 100), marker: { color: C.uls }, text: rows.map((r) => pct(r.found.value, 0)), textposition: "outside", cliponaxis: false },
    { type: "bar", name: "smallest fifth by BHD", x: ms2, y: rows.map((r) => (r.small ? r.small.value * 100 : null)), marker: { color: C.ground }, text: rows.map((r) => (r.small ? pct(r.small.value, 0) : "")), textposition: "outside", cliponaxis: false }],
  { barmode: "group", yaxis: { title: "Field trees found (%)", range: [0, 110] }, xaxis: { tickfont: { size: 10 } } }, { cls: "short" });
  parent.appendChild(mkTable([{ k: "name", label: "Method", cls: "l" }, { k: "prec", label: "Precision" }, { k: "f1", label: "F1" }, { k: "rm", label: "Height RMSE vs Maßband" }, { k: "md", label: "Median" }],
    rows.map((r, i) => ({ name: ms2[i], prec: r.prec ? pct(r.prec.value, 0) : "–", f1: r.f1 ? f(r.f1.value, 2) : "–", rm: r.rm ? fmtVal(r.rm) : "–", md: r.md ? fmtVal(r.md) : "–" }))));
  parent.appendChild(h("p", { class: "cap", text: rows.map((r) => stmt("baseline." + r.m + ".found")).join(" ") }));
}
function unmatchedChart(parent) {
  const u = D.unmatched_instances.filter((r) => F.p === "all" || String(r.parzelle) === F.p);
  if (!u.length) return noData(parent);
  const g = [...groupBy(u, (r) => r.reading || "no reading")].sort((a, b) => b[1].length - a[1].length);
  plot(parent, [{ type: "bar", orientation: "h", y: g.map((x) => x[0]), x: g.map((x) => x[1].length), marker: { color: C.uls }, text: g.map((x) => String(x[1].length)), textposition: "outside", cliponaxis: false }],
    { xaxis: { title: "Number of ULS instances" }, yaxis: { autorange: "reversed", automargin: true }, margin: { l: 260, r: 40 } }, { cls: "short" });
}
function weakPart(root) {
  const g = h("div", { class: "grid" });
  root.appendChild(g);
  let c = card(g, "Small trees under a neighbour's crown", "Detection falls with tree size, and merged crowns are the main cause of missed trees.");
  detectionByClassChart(c);
  ["detection_model.rate_0_200_mm|detected_only", "baseline.best_local_maxima.found_smallest_fifth|", "causes.not_detected.merged_with_a_neighbour|"].forEach((k) => { const [id, s] = k.split("|"); const t = stmt(id, s || undefined); if (t) c.appendChild(h("p", { class: "cap", text: t })); });
  c = card(g, "Detection rate differs between Parzellen", "Share of field trees detected per Parzelle.");
  const ps = parzF().filter((p) => isNum(p.detection_rate)).sort((a, b) => a.detection_rate - b.detection_rate);
  plot(c, [{ type: "bar", x: ps.map((p) => String(p.parzelle)), y: ps.map((p) => p.detection_rate * 100), marker: { color: C.uls }, hovertemplate: "Parzelle %{x}: %{y:.0f} %<extra></extra>" }],
    { xaxis: { title: "Parzelle (sorted by detection rate)", type: "category", tickfont: { size: 8 } }, yaxis: { title: "Trees detected (%)", range: [0, 105] } }, { cls: "short" });
  c.appendChild(h("p", { class: "cap", text: stmt("stand.detection_rate_range_over_plots", "detected_only") + " " + stmt("detection.parzelle_rate_min", "detected_only") }));
  c = card(g, "BHD: ULS minus field by BHD class", "Differences per class (box = middle half, line = median). Matched trees with an ok circle fit.");
  const ms = matchedF().filter((t) => t.dbhq === "ok" && isNum(t.dbhU) && isNum(t.dbh) && t.cls);
  const classes = unique(ms.map((t) => t.cls)).sort((a, b) => clsKey(a) - clsKey(b));
  plot(c, classes.map((k) => ({ type: "box", name: k, y: ms.filter((t) => t.cls === k).map((t) => (t.dbhU - t.dbh) / 10), marker: { color: C.uls }, line: { width: 1.2 }, boxpoints: false })),
    { xaxis: { title: "Field BHD class (mm)" }, yaxis: { title: "ULS minus field BHD (cm)", zeroline: true }, showlegend: false }, { cls: "short" });
  c.appendChild(h("p", { class: "cap", text: stmt("identity.dbh_all_field_trees.all_matched.bias", "all_matched") }));
  c = card(g, "Height: one tree off by a lot", "The sample tree with the largest ULS-Maßband difference (see the Height tab).");
  const all = SAMPLES.filter((s) => isNum(s.diff_uls_minus_massband_m));
  if (all.length) {
    const w = all.reduce((a, b) => (Math.abs(b.diff_uls_minus_massband_m) > Math.abs(a.diff_uls_minus_massband_m) ? b : a), all[0]);
    c.appendChild(h("p", { text: `${w.tree_id}: ULS minus Maßband = ${f(w.diff_uls_minus_massband_m, 1, true)} m. ${w.explanation || ""}` }));
  }
  c = card(g, "Harvest record against LiDAR", "Trees whose harvest record disagrees with what the ULS point clouds show.");
  ["harvest.agreement", "harvest.marked_but_standing", "harvest.not_marked_but_gone", "ledger.harvest_record_disagrees"].forEach((id) => { const t = stmt(id); if (t) c.appendChild(h("p", { class: "cap", text: t })); });
  c = card(g, "Questions that remain open", "From the data-quality ledger.");
  const open = ["ledger.sample_tree_stands_after_harvest", "ledger.sample_tree_remark", "ledger.sample_tree_height_differs", "ledger.identity_ambiguous", "ledger.identity_rejected", "ledger.unmatched"];
  open.forEach((id) => { const t = stmt(id); if (t) c.appendChild(h("p", { class: "cap", text: t })); });
}

/* ======================================================================
   Sample explorer
   ====================================================================== */
let selectedSample = null;
function renderSamples(root) {
  root.appendChild(h("p", { class: "lede", text: "The felled sample trees, one by one. Click a row or a case to see the three heights, the identity of the match and what the analysis says about it." }));
  const rows = samplesF();
  const strip = h("div", { class: "notable" });
  root.appendChild(strip);
  const withBoth = SAMPLES.filter((s) => isNum(s.diff_uls_minus_massband_m));
  const withVM = SAMPLES.filter((s) => isNum(s.diff_vertex_minus_massband_m));
  const best = withBoth.length ? withBoth.reduce((a, b) => (Math.abs(b.diff_uls_minus_massband_m) < Math.abs(a.diff_uls_minus_massband_m) ? b : a)) : null;
  const worstU = withBoth.length ? withBoth.reduce((a, b) => (Math.abs(b.diff_uls_minus_massband_m) > Math.abs(a.diff_uls_minus_massband_m) ? b : a)) : null;
  const worstV = withVM.length ? withVM.reduce((a, b) => (Math.abs(b.diff_vertex_minus_massband_m) > Math.abs(a.diff_vertex_minus_massband_m) ? b : a)) : null;
  const openQ = SAMPLES.filter((s) => s.cause).sort((a, b) => Math.abs(b.diff_uls_minus_massband_m || 0) - Math.abs(a.diff_uls_minus_massband_m || 0))[0] || SAMPLES.find((s) => s.explanation);
  const nc = (k, s, v, d) => s && strip.appendChild(h("button", { type: "button", onclick: () => { selectedSample = s; HASH.tree = s.tree_id; writeHash(); drawDetail(); tbl.redraw(); document.getElementById("detail-anchor").scrollIntoView({ behavior: "smooth", block: "nearest" }); } },
    h("div", { class: "k", text: k }), h("div", { class: "v", text: s.tree_id }), h("div", { class: "d", text: d })));
  nc("Best agreement", best, null, best && `ULS minus Maßband ${f(best.diff_uls_minus_massband_m, 2, true)} m`);
  nc("Largest ULS-Maßband difference", worstU, null, worstU && `${f(worstU.diff_uls_minus_massband_m, 1, true)} m, identity ${worstU.identity_class}`);
  nc("Largest Vertex-Maßband difference", worstV, null, worstV && `${f(worstV.diff_vertex_minus_massband_m, 2, true)} m`);
  nc("Open field question", openQ, null, openQ && (openQ.cause || "explanation in the detail panel"));
  const g = h("div", { class: "grid wide" });
  root.appendChild(g);
  const ct = card(root, "Sample trees (" + rows.length + " shown)", "Click a column to sort, a row for details. Heights in m, differences in m.");
  const fm = (k, nd = 2, sg = false) => (r) => f(r[k], nd, sg);
  const tbl = mkTable([
    { k: "tree_id", label: "Tree", cls: "l" }, { k: "parzelle", label: "Parzelle" }, { k: "height_massband_m", label: "Maßband (m)", fmt: fm("height_massband_m") }, { k: "height_vertex_m", label: "Vertex (m)", fmt: fm("height_vertex_m") },
    { k: "height_uls_m", label: "ULS (m)", fmt: fm("height_uls_m") }, { k: "diff_uls_minus_massband_m", label: "ULS − Maßband (m)", fmt: fm("diff_uls_minus_massband_m", 2, true) },
    { k: "diff_vertex_minus_massband_m", label: "Vertex − Maßband (m)", fmt: fm("diff_vertex_minus_massband_m", 2, true) }, { k: "diff_uls_minus_vertex_m", label: "ULS − Vertex (m)", fmt: fm("diff_uls_minus_vertex_m", 2, true) },
    { k: "identity_class", label: "Identity", cls: "l" }, { k: "detection", label: "Detection", cls: "l" },
  ], rows, { scroll: true, onRow: (r) => { selectedSample = r; HASH.tree = r.tree_id; writeHash(); drawDetail(); tbl.redraw(); }, selected: () => selectedSample });
  ct.appendChild(tbl);
  const anchor = h("div", { id: "detail-anchor" });
  root.appendChild(anchor);
  const det = card(root, null, null);
  const body = h("div");
  det.appendChild(body);
  function drawDetail() {
    body.innerHTML = "";
    const s = selectedSample;
    if (!s) { body.appendChild(h("p", { class: "empty", text: "Select a sample tree above." })); return; }
    body.appendChild(h("h3", { text: "Sample tree " + s.tree_id + " (Parzelle " + s.parzelle + ", " + s.species + ")" }));
    const d = h("div", { class: "detail" });
    body.appendChild(d);
    const left = h("div"), right = h("div");
    d.append(left, right);
    const hs = [["Maßband", s.height_massband_m, C.ground], ["Vertex", s.height_vertex_m, C.vertex], ["ULS", s.height_uls_m, C.uls]].filter((x) => isNum(x[1]));
    plot(left, [{ type: "bar", x: hs.map((x) => x[0]), y: hs.map((x) => x[1]), marker: { color: hs.map((x) => x[2]) }, text: hs.map((x) => f(x[1], 2) + " m"), textposition: "outside", cliponaxis: false }],
      { yaxis: { title: "Tree height (m)", rangemode: "tozero" } }, { cls: "short" });
    left.appendChild(h("dl", { class: "kv" },
      h("dt", { text: "ULS − Maßband" }), h("dd", { text: isNum(s.diff_uls_minus_massband_m) ? f(s.diff_uls_minus_massband_m, 2, true) + " m" : "no Maßband length" }),
      h("dt", { text: "Vertex − Maßband" }), h("dd", { text: isNum(s.diff_vertex_minus_massband_m) ? f(s.diff_vertex_minus_massband_m, 2, true) + " m" : "no both" }),
      h("dt", { text: "ULS − Vertex" }), h("dd", { text: isNum(s.diff_uls_minus_vertex_m) ? f(s.diff_uls_minus_vertex_m, 2, true) + " m" : "no Vertex length" }),
      h("dt", { text: "Identity class" }), h("dd", null, h("span", { class: "badge", text: s.identity_class }), " score " + f(s.identity_score, 2)),
      h("dt", { text: "Detection / tier" }), h("dd", { text: (s.detection || "–") + " / " + (s.tier_method || "–") }),
      h("dt", { text: "BHD field / ULS" }), h("dd", { text: `${f(s.dbh_field_mm, 0)} mm / ${f(s.dbh_uls_mm, 0)} mm` }),
      h("dt", { text: "Cause" }), h("dd", { text: s.cause || "none recorded" }),
      h("dt", { text: "Field remark" }), h("dd", { text: s.remark || "none" })));
    if (s.explanation) right.appendChild(h("div", { class: "callout info" }, h("b", { text: "Generated explanation. " }), s.explanation));
    const pts = D.sample_points && D.sample_points[s.tree_id];
    const figs = s.figures || [];
    if (!pts && !figs.length) { right.appendChild(h("p", { class: "empty", text: "No figure or point profile is available for this tree." })); return; }
    const viz = h("div", { class: "viz-full" });
    body.appendChild(viz);
    figs.forEach((src) => {
      viz.appendChild(h("h4", { text: /chm_before_after/.test(src) ? "Canopy height model before and after" : "Raster and point cloud" }));
      viz.appendChild(h("a", { class: "fig", href: src, target: "_blank", rel: "noopener", title: "Open at full resolution" },
        h("img", { src, alt: "Figure for sample tree " + s.tree_id, loading: "lazy" })));
    });
    if (pts) {
      const rowsP = pts.rows.map((r) => { const o = {}; pts.cols.forEach((c, i) => (o[c] = r[i])); return o; });
      const groups = unique(rowsP.map((r) => r.group));
      const is3d = HASH.view3d === "1";
      const seg = h("div", { class: "seg", role: "group", "aria-label": "Point view" });
      [["2D side view", false], ["3D", true]].forEach(([lab, v]) => seg.appendChild(h("button", { type: "button", "aria-pressed": String(is3d === v),
        onclick: () => { if (is3d !== v) { HASH.view3d = v ? "1" : ""; writeHash(); drawDetail(); } } , text: lab })));
      viz.appendChild(h("div", { class: "viz-head" }, h("h4", { text: "Point profile" }), seg));
      viz.appendChild(h("div", { class: "cap", text: is3d
        ? "Points within the sample radius around the stem, by group, at true scale. Drag to rotate, scroll to zoom, right-drag to pan. The vertical line is the field stem position; the rings mark the Maßband and ULS heights."
        : "Side view of the points around this tree, by group. Scroll to zoom, drag to pan, double-click to reset." }));
      const byG = (gr, k) => rowsP.filter((r) => r.group === gr).map((r) => r[k]);
      if (!is3d) {
        plot(viz, groups.map((gr, i) => ({ type: "scatter", mode: "markers", name: gr, x: byG(gr, "x_rel"), y: byG(gr, "h"), marker: { color: C.c[i % 8], size: 4 } })),
          { xaxis: { title: "Distance from the stem, side view (m)", scaleanchor: "y" }, yaxis: { title: "Height above ground (m)" } }, { cls: "profile", config: { displayModeBar: true, scrollZoom: true } });
      } else {
        const hmax = Math.max(...rowsP.map((r) => r.h), s.height_massband_m || 0, s.height_uls_m || 0);
        const ring = (z, name, color) => { const a = [...Array(49).keys()].map((k) => (k / 48) * 2 * Math.PI), r = 1.5;
          return { type: "scatter3d", mode: "lines", name: name + " " + f(z, 2) + " m", x: a.map((t) => r * Math.cos(t)), y: a.map((t) => r * Math.sin(t)), z: a.map(() => z), line: { color, width: 6 }, hoverinfo: "name" }; };
        const tr = groups.map((gr, i) => ({ type: "scatter3d", mode: "markers", name: gr, x: byG(gr, "x_rel"), y: byG(gr, "y_rel"), z: byG(gr, "h"),
          marker: { color: C.c[i % 8], size: 2.5, opacity: 0.85 }, hovertemplate: "x %{x:.2f} m, y %{y:.2f} m<br>h %{z:.2f} m<extra>" + gr + "</extra>" }));
        tr.push({ type: "scatter3d", mode: "lines", name: "field stem position", x: [0, 0], y: [0, 0], z: [0, hmax], line: { color: C.ink2, width: 4, dash: "dash" }, hoverinfo: "name" });
        if (isNum(s.height_massband_m)) tr.push(ring(s.height_massband_m, "Maßband", C.ground));
        if (isNum(s.height_uls_m)) tr.push(ring(s.height_uls_m, "ULS", C.uls));
        const ax = (title) => ({ title: { text: title }, gridcolor: C.grid, zerolinecolor: C.line, linecolor: C.line, backgroundcolor: "rgba(0,0,0,0)", showbackground: false });
        plot(viz, tr, { margin: { l: 0, r: 0, t: 0, b: 0 }, legend: { y: 1, yanchor: "top", orientation: "v" },
          scene: { aspectmode: "data", xaxis: ax("East from stem (m)"), yaxis: ax("North from stem (m)"), zaxis: ax("Height above ground (m)"), camera: { eye: { x: 1.6, y: -1.6, z: 0.5 } } },
          uirevision: s.tree_id }, { cls: "profile", config: { displayModeBar: true } });
      }
    }
  }
  const fromHash = HASH.tree && SAMPLES.find((x) => x.tree_id === HASH.tree);
  if (fromHash) selectedSample = fromHash;
  else if (!selectedSample || !rows.includes(selectedSample)) selectedSample = worstU || rows[0] || null;
  drawDetail();
  tbl.redraw();
}

/* ======================================================================
   Wood volume
   ====================================================================== */
function renderVolume(root) {
  root.appendChild(h("div", { class: "callout" }, h("b", { text: "Read this first. " }),
    "The volume function is provisional (rough form factors, not from a cited source). Both volumes use the ULS height, so differences between field volume and ULS volume come from the BHD only. Trees without a ULS height use a modelled height. Stand-level height accuracy rests on the sample trees only: this tab does not test ULS height."));
  const sp = D.volume.per_species || [], minN = D.volume.min_trees_per_species;
  const g = h("div", { class: "grid tiles" });
  root.appendChild(g);
  g.append(tile("Stand volume, field", S("volume.total.field_with_modelled.stand")), tile("Stand volume, ULS", S("volume.total.uls.stand")), tile("Gap ULS minus field", S("volume.gap.total")),
    tile("Volume per tree, ULS minus field: bias", S("volume.tree_error." + F.sub + ".all.bias", F.sub), { follows: true, extra: [{ label: "relative", e: S("volume.tree_error." + F.sub + ".all.relative_bias", F.sub) }] }));
  const g1 = h("div", { class: "grid wide" });
  root.appendChild(g1);
  let c = card(g1, "Per species", "Total volume per species. Species with fewer than " + minN + " trees are shown as totals with their tree count, without accuracy statistics.");
  const norm = (s) => s.toLowerCase().replace(/ /g, "_");
  const rowsS = sp.map((r) => {
    const big = r.n_trees >= minN, id = (m) => S("volume.tree_error." + F.sub + ".species_" + norm(r.species) + "." + m, F.sub);
    return { species: r.species, n: r.n_trees, vf: r.volume_field_with_modelled_m3, vu: r.volume_uls_m3, rel: r.volume_field_with_modelled_m3 ? (r.volume_uls_m3 / r.volume_field_with_modelled_m3 - 1) * 100 : null,
      big, bias: big ? id("bias") : null, rmse: big ? id("rmse") : null, med: big ? id("median") : null, mad: big ? id("mad") : null };
  }).sort((a, b) => b.vf - a.vf);
  const cell = (e) => (e ? f(e.value, 3) + (isNum(e.ci_low) ? ` (${f(e.ci_low, 3)} to ${f(e.ci_high, 3)})` : "") : "totals only");
  c.appendChild(mkTable([{ k: "species", label: "Species", cls: "l" }, { k: "n", label: "Trees" }, { k: "vf", label: "Field (m³)", fmt: (r) => f(r.vf, 1) }, { k: "vu", label: "ULS (m³)", fmt: (r) => f(r.vu, 1) },
    { k: "rel", label: "ULS vs field (%)", fmt: (r) => f(r.rel, 1, true) }, { k: "bias", label: "Bias per tree (m³)", fmt: (r) => cell(r.bias), sort: (r) => r.bias && r.bias.value },
    { k: "rmse", label: "RMSE (m³)", fmt: (r) => (r.big ? f(r.rmse && r.rmse.value, 3) : "n < " + minN) }, { k: "med", label: "Median (m³)", fmt: (r) => (r.big ? f(r.med && r.med.value, 3) : "n < " + minN) }, { k: "mad", label: "MAD (m³)", fmt: (r) => (r.big ? f(r.mad && r.mad.value, 3) : "n < " + minN) }], rowsS));
  c.appendChild(h("p", { class: "legend-note", text: "Field volume includes modelled heights for trees ULS did not find; ULS volume covers matched trees only. Accuracy follows the subset filter (" + F.sub.replace("_", " ") + ")." }));
  c = card(g1, "Per Parzelle: volume per hectare", "Field (with modelled heights) against ULS, m³ per ha. Follows the Parzelle filter.");
  const ps = parzF().filter((p) => isNum(p.volume_uls_m3_per_ha));
  plot(c, [{ type: "bar", name: "field", x: ps.map((p) => String(p.parzelle)), y: ps.map((p) => p.volume_field_with_modelled_m3_per_ha), marker: { color: C.ground } },
    { type: "bar", name: "ULS", x: ps.map((p) => String(p.parzelle)), y: ps.map((p) => p.volume_uls_m3_per_ha), marker: { color: C.uls } }],
  { barmode: "group", xaxis: { title: "Parzelle", type: "category", tickfont: { size: 9 } }, yaxis: { title: "Wood volume (m³ per ha)" } });
  const g2 = h("div", { class: "grid" });
  root.appendChild(g2);
  c = card(g2, "Per tree: ULS volume against field volume", "Each point is one matched tree; both volumes use the ULS height.");
  const ts = matchedF().filter((t) => isNum(t.vF) && isNum(t.vU));
  const mx = Math.max(0.5, ...ts.map((t) => Math.max(t.vF, t.vU))) * 1.05;
  plot(c, [{ type: "scatter", mode: "lines", x: [0, mx], y: [0, mx], line: { color: C.ink2, dash: "dot", width: 1 }, name: "1:1 line", hoverinfo: "skip" },
    ...DET.slice(0, 2).map((d) => { const gg = ts.filter((t) => t.det === d); return { type: "scatter", mode: "markers", name: d + " (n=" + gg.length + ")", x: gg.map((t) => t.vF), y: gg.map((t) => t.vU), text: gg.map((t) => t.id), marker: { color: C.det[d], size: 4, opacity: 0.6 }, hovertemplate: "%{text}<br>field %{x:.3f} m³, ULS %{y:.3f} m³<extra></extra>" }; })],
  { xaxis: { title: "Field volume per tree (m³)", range: [0, mx] }, yaxis: { title: "ULS volume per tree (m³)", range: [0, mx], scaleanchor: "x" } }, { cls: "tall" });
  c = card(g2, "Bias by BHD class", "ULS minus field volume per tree (m³) with 95 % bootstrap interval. Classes follow the subset filter.");
  const cls = D.stats.filter((e) => e.id.startsWith("volume.tree_error." + F.sub + ".class_") && e.id.endsWith(".bias")).map((e) => ({ e, id: e.id.split(".")[3].replace("class_", ""), rel: S(e.id.replace(".bias", ".relative_bias"), F.sub) }));
  cls.sort((a, b) => parseInt(a.id) - parseInt(b.id));
  const lab = (id) => id.replace(/_mm$/, "").replace("_", "–") + " mm";
  if (cls.length) {
    plot(c, [{ type: "bar", x: cls.map((x) => lab(x.id) + "<br>n=" + x.e.n), y: cls.map((x) => x.e.value), marker: { color: C.uls }, text: cls.map((x) => f(x.e.value, 3, true)), textposition: "outside", cliponaxis: false,
      error_y: { type: "data", symmetric: false, array: cls.map((x) => (isNum(x.e.ci_high) ? x.e.ci_high - x.e.value : 0)), arrayminus: cls.map((x) => (isNum(x.e.ci_low) ? x.e.value - x.e.ci_low : 0)), color: C.ink2, thickness: 1.2 },
      customdata: cls.map((x) => [x.rel ? x.rel.value * 100 : null]), hovertemplate: "%{x}<br>bias %{y:.3f} m³ (%{customdata[0]:.1f} %)<extra></extra>" }],
    { xaxis: { title: "Field BHD class" }, yaxis: { title: "Volume bias ULS minus field (m³ per tree)", zeroline: true } }, { cls: "short" });
    c.appendChild(h("p", { class: "cap", text: stmt(cls[0].e.id, F.sub) }));
  } else noData(c);
  c = card(g2, "Volume-weighted detection", "Share of trees against share of the field volume, by detection status. Large trees are found more often than small ones, so the volume share is higher than the tree share.");
  const sh = DET.map((d, i) => ({ d, v: S("volume.detection_share." + d.replace(" ", "_")), n: S("detection.count_" + d.replace(" ", "_")) }));
  if (sh.every((x) => x.v && x.n)) {
    const tot = sum(sh.map((x) => x.n.value));
    plot(c, [{ type: "bar", name: "share of trees", x: sh.map((x) => x.d), y: sh.map((x) => (x.n.value / tot) * 100), marker: { color: C.ground }, text: sh.map((x) => pct(x.n.value / tot)), textposition: "outside", cliponaxis: false },
      { type: "bar", name: "share of field volume", x: sh.map((x) => x.d), y: sh.map((x) => x.v.value * 100), marker: { color: C.uls }, text: sh.map((x) => pct(x.v.value)), textposition: "outside", cliponaxis: false }],
    { barmode: "group", yaxis: { title: "Share of the stand (%)", range: [0, 110] } }, { cls: "short" });
    c.appendChild(h("p", { class: "cap", text: stmt("volume.detection_share.detected") }));
  } else noData(c);
  c = card(g2, "Where the stand-volume gap comes from", "Signed parts of the gap ULS minus field. The three parts add up to the total.");
  gapChart(c);
  c.appendChild(h("p", { class: "cap", text: stmt("volume.gap.trees_missed") }));
  c = card(g2, "Sample trees: volume with three heights", "Volume at the same field BHD with the Maßband height, the Vertex height and the ULS height (m³ per tree). Follows the filters.");
  const ss = samplesF().filter((s) => isNum(s.wood_volume_field_m3)).sort((a, b) => (a.tree_id > b.tree_id ? 1 : -1));
  if (ss.length) {
    const ids = ss.map((s) => s.tree_id);
    plot(c, [{ type: "scatter", mode: "markers", name: "Maßband height", x: ids, y: ss.map((s) => s.wood_volume_massband_m3), marker: { color: C.ground, size: 9, symbol: "square" } },
      { type: "scatter", mode: "markers", name: "Vertex height", x: ids, y: ss.map((s) => s.wood_volume_vertex_m3), marker: { color: C.vertex, size: 9, symbol: "diamond" } },
      { type: "scatter", mode: "markers", name: "ULS height", x: ids, y: ss.map((s) => s.wood_volume_field_m3), marker: { color: C.uls, size: 9 } }],
    { xaxis: { title: "Sample tree", type: "category", tickangle: -60, tickfont: { size: 9 } }, yaxis: { title: "Volume per tree (m³)" }, margin: { b: 70 } }, { cls: "tall" });
    ["volume.sample.uls_height_vs_massband.bias", "volume.sample.uls_height_vs_vertex.bias", "volume.sample.vertex_vs_massband.bias"].forEach((id) => { const t = stmt(id); if (t) c.appendChild(h("p", { class: "cap", text: t })); });
  } else noData(c, "No sample trees match the current filters.");
  c = card(g2, "Sensitivity of the stand volume", "Change of the total stand volume (m³) when one input moves by the given amount.");
  const sens = ["height_plus", "height_minus", "dbh_plus", "dbh_minus", "form_factor_plus", "form_factor_minus"].map((k) => ({ k, fe: S("volume.sensitivity." + k + ".field"), ue: S("volume.sensitivity." + k + ".uls") })).filter((x) => x.fe && x.ue);
  const slabel = { height_plus: "height +1 m", height_minus: "height −1 m", dbh_plus: "BHD +10 mm", dbh_minus: "BHD −10 mm", form_factor_plus: "form factor +0.02", form_factor_minus: "form factor −0.02" };
  if (sens.length) {
    plot(c, [{ type: "bar", orientation: "h", name: "field volume", y: sens.map((x) => slabel[x.k] || x.k), x: sens.map((x) => x.fe.value), marker: { color: C.ground } },
      { type: "bar", orientation: "h", name: "ULS volume", y: sens.map((x) => slabel[x.k] || x.k), x: sens.map((x) => x.ue.value), marker: { color: C.uls } }],
    { barmode: "group", xaxis: { title: "Change of stand volume (m³)", zeroline: true }, yaxis: { autorange: "reversed" }, margin: { l: 120 } }, { cls: "short" });
    c.appendChild(h("p", { class: "cap", text: stmt("volume.sensitivity.height_plus.field") + " " + stmt("volume.sensitivity.dbh_plus.field") }));
  } else noData(c);
  c = card(g2, "Harvested and remaining volume per Parzelle", "Volume of trees marked removed in the harvest and of the remaining stand, field against ULS (m³).");
  const hp = parzF().filter((p) => isNum(p.volume_field_removed_m3));
  if (hp.length) {
    const x = hp.map((p) => String(p.parzelle));
    plot(c, [{ type: "bar", name: "field removed", x, y: hp.map((p) => p.volume_field_removed_m3), marker: { color: C.ground }, offsetgroup: "f" }, { type: "bar", name: "field remaining", x, y: hp.map((p) => p.volume_field_remaining_m3), marker: { color: C.c[3] }, offsetgroup: "f", base: hp.map((p) => p.volume_field_removed_m3) },
      { type: "bar", name: "ULS removed", x, y: hp.map((p) => p.volume_uls_removed_m3), marker: { color: C.uls }, offsetgroup: "u" }, { type: "bar", name: "ULS remaining", x, y: hp.map((p) => p.volume_uls_remaining_m3), marker: { color: C.c[6] }, offsetgroup: "u", base: hp.map((p) => p.volume_uls_removed_m3) }],
    { barmode: "group", xaxis: { title: "Parzelle", type: "category", tickfont: { size: 9 } }, yaxis: { title: "Volume (m³)" } });
    c.appendChild(h("p", { class: "cap", text: stmt("volume.harvest.stand.field_removed_share") + " " + stmt("volume.harvest.stand.uls_removed_share") }));
  } else noData(c);
  c = card(g2, "Height-diameter curve check (leave-one-out)", "Modelled heights for trees without a ULS height come from these curves; the table shows how well the curve predicts a left-out tree.");
  const hc = (D.volume.height_curves || []);
  if (hc.length) c.appendChild(mkTable([{ k: "who", label: "Curve", cls: "l" }, { k: "n", label: "Trees" }, { k: "bias", label: "LOO bias (m)" }, { k: "rmse", label: "LOO RMSE (m)" }, { k: "med", label: "LOO median (m)" }, { k: "mad", label: "LOO MAD (m)" }],
    hc.map((r) => ({ who: r.scope === "pooled" ? "all species" : r.species, n: r.n, bias: f(r.loo_bias_m, 2, true), rmse: f(r.loo_rmse_m, 2), med: f(r.loo_median_m, 2, true), mad: f(r.loo_mad_m, 2) }))));
  else noData(c);
  c.appendChild(h("p", { class: "cap", text: stmt("volume.modelled_trees") + " " + stmt("volume.field_excluded") }));
  root.appendChild(h("h2", { text: "Downloads" }));
  root.appendChild(downloadRow());
}

/* ======================================================================
   Wood volume (GRI): the volume functions are evaluated here so that the species groups can be changed
   ====================================================================== */
const G = D.volume_gri || { available: false };
const GT = G.available ? G.trees.rows.map((r) => { const o = {}; G.trees.cols.forEach((c, i) => (o[c] = r[i])); return o; }) : [];
const GRI_MIN_D_CM = 6.1, GRI_MIN_H_M = 1.3;
// GRI volume in m3 (Franz et al. 1973, as v_gri() in ForestElementsR): code = species group, dmm = DBH in mm, hm = height in m.
function griVolume(code, dmm, hm) {
  const c = G.coefficients[code];
  if (!c || !isNum(dmm) || !isNum(hm)) return NaN;
  const d = dmm / 10;
  if (d < GRI_MIN_D_CM || hm < GRI_MIN_H_M) return 0;
  const ld = Math.log(d), lh = Math.log(hm), a = [1, ld, ld * ld], b = [1, lh, lh * lh];
  let s = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += c[3 * i + j] * a[i] * b[j];
  return Math.max(0, Math.PI / 40000 * d * d * Math.exp(s));
}
// Three volumes per tree (plus the sample-tree volumes at the tape and Vertex heights) for a species -> group mapping.
// Efm o. R. (harvest volume under bark) = Vfm m. R. x factor of the species group (v_red_harvest_ubark in ForestElementsR).
const GEFM = G.available && G.efm_factors && Object.keys(G.efm_factors).length ? G.efm_factors : null;
function griCalc(map) {
  return GT.map((t) => {
    const code = map[t.sp], hF = t.hs === 1 ? t.hU : t.hs === 2 ? t.hMod : null, fac = GEFM && isNum(GEFM[code]) ? GEFM[code] : NaN;
    const r = { t, vF: griVolume(code, t.dF, hF), vU: t.hs === 1 ? griVolume(code, t.dU, t.hU) : NaN, vP: griVolume(code, t.dF, t.hP),
      vM: griVolume(code, t.dF, t.hM), vV: griVolume(code, t.dF, t.hV), vH: griVolume(code, t.dF, t.hU) };
    r.eF = r.vF * fac; r.eU = r.vU * fac; r.eP = r.vP * fac;
    return r;
  });
}
const griSum = (rows, k) => sum(rows.map((r) => r[k]).filter(isNum));
const griDefaultMap = () => { const m = {}; speciesOrder.forEach((s) => (m[s] = G.default_codes[s] != null ? String(G.default_codes[s]) : "")); return m; };
let griMap = null;
function griReadHash(def) {
  const m = Object.assign({}, def);
  (HASH.gri || "").split(",").forEach((kv) => { const i = kv.lastIndexOf(":"), s = kv.slice(0, i), v = kv.slice(i + 1); if (i > 0 && s in m && (v === "" || G.coefficients[v])) m[s] = v; });
  return m;
}

function renderVolumeGri(root) {
  root.appendChild(h("div", { class: "callout" }, h("b", { text: "Read this first. " }),
    "Standing merchantable volume over bark (Derbholz mit Rinde) from the GRI volume functions of Franz et al. (1973) as implemented in v_gri() of the R package ForestElementsR (Biber and Toraño Caicoya 2025). It is reported as the Vorrat in Vorratsfestmeter, \"Vfm m. R.\" (m³ standing volume over bark). The harvest volume under bark, Erntefestmeter \"Efm o. R.\", is the Vfm multiplied by a reduction factor of the species group (harvest loss and bark, v_red_harvest_ubark in ForestElementsR). Trees below 6.1 cm DBH have a volume of 0. Three volumes are computed for every tree. Field: field DBH and ULS height (a modelled height where no ULS instance is matched). ULS: ULS DBH and ULS height, matched trees only. Practice: field DBH and a height from a height-diameter curve fitted on the field-measured standing heights and applied to all trees, which is what an inventory without laser scanning would compute."));
  if (!G.available) {
    root.appendChild(h("div", { class: "callout info" }, h("b", { text: "Not available for this build. " }),
      "To enable this tab, create the coefficient file with ", h("code", { text: "Rscript scripts/export_gri_coefficients.R" }), ", set ", h("code", { text: "volume_gri.species_codes" }),
      " in the site configuration (field species to GRI species group), run the pipeline stage ", h("code", { text: "volume_gri" }), " and rebuild the dashboard."));
    return;
  }
  const def = griDefaultMap(), species = speciesOrder.filter((s) => s in def);
  if (!griMap) griMap = griReadHash(def);
  const isDefault = () => species.every((s) => griMap[s] === def[s]);
  const groupName = (code) => { const g = G.groups.find((x) => String(x.code) === code); return g ? code + " " + g.name : "none"; };
  const calcDef = griCalc(def);
  // consistency of the browser calculation with the Python statistics (default mapping)
  const checks = [["volume_gri.total.field_with_modelled.stand", "vF"], ["volume_gri.total.uls.stand", "vU"], ["volume_gri.total.practice.stand", "vP"]]
    .concat(GEFM ? [["volume_gri.efm.total.field_with_modelled.stand", "eF"], ["volume_gri.efm.total.uls.stand", "eU"], ["volume_gri.efm.total.practice.stand", "eP"]] : [])
    .map(([id, k]) => ({ id, js: griSum(calcDef, k), py: S(id) && S(id).value })).filter((x) => isNum(x.py) && x.py !== 0);
  const maxdiff = checks.length ? Math.max(...checks.map((x) => Math.abs(x.js / x.py - 1))) : NaN;
  if (isNum(maxdiff)) root.setAttribute("data-gri-maxdiff", String(maxdiff));

  const pc = card(root, "Parameters", "GRI species group used for each field species. Changing a group recomputes all three volumes (Vfm m. R.) and the Efm o. R. for every tree in the browser.");
  const selects = {};
  const grid = h("div", { class: "params" });
  species.forEach((s) => {
    const sel = h("select", { "aria-label": "GRI species group for " + s }, h("option", { value: "", text: "no group (no volume)" }),
      G.groups.map((g) => h("option", { value: String(g.code), text: g.code + " " + g.name })));
    sel.value = griMap[s];
    sel.addEventListener("change", () => { griMap[s] = sel.value; HASH.gri = isDefault() ? "" : species.map((x) => x + ":" + griMap[x]).join(","); writeHash(); draw(); });
    selects[s] = sel;
    grid.appendChild(h("label", null, s + " ", sel));
  });
  pc.appendChild(grid);
  pc.appendChild(h("button", { class: "btn", type: "button", onclick: () => { griMap = Object.assign({}, def); species.forEach((s) => (selects[s].value = def[s])); HASH.gri = ""; writeHash(); draw(); } }, "Reset to defaults"));
  pc.appendChild(h("p", { class: "legend-note", text: "Defaults come from the site configuration. A species without a group gets no volume. The chosen groups are kept in the page address (key gri) when they differ from the defaults." }));

  if (isNum(maxdiff) && maxdiff > 0.001) root.appendChild(h("div", { class: "callout warn" }, h("b", { text: "Consistency warning. " }),
    "With the default groups the browser calculation differs from the Python statistics by " + (maxdiff * 100).toFixed(3) + " % (more than 0.1 %)."));
  const dyn = h("div");
  root.appendChild(dyn);
  root.appendChild(h("p", { class: "legend-note", text: G.method }));
  let incRedraw = null;

  function draw() {
    dyn.innerHTML = "";
    const calc = isDefault() ? calcDef : griCalc(griMap), dflt = isDefault();
    const fk = (value, unit, n, statement) => ({ value, unit, n, statement });
    const tot = { F: griSum(calc, "vF"), U: griSum(calc, "vU"), P: griSum(calc, "vP"), eF: griSum(calc, "eF"), eU: griSum(calc, "eU"), eP: griSum(calc, "eP") };
    const nOf = (k) => calc.filter((r) => isNum(r[k])).length, area = G.area_total_ha;
    const perHa = (v) => (area ? fk(v / area, "m3/ha") : null);
    if (!dflt) dyn.appendChild(h("div", { class: "callout info" }, h("span", { class: "badge", text: "Recalculated" }), " Recalculated in the browser with a changed species mapping; no confidence intervals."));
    const g = h("div", { class: "grid tiles" });
    dyn.appendChild(g);
    const vfm = (e, nd = 1) => (e && isNum(e.value) ? f(e.value, nd) + " Vfm" : "–");
    const vfmHaTxt = (e) => (e && isNum(e.value) ? f(e.value, 1) + " Vfm/ha" : null);
    const volTile = (label, key, id, k) => {
      const e = dflt ? S(id) : fk(tot[key], "m3", nOf(k), `With the changed species groups, the ${label.toLowerCase()} is ${f(tot[key], 1)} Vfm m. R. from ${nOf(k)} trees.`);
      const ph = dflt ? S(id.replace("total.", "per_ha.")) : perHa(tot[key]);
      return tile(label, e, { value: vfm(e), extra: [{ label: "per ha", e: ph, text: vfmHaTxt(ph) }] });
    };
    const efmTile = (label, key, id, k) => {
      const e = dflt && S(id) ? S(id) : fk(tot[key], "m3", nOf(k), `With the changed species groups, the ${label.toLowerCase()} is ${f(tot[key], 1)} Efm o. R. from ${nOf(k)} trees.`);
      const ph = dflt && S(id.replace("total.", "per_ha.")) ? S(id.replace("total.", "per_ha.")) : perHa(tot[key]);
      return tile(label, e, { value: e && isNum(e.value) ? f(e.value, 1) + " Efm" : "–", extra: [{ label: "per ha", e: ph, text: ph && isNum(ph.value) ? f(ph.value, 1) + " Efm/ha" : null }] });
    };
    const dU = tot.U - tot.F, dP = tot.P - tot.F, oldF = S("volume.total.field_with_modelled.stand");
    const gapE = dflt ? S("volume_gri.gap.total") : fk(dU, "m3", nOf("vU"), `With the changed species groups the ULS stand volume is ${f(Math.abs(dU), 1)} m³ ${dU >= 0 ? "above" : "below"} the field stand volume.`);
    const prE = dflt ? S("volume_gri.practice_vs_field.relative_bias", "all") : fk(tot.F ? dP / tot.F : NaN, "share", nOf("vP"), `With the changed species groups the practice stand volume is ${f(Math.abs(dP), 1)} m³ ${dP >= 0 ? "above" : "below"} the field stand volume.`);
    const oldE = oldF && isNum(oldF.value) ? (dflt && S("volume_gri.vs_old.field") ? S("volume_gri.vs_old.field") : fk(tot.F - oldF.value, "m3", nOf("vF"), `The GRI field stand volume is ${f(tot.F, 1)} m³, against ${f(oldF.value, 1)} m³ with the old method.`)) : null;
    g.append(volTile("Vorrat (Vfm m. R.), field", "F", "volume_gri.total.field_with_modelled.stand", "vF"), volTile("Vorrat (Vfm m. R.), ULS", "U", "volume_gri.total.uls.stand", "vU"),
      volTile("Vorrat (Vfm m. R.), practice", "P", "volume_gri.total.practice.stand", "vP"));
    if (GEFM) g.append(efmTile("Harvest volume (Efm o. R.), field", "eF", "volume_gri.efm.total.field_with_modelled.stand", "eF"), efmTile("Harvest volume (Efm o. R.), ULS", "eU", "volume_gri.efm.total.uls.stand", "eU"),
      efmTile("Harvest volume (Efm o. R.), practice", "eP", "volume_gri.efm.total.practice.stand", "eP"));
    g.append(
      tile("ULS minus field", gapE, { value: f(dU, 1, true) + " Vfm", noci: true, extra: [{ label: "relative", e: fk(tot.F ? dU / tot.F : NaN, "share") }], statement: gapE && gapE.statement }),
      tile("Practice minus field", prE, { value: pct(tot.F ? dP / tot.F : NaN, 1), extra: [{ label: "absolute", text: f(dP, 1, true) + " Vfm", e: fk(dP, "m3") }] }),
      tile("GRI vs old method (field total)", oldE, { value: oldE && pct(tot.F / oldF.value - 1), noci: true }));

    // per species
    const g1 = h("div", { class: "grid wide" });
    dyn.appendChild(g1);
    let c = card(g1, "Per species", "Total Vorrat (Vfm m. R.) per species with the group used (all trees; the Parzelle filter does not apply), and the harvest volume (Efm o. R.) of the field volume with the factor of the group. Differences are relative to the field volume.");
    const rowsS = species.map((s) => { const r = calc.filter((x) => x.t.sp === s), v = { F: griSum(r, "vF"), U: griSum(r, "vU"), P: griSum(r, "vP"), E: griSum(r, "eF") };
      const code = griMap[s], fac = GEFM && isNum(GEFM[code]) ? GEFM[code] : null;
      return { s, n: r.length, grp: groupName(griMap[s]), F: v.F, U: v.U, P: v.P, E: GEFM ? v.E : null, fac, du: v.F ? (v.U / v.F - 1) * 100 : null, dp: v.F ? (v.P / v.F - 1) * 100 : null }; }).sort((a, b) => b.F - a.F);
    c.appendChild(mkTable([{ k: "s", label: "Species", cls: "l" }, { k: "n", label: "Trees" }, { k: "grp", label: "GRI group", cls: "l" }, { k: "F", label: "Field (Vfm)", fmt: (r) => f(r.F, 1) },
      { k: "U", label: "ULS (Vfm)", fmt: (r) => f(r.U, 1) }, { k: "P", label: "Practice (Vfm)", fmt: (r) => f(r.P, 1) },
      ...(GEFM ? [{ k: "E", label: "Field (Efm o. R.)", fmt: (r) => f(r.E, 1) }, { k: "fac", label: "Efm / Vfm", fmt: (r) => f(r.fac, 3) }] : []),
      { k: "du", label: "ULS vs field (%)", fmt: (r) => f(r.du, 1, true) }, { k: "dp", label: "Practice vs field (%)", fmt: (r) => f(r.dp, 1, true) }], rowsS));
    c.appendChild(h("p", { class: "legend-note", text: "The ULS volume covers matched trees only; the field and practice volumes cover all trees with a valid field DBH." }));

    // per Parzelle
    c = card(g1, "Per Parzelle: Vorrat per hectare", "Field, ULS and practice Vorrat (Vfm m. R. per ha). Follows the Parzelle filter.");
    const ps = parzF().map((p) => String(p.parzelle)).filter((p) => isNum(G.areas_ha && G.areas_ha[p]));
    const ph = (k) => ps.map((p) => sum(calc.filter((r) => String(r.t.p) === p).map((r) => r[k]).filter(isNum)) / G.areas_ha[p]);
    plot(c, [{ type: "bar", name: "field", x: ps, y: ph("vF"), marker: { color: C.ground } }, { type: "bar", name: "ULS", x: ps, y: ph("vU"), marker: { color: C.uls } },
      { type: "bar", name: "practice", x: ps, y: ph("vP"), marker: { color: C.practice } }],
    { barmode: "group", xaxis: { title: "Parzelle", type: "category", tickfont: { size: 9 } }, yaxis: { title: "Vorrat (Vfm m. R. per ha)" } });

    // per-tree scatter
    const g2 = h("div", { class: "grid" });
    dyn.appendChild(g2);
    const inP = (r) => F.p === "all" || String(r.t.p) === F.p;
    const scat = (title, cap, k, label, color) => {
      const cc = card(g2, title, cap), ts = calc.filter((r) => inP(r) && isNum(r.vF) && isNum(r[k]));
      const mx = Math.max(0.5, ...ts.map((r) => Math.max(r.vF, r[k]))) * 1.05;
      plot(cc, [{ type: "scatter", mode: "lines", x: [0, mx], y: [0, mx], line: { color: C.ink2, dash: "dot", width: 1 }, name: "1:1 line", hoverinfo: "skip" },
        { type: "scatter", mode: "markers", name: label + " (n=" + ts.length + ")", x: ts.map((r) => r.vF), y: ts.map((r) => r[k]), text: ts.map((r) => r.t.id), marker: { color, size: 4, opacity: 0.6 }, hovertemplate: "%{text}<br>field %{x:.3f} Vfm, " + label + " %{y:.3f} Vfm<extra></extra>" }],
      { xaxis: { title: "Field Vorrat per tree (Vfm m. R.)", range: [0, mx] }, yaxis: { title: label + " Vorrat per tree (Vfm m. R.)", range: [0, mx], scaleanchor: "x" } }, { cls: "tall" });
    };
    scat("Per tree: practice volume against field volume", "Each point is one tree with both volumes. The practice height comes from the field height-diameter curve. Follows the Parzelle filter.", "vP", "practice", C.practice);
    scat("Per tree: ULS volume against field volume", "Each point is one matched tree; both volumes use the ULS height, so the difference comes from the DBH. Follows the Parzelle filter.", "vU", "ULS", C.uls);

    // height-diameter
    c = card(g2, "Heights against field DBH", "Field-measured standing heights, the heights of the practice curve and the ULS heights, by field DBH. These heights do not depend on the species groups.");
    const hr = GT.filter((t) => (F.p === "all" || String(t.p) === F.p) && isNum(t.dF));
    const hp = (k, sel) => { const a = hr.filter((t) => isNum(t[k]) && (!sel || sel(t))); return { x: a.map((t) => t.dF / 10), y: a.map((t) => t[k]), text: a.map((t) => t.id) }; };
    const hm = (name, o, mk) => Object.assign({ type: "scatter", mode: "markers", name: name + " (n=" + o.x.length + ")", hovertemplate: "%{text}<br>DBH %{x:.1f} cm, height %{y:.1f} m<extra>" + name + "</extra>" }, o, { marker: mk });
    plot(c, [hm("field-measured", hp("hF"), { color: C.ground, size: 5, opacity: 0.6 }), hm("practice curve", hp("hP"), { color: C.practice, size: 4, symbol: "x", opacity: 0.8 }),
      hm("ULS", hp("hU", (t) => t.hs === 1), { color: C.uls, size: 4, opacity: 0.5 })],
    { xaxis: { title: "Field DBH (cm)" }, yaxis: { title: "Tree height (m)" } }, { cls: "tall" });

    // sample trees
    c = card(g2, "Sample trees: volume with three heights", "Vorrat at the same field DBH with the Maßband height, the Vertex height and the ULS height (Vfm m. R. per tree). Follows the Parzelle filter.");
    const ss = calc.filter((r) => inP(r) && isNum(r.vH) && (isNum(r.vM) || isNum(r.vV))).sort((a, b) => (a.t.id > b.t.id ? 1 : -1)), ids = ss.map((r) => r.t.id);
    if (ss.length) plot(c, [{ type: "scatter", mode: "markers", name: "Maßband height", x: ids, y: ss.map((r) => r.vM), marker: { color: C.ground, size: 9, symbol: "square" } },
      { type: "scatter", mode: "markers", name: "Vertex height", x: ids, y: ss.map((r) => r.vV), marker: { color: C.vertex, size: 9, symbol: "diamond" } },
      { type: "scatter", mode: "markers", name: "ULS height", x: ids, y: ss.map((r) => r.vH), marker: { color: C.uls, size: 9 } }],
    { xaxis: { title: "Sample tree", type: "category", tickangle: -60, tickfont: { size: 9 } }, yaxis: { title: "Vorrat per tree (Vfm m. R.)" }, margin: { b: 70 } }, { cls: "tall" });
    else noData(c, "No sample trees match the current filter.");

    // consistency
    dyn.appendChild(h("p", { class: "legend-note", text: isNum(maxdiff)
      ? "Consistency: with the default groups the browser calculation differs from the Python statistics by at most " + (maxdiff * 100).toFixed(4) + " % in the stand totals (Vfm m. R. and Efm o. R.; field, ULS, practice)." : "No Python statistics are available to check the browser calculation against." }));
    if (incRedraw) incRedraw(calc);
  }
  incRedraw = buildIncrement(root, { isDefault, area: G.area_total_ha });
  draw();
}

/* ======================================================================
   Increment (Zuwachs) of the GRI volume tab: three methods, stand age as an input
   ====================================================================== */
const INC = { init: false, age: null, ages: {}, zm: "all" };
const INC_METHODS = [["yield", "Yield table", 4], ["growth", "Tree growth functions", 5], ["measured", "Measured 2020–2025/26", 6]];
const INC_MODES = INC_METHODS.concat([["all", "Compare all", null]]);
const incLabel = (k) => INC_METHODS.find((m) => m[0] === k)[1];
const incCol = (k) => C.c[INC_METHODS.find((m) => m[0] === k)[2]];
const incTextKey = { yield: "yield_table", growth: "tree_growth", measured: "measured" };

function buildIncrement(root, info) {
  const I = G.increment;
  const sec = h("section", { class: "inc" });
  root.appendChild(sec);
  sec.appendChild(h("h2", { text: "Increment (Zuwachs)" }));
  if (!I || !I.available) {
    sec.appendChild(h("div", { class: "callout info" }, h("b", { text: "Increment not available for this build. " }),
      "The increment tables are missing. Run the pipeline stage ", h("code", { text: "increment" }), " (it needs the exported ForestElementsR files) and rebuild the dashboard."));
    return null;
  }
  const BA = I.by_age, [a0, a1] = I.age_grid, ref = I.reference_year, pk = Object.keys(BA).filter((k) => k !== "stand");
  const M = {};
  I.measured.forEach((r) => (M[String(r.parzelle)] = r));
  const nF = GT.filter((t) => isNum(t.dF)).length;
  // Measured increment (two surveys): recalculated from the ledger columns for any growing seasons and share of the period of the trees removed earlier.
  // pai = (survivor_growth + share x v_removed_2020 x survivor_rel_growth) / seasons / area, summed over the Parzellen of `keys` (null = all).
  const GS0 = Object.assign({}, I.growing_seasons || {}), syears = Object.keys(GS0).sort(), RG0 = isNum(I.removed_growth_share) ? I.removed_growth_share : 0;
  const mRows = I.measured.filter((r) => String(r.parzelle) !== "stand");
  const calcM = (keys, gs, rg) => {
    const o = { area: 0, num: 0, t1: 0, t2: 0, rem: 0, rem0: 0, stat: 0, dbh: 0, nS: 0, efm: 0, nR: 0, nM: 0, nX: 0, seasons: new Set(), n: 0 };
    mRows.forEach((r) => {
      if (keys && !keys.includes(String(r.parzelle))) return;
      const sec = gs[String(r.survey_year)];
      if (!(sec > 0) || !(r.area_ha > 0)) return;
      const rel = isNum(r.survivor_rel_growth) ? r.survivor_rel_growth : 0, d0 = r.growing_seasons || sec;
      o.n++; o.area += r.area_ha; o.seasons.add(sec);
      o.num += (r.survivor_growth_m3 + rg * r.v_removed_2020_m3 * rel) / sec;
      o.t1 += r.v_alive_t1_m3; o.t2 += r.v_alive_t2_m3; o.rem0 += r.v_removed_2020_m3;
      o.rem += r.v_removed_2020_m3 * (1 + rg * rel);
      o.stat += (r.pai_static_curve_m3_ha_yr || 0) * r.area_ha * d0 / sec;
      if (isNum(r.dbh_increment_mm_yr)) { o.dbh += r.dbh_increment_mm_yr * d0 / sec * r.n_survivors; o.nS += r.n_survivors; }
      o.efm += (r.efm_factor || 0) * r.area_ha;
      o.nR += r.n_removed_earlier; o.nM += r.n_missing; o.nX += r.n_excluded; o.ns = (o.ns || 0) + r.n_survivors;
    });
    if (!o.n) return null;
    o.pai = o.num / o.area; o.efmf = o.efm / o.area; o.paiE = o.pai * o.efmf;
    o.paiStatic = o.stat / o.area; o.dbhMm = o.nS ? o.dbh / o.nS : NaN;
    o.incr = o.t2 + o.rem - o.t1;
    return o;
  };
  const allKeys = mRows.map((r) => String(r.parzelle));
  sec.setAttribute("data-increment-measured", (() => { const c = calcM(null, GS0, RG0); return c ? c.pai.toFixed(4) : ""; })());
  if (!INC.init) {
    INC.init = true;
    const ha = HASH.age;
    INC.age = ha !== undefined ? (ha !== "" && isFinite(+ha) ? +ha : null) : I.default_age;
    INC.ages = {};
    if (HASH.ages !== undefined) (HASH.ages || "").split(",").forEach((kv) => { const [p, a] = kv.split(":"); if (pk.includes(p) && a !== "" && isFinite(+a)) INC.ages[p] = +a; });
    else Object.entries(I.default_age_per_parzelle || {}).forEach(([p, a]) => { if (pk.includes(p) && isNum(a)) INC.ages[p] = a; });
    INC.zm = INC_MODES.some((m) => m[0] === HASH.zm) ? HASH.zm : "all";
    INC.gs = Object.assign({}, GS0);
    if (HASH.gs) HASH.gs.split(",").forEach((kv) => { const [y, s] = kv.split(":"); if (y in GS0 && s !== "" && isFinite(+s) && +s >= 1 && +s <= 10) INC.gs[y] = Math.round(+s); });
    INC.rg = HASH.rg !== undefined && HASH.rg !== "" && isFinite(+HASH.rg) ? Math.min(1, Math.max(0, +HASH.rg)) : RG0;
  }
  const gsDefault = () => syears.every((y) => INC.gs[y] === GS0[y]) && INC.rg === RG0;
  const sync = () => {
    HASH.gs = syears.every((y) => INC.gs[y] === GS0[y]) ? "" : syears.map((y) => y + ":" + INC.gs[y]).join(",");
    HASH.rg = INC.rg === RG0 ? "" : String(INC.rg);
    HASH.age = isNum(INC.age) ? String(INC.age) : "";
    HASH.ages = pk.filter((p) => isNum(INC.ages[p])).map((p) => p + ":" + INC.ages[p]).join(",");
    HASH.zm = INC.zm === "all" ? "" : INC.zm;
    writeHash();
  };
  const clampAge = (a) => Math.min(a1, Math.max(a0, Math.round(a)));
  const FIELDS = ["si", "ex", "ao", "pt", "mt", "sl", "py", "pg", "pye", "pge"];
  const rowAt = (k, age) => { const e = BA[k], i = clampAge(age) - a0, r = {}; FIELDS.forEach((c) => (r[c] = e[c][i])); return r; };
  const agePz = (p) => (isNum(INC.ages[p]) ? INC.ages[p] : isNum(INC.age) ? INC.age : null);
  const anyOverride = () => pk.some((p) => isNum(INC.ages[p]));
  // Stand values: the precomputed stand row when all Parzellen share the trial age, otherwise the area-weighted mean over the Parzellen.
  const standAt = () => {
    if (!anyOverride()) return isNum(INC.age) ? Object.assign(rowAt("stand", INC.age), { yt: BA.stand.yield_table }) : null;
    const as = pk.map(agePz);
    if (as.some((a) => a == null)) return null;
    const rows = pk.map((p, i) => rowAt(p, as[i])), w = pk.map((p) => G.areas_ha[p]);
    const wm = (c) => { let s = 0, ws = 0; rows.forEach((r, i) => { if (isNum(r[c]) && isNum(w[i])) { s += r[c] * w[i]; ws += w[i]; } }); return ws ? s / ws : null; };
    const out = {};
    ["si", "sl", "pt", "mt", "py", "pg", "pye", "pge"].forEach((c) => (out[c] = wm(c)));
    out.ex = rows.some((r) => r.ex === 1) ? 1 : 0; out.ao = rows.some((r) => r.ao === 1) ? 1 : 0;
    out.yt = unique(pk.map((p) => BA[p].yield_table).filter(Boolean)).join("; ");
    return out;
  };

  // ----- controls (built once, so that typing keeps the focus) -----
  const ctl = card(sec, "Parameters of the increment", "Increment of the standing volume in Vfm m. R. per ha and year; the harvest volume (Efm o. R.) follows with the volume-weighted reduction factor. The yield table and the tree growth functions need the approximate stand age, the measured increment does not.");
  const segBtns = INC_MODES.map(([k, lab]) => h("button", { type: "button", "aria-pressed": String(INC.zm === k), "data-method": k, text: lab, onclick: () => {
    INC.zm = k; segBtns.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.method === k))); sync(); showParams(); redraw(); } }));
  const ageInput = (value, onChange, label) => {
    const inp = h("input", { type: "number", min: String(a0), max: String(a1), step: "1", inputmode: "numeric", "aria-label": label });
    inp.value = isNum(value) ? String(value) : "";
    inp.addEventListener("input", () => { const v = inp.value.trim(); onChange(v !== "" && isFinite(+v) ? +v : null); sync(); redraw(); });
    return inp;
  };
  const trial = ageInput(INC.age, (v) => (INC.age = v), `Stand age (years, in ${ref})`);
  const pzInputs = {};
  const pzGrid = h("div", { class: "params" });
  pk.forEach((p) => { pzInputs[p] = ageInput(INC.ages[p], (v) => { if (v == null) delete INC.ages[p]; else INC.ages[p] = v; }, "Age of Parzelle " + p); pzGrid.appendChild(h("label", null, "Parzelle " + p, pzInputs[p])); });
  const det = h("details", { class: "inc-ages", open: anyOverride() ? "" : null }, h("summary", { text: "Age per Parzelle (optional)" }),
    h("p", { class: "legend-note", text: "An empty field uses the stand age above. The stand value is the area-weighted mean over the Parzellen, each at its own age." }), pzGrid,
    h("button", { class: "btn", type: "button", text: "Clear", onclick: () => { INC.ages = {}; pk.forEach((p) => (pzInputs[p].value = "")); sync(); redraw(); } }));
  ctl.appendChild(h("div", { class: "inc-row" }, h("div", { class: "seg", role: "group", "aria-label": "Increment method" }, segBtns), h("label", null, `Stand age (years, in ${ref})`, trial)));
  ctl.appendChild(det);
  // parameters of the measured method: growing seasons per survey year, growth of the trees removed earlier
  const gsInputs = {}, mParams = h("div", { class: "inc-row inc-measured" });
  syears.forEach((y) => {
    const ps = mRows.filter((r) => String(r.survey_year) === y).map((r) => +r.parzelle).filter((v) => isFinite(v));
    const span = ps.length ? ` (Parzellen ${Math.min(...ps)}–${Math.max(...ps)})` : "";
    const inp = h("input", { type: "number", min: "1", max: "10", step: "1", inputmode: "numeric", "aria-label": `Growing seasons 2020 to the ${y} survey` });
    inp.value = String(INC.gs[y]);
    inp.addEventListener("input", () => { const v = inp.value.trim(); if (v !== "" && isFinite(+v) && +v >= 1 && +v <= 10) { INC.gs[y] = Math.round(+v); sync(); redraw(); } });
    gsInputs[y] = inp;
    mParams.appendChild(h("label", null, `Growing seasons 2020 to the ${y} survey${span}`, inp));
  });
  const rgOut = h("b", { text: f(INC.rg, 1) });
  const rgInp = h("input", { type: "range", min: "0", max: "1", step: "0.1", "aria-label": "Growth of the trees removed earlier (share of the period)" });
  rgInp.value = String(INC.rg);
  rgInp.addEventListener("input", () => { INC.rg = Math.round(+rgInp.value * 10) / 10; rgOut.textContent = f(INC.rg, 1); sync(); redraw(); });
  mParams.appendChild(h("label", null, "Growth of the trees removed earlier (share of the period)", rgInp, rgOut));
  mParams.appendChild(h("button", { class: "btn", type: "button", text: "Reset", onclick: () => {
    syears.forEach((y) => { INC.gs[y] = GS0[y]; gsInputs[y].value = String(GS0[y]); });
    INC.rg = RG0; rgInp.value = String(RG0); rgOut.textContent = f(RG0, 1); sync(); redraw(); } }));
  const mHelp = h("p", { class: "legend-note", text: "The inventory records neither the number of growing seasons between the surveys nor when the trees flagged as removed left the stand. 0 = removed at the 2020 survey, no growth; 1 = grew like the surviving trees until the current survey." });
  const showParams = () => { const on = INC.zm === "measured" || INC.zm === "all"; mParams.hidden = !on; mHelp.hidden = !on; };
  ctl.appendChild(mParams);
  ctl.appendChild(mHelp);
  showParams();
  const dyn = h("div");
  sec.appendChild(dyn);
  let last = null;

  function redraw() {
    dyn.innerHTML = "";
    if (!last) return;
    const calc = last, area = info.area, vfmHa = area ? griSum(calc, "vF") / area : NaN;
    const methods = INC.zm === "all" ? INC_METHODS.map((m) => m[0]) : [INC.zm];
    const needAge = methods.some((m) => m !== "measured"), st = standAt(), hasAge = isNum(INC.age);
    if (hasAge) sec.setAttribute("data-increment-age", String(INC.age)); else sec.removeAttribute("data-increment-age");
    sec.setAttribute("data-increment-methods", methods.join(","));
    const note = (cls, bold, text) => dyn.appendChild(h("div", { class: "callout " + cls }, h("b", { text: bold + " " }), text));
    if (needAge && !st) note("info", "Stand age needed.", hasAge || anyOverride() ? "Enter the stand age, or an age for every Parzelle, to show the predicted increment." : "Enter the approximate stand age to show the predicted increment");
    if (hasAge && (INC.age < a0 || INC.age > a1)) note("warn", "Age outside the range.", `The age grid covers ${a0} to ${a1} years; the nearest age is used.`);

    const mKeys = parzF().map((p) => String(p.parzelle)).filter((p) => allKeys.includes(p)), cm = calcM(mKeys, INC.gs, INC.rg);
    const mExact = gsDefault() && F.p === "all";
    // values of the stand per method: v in Vfm/ha/yr, e in Efm/ha/yr
    const vals = (m) => {
      if (m === "measured") return cm ? { v: cm.pai, e: cm.paiE } : null;
      return st ? (m === "yield" ? { v: st.py, e: st.pye } : { v: st.pg, e: st.pge }) : null;
    };
    const caveat = (m) => (I.methods && I.methods[incTextKey[m]]) || "";
    const entry = (value, statement, n) => ({ value, unit: "", n: n != null ? n : nF, statement });
    const share = (v) => (isNum(v) && isNum(vfmHa) && vfmHa ? v / vfmHa : NaN);
    const vf = (v, nd = 2) => (isNum(v) ? f(v, nd) + " Vfm/ha/yr" : "–");
    const ef = (v, nd = 2) => (isNum(v) ? f(v, nd) + " Efm/ha/yr" : "–");
    const g = h("div", { class: "grid tiles" });
    dyn.appendChild(g);
    const statFor = (m, text) => { const s = m === "measured" && mExact ? S("increment.measured.stand.pai") : null;
      if (s && isNum(s.value)) return Object.assign({}, s, { statement: text });
      return entry(vals(m) && vals(m).v, text, m === "measured" && cm ? cm.ns : undefined); };
    const mTile = (m, label, kind) => {
      const x = vals(m);
      if (!x) return tile(label, entry(NaN, "needs the stand age"), { value: "–", noci: true, statement: "Needs the stand age: enter it above.", follows: true });
      const txt = `${incLabel(m)}: the stand increases by ${f(x.v, 2)} Vfm m. R. per ha and year` + (isNum(x.e) ? ` (${f(x.e, 2)} Efm o. R.)` : "") +
        (isNum(share(x.v)) ? `, ${pct(share(x.v), 2)} of the standing Vorrat of ${f(vfmHa, 0)} Vfm/ha per year` : "") + ".";
      if (kind === "compact") return tile(label, statFor(m, txt), { value: vf(x.v), statement: txt, follows: true,
        extra: [{ label: "Efm o. R.", text: ef(x.e) }, { label: "trial area", text: isNum(x.v) && area ? f(x.v * area, 1) + " Vfm/yr" : "–" }, { label: "of the Vorrat", text: pct(share(x.v), 2) + " /yr" }] });
      if (kind === "vfm") return tile(label, statFor(m, txt), { value: vf(x.v), statement: txt, follows: true });
      if (kind === "efm") return tile(label, entry(x.e, "Harvest volume under bark: the Vfm increment times the volume-weighted reduction factor of the species."), { value: ef(x.e), noci: true, follows: true });
      if (kind === "total") return tile(label, entry(x.v, `Increment of the ${f(area, 2)} ha trial area per year (Vfm m. R.)` + (isNum(x.e) ? `; ${f(x.e * area, 1)} Efm o. R. per year.` : ".")),
        { value: isNum(x.v) && area ? f(x.v * area, 1) + " Vfm/yr" : "–", noci: true, follows: true, extra: [{ label: "Efm o. R.", text: isNum(x.e) && area ? f(x.e * area, 1) + " Efm/yr" : "–" }] });
      return tile(label, entry(share(x.v), `The increment divided by the standing Vorrat of the field trees (${f(vfmHa, 1)} Vfm/ha from the volume section above, which follows the species groups).`),
        { value: pct(share(x.v), 2) + " per year", noci: true, follows: true });
    };
    if (INC.zm === "all") methods.forEach((m) => g.appendChild(mTile(m, "Increment, " + incLabel(m), "compact")));
    else {
      const m = INC.zm;
      g.append(mTile(m, "Increment (Vfm m. R. per ha and year)", "vfm"), mTile(m, "Increment (Efm o. R. per ha and year)", "efm"),
        mTile(m, "Total increment per year, trial area", "total"), mTile(m, "Increment as share of the Vorrat", "share"));
    }
    if (methods.includes("measured") && cm) {
      const sTxt = syears.map((y) => `${INC.gs[y]} (${y})`).join(", ");
      g.append(tile("Increment with one height curve for both dates (comparison)", entry(cm.paiStatic, `Volume growth of the surviving trees alone, with the height-diameter curve of the current survey at both dates, per ha and year (Vfm m. R.); rescaled to ${sTxt} growing seasons. A comparison; heights grow only along the curve.`, cm.ns),
        { value: vf(cm.paiStatic), noci: true, follows: true }),
        tile("Mean diameter increment of the surviving trees", entry(cm.dbhMm, `Mean diameter increment of the surviving trees (mm per year), rescaled to ${sTxt} growing seasons.`, cm.ns),
          { value: isNum(cm.dbhMm) ? f(cm.dbhMm, 2) + " mm/yr" : "–", noci: true, follows: true }));
      if (!mExact) dyn.appendChild(h("p", { class: "legend-note", text: "Measured increment recalculated with changed parameters or for a Parzelle selection; no confidence interval." }));
    }
    if (methods.includes("yield") && st) {
      const gehr = /gehrhardt/i.test(st.yt || "");
      const siTxt = `Site index of the stand from the height of the quadratic-mean-diameter tree at the assumed age (${st.yt || "yield table"}).` +
        (gehr ? " In the spruce table of Gehrhardt (1921) the site index is a yield class: a lower value means a better site, and it is not a height." : "") +
        (st.ex === 1 ? " The site index lies outside the table, so the table values are extrapolated." : "");
      g.append(tile("Site index", entry(st.si, siTxt), { value: f(st.si, 2), noci: true, follows: true, extra: [{ label: "yield table", text: st.yt || "–" }] }),
        tile("Stocking level", entry(st.sl, "Basal area of the stand over the table basal area at the site index and age; the table increment is multiplied by it where it is below 1."), { value: f(st.sl, 2), noci: true, follows: true }),
        tile("Table increment, before stocking level", entry(st.pt, "Increment of the yield table at the site index of the stand, before the stocking level is applied."), { value: vf(st.pt), noci: true, follows: true }),
        tile("Mean annual increment (dGZ), table", entry(st.mt, "Mean annual increment of the yield table at the site index of the stand."), { value: vf(st.mt), noci: true, follows: true }));
    }

    // ledger of the measured increment
    if (methods.includes("measured") && cm) {
      const lc = card(dyn, "Ledger of the measured increment", "Increment = V alive at the current survey + V removed - V alive at the 2020 survey, per ha of the surveyed plot area (Vfm m. R.). Follows the Parzelle filter and the parameters above.");
      lc.setAttribute("data-ledger", "measured");
      const ph = (v) => (isNum(v) ? f(v / cm.area, 2) : "–"), one = cm.seasons.size === 1 ? [...cm.seasons][0] : null;
      const rows = [
        { l: "V alive at the 2020 survey, V(t1)", v: ph(cm.t1) + " Vfm/ha", n: "all trees alive in 2020, GRI volume with the 2020 height-diameter curve" },
        { l: "V alive at the current survey, V(t2)", v: ph(cm.t2) + " Vfm/ha", n: "the surviving trees, GRI volume with the current height-diameter curve" },
        { l: "V removed", v: ph(cm.rem) + " Vfm/ha", n: `2020 volume of the trees removed earlier (${ph(cm.rem0)} Vfm/ha) plus ${f(INC.rg * 100, 0)} % of the relative growth of the surviving trees` },
        { l: "Increment over the period", v: ph(cm.incr) + " Vfm/ha", n: "V(t2) + V removed - V(t1)" },
        { l: "Periodic annual increment, Vfm m. R.", v: f(cm.pai, 2) + " Vfm/ha/yr", n: one ? `increment / ${one} growing seasons` : "increment / growing seasons of each Parzelle" },
        { l: "Periodic annual increment, Efm o. R.", v: f(cm.paiE, 2) + " Efm/ha/yr", n: `Vfm increment x volume-weighted reduction factor ${f(cm.efmf, 3)}` }];
      lc.appendChild(mkTable([{ k: "l", label: "Term", cls: "l" }, { k: "v", label: "Value" }, { k: "n", label: "Calculation", cls: "l" }], rows));
      lc.appendChild(h("p", { class: "legend-note", text: `Surviving trees: ${cm.ns}; trees removed earlier: ${cm.nR}; missing trees: ${cm.nM}; trees excluded for an implausible diameter change: ${cm.nX}; area ${f(cm.area, 2)} ha.` }));
    }

    // per Parzelle
    const grid = h("div", { class: "grid wide" });
    dyn.appendChild(grid);
    const ps = parzF().map((p) => String(p.parzelle)).filter((p) => BA[p]);
    const yv = (m, p) => { if (m === "measured") { const c1 = calcM([p], INC.gs, INC.rg); return c1 ? c1.pai : null; } const a = agePz(p); if (a == null) return null; const r = rowAt(p, a); return m === "yield" ? r.py : r.pg; };
    let c = card(grid, INC.zm === "all" ? "Per Parzelle: increment by the three methods" : "Per Parzelle: increment, " + incLabel(INC.zm), "Increment of the standing volume (Vfm m. R. per ha and year). Follows the Parzelle filter. Parzellen without an age have no value for the age-dependent methods.");
    plot(c, methods.map((m) => ({ type: "bar", name: incLabel(m), x: ps, y: ps.map((p) => yv(m, p)), marker: { color: incCol(m) } })),
      { barmode: "group", xaxis: { title: "Parzelle", type: "category", tickfont: { size: 9 } }, yaxis: { title: "Increment (Vfm m. R. per ha and year)" } });
    if (INC.zm === "all") {
      const rows = ps.map((p) => { const a = agePz(p), r = a == null ? null : rowAt(p, a);
        return { p, age: a, sp: BA[p].main_species, yt: BA[p].yield_table, si: r && r.si, sl: r && r.sl, y: yv("yield", p), g: yv("growth", p), m: yv("measured", p) }; });
      c.appendChild(mkTable([{ k: "p", label: "Parzelle" }, { k: "age", label: "Age used", fmt: (r) => (r.age == null ? "–" : String(r.age)) }, { k: "sp", label: "Main species", cls: "l" },
        { k: "yt", label: "Yield table", cls: "l" }, { k: "si", label: "Site index", fmt: (r) => f(r.si, 2) }, { k: "sl", label: "Stocking level", fmt: (r) => f(r.sl, 2) },
        { k: "y", label: "Yield table (Vfm/ha/yr)", fmt: (r) => f(r.y, 2) }, { k: "g", label: "Tree growth (Vfm/ha/yr)", fmt: (r) => f(r.g, 2) }, { k: "m", label: "Measured (Vfm/ha/yr)", fmt: (r) => f(r.m, 2) }], rows));
    }

    // against age
    const key = F.p !== "all" && BA[F.p] ? F.p : "stand", e = BA[key];
    c = card(grid, "Increment against assumed stand age", `Predicted increment for ${key === "stand" ? "the stand" : "Parzelle " + key} for every stand age of the grid, with the measured increment as a horizontal line: it shows how much the prediction depends on the age. Dashed: the site index lies outside the yield table (extrapolated). Follows the Parzelle filter.`);
    const xs = []; for (let a = a0; a <= a1; a++) xs.push(a);
    const tr = [];
    if (e.ex.some((v) => v === 1)) tr.push({ type: "scatter", mode: "lines", name: "Yield table, site index extrapolated", x: xs, y: e.py, line: { color: incCol("yield"), width: 2, dash: "dash" }, hovertemplate: "age %{x}: %{y:.2f} Vfm/ha/yr<extra>Yield table (extrapolated)</extra>" });
    tr.push({ type: "scatter", mode: "lines", name: incLabel("yield"), x: xs, y: e.py.map((v, i) => (e.ex[i] === 1 ? null : v)), line: { color: incCol("yield"), width: 2.5 }, hovertemplate: "age %{x}: %{y:.2f} Vfm/ha/yr<extra>Yield table</extra>" },
      { type: "scatter", mode: "lines", name: incLabel("growth"), x: xs, y: e.pg, line: { color: incCol("growth"), width: 2.5 }, hovertemplate: "age %{x}: %{y:.2f} Vfm/ha/yr<extra>Tree growth functions</extra>" });
    const mr = calcM(key === "stand" ? allKeys : [key], INC.gs, INC.rg);
    if (mr && isNum(mr.pai)) tr.push({ type: "scatter", mode: "lines", name: incLabel("measured"), x: [a0, a1], y: [mr.pai, mr.pai], line: { color: incCol("measured"), width: 2.5, dash: "dot" }, hovertemplate: "%{y:.2f} Vfm/ha/yr<extra>Measured</extra>" });
    const mk = key === "stand" ? INC.age : agePz(key), lay = { xaxis: { title: `Assumed stand age in ${ref} (years)` }, yaxis: { title: "Increment (Vfm m. R. per ha and year)" } };
    if (isNum(mk)) {
      lay.shapes = [{ type: "line", xref: "x", yref: "paper", x0: mk, x1: mk, y0: 0, y1: 1, line: { color: C.ink2, width: 1.5, dash: "dot" } }];
      lay.annotations = [{ x: mk, y: 1, xref: "x", yref: "paper", text: "chosen age " + mk, showarrow: false, xanchor: "left", yanchor: "top", font: { size: 11, color: C.ink2 } }];
    }
    plot(c, tr, lay, { cls: "tall" });

    // notes
    if (methods.includes("yield") && st && st.ex === 1) note("warn", "Site index extrapolated.", `At this age the site index lies outside the yield table (${st.yt || "yield table"}), so the table values are extrapolated.`);
    if (methods.includes("yield") && st && st.ao === 1) note("warn", "Age outside the yield table.", "The age lies outside the age range of the table; the nearest tabulated age is used.");
    methods.forEach((m) => { if (caveat(m)) dyn.appendChild(h("p", { class: "legend-note" }, h("span", { class: "inc-key", style: "background:" + incCol(m) }), caveat(m))); });
    if (methods.includes("measured")) [
      "The trees felled in the 2026 harvest were measured before felling and count as alive at the current survey.",
      "The reference area is the surveyed plot area (convex hull of the four corner points, about 48 m x 30 m per Parzelle, gaps included).",
      "No ingrowth and no removal dates are recorded."].forEach((t) => dyn.appendChild(h("p", { class: "legend-note", text: t })));
    if (!info.isDefault()) note("info", "Species assignment.", "The increment uses the configured species assignment and does not follow the species-group dropdowns above; only the share of the Vorrat does.");
  }
  return (calc) => { last = calc; redraw(); };
}

/* ======================================================================
   Methods and caveats
   ====================================================================== */
function mdToHtml(md) {
  const lines = md.replace(/\r/g, "").split("\n");
  let out = "", inList = false, para = [];
  const inl = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  const flush = () => { if (para.length) { out += "<p>" + inl(para.join(" ")) + "</p>"; para = []; } };
  const closeList = () => { if (inList) { out += "</ul>"; inList = false; } };
  for (const line of lines) {
    let m;
    if ((m = line.match(/^(#{1,4})\s+(.*)/))) { flush(); closeList(); const lv = Math.min(4, m[1].length + 1); out += `<h${lv}>${inl(m[2])}</h${lv}>`; }
    else if ((m = line.match(/^[-*]\s+(.*)/))) { flush(); if (!inList) { out += "<ul>"; inList = true; } out += "<li>" + inl(m[1]) + "</li>"; }
    else if (/^\s{2,}\S/.test(line) && inList) { out = out.replace(/<\/li>$/, " " + inl(line.trim()) + "</li>"); }
    else if (!line.trim()) { flush(); closeList(); }
    else { closeList(); para.push(line.trim()); }
  }
  flush(); closeList();
  return out;
}
function renderMethods(root) {
  root.appendChild(h("p", { class: "lede", text: "How the numbers are made, what they test and what they do not." }));
  const c0 = card(root, "Metrics, defined once", null);
  c0.appendChild(mkTable([{ k: "m", label: "Metric", cls: "l" }, { k: "d", label: "Definition", cls: "wrap l" }], [
    { m: "Bias", d: "Mean of (ULS minus reference). Positive: ULS is higher or larger." }, { m: "RMSE", d: "Root of the mean squared difference. One large error can dominate it." },
    { m: "Median", d: "Middle value of the differences; the typical tree." }, { m: "MAD", d: "Median absolute deviation of the differences around their median; robust spread." },
    { m: "95 % CI", d: "Bootstrap interval (resampling trees) for bias, RMSE, median and MAD; Wilson interval for shares." },
    { m: "Bland-Altman", d: "Difference of two methods against their mean; limits of agreement are bias ± 1.96 SD." },
    { m: "all_matched / detected_only", d: "all_matched: detected and uncertain matches. detected_only: only matches the analysis calls detected." },
    { m: "Noise floor", d: "Difference between Vertex and Maßband: what ground measurement itself cannot resolve." },
    { m: "Maßband is the reference", d: "Only the tape length on the felled tree is the height reference. The Vertex is also measured on lying trees and is evaluated separately." },
  ]));
  (D.methods || []).forEach((m) => { const c = card(root, null, null); c.appendChild(h("div", { class: "md", html: mdToHtml(m.text) })); });
  const co = card(root, "Open questions and what is not done", null);
  const ul = h("ul");
  const open = D.ledger.filter((r) => r.issue === "sample_tree_stands_after_harvest" || r.issue === "sample_tree_remark");
  open.forEach((r) => ul.appendChild(h("li", { text: `${r.tree_id}: ${r.detail}` })));
  ["Point-density and canopy-model resolution sensitivity: not done.", "Flight comparison and photogrammetry heights: not done in this version.", "Growth and assortments: not done.", "No correction model for ULS height is fitted on so few sample trees."].forEach((t) => ul.appendChild(h("li", { text: t })));
  if (META.missing_tables && META.missing_tables.length) ul.appendChild(h("li", { text: "Data files not available for this build: " + META.missing_tables.join(", ") + "." }));
  co.appendChild(ul);
  co.appendChild(h("p", { class: "legend-note", text: "Built " + META.built + " for site " + META.site_id + " (" + META.site_name + ")." }));
}

/* ======================================================================
   Shell: header, tabs, filters, render loop
   ====================================================================== */
function buildFilters() {
  const el = document.getElementById("filters");
  const plots = unique(T.map((t) => t.p)).sort((a, b) => a - b);
  const sp = h("select", { "aria-label": "Parzelle" }, h("option", { value: "all", text: "All Parzellen" }), plots.map((p) => h("option", { value: String(p), text: "Parzelle " + p })));
  const si = h("select", { "aria-label": "Identity class" }, h("option", { value: "all", text: "all classes" }), h("option", { value: "confirmed+probable", text: "confirmed + probable" }), IDC.map((c) => h("option", { value: c, text: c })));
  const ss = h("select", { "aria-label": "Trees included" }, h("option", { value: "all_matched", text: "all matched (detected + uncertain)" }), h("option", { value: "detected_only", text: "detected only" }));
  sp.value = F.p; si.value = F.idc; ss.value = F.sub;
  const upd = () => { F.p = sp.value; F.idc = si.value; F.sub = ss.value; renderActive(); };
  [sp, si, ss].forEach((x) => x.addEventListener("change", upd));
  const reset = h("button", { class: "btn", type: "button", onclick: () => { sp.value = "all"; si.value = "all"; ss.value = "all_matched"; upd(); } }, "Reset");
  el.append(h("strong", { text: "Filters:" }), h("label", null, "Parzelle ", sp), h("label", null, "Identity ", si), h("label", null, "Trees ", ss), reset);
}
let active = "overview";
function renderActive() {
  const main = document.getElementById("main");
  loadColors();
  chartCount = 0; tableCount = 0;
  main.innerHTML = "";
  const t = TABS.find((x) => x[0] === active) || TABS[0];
  const note = document.getElementById("filter-note");
  if (filtersActive()) {
    note.hidden = false;
    note.textContent = `Filters active (${F.p === "all" ? "all Parzellen" : "Parzelle " + F.p}, identity: ${F.idc}, ${F.sub.replace("_", " ")}). Charts that use single trees follow them; tiles, the accuracy table and other numbers from the analysis files (stats.json) cannot and say so.`;
  } else note.hidden = true;
  try { t[2](main); } catch (e) { __report("tab " + t[0] + ": " + e.message + (e.stack ? " | " + e.stack.split("\n")[1] : "")); main.appendChild(h("div", { class: "err", text: "This tab could not be drawn: " + e.message })); }
  document.body.dataset.rendered = String(chartCount);
  document.body.dataset.tables = String(tableCount);
  document.body.dataset.tab = active;
  setTimeout(() => { document.body.dataset.vw = String(window.innerWidth); document.body.dataset.sw = String(document.documentElement.scrollWidth); }, 1500);
  [...document.querySelectorAll("#tabs button")].forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === active)));
}
function setTab(id, push) {
  active = id; HASH.tab = id; if (push !== false) writeHash();
  renderActive();
  window.scrollTo(0, 0);
}
function init() {
  Object.assign(HASH, readHash());
  if (HASH.theme === "dark" || HASH.theme === "light") document.documentElement.dataset.theme = HASH.theme;
  else try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage blocked */ }
  if (HASH.p) F.p = HASH.p;
  document.getElementById("title").textContent = META.title;
  document.getElementById("subtitle").textContent = `${META.site_name} · built ${META.built} · ${T.length} field trees in ${unique(T.map((t) => t.p)).length} Parzellen`;
  document.title = META.title;
  const nav = document.getElementById("tabs");
  TABS.forEach(([id, label]) => nav.appendChild(h("button", { type: "button", role: "tab", "data-tab": id, onclick: () => setTab(id) }, label)));
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const nx = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = nx;
    try { localStorage.setItem("theme", nx); } catch (e) { /* ignore */ }
    renderActive();
  });
  window.addEventListener("hashchange", () => { const hh = readHash(); if (hh.tab && hh.tab !== active && TABS.some((t) => t[0] === hh.tab)) { Object.assign(HASH, hh); active = hh.tab; renderActive(); } });
  buildFilters();
  active = TABS.some((t) => t[0] === HASH.tab) ? HASH.tab : "overview";
  renderActive();
}
try { init(); } catch (e) { __report("init: " + e.message); }
})();
