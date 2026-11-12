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
  C.uls = C.c[0]; C.ground = C.c[1]; C.vertex = C.c[2];
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
  if (o.extra) t.appendChild(h("div", { class: "extra" }, o.extra.filter((x) => x.e).map((x) => h("span", null, x.label + " ", h("b", { text: fmtVal(x.e) })))));
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
  ["samples", "Sample explorer", renderSamples], ["volume", "Wood volume", renderVolume], ["methods", "Methods and caveats", renderMethods],
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
