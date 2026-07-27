"""Build a self-contained interactive HTML explorer for the P(homozygous) scores.

A slider sets a threshold on the per-embryo P(homozygous) score; traces below it are filtered
out live, so you can watch which trajectories survive as the bar is raised. A companion bar
chart shows how many embryos remain in each zygosity at the current threshold -- that is the
readout that matters, because if the score were informative the wildtype bar would empty out
long before the homozygous bar.

Everything (data + JS) is inlined: no CDN, no server. Download the .html and open it.

Run:
    conda run -n segmentation_grounded_sam --no-capture-output python \\
        results/mcolon/20260715_lab_meeting_scratch/27_build_threshold_explorer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SOURCE = PROJECT_ROOT / "results/mcolon/20251219_b9d2_phenotype_extraction/data/b9d2_labeled_data.csv"
SCORES = RUN_DIR / "figures/homo_vs_wt_probability/p_homo_scores.csv"
OUT = RUN_DIR / "figures/homo_vs_wt_probability/threshold_explorer.html"

ID_COL = "embryo_id"
TIME_COL = "predicted_stage_hpf"
FEATURES = ["baseline_deviation_normalized", "total_length_um"]
FEATURE_LABEL = {
    "baseline_deviation_normalized": "curvature (baseline deviation, normalized)",
    "total_length_um": "total length (µm)",
}

ZYG_ORDER = ["b9d2_wildtype", "b9d2_heterozygous", "b9d2_homozygous", "b9d2_unknown"]
ZYG_LABEL = {
    "b9d2_wildtype": "wildtype",
    "b9d2_heterozygous": "heterozygous",
    "b9d2_homozygous": "homozygous",
    "b9d2_unknown": "unknown",
}
ZYG_COLOR = {
    "wildtype": "#7F7F7F",
    "heterozygous": "#F7B267",
    "homozygous": "#B2182B",
    "unknown": "#4C9F70",
}

# Downsample each trajectory to keep the payload small enough to inline comfortably.
MAX_POINTS = 60


def build_payload() -> dict:
    df = pd.read_csv(SOURCE, low_memory=False)
    scores = pd.read_csv(SCORES)

    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df = df.dropna(subset=[TIME_COL, *FEATURES])

    score_map = scores.set_index(ID_COL)["p_homo_embryo"].to_dict()
    pheno_map = scores.set_index(ID_COL)["cluster_categories"].to_dict()

    embryos = []
    for eid, grp in df.groupby(ID_COL):
        if eid not in score_map:
            continue
        grp = grp.sort_values(TIME_COL)
        if len(grp) > MAX_POINTS:
            idx = np.linspace(0, len(grp) - 1, MAX_POINTS).astype(int)
            grp = grp.iloc[idx]
        geno = grp["genotype"].iloc[0]
        embryos.append({
            "id": eid,
            "zyg": ZYG_LABEL.get(geno, "unknown"),
            "p": round(float(score_map[eid]), 4),
            "pheno": str(pheno_map.get(eid, "")),
            "t": [round(float(v), 2) for v in grp[TIME_COL]],
            "y0": [round(float(v), 4) for v in grp[FEATURES[0]]],
            "y1": [round(float(v), 1) for v in grp[FEATURES[1]]],
        })

    return {
        "embryos": embryos,
        "zygOrder": [ZYG_LABEL[z] for z in ZYG_ORDER],
        "zygColor": ZYG_COLOR,
        "featureLabels": [FEATURE_LABEL[f] for f in FEATURES],
    }


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>b9d2 — P(homozygous) threshold explorer</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --line: #e2e2e2; --panel: #fafafa;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#15171a; --fg:#e8e8e8; --muted:#9aa0a6; --line:#2c3034; --panel:#1c1f23; }
  }
  * { box-sizing: border-box; }
  body {
    margin:0; padding:24px; background:var(--bg); color:var(--fg);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  }
  h1 { font-size:19px; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:20px; max-width:900px; }
  .controls {
    background:var(--panel); border:1px solid var(--line); border-radius:8px;
    padding:16px 18px; margin-bottom:20px; display:flex; gap:22px;
    align-items:center; flex-wrap:wrap;
  }
  .controls label { font-weight:600; white-space:nowrap; }
  input[type=range] { width:min(420px, 46vw); accent-color:#B2182B; }
  .val {
    font-variant-numeric:tabular-nums; font-weight:700; font-size:17px;
    min-width:58px; display:inline-block;
  }
  .chk { display:flex; align-items:center; gap:6px; font-weight:500; }
  .layout { display:grid; grid-template-columns:minmax(0,1fr) 300px; gap:20px; align-items:start; }
  @media (max-width: 980px) { .layout { grid-template-columns:1fr; } }
  .card {
    background:var(--panel); border:1px solid var(--line);
    border-radius:8px; padding:14px; overflow-x:auto;
  }
  .card h2 { font-size:13px; margin:0 0 10px; color:var(--muted); font-weight:600;
             text-transform:uppercase; letter-spacing:.04em; }
  svg { display:block; max-width:100%; height:auto; }
  .axis { stroke:var(--line); }
  .tick { fill:var(--muted); font-size:10px; }
  .albl { fill:var(--muted); font-size:11px; }
  table { border-collapse:collapse; width:100%; font-size:12.5px; }
  th, td { text-align:left; padding:5px 6px; border-bottom:1px solid var(--line); }
  th { color:var(--muted); font-weight:600; }
  td.n { text-align:right; font-variant-numeric:tabular-nums; }
  .swatch { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }
  .hint { color:var(--muted); font-size:12px; margin-top:10px; }
  .facet-row { display:flex; gap:6px; flex-wrap:nowrap; overflow-x:auto; padding-bottom:4px; }
  #tip {
    position:fixed; pointer-events:none; z-index:50; display:none;
    background:var(--fg); color:var(--bg); padding:7px 10px; border-radius:6px;
    font-size:12px; line-height:1.45; box-shadow:0 3px 12px rgba(0,0,0,.28);
    max-width:280px;
  }
  #tip b { font-size:12.5px; }
  #tip .k { opacity:.7; }
  .facet-row svg { flex:0 0 auto; }
  select {
    font:inherit; padding:5px 8px; border-radius:6px;
    border:1px solid var(--line); background:var(--bg); color:var(--fg);
  }
</style>
</head>
<body>
<h1>b9d2 — P(homozygous) threshold explorer</h1>
<div class="sub">
  Each line is one embryo, colored by genotype. The slider filters on the per-embryo
  <em>P(homozygous)</em> score; flip the operator to keep the high (mutant-like) or the low
  (wildtype-like) end. If the score were informative, keeping the high end would empty the
  wildtype bar while the homozygous bar held &mdash; and vice versa. Counts update live on the
  right. Hover any trace for its embryo ID, curated phenotype and score.
</div>

<div class="controls">
  <label for="thr">P(homozygous)</label>
  <select id="dir">
    <option value="ge">&ge; (keep high / mutant-like)</option>
    <option value="le">&le; (keep low / wildtype-like)</option>
  </select>
  <input type="range" id="thr" min="0" max="1" step="0.01" value="0">
  <span class="val" id="thrVal">0.00</span>
  <span class="chk"><input type="checkbox" id="dim" checked>
    <label for="dim" style="font-weight:500">fade filtered-out traces</label></span>
  <span class="chk"><input type="checkbox" id="split" checked>
    <label for="split" style="font-weight:500">split by zygosity</label></span>
</div>

<div id="tip"></div>

<div class="layout">
  <div class="card" id="plots"></div>
  <div>
    <div class="card">
      <h2>Embryos remaining</h2>
      <svg id="bars" width="270" height="180"></svg>
      <table id="tbl"></table>
      <div class="hint" id="tot"></div>
    </div>
  </div>
</div>

<script>
const DATA = __PAYLOAD__;
const W = 640, H = 210, M = {t:12, r:14, b:38, l:60};

function extent(arr){ let lo=Infinity, hi=-Infinity;
  for(const v of arr){ if(v<lo)lo=v; if(v>hi)hi=v; } return [lo,hi]; }

// Fixed domains so the axes don't jump around as traces are filtered.
const allT = [], allY0 = [], allY1 = [];
for(const e of DATA.embryos){
  for(const v of e.t) allT.push(v);
  for(const v of e.y0) allY0.push(v);
  for(const v of e.y1) allY1.push(v);
}
const DOM = { t: extent(allT), y: [extent(allY0), extent(allY1)] };

function mkScale(dom, lo, hi){
  const [a,b] = dom, span = (b-a)||1;
  return v => lo + (v-a)/span*(hi-lo);
}

const SVGNS = 'http://www.w3.org/2000/svg';
function el(tag, attrs){
  const n = document.createElementNS(SVGNS, tag);
  for(const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}

/* Draw one panel. `subset` is the list of [embryo, globalIndex] pairs it contains; the
   scales stay GLOBAL so panels are directly comparable to each other. */
function drawPanel(host, fi, subset, title, w){
  const x = mkScale(DOM.t, M.l, w-M.r);
  const y = mkScale(DOM.y[fi], H-M.b, M.t);

  const svg = el('svg', {viewBox:`0 0 ${w} ${H}`, width:w, height:H});

  if(title){
    const tt = el('text', {x:w/2, y:11, 'text-anchor':'middle', class:'albl'});
    tt.style.fontWeight = '700'; tt.textContent = title; svg.appendChild(tt);
  }

  svg.appendChild(el('path', {
    d:`M${M.l},${M.t} L${M.l},${H-M.b} L${w-M.r},${H-M.b}`,
    fill:'none', class:'axis', 'stroke-width':'1'
  }));

  for(let i=0;i<=4;i++){
    const tv = DOM.t[0] + (DOM.t[1]-DOM.t[0])*i/4;
    const tx = el('text', {x:x(tv), y:H-M.b+14, 'text-anchor':'middle', class:'tick'});
    tx.textContent = tv.toFixed(0); svg.appendChild(tx);

    const yv = DOM.y[fi][0] + (DOM.y[fi][1]-DOM.y[fi][0])*i/4;
    const ty = el('text', {x:M.l-6, y:y(yv)+3, 'text-anchor':'end', class:'tick'});
    ty.textContent = (DOM.y[fi][1] > 100) ? yv.toFixed(0) : yv.toFixed(2);
    svg.appendChild(ty);
  }
  const xl = el('text', {x:(M.l+w-M.r)/2, y:H-4, 'text-anchor':'middle', class:'albl'});
  xl.textContent = 'Hours post fertilization'; svg.appendChild(xl);

  const g = el('g', {class:'traces'}); svg.appendChild(g);
  // Hit targets live in their own layer ABOVE the visible traces.
  const hits = el('g', {class:'hitlayer'}); svg.appendChild(hits);
  for(const [e, ei] of subset){
    const ys = fi === 0 ? e.y0 : e.y1;
    let d = '';
    for(let i=0;i<e.t.length;i++) d += (i?'L':'M') + x(e.t[i]).toFixed(1) + ',' + y(ys[i]).toFixed(1);
    const p = el('path', {d, fill:'none', stroke:DATA.zygColor[e.zyg], 'stroke-width':'1.1'});
    p.dataset.ei = ei;
    // A wide transparent copy on top: gives the thin line a fat hit target so hover
    // actually catches it (a 1.1px stroke is nearly impossible to hit precisely).
    const hit = el('path', {d, fill:'none', stroke:'transparent', 'stroke-width':'10',
                            'pointer-events':'stroke'});
    hit.dataset.ei = ei; hit.classList.add('hit');
    g.appendChild(p);
    hits.appendChild(hit);
  }
  host.appendChild(svg);
}

function buildPlots(){
  const host = document.getElementById('plots');
  host.innerHTML = '';
  const split = document.getElementById('split').checked;
  const indexed = DATA.embryos.map((e,i)=>[e,i]);

  DATA.featureLabels.forEach((lab, fi) => {
    const h2 = document.createElement('h2'); h2.textContent = lab; host.appendChild(h2);
    if(!split){
      drawPanel(host, fi, indexed, null, W);
    } else {
      const row = document.createElement('div');
      row.className = 'facet-row';
      host.appendChild(row);
      const pw = Math.max(210, Math.floor(W / DATA.zygOrder.length) + 40);
      DATA.zygOrder.forEach(z => {
        drawPanel(row, fi, indexed.filter(([e]) => e.zyg === z), z, pw);
      });
    }
  });
}

function buildBars(){
  const svg = document.getElementById('bars');
  svg.innerHTML = '';
  const zs = DATA.zygOrder, BW = 250, BH = 26, GAP = 12;
  svg.setAttribute('height', zs.length*(BH+GAP)+10);
  zs.forEach((z,i)=>{
    const yy = i*(BH+GAP)+4;
    const bg = document.createElementNS('http://www.w3.org/2000/svg','rect');
    bg.setAttribute('x',0); bg.setAttribute('y',yy);
    bg.setAttribute('width',BW); bg.setAttribute('height',BH);
    bg.setAttribute('fill','currentColor'); bg.setAttribute('opacity','0.07');
    bg.setAttribute('rx','3'); svg.appendChild(bg);

    const bar = document.createElementNS('http://www.w3.org/2000/svg','rect');
    bar.setAttribute('x',0); bar.setAttribute('y',yy); bar.setAttribute('height',BH);
    bar.setAttribute('fill',DATA.zygColor[z]); bar.setAttribute('rx','3');
    bar.id = 'bar-'+z; svg.appendChild(bar);

    const lbl = document.createElementNS('http://www.w3.org/2000/svg','text');
    lbl.setAttribute('x',6); lbl.setAttribute('y',yy+BH/2+4);
    lbl.setAttribute('class','albl'); lbl.setAttribute('fill','#fff');
    lbl.style.fontWeight = '600'; lbl.id='lbl-'+z; svg.appendChild(lbl);
  });
}

const TOTALS = {};
for(const e of DATA.embryos) TOTALS[e.zyg] = (TOTALS[e.zyg]||0)+1;

function update(){
  const thr = parseFloat(document.getElementById('thr').value);
  const fade = document.getElementById('dim').checked;
  const ge = document.getElementById('dir').value === 'ge';
  document.getElementById('thrVal').textContent = thr.toFixed(2);

  const keep = DATA.embryos.map(e => ge ? (e.p >= thr) : (e.p <= thr));

  document.querySelectorAll('#plots path[data-ei]:not(.hit)').forEach(p=>{
    const on = keep[+p.dataset.ei];
    p.style.display = (!on && !fade) ? 'none' : '';
    p.setAttribute('opacity', on ? '0.75' : '0.06');
  });
  // Filtered-out traces must not be hoverable, or the tooltip reports invisible embryos.
  document.querySelectorAll('#plots path.hit').forEach(h=>{
    h.style.display = keep[+h.dataset.ei] ? '' : 'none';
  });

  const counts = {};
  DATA.embryos.forEach((e,i)=>{ if(keep[i]) counts[e.zyg]=(counts[e.zyg]||0)+1; });

  const BW = 250;
  let rows = '<tr><th>zygosity</th><th class="n">kept</th><th class="n">of</th><th class="n">%</th></tr>';
  DATA.zygOrder.forEach(z=>{
    const n = counts[z]||0, tot = TOTALS[z]||0;
    const frac = tot ? n/tot : 0;
    const bar = document.getElementById('bar-'+z);
    if(bar) bar.setAttribute('width', (BW*frac).toFixed(1));
    const lbl = document.getElementById('lbl-'+z);
    if(lbl) lbl.textContent = `${z}  ${n}/${tot}`;
    rows += `<tr><td><span class="swatch" style="background:${DATA.zygColor[z]}"></span>${z}</td>`
          + `<td class="n">${n}</td><td class="n">${tot}</td>`
          + `<td class="n">${(frac*100).toFixed(0)}%</td></tr>`;
  });
  document.getElementById('tbl').innerHTML = rows;

  const kept = keep.filter(Boolean).length;
  const op = ge ? '\\u2265' : '\\u2264';
  document.getElementById('tot').textContent =
    `${kept} of ${DATA.embryos.length} embryos with P(homo) ${op} ${thr.toFixed(2)}`;
}

/* Tooltip. Delegated on the container so it keeps working after buildPlots() re-renders.
   Hovering a hit-path also thickens its visible twin, so you can see WHICH trace you're on. */
const tip = document.getElementById('tip');
let hoveredEi = null;

function visiblePathsFor(ei){
  return document.querySelectorAll(`#plots path[data-ei="${ei}"]:not(.hit)`);
}
function setHover(ei){
  if(hoveredEi === ei) return;
  if(hoveredEi !== null)
    visiblePathsFor(hoveredEi).forEach(p => p.setAttribute('stroke-width','1.1'));
  hoveredEi = ei;
  if(ei !== null)
    visiblePathsFor(ei).forEach(p => p.setAttribute('stroke-width','3.2'));
}

document.getElementById('plots').addEventListener('mousemove', ev => {
  const hit = ev.target.closest('path.hit');
  if(!hit){ tip.style.display='none'; setHover(null); return; }
  const e = DATA.embryos[+hit.dataset.ei];
  setHover(+hit.dataset.ei);
  tip.innerHTML = `<b>${e.id}</b><br>`
    + `<span class="k">zygosity:</span> ${e.zyg}<br>`
    + `<span class="k">curated:</span> ${e.pheno || 'n/a'}<br>`
    + `<span class="k">P(homo):</span> ${e.p.toFixed(3)}`;
  tip.style.display = 'block';
  // Keep the tooltip inside the viewport.
  const pad = 14, r = tip.getBoundingClientRect();
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if(x + r.width  > innerWidth)  x = ev.clientX - r.width  - pad;
  if(y + r.height > innerHeight) y = ev.clientY - r.height - pad;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
});
document.getElementById('plots').addEventListener('mouseleave', () => {
  tip.style.display = 'none'; setHover(null);
});

buildPlots(); buildBars();
document.getElementById('thr').addEventListener('input', update);
document.getElementById('dim').addEventListener('change', update);
document.getElementById('split').addEventListener('change', ()=>{ buildPlots(); update(); });
// Flipping direction: reset the slider to the end that keeps everything, so the view never
// starts out empty.
document.getElementById('dir').addEventListener('change', (ev)=>{
  document.getElementById('thr').value = ev.target.value === 'ge' ? 0 : 1;
  update();
});
update();
</script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload()
    html = HTML.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"Saved: {OUT.relative_to(RUN_DIR)}  ({len(payload['embryos'])} embryos, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
