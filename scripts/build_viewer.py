"""Build a self-contained HTML player for a run: the video with tracked boxes overlaid, the
entity graph as it changes, and the reducer's state intervals on a seekable timeline.

Usage: python scripts/build_viewer.py <video_path> [out.html] [--schema NAME]

graph.report() says how much was extracted and scripts/dump_graph.py says what it claims; this
says whether the claims match the footage they came from, which is the only way to judge stage 3.
Reads the stage caches written by a previous fit() — it never calls the detector or the VLM.
"""

import base64
import html
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from core.cache import artifacts
from core.cache.keys import file_hash, stage_key
from core.detect.base import TrackedBox
from core.graph.kuzu_store import KuzuStore
from core.pipeline import PipelineConfig, _confirmed_tracks
from core.preprocessing.filter import DualPhaseFilter
from core.schema.builtin.meva import MEVA
from core.schema.builtin.warehouse import WAREHOUSE

SCHEMAS = {"warehouse": WAREHOUSE, "meva": MEVA}


def _keys(video_path: Path, config: PipelineConfig) -> tuple[str, str, str]:
    """Mirror VideoPipeline.fit's key chain so the viewer reads exactly what the run wrote."""
    video_hash = file_hash(video_path)
    filter_key = stage_key(video_hash, "filter", asdict(config.filter))
    detect_cfg = {
        "backend": config.detect_backend,
        **asdict(config.detect_config),
        "entity_classes": list(config.schema.entity_classes),
    }
    detect_key = stage_key(video_hash, "detect", detect_cfg, filter_key)
    extract_cfg = {
        **asdict(config.extract),
        "relations": sorted(config.schema.instant_relation_names),
    }
    extract_key = stage_key(video_hash, "extract", extract_cfg, detect_key)
    return filter_key, detect_key, extract_key


def _states(graph_dir: Path) -> list[dict]:
    store = KuzuStore(graph_dir / "graph.kuzu")
    return store.execute(
        "MATCH (s:Entity)-[:SUBJECT_OF]->(st:State)-[:OBJECT_OF]->(o:Entity) "
        "RETURN s.id AS subj, st.type AS type, o.id AS obj, st.valid_from AS start, "
        "st.valid_until AS stop, st.flags AS flags ORDER BY st.valid_from",  # `end` is reserved
        {},
    )


def build(video_path: Path, out_path: Path, schema=WAREHOUSE) -> None:
    config = PipelineConfig(schema=schema)
    _, detect_key, extract_key = _keys(video_path, config)
    cache_dir = config.cache_dir

    if not artifacts.exists(cache_dir, detect_key):
        sys.exit(f"no cached detections for {video_path} — run scripts/run_demo.py first")

    tracked = _confirmed_tracks(
        [TrackedBox(**row) for row in artifacts.load(cache_dir, detect_key)], config.min_track_hits
    )
    extracted = artifacts.load(cache_dir, extract_key)
    graph_dir = cache_dir / "graphs" / extract_key
    states = _states(graph_dir) if (graph_dir / "graph.kuzu").exists() else []

    # bbox coordinates are in source-frame pixels; the player needs that size to scale the overlay
    keyframes = DualPhaseFilter(config.filter).select(video_path)
    width, height = keyframes.frames[0].image.size
    duration = max((kf.timestamp for kf in keyframes.frames), default=1.0)

    by_id: dict[str, dict] = {}
    for t in sorted(tracked, key=lambda b: b.timestamp):
        track = by_id.setdefault(t.local_id, {"cls": t.class_label, "pts": []})
        track["pts"].append([round(t.timestamp, 3), *t.bbox])

    payload = {
        "meta": {"w": width, "h": height, "duration": duration, "stride": _stride(keyframes)},
        "tracks": by_id,
        "events": [
            [round(e["timestamp"], 3), e["source_id"], e["relation"], e["target_id"]]
            for e in sorted(extracted["events"], key=lambda e: e["timestamp"])
        ],
        "states": [
            [s["subj"], s["type"], s["obj"], s["start"], s["stop"], s["flags"] or ""]
            for s in states
        ],
    }

    video_b64 = base64.b64encode(video_path.read_bytes()).decode()
    flagged = sum(1 for s in states if s["flags"])
    summary = (
        f"{len(keyframes.frames)} keyframes &middot; {len(by_id)} tracks &middot; "
        f"{len(payload['events'])} events &middot; {len(states)} states ({flagged} flagged)"
    )

    out_path.write_text(
        _TEMPLATE.replace("{{TITLE}}", html.escape(video_path.name))
        .replace("{{SUMMARY}}", summary)
        .replace("{{VIDEO}}", video_b64)
        .replace("{{DATA}}", json.dumps(payload))
    )
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def _stride(keyframes) -> float:
    """Median gap between keyframes — the overlay stops drawing a track beyond one of these."""
    ts = [kf.timestamp for kf in keyframes.frames]
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]))
    return gaps[len(gaps) // 2] if gaps else 1.0


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
:root{--bg:#fbfaf9;--fg:#1a1a18;--dim:#6b6b66;--line:#e2e0dc;--card:#fff;
--bar:#4a6fa5;--flag:#c2684a;--accent:#2f6b4f;--head:#1a1a18}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#16161a;--fg:#e8e6e3;--dim:#8a8a84;--line:#2c2c32;--card:#1d1d22;
--bar:#6b94c9;--flag:#d4805e;--accent:#5aa37c;--head:#e8e6e3}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 ui-sans-serif,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:20px;margin:0 0 3px}
.summary{color:var(--dim);font-size:13px;margin:0 0 20px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px;margin-bottom:16px}
.panel h3{font-size:12px;text-transform:uppercase;letter-spacing:.06em;
color:var(--dim);margin:0 0 12px;display:flex;justify-content:space-between}
.clock{font-family:ui-monospace,monospace;color:var(--fg);letter-spacing:0}
#tl{position:relative;cursor:pointer}
.trow{display:flex;align-items:center;gap:10px;margin-bottom:4px}
.tlabel{width:230px;flex:none;font-size:11px;font-family:ui-monospace,monospace;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--dim)}
.ttrack{position:relative;flex:1;height:13px;background:var(--bg);
border:1px solid var(--line);border-radius:3px}
.tbar{position:absolute;top:0;bottom:0;background:var(--bar);border-radius:2px;
min-width:3px;opacity:.5;transition:opacity .1s}
.tbar.flagged{background:var(--flag)}
.tbar.live{opacity:1}
#head{position:absolute;top:0;bottom:0;width:2px;background:var(--flag);
pointer-events:none;z-index:2}
.cols{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(0,1fr);gap:16px;
align-items:start}
.stage{position:relative;line-height:0}
video{width:100%;height:auto;border-radius:6px;display:block;background:#000}
#ov{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#g{width:100%;height:auto;display:block}
.live{font-size:12px;font-family:ui-monospace,monospace}
.live li{list-style:none;padding:4px 0;border-bottom:1px solid var(--line);
display:flex;gap:7px;flex-wrap:wrap}
.live li:last-child{border-bottom:none}
.live ul{margin:0;padding:0}
.rel{color:var(--accent);font-weight:600}
.fire{color:var(--flag);font-weight:600}
.none{color:var(--dim);font-style:italic;font-family:inherit;font-size:12px}
.sub{font-size:11px;color:var(--dim);margin:12px 0 5px;text-transform:uppercase;
letter-spacing:.05em}
#sg{width:100%;height:auto;display:block}
.legend{font-size:11px;color:var(--dim);margin:10px 0 0;display:flex;
align-items:center;gap:4px;flex-wrap:wrap}
.k{display:inline-block;width:18px;height:3px;border-radius:2px;vertical-align:middle}
.edge{cursor:pointer;transition:stroke-opacity .12s}
#sg{cursor:grab;touch-action:none;user-select:none}
#sg.grabbing{cursor:grabbing}
#sg .node{cursor:grab}
#sg .faded{opacity:.07}
#sg text{pointer-events:none}
@media(max-width:900px){.cols{grid-template-columns:1fr}.tlabel{width:140px}}
</style></head><body><div class="wrap">
<h1>{{TITLE}}</h1>
<p class="summary">{{SUMMARY}}</p>

<div class="panel"><h3>State intervals <span class="clock" id="clock">0.00s</span></h3>
<div id="tl"><div id="head"></div></div></div>

<div class="cols">
  <div class="panel"><h3>Tracked detections</h3>
    <div class="stage"><video id="v" controls playsinline>
      <source src="data:video/mp4;base64,{{VIDEO}}" type="video/mp4"></video>
      <canvas id="ov"></canvas></div>
  </div>
  <div class="panel"><h3>Entity graph <span class="clock">now</span></h3>
    <svg id="g" viewBox="0 0 400 400"></svg>
    <p class="sub">On screen</p><div class="live" id="now"></div>
    <p class="sub">Events</p><div class="live" id="evs"></div>
  </div>
</div>

<div class="panel"><h3>All relations across the clip
  <span class="clock" id="scount"></span></h3>
  <svg id="sg" viewBox="0 0 900 520"></svg>
  <p class="legend"><span class="k" style="background:var(--accent)"></span>event
  &nbsp;<span class="k" style="background:var(--bar)"></span>state
  &nbsp;<span class="k" style="background:var(--flag)"></span>flagged state only
  &nbsp;&mdash; thickness = times observed &middot; drag to rearrange, scroll to zoom,
  hover to isolate a node</p>
</div>
</div>
<script>
const D = {{DATA}};
const v = document.getElementById('v'), ov = document.getElementById('ov');
const ctx = ov.getContext('2d'), g = document.getElementById('g');
const tl = document.getElementById('tl'), head = document.getElementById('head');
const clock = document.getElementById('clock');
const MAXGAP = D.meta.stride * 1.6;   // stop drawing a track across a longer absence
const FLASH = 0.5;                    // seconds an event stays lit
// The last keyframe is not the end of the video; rescale once the real duration is known,
// or the playhead runs past bars that were laid out against the keyframe span.
let DUR = D.meta.duration;

const hue = id => { let h = 0; for (const c of id) h = (h * 31 + c.charCodeAt(0)) % 360; return h; };
const col = (id, a = 1) => `hsla(${hue(id)},70%,55%,${a})`;

// ---- timeline -------------------------------------------------------------
const bars = D.states.map(([subj, type, obj, start, stop, flags]) => {
  const row = document.createElement('div'); row.className = 'trow';
  const lab = document.createElement('div'); lab.className = 'tlabel';
  lab.textContent = `${subj} ${type} ${obj}`;
  const track = document.createElement('div'); track.className = 'ttrack';
  const bar = document.createElement('div');
  bar.className = 'tbar' + (flags ? ' flagged' : '');
  bar.title = `${subj} ${type} ${obj}  ${start.toFixed(2)}..` +
              (stop === null ? 'open' : stop.toFixed(2)) + (flags ? `  [${flags}]` : '');
  track.appendChild(bar); row.append(lab, track); tl.appendChild(row);
  return { bar, start, stop };
});
function layoutBars() {
  for (const b of bars) {
    const end = b.stop === null ? DUR : b.stop;
    b.bar.style.left = (100 * b.start / DUR) + '%';
    b.bar.style.width = Math.max(100 * (end - b.start) / DUR, 0.7) + '%';
  }
}
layoutBars();
tl.addEventListener('click', e => {
  const t = tl.querySelector('.ttrack'); if (!t) return;
  const r = t.getBoundingClientRect();
  v.currentTime = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * DUR;
});

// ---- graph layout: one ring, grouped by class ------------------------------
const ids = Object.keys(D.tracks).sort((a, b) =>
  (D.tracks[a].cls + a).localeCompare(D.tracks[b].cls + b));
const pos = {};
ids.forEach((id, i) => {
  const a = (i / ids.length) * 2 * Math.PI - Math.PI / 2;
  pos[id] = { x: 200 + 150 * Math.cos(a), y: 200 + 150 * Math.sin(a) };
});
const svgns = 'http://www.w3.org/2000/svg';
const mk = (n, at) => { const e = document.createElementNS(svgns, n);
  for (const k in at) e.setAttribute(k, at[k]); return e; };

// ---- static graph: every relation the run ever observed ---------------------
// The ring above answers "what is true now"; this answers "what was ever claimed", which is
// how you spot a pair the pipeline related once by accident versus consistently.
const agg = new Map();
const addEdge = (a, rel, b, kind, t0) => {
  if (!(a in D.tracks) || !(b in D.tracks)) return;
  const k = a + '|' + rel + '|' + b;
  const e = agg.get(k) || { a, b, rel, kind, n: 0, t0: Infinity };
  e.n++; e.t0 = Math.min(e.t0, t0);
  if (kind === 'event') e.kind = 'event';            // an event outranks a state label
  else if (e.kind !== 'event' && kind === 'state') e.kind = 'state';
  agg.set(k, e);
};
for (const [t, a, rel, b] of D.events) addEdge(a, rel, b, 'event', t);
for (const [a, type, b, st, sp, flags] of D.states)
  addEdge(a, type, b, flags ? 'flagged' : 'state', st);
const edges = [...agg.values()];

const sg = document.getElementById('sg');
const SW = 900, SH = 520;
const root = mk('g', {}); sg.appendChild(root);
const eLayer = mk('g', {}), nLayer = mk('g', {});
root.append(eLayer, nLayer);

// Obsidian-style force layout: repulsion between every pair, springs along observed relations,
// mild gravity. 15-30 nodes, so the naive O(n^2) pass is cheaper than any spatial index.
const deg = {}; for (const id of ids) deg[id] = 0;
for (const e of edges) { deg[e.a]++; deg[e.b]++; }
const N = {};
ids.forEach((id, i) => {
  const a = (i / ids.length) * 2 * Math.PI;
  N[id] = { x: SW / 2 + 140 * Math.cos(a) + Math.random(), y: SH / 2 + 140 * Math.sin(a),
            vx: 0, vy: 0, r: 5 + Math.min(Math.sqrt(deg[id]) * 2.6, 11) };
});
const nbr = {}; for (const id of ids) nbr[id] = new Set([id]);
for (const e of edges) { nbr[e.a].add(e.b); nbr[e.b].add(e.a); }

const EDGE_COL = { event: 'var(--accent)', state: 'var(--bar)', flagged: 'var(--flag)' };
const drawn = edges.map(e => {
  const path = mk('path', { fill: 'none', stroke: EDGE_COL[e.kind], class: 'edge',
    'stroke-width': Math.min(1 + Math.log2(e.n + 1) * 1.3, 6), 'stroke-linecap': 'round' });
  path.appendChild(Object.assign(mk('title', {}),
    { textContent: `${e.a} ${e.rel} ${e.b}  \u00d7${e.n}  first ${e.t0.toFixed(2)}s` }));
  eLayer.appendChild(path);
  return { el: path, e };
});
const nodeEls = ids.map(id => {
  const g2 = mk('g', { class: 'node' });
  const c = mk('circle', { r: N[id].r, fill: col(id), stroke: 'var(--card)', 'stroke-width': 1.5 });
  const t = Object.assign(mk('text', { 'font-size': 10, 'text-anchor': 'middle',
    'font-family': 'ui-monospace, monospace', fill: 'var(--fg)' }), { textContent: id });
  g2.append(c, t);
  g2.appendChild(Object.assign(mk('title', {}),
    { textContent: `${id} \u00b7 ${D.tracks[id].cls} \u00b7 ${deg[id]} relations` }));
  nLayer.appendChild(g2);
  return { id, g: g2, t };
});

let heat = 1;
function tick() {
  const K = 118;
  for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) {
    const a = N[ids[i]], b = N[ids[j]];
    let dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 0.01;
    const f = (K * K) / (d * d) * 0.9;
    dx /= d; dy /= d; a.vx -= dx * f; a.vy -= dy * f; b.vx += dx * f; b.vy += dy * f;
  }
  for (const e of edges) {
    const a = N[e.a], b = N[e.b];
    let dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 0.01;
    const f = (d * d) / K * 0.0016 * Math.min(e.n, 4);
    dx /= d; dy /= d; a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
  }
  for (const id of ids) {
    const n = N[id];
    if (n.fixed) { n.vx = n.vy = 0; continue; }
    n.vx += (SW / 2 - n.x) * 0.004; n.vy += (SH / 2 - n.y) * 0.004;
    n.vx *= 0.82; n.vy *= 0.82;
    n.x += Math.max(-12, Math.min(12, n.vx)) * heat;
    n.y += Math.max(-12, Math.min(12, n.vy)) * heat;
    n.x = Math.max(24, Math.min(SW - 24, n.x)); n.y = Math.max(20, Math.min(SH - 20, n.y));
  }
  heat *= 0.985;
  paintPositions();
}
function paintPositions() {
  for (const { el, e } of drawn) {
    const a = N[e.a], b = N[e.b];
    el.setAttribute('d', `M${a.x.toFixed(1)},${a.y.toFixed(1)} L${b.x.toFixed(1)},${b.y.toFixed(1)}`);
  }
  for (const { id, g: g2, t } of nodeEls) {
    g2.setAttribute('transform', `translate(${N[id].x.toFixed(1)},${N[id].y.toFixed(1)})`);
    t.setAttribute('y', -N[id].r - 4);
  }
}
let simRaf;
const settle = () => { tick(); if (heat > 0.02) simRaf = requestAnimationFrame(settle); };
const reheat = () => { heat = Math.max(heat, 0.45);
  cancelAnimationFrame(simRaf); simRaf = requestAnimationFrame(settle); };
settle();

// hover isolates a node and its neighbours; drag moves it; wheel zooms; background pans
let view = { k: 1, x: 0, y: 0 };
const applyView = () =>
  root.setAttribute('transform', `translate(${view.x},${view.y}) scale(${view.k})`);
nodeEls.forEach(({ id, g: g2 }) => {
  g2.addEventListener('mouseenter', () => {
    for (const n of nodeEls) n.g.classList.toggle('faded', !nbr[id].has(n.id));
    for (const { el, e } of drawn)
      el.classList.toggle('faded', e.a !== id && e.b !== id);
  });
  g2.addEventListener('mouseleave', () => {
    for (const n of nodeEls) n.g.classList.remove('faded');
    for (const { el } of drawn) el.classList.remove('faded');
  });
  g2.addEventListener('pointerdown', ev => {
    ev.stopPropagation(); g2.setPointerCapture(ev.pointerId);
    const move = m => {
      const r = sg.getBoundingClientRect();
      N[id].fixed = true;
      N[id].x = ((m.clientX - r.left) / r.width * SW - view.x) / view.k;
      N[id].y = ((m.clientY - r.top) / r.height * SH - view.y) / view.k;
      paintPositions();
    };
    const up = () => { N[id].fixed = false; reheat();
      g2.removeEventListener('pointermove', move); g2.removeEventListener('pointerup', up); };
    g2.addEventListener('pointermove', move); g2.addEventListener('pointerup', up);
  });
});
sg.addEventListener('wheel', ev => {
  ev.preventDefault();
  const k = Math.max(0.4, Math.min(4, view.k * (ev.deltaY < 0 ? 1.12 : 0.89)));
  view.x = SW / 2 - (SW / 2 - view.x) * (k / view.k);
  view.y = SH / 2 - (SH / 2 - view.y) * (k / view.k);
  view.k = k; applyView();
}, { passive: false });
sg.addEventListener('pointerdown', ev => {
  sg.classList.add('grabbing');
  const x0 = ev.clientX, y0 = ev.clientY, vx = view.x, vy = view.y;
  const r = sg.getBoundingClientRect();
  const move = m => { view.x = vx + (m.clientX - x0) / r.width * SW;
    view.y = vy + (m.clientY - y0) / r.height * SH; applyView(); };
  const up = () => { sg.classList.remove('grabbing');
    window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
  window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
});

document.getElementById('scount').textContent =
  `${edges.length} distinct \u00b7 ${ids.length} nodes`;

function paintStatic(t, liveKeys) {
  for (const { el, e } of drawn) {
    const live = liveKeys.has(e.a + '|' + e.rel + '|' + e.b);
    el.setAttribute('stroke-opacity', live ? 0.95 : t >= e.t0 ? 0.4 : 0.1);
  }
}

// ---- per-frame state -------------------------------------------------------
function boxAt(track, t) {
  const p = track.pts;
  if (!p.length || t < p[0][0] - MAXGAP || t > p[p.length - 1][0] + MAXGAP) return null;
  let i = 0; while (i < p.length - 1 && p[i + 1][0] <= t) i++;
  const a = p[i], b = p[Math.min(i + 1, p.length - 1)];
  if (b[0] - a[0] > MAXGAP && t > a[0]) return null;
  if (Math.abs(t - a[0]) > MAXGAP && b === a) return null;
  const f = b[0] > a[0] ? Math.max(0, Math.min(1, (t - a[0]) / (b[0] - a[0]))) : 0;
  return [1, 2, 3, 4].map(k => a[k] + (b[k] - a[k]) * f);
}

function draw() {
  const t = v.currentTime;
  clock.textContent = t.toFixed(2) + 's';
  head.style.left = (100 * t / DUR) + '%';
  for (const b of bars) b.bar.classList.toggle(
    'live', t >= b.start && t <= (b.stop === null ? DUR : b.stop));

  // overlay
  const r = v.getBoundingClientRect();
  if (ov.width !== Math.round(r.width) || ov.height !== Math.round(r.height)) {
    ov.width = Math.round(r.width); ov.height = Math.round(r.height);
  }
  const sx = ov.width / D.meta.w, sy = ov.height / D.meta.h;
  ctx.clearRect(0, 0, ov.width, ov.height);
  ctx.lineWidth = 2; ctx.font = '600 12px ui-monospace, monospace';
  const visible = [];
  for (const id of ids) {
    const bb = boxAt(D.tracks[id], t); if (!bb) continue;
    visible.push(id);
    const [x0, y0, x1, y1] = [bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy];
    ctx.strokeStyle = col(id); ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    const w = ctx.measureText(id).width + 8;
    ctx.fillStyle = col(id); ctx.fillRect(x0, Math.max(0, y0 - 15), w, 15);
    ctx.fillStyle = '#fff'; ctx.fillText(id, x0 + 4, Math.max(11, y0 - 4));
  }

  // graph: nodes on screen now, edges = states holding now, flash = event firing
  const vis = new Set(visible);
  const active = D.states.filter(([s, , o, st, sp]) =>
    t >= st && t <= (sp === null ? DUR : sp) + 0.001);
  const firing = D.events.filter(e => t >= e[0] && t < e[0] + FLASH);
  g.textContent = '';
  for (const [s, type, o] of active) {
    if (!pos[s] || !pos[o]) continue;
    g.appendChild(mk('line', { x1: pos[s].x, y1: pos[s].y, x2: pos[o].x, y2: pos[o].y,
      stroke: 'var(--bar)', 'stroke-width': 1.5, 'stroke-opacity': .55 }));
  }
  for (const [, s, rel, o] of firing) {
    if (!pos[s] || !pos[o]) continue;
    g.appendChild(mk('line', { x1: pos[s].x, y1: pos[s].y, x2: pos[o].x, y2: pos[o].y,
      stroke: 'var(--flag)', 'stroke-width': 3 }));
    g.appendChild(Object.assign(mk('text', {
      x: (pos[s].x + pos[o].x) / 2, y: (pos[s].y + pos[o].y) / 2 - 4,
      fill: 'var(--flag)', 'font-size': 10, 'font-family': 'ui-monospace, monospace',
      'text-anchor': 'middle' }), { textContent: rel }));
  }
  for (const id of ids) {
    const on = vis.has(id), p = pos[id];
    g.appendChild(mk('circle', { cx: p.x, cy: p.y, r: on ? 7 : 3.5,
      fill: on ? col(id) : 'var(--line)' }));
    g.appendChild(Object.assign(mk('text', {
      x: p.x, y: p.y - 11, fill: on ? 'var(--fg)' : 'var(--dim)', 'font-size': 9,
      'font-family': 'ui-monospace, monospace', 'text-anchor': 'middle' }),
      { textContent: id }));
  }

  const liveKeys = new Set();
  for (const [a, type, b] of active) liveKeys.add(a + '|' + type + '|' + b);
  for (const [, a, rel, b] of firing) liveKeys.add(a + '|' + rel + '|' + b);
  paintStatic(t, liveKeys);

  document.getElementById('now').innerHTML = visible.length
    ? '<ul>' + visible.map(id =>
        `<li><span style="color:${col(id)}">&#9632;</span> ${id}` +
        `<span style="color:var(--dim)">${D.tracks[id].cls}</span></li>`).join('') + '</ul>'
    : '<p class="none">nothing tracked</p>';
  document.getElementById('evs').innerHTML = firing.length
    ? '<ul>' + firing.map(([ts, s, rel, o]) =>
        `<li class="fire">${s} <span class="rel">${rel}</span> ${o}` +
        `<span style="color:var(--dim)">${ts.toFixed(2)}s</span></li>`).join('') + '</ul>'
    : '<p class="none">no event firing</p>';
}

// drive on rAF while playing; still redraw on seek/resize when paused
let raf;
const loop = () => { draw(); raf = requestAnimationFrame(loop); };
v.addEventListener('play', () => { cancelAnimationFrame(raf); loop(); });
v.addEventListener('pause', () => { cancelAnimationFrame(raf); draw(); });
v.addEventListener('loadedmetadata', () => {
  if (isFinite(v.duration) && v.duration > DUR) { DUR = v.duration; layoutBars(); }
  draw();
});
['seeked', 'loadeddata', 'timeupdate'].forEach(e => v.addEventListener(e, draw));
window.addEventListener('resize', draw);
draw();
</script></body></html>
"""


if __name__ == "__main__":
    argv = sys.argv[1:]
    schema = WAREHOUSE
    if "--schema" in argv:
        i = argv.index("--schema")
        schema = SCHEMAS[argv[i + 1]]
        del argv[i : i + 2]
    video = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 else Path("run_view.html")
    build(video, out, schema)
