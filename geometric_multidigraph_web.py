"""Interactive web UI for random geometric multidigraph generation.

Run from the repo root::

    python geometric_multidigraph_web.py

Then open http://127.0.0.1:8765/ in a browser.
"""

from __future__ import annotations

import base64
import json
import random
import tempfile
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import matplotlib

matplotlib.use("Agg")

from graph_export import graph_stats
from geometric_multidigraph_config import (
    GeometricMultidigraphConfig,
    POSITION_DISTRIBUTIONS,
)
from geometric_multidigraph_plot import plot_geometric_rgg
from random_geometric_graph import DEFAULT_PROTOCOLS, generate

_EXTRA_POSITION_DIST_PANELS = """
      <motion.div class="dist-panel" data-dist="disk">
        <p class="dist-intro">Uniform random points inside a disk centered at (cx, cy).</p>
        <motion.div class="param"><label class="param-label" for="disk_cx">cx</label><input type="range" id="disk_cx" min="0" max="1" step="0.05" value="0.5" /><input type="number" class="param-num" data-range="disk_cx" /><p class="param-hint">Disk center x.</p></div>
        <div class="param"><label class="param-label" for="disk_cy">cy</label><input type="range" id="disk_cy" min="0" max="1" step="0.05" value="0.5" /><input type="number" class="param-num" data-range="disk_cy" /><p class="param-hint">Disk center y.</p></motion.div>
        <div class="param"><label class="param-label" for="disk_radius">radius</label><input type="range" id="disk_radius" min="0.05" max="0.75" step="0.01" value="0.45" /><input type="number" class="param-num" data-range="disk_radius" /><p class="param-hint">Disk radius (placement region, not graph connection radius).</p></div>
      </div>

      <div class="dist-panel" data-dist="annulus">
        <p class="dist-intro">Uniform area on a ring: r_inner &le; r &le; r_outer around (cx, cy).</p>
        <div class="param"><label class="param-label" for="annulus_cx">cx</label><input type="range" id="annulus_cx" min="0" max="1" step="0.05" value="0.5" /><input type="number" class="param-num" data-range="annulus_cx" /></div>
        <div class="param"><label class="param-label" for="annulus_cy">cy</label><input type="range" id="annulus_cy" min="0" max="1" step="0.05" value="0.5" /><input type="number" class="param-num" data-range="annulus_cy" /></div>
        <div class="param"><label class="param-label" for="r_inner">r_inner</label><input type="range" id="r_inner" min="0" max="0.5" step="0.01" value="0.15" /><input type="number" class="param-num" data-range="r_inner" /></motion.div>
        <div class="param"><label class="param-label" for="r_outer">r_outer</label><input type="range" id="r_outer" min="0.05" max="0.75" step="0.01" value="0.5" /><input type="number" class="param-num" data-range="r_outer" /></div>
      </div>

      <div class="dist-panel" data-dist="polygon">
        <p class="dist-intro">Uniform area inside a regular polygon (circumradius).</p>
        <div class="param"><label class="param-label" for="poly_sides">sides</label><input type="range" id="poly_sides" min="3" max="12" step="1" value="6" /><input type="number" class="param-num" data-range="poly_sides" /></div>
        <div class="param"><label class="param-label" for="poly_cx">cx</label><input type="range" id="poly_cx" min="0" max="1" step="0.05" value="0.5" /><input type="number" class="param-num" data-range="poly_cx" /></div>
        <div class="param"><label class="param-label" for="poly_cy">cy</label><input type="range" id="poly_cy" min="0" max="1" step="0.05" value="0.5" /><input type="number" class="param-num" data-range="poly_cy" /></div>
        <div class="param"><label class="param-label" for="poly_radius">radius</label><input type="range" id="poly_radius" min="0.05" max="0.75" step="0.01" value="0.45" /><input type="number" class="param-num" data-range="poly_radius" /></div>
      </div>

      <div class="dist-panel" data-dist="arc">
        <p class="dist-intro">Points on a circular arc; jitter adds normal offset.</p>
        <div class="param"><label class="param-label" for="arc_cx">cx</label><input type="range" id="arc_cx" min="0" max="1" step="0.05" value="0.5" /><input type="number" class="param-num" data-range="arc_cx" /></div>
        <div class="param"><label class="param-label" for="arc_cy">cy</label><input type="range" id="arc_cy" min="0" max="1" step="0.05" value="0.5" /><input type="number" class="param-num" data-range="arc_cy" /></div>
        <motion.div class="param"><label class="param-label" for="arc_radius">radius</label><input type="range" id="arc_radius" min="0.05" max="0.75" step="0.01" value="0.45" /><input type="number" class="param-num" data-range="arc_radius" /></div>
        <motion.div class="param"><label class="param-label" for="angle_start_deg">angle_start (&deg;)</label><input type="range" id="angle_start_deg" min="-180" max="180" step="5" value="0" /><input type="number" class="param-num" data-range="angle_start_deg" /></div>
        <div class="param"><label class="param-label" for="angle_end_deg">angle_end (&deg;)</label><input type="range" id="angle_end_deg" min="-180" max="180" step="5" value="180" /><input type="number" class="param-num" data-range="angle_end_deg" /></div>
        <div class="param"><label class="param-label" for="arc_jitter">jitter</label><input type="range" id="arc_jitter" min="0" max="0.15" step="0.005" value="0.02" /><input type="number" class="param-num" data-range="arc_jitter" /></div>
      </div>
""".replace("<motion.div", "<div").replace("</motion.div>", "</div>")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Geometric multidigraph explorer</title>
  <style>
    :root {
      --bg: #1a1b1e;
      --panel: #25262b;
      --border: #3d3f46;
      --text: #e8e8ea;
      --muted: #9a9ca3;
      --accent: #5c7cfa;
      --accent-hover: #748ffc;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    header {
      padding: 1rem 1.25rem;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }
    header h1 { margin: 0; font-size: 1.25rem; font-weight: 600; }
    header p { margin: 0.35rem 0 0; color: var(--muted); font-size: 0.875rem; }
    .layout {
      display: grid;
      grid-template-columns: minmax(280px, 340px) 1fr;
      gap: 0;
      min-height: calc(100vh - 72px);
    }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
    }
    aside {
      padding: 1rem 1.25rem 2rem;
      border-right: 1px solid var(--border);
      background: var(--panel);
      overflow-y: auto;
      max-height: calc(100vh - 72px);
    }
    main {
      padding: 1rem 1.25rem 2rem;
      overflow-y: auto;
    }
    fieldset {
      border: 1px solid var(--border);
      border-radius: 8px;
      margin: 0 0 1rem;
      padding: 0.75rem 1rem 1rem;
    }
    legend {
      padding: 0 0.35rem;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }
    .fieldset-hint {
      margin: 0 0 0.75rem;
      font-size: 0.78rem;
      line-height: 1.4;
      color: var(--muted);
    }
    .param {
      display: grid;
      grid-template-columns: minmax(5.5rem, auto) 1fr 4.75rem;
      align-items: center;
      gap: 0.45rem 0.5rem;
      margin: 0.65rem 0;
      font-size: 0.875rem;
    }
    .param-hint {
      grid-column: 1 / -1;
      margin: 0;
      font-size: 0.72rem;
      line-height: 1.38;
      color: var(--muted);
    }
    .param-label { color: var(--text); }
    .dist-intro {
      margin: 0.5rem 0 0.35rem;
      font-size: 0.72rem;
      line-height: 1.38;
      color: var(--muted);
    }
    .param input[type="range"] { width: 100%; accent-color: var(--accent); }
    .param-num {
      width: 100%;
      padding: 0.3rem 0.35rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      font-size: 0.8rem;
      font-variant-numeric: tabular-nums;
      text-align: right;
    }
    .param-num:focus {
      outline: none;
      border-color: var(--accent);
    }
    select {
      width: 100%;
      margin-top: 0.35rem;
      padding: 0.4rem 0.5rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      font-size: 0.875rem;
    }
    .dist-panel { display: none; }
    .dist-panel.active { display: block; }
    .checkbox-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin: 0.55rem 0;
      font-size: 0.875rem;
    }
    button.primary {
      width: 100%;
      margin-top: 0.5rem;
      padding: 0.65rem 1rem;
      border: none;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
    }
    button.primary:hover { background: var(--accent-hover); }
    button.primary:disabled { opacity: 0.6; cursor: wait; }
    .status { font-size: 0.8rem; color: var(--muted); margin-top: 0.5rem; min-height: 1.2em; }
    .figure-wrap {
      background: #cfcfcf;
      border-radius: 8px;
      padding: 0.5rem;
      min-height: 200px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .figure-wrap img {
      max-width: 100%;
      height: auto;
      border-radius: 4px;
    }
    .figure-wrap .placeholder { color: #555; font-size: 0.9rem; }
    .stats {
      margin-top: 1rem;
      font-family: ui-monospace, "Cascadia Code", monospace;
      font-size: 0.75rem;
      line-height: 1.45;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1rem;
      white-space: pre-wrap;
      overflow-x: auto;
    }
  </style>
</head>
<body>
  <header>
    <h1>Random geometric multidigraph</h1>
    <p>Adjust parameters, then click <strong>Regenerate</strong> — same pipeline as <code>random_geometric_graph.py</code>.</p>
  </header>
  <div class="layout">
  <aside>
    <fieldset>
      <legend>Graph</legend>
      <p class="fieldset-hint">Undirected geometric graph on sampled positions; directed service edges follow from each node&rsquo;s protocol profile.</p>
      <div class="param">
        <label class="param-label" for="n">n (nodes)</label>
        <input type="range" id="n" min="5" max="80" step="1" value="30" />
        <input type="number" class="param-num" data-range="n" />
        <p class="param-hint">How many nodes (and matching protocol lists) to generate.</p>
      </div>
      <div class="param">
        <label class="param-label" for="position_seed">position_seed</label>
        <input type="range" id="position_seed" min="0" max="199" step="1" value="0" />
        <input type="number" class="param-num" data-range="position_seed" />
        <p class="param-hint">RNG for node coordinates only. Fix this and change CRP alphas to explore profiles on the same layout.</p>
      </div>
      <div class="param">
        <label class="param-label" for="profile_seed">profile_seed</label>
        <input type="range" id="profile_seed" min="0" max="199" step="1" value="0" />
        <input type="number" class="param-num" data-range="profile_seed" />
        <p class="param-hint">RNG for CRP profile sampling only; does not move nodes when position_seed is fixed.</p>
      </div>
      <div class="param">
        <label class="param-label" for="radius">radius</label>
        <input type="range" id="radius" min="0.05" max="1.5" step="0.01" value="0.25" />
        <input type="number" class="param-num" data-range="radius" />
        <p class="param-hint">Distance threshold in position space: larger values connect more node pairs in the geometric graph (more candidate service links).</p>
      </div>
    </fieldset>

    <fieldset>
      <legend>CRP profiles</legend>
      <p class="fieldset-hint">Nested Chinese Restaurant Process: each node gets a profile (list of protocols). Outer step clones an existing profile or invents a new one; inner step fills new profiles with tags. Node&nbsp;0 is seeded with protocol_1&ndash;3 (fixed).</p>
      <div class="param">
        <label class="param-label" for="alpha_outer">alpha_outer</label>
        <input type="range" id="alpha_outer" min="0.5" max="64" step="0.5" value="4" />
        <input type="number" class="param-num" data-range="alpha_outer" />
        <p class="param-hint">Outer concentration. Higher &rarr; more often invent a new profile type instead of copying an existing one (more diversity of service bundles across nodes).</p>
      </div>
      <div class="param">
        <label class="param-label" for="alpha_inner">alpha_inner</label>
        <input type="range" id="alpha_inner" min="0.5" max="64" step="0.5" value="8" />
        <input type="number" class="param-num" data-range="alpha_inner" />
        <p class="param-hint">Inner concentration when building a <em>new</em> profile. Higher &rarr; more fresh draws from the protocol catalog vs reusing tags on the tape. Ignored when a profile is cloned.</p>
      </div>
      <div class="param">
        <label class="param-label" for="mean_tags">mean_tags</label>
        <input type="range" id="mean_tags" min="0.5" max="10" step="0.5" value="4" />
        <input type="number" class="param-num" data-range="mean_tags" />
        <p class="param-hint">Poisson mean for how many tag draws run when inventing a new profile (typical bundle size). Not used when the outer step copies an existing profile.</p>
      </div>
    </fieldset>

    <fieldset>
      <legend>Node positions</legend>
      <p class="fieldset-hint">Where nodes sit on the unit square before <code>random_geometric_graph</code> wiring. Only the active distribution&rsquo;s sliders apply.</p>
      <label for="pos_distribution">distribution</label>
      <select id="pos_distribution">
        <option value="uniform" selected>uniform</option>
        <option value="gaussian">gaussian</option>
        <option value="beta">beta</option>
        <option value="grid">grid</option>
        <option value="disk">disk</option>
        <option value="annulus">annulus</option>
        <option value="polygon">polygon</option>
        <option value="arc">arc</option>
      </select>

      <div class="dist-panel active" data-dist="uniform">
        <p class="dist-intro">Uniform random points in the rectangle [x0, x1] &times; [y0, y1].</p>
        <div class="param"><label class="param-label" for="x0">x0</label><input type="range" id="x0" min="0" max="0.9" step="0.05" value="0" /><input type="number" class="param-num" data-range="x0" /><p class="param-hint">Left edge of the sampling box (x).</p></div>
        <div class="param"><label class="param-label" for="x1">x1</label><input type="range" id="x1" min="0.1" max="1" step="0.05" value="1" /><input type="number" class="param-num" data-range="x1" /><p class="param-hint">Right edge of the sampling box (x).</p></div>
        <div class="param"><label class="param-label" for="y0">y0</label><input type="range" id="y0" min="0" max="0.9" step="0.05" value="0" /><input type="number" class="param-num" data-range="y0" /><p class="param-hint">Bottom edge of the sampling box (y).</p></div>
        <div class="param"><label class="param-label" for="y1">y1</label><input type="range" id="y1" min="0.1" max="1" step="0.05" value="1" /><input type="number" class="param-num" data-range="y1" /><p class="param-hint">Top edge of the sampling box (y).</p></div>
      </div>

      <div class="dist-panel" data-dist="gaussian">
        <p class="dist-intro">Gaussian cluster around (mu_x, mu_y); coordinates are not clamped to [0, 1].</p>
        <div class="param"><label class="param-label" for="mu_x">mu_x</label><input type="range" id="mu_x" min="0" max="1" step="0.05" value="0.5" /><input type="number" class="param-num" data-range="mu_x" /><p class="param-hint">Mean x of the Gaussian cluster.</p></div>
        <div class="param"><label class="param-label" for="mu_y">mu_y</label><input type="range" id="mu_y" min="0" max="1" step="0.05" value="0.5" /><input type="number" class="param-num" data-range="mu_y" /><p class="param-hint">Mean y of the Gaussian cluster.</p></div>
        <div class="param"><label class="param-label" for="sigma">sigma</label><input type="range" id="sigma" min="0.02" max="0.45" step="0.01" value="0.15" /><input type="number" class="param-num" data-range="sigma" /><p class="param-hint">Standard deviation of each coordinate (spread of the cluster).</p></div>
      </div>

      <div class="dist-panel" data-dist="beta">
        <p class="dist-intro">Beta-distributed coordinates on [0, 1]&sup2;; &alpha;=&beta;&gt;1 pulls mass toward the center, &lt;1 toward corners.</p>
        <div class="param"><label class="param-label" for="beta_alpha">alpha</label><input type="range" id="beta_alpha" min="0.2" max="12" step="0.1" value="2" /><input type="number" class="param-num" data-range="beta_alpha" /><p class="param-hint">Beta distribution shape parameter &alpha; (both axes).</p></div>
        <div class="param"><label class="param-label" for="beta_shape">beta</label><input type="range" id="beta_shape" min="0.2" max="12" step="0.1" value="2" /><input type="number" class="param-num" data-range="beta_shape" /><p class="param-hint">Beta distribution shape parameter &beta; (both axes).</p></div>
      </div>

      <div class="dist-panel" data-dist="grid">
        <p class="dist-intro">Nodes on a lattice with random jitter; jitter=0 is a perfect grid.</p>
        <div class="param"><label class="param-label" for="jitter">jitter</label><input type="range" id="jitter" min="0" max="0.25" step="0.01" value="0.03" /><input type="number" class="param-num" data-range="jitter" /><p class="param-hint">Max random offset added to each grid cell center.</p></div>
        <div class="param"><label class="param-label" for="cols">cols</label><input type="range" id="cols" min="2" max="20" step="1" value="6" /><input type="number" class="param-num" data-range="cols" /><p class="param-hint">Number of columns in the grid (rows chosen to fit n).</p></div>
      </div>
""" + _EXTRA_POSITION_DIST_PANELS + """
    </fieldset>

    <fieldset>
      <legend>Plot</legend>
      <p class="fieldset-hint">Matplotlib rendering options for the preview figure (not used in graph generation).</p>
      <div class="param">
        <label class="param-label" for="sep">edge offset (sep)</label>
        <input type="range" id="sep" min="0.005" max="0.04" step="0.001" value="0.018" />
        <input type="number" class="param-num" data-range="sep" />
        <p class="param-hint">Parallel offset for directed edges so opposite directions do not overlap in the drawing.</p>
      </div>
      <div class="param">
        <label class="param-label" for="dpi">dpi</label>
        <input type="range" id="dpi" min="72" max="200" step="4" value="120" />
        <input type="number" class="param-num" data-range="dpi" />
        <p class="param-hint">Resolution of the PNG preview (dots per inch).</p>
      </div>
      <div class="checkbox-row">
        <input type="checkbox" id="per_protocol" checked />
        <label for="per_protocol" style="display:inline;margin:0">One panel per protocol</label>
      </div>
      <p class="fieldset-hint" style="margin-top:0.25rem">When checked, draw a separate subplot for each protocol that has at least one edge; otherwise one combined view.</p>
    </fieldset>

    <button type="button" class="primary" id="regenerate">Regenerate</button>
    <div class="status" id="status">Ready.</div>
  </aside>

  <main>
    <div class="figure-wrap" id="figure-wrap">
      <span class="placeholder">Click Regenerate to generate a graph.</span>
    </div>
    <pre class="stats" id="stats"></pre>
  </main>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    const statusEl = $("status");
    const statsEl = $("stats");
    const figureWrap = $("figure-wrap");
    let inflight = null;

    function initParamControls() {
      document.querySelectorAll('input[type="range"]').forEach((range) => {
        const num = document.querySelector('input.param-num[data-range="' + range.id + '"]');
        if (!num) return;
        num.min = range.min;
        num.max = range.max;
        num.step = range.step;
        num.value = range.value;

        const clamp = (v) => {
          const lo = parseFloat(range.min);
          const hi = parseFloat(range.max);
          return Math.min(hi, Math.max(lo, v));
        };

        range.addEventListener("input", () => {
          num.value = range.value;
        });

        const applyFromNum = () => {
          let v = parseFloat(num.value);
          if (Number.isNaN(v)) {
            num.value = range.value;
            return;
          }
          v = clamp(v);
          range.value = String(v);
          num.value = String(v);
        };

        num.addEventListener("change", applyFromNum);
        num.addEventListener("keydown", (e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            applyFromNum();
          }
        });
      });
    }

    function showDistPanels() {
      const dist = $("pos_distribution").value;
      document.querySelectorAll(".dist-panel").forEach((p) => {
        p.classList.toggle("active", p.dataset.dist === dist);
      });
    }

    function collectParams() {
      const dist = $("pos_distribution").value;
      const p = {
        n: +$("n").value,
        position_seed: +$("position_seed").value,
        profile_seed: +$("profile_seed").value,
        radius: +$("radius").value,
        alpha_outer: +$("alpha_outer").value,
        alpha_inner: +$("alpha_inner").value,
        mean_tags: +$("mean_tags").value,
        initial_profiles: null,
        pos_distribution: dist,
        per_protocol: $("per_protocol").checked,
        sep: +$("sep").value,
        dpi: +$("dpi").value,
      };
      if (dist === "uniform") {
        Object.assign(p, {
          x0: +$("x0").value, x1: +$("x1").value,
          y0: +$("y0").value, y1: +$("y1").value,
        });
      } else if (dist === "gaussian") {
        Object.assign(p, {
          mu_x: +$("mu_x").value, mu_y: +$("mu_y").value, sigma: +$("sigma").value,
        });
      } else if (dist === "beta") {
        Object.assign(p, { alpha: +$("beta_alpha").value, beta: +$("beta_shape").value });
      } else if (dist === "grid") {
        Object.assign(p, { jitter: +$("jitter").value, cols: +$("cols").value });
      } else if (dist === "disk") {
        Object.assign(p, {
          cx: +$("disk_cx").value, cy: +$("disk_cy").value,
          disk_radius: +$("disk_radius").value,
        });
      } else if (dist === "annulus") {
        Object.assign(p, {
          cx: +$("annulus_cx").value, cy: +$("annulus_cy").value,
          r_inner: +$("r_inner").value, r_outer: +$("r_outer").value,
        });
      } else if (dist === "polygon") {
        Object.assign(p, {
          sides: +$("poly_sides").value,
          cx: +$("poly_cx").value, cy: +$("poly_cy").value,
          poly_radius: +$("poly_radius").value,
        });
      } else if (dist === "arc") {
        Object.assign(p, {
          cx: +$("arc_cx").value, cy: +$("arc_cy").value,
          arc_radius: +$("arc_radius").value,
          angle_start_deg: +$("angle_start_deg").value,
          angle_end_deg: +$("angle_end_deg").value,
          jitter: +$("arc_jitter").value,
        });
      }
      return p;
    }

    async function regenerate() {
      if (inflight) inflight.abort();
      const ctrl = new AbortController();
      inflight = ctrl;
      $("regenerate").disabled = true;
      statusEl.textContent = "Generating…";

      try {
        const res = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(collectParams()),
          signal: ctrl.signal,
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.error || res.statusText);
        }
        const data = await res.json();
        figureWrap.innerHTML = '<img alt="Graph visualization" src="data:image/png;base64,' + data.png + '" />';
        statsEl.textContent = data.stats_text;
        statusEl.textContent = "Done — " + data.nodes + " nodes, " + data.edges + " edges.";
      } catch (e) {
        if (e.name === "AbortError") return;
        statusEl.textContent = "Error: " + e.message;
        figureWrap.innerHTML = '<span class="placeholder">' + e.message + '</span>';
      } finally {
        $("regenerate").disabled = false;
        if (inflight === ctrl) inflight = null;
      }
    }

    initParamControls();
    showDistPanels();

    $("pos_distribution").addEventListener("change", showDistPanels);
    $("regenerate").addEventListener("click", regenerate);
  </script>
</body>
</html>
"""


def _format_stats_bundle(stats: dict, *, heading: str) -> str:
    lines = [heading, "-" * len(heading)]
    lines.append("graph_stats")
    for key, val in stats["graph_stats"].items():
        lines.append(f"  {key:45} {val}")
    lines.append("node_stats_discrete")
    for metric, summary in stats["node_stats_discrete"].items():
        lines.append(f"  {metric}")
        if summary is None:
            lines.append("    (none)")
            continue
        for sk, sv in summary.items():
            if isinstance(sv, float):
                lines.append(f"    {sk:12} {sv:.6g}")
            else:
                lines.append(f"    {sk:12} {sv}")
    return "\n".join(lines)


def generate_from_params(body: dict[str, Any]) -> dict[str, Any]:
    n = int(body["n"])
    if n < 2 or n > 80:
        raise ValueError("n must be between 2 and 80")

    dist = body["pos_distribution"]
    if dist not in POSITION_DISTRIBUTIONS:
        raise ValueError(f"pos_distribution must be one of {POSITION_DISTRIBUTIONS}")

    per_protocol = bool(body["per_protocol"])
    sep = float(body["sep"])
    dpi = float(body["dpi"])

    config = GeometricMultidigraphConfig.from_mapping(
        {**body, "protocols": list(DEFAULT_PROTOCOLS)}
    )
    result = generate(config)
    G = result.graph
    catalog = result.catalog

    with tempfile.TemporaryDirectory() as tmp:
        png_path = Path(tmp) / "graph.png"
        plot_geometric_rgg(
            G,
            per_protocol=per_protocol,
            sep=sep,
            save_path=png_path,
            dpi=dpi,
            show=False,
            proto_order=catalog,
            config=config,
        )
        png_b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")

    stats = graph_stats(G)
    edge_counts = result.metadata["protocol_edge_counts"]
    profile_counts = result.metadata["protocol_profile_node_counts"]

    sections = [
        _format_stats_bundle(stats, heading="Full multigraph (all protocols)"),
        "",
        "protocol_edge_counts:",
        *[f"  {p:20} {edge_counts.get(p, 0)}" for p in catalog],
        "",
        "protocol_profile_node_counts:",
        *[f"  {p:20} {profile_counts.get(p, 0)}" for p in catalog],
    ]

    return {
        "png": png_b64,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "stats_text": "\n".join(sections),
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/generate":
            self.send_error(404)
            return
        try:
            payload = self._read_json_body()
            result = generate_from_params(payload)
            self._send_json(200, result)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})


def main() -> None:
    host, port = "127.0.0.1", 8765
    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    print(f"Serving {url}  (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
