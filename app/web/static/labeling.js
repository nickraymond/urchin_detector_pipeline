// Labeling page: two grids of frames with their SAM masks, the preview (S3 bite 5) and
// "Labeled so far" (S4 bite 6c, newest first, refreshed while Run goes on). One class legend
// state for both grids and the large viewer (S2's viewer.js): click a class to show or hide its
// masks; "only" shows that class alone (again: all). The SAM job boxes update themselves
// (live.js).
(function () {
  "use strict";
  const listEl = document.getElementById("frame-list");
  if (!listEl) return;
  const {base, units, frames: previewFrames} = JSON.parse(listEl.textContent);
  const hidden = new Set();
  let map = false;
  const big = document.querySelector(".v-legend");
  const grids = [];
  let open = null;   // the grid whose frames the large viewer moves through

  function query(large, v) {
    const q = new URLSearchParams();  // one hide per unit: a class name may hold a comma
    if (map) q.append("map", "1");    // the class map shows every class
    else hidden.forEach(k => q.append("hide", k));
    if (large) q.append("size", "large");
    if (v) q.append("v", v);          // a new results file is a new picture (no stale cache)
    return q.toString() ? "?" + q.toString() : "";
  }
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  // counts: {unit key: masks or null (not run)} for one frame, or summed over a grid
  function render(box, counts, where) {
    box.replaceChildren();
    const m = el("label", "lg-map");
    const cb = el("input");
    cb.type = "checkbox"; cb.checked = map; cb.dataset.act = "map";
    m.append(cb, el("b", null, " final class map"));
    m.title = "One colour per pixel after the overlap rules";
    box.append(m);
    units.forEach(u => {
      const item = el("span", "lg-item" + (hidden.has(u.key) && !map ? " off" : ""));
      const b = el("button", "lg-class");
      b.type = "button"; b.dataset.act = "toggle"; b.dataset.key = u.key; b.disabled = map;
      b.title = (hidden.has(u.key) ? "Show " : "Hide ") + u.key + " masks";
      const n = counts[u.key];
      const sw = el("span", "swatch"); sw.style.background = u.colour;
      b.append(sw, " " + u.key + (u.untagged ? " (untagged)" : "") + ": " + (n == null ? "not run" : n) +
               (u.kind === "object" ? " ☐" : ""));
      const o = el("button", "lg-only", "only");
      o.type = "button"; o.dataset.act = "only"; o.dataset.key = u.key; o.disabled = map;
      o.title = "Show only " + u.key + " (click again: all classes)";
      item.append(b, o);
      box.append(item);
    });
    box.append(el("span", "lg-note", "☐ objects are boxed; areas (kelp, rock) are not" + where));
  }
  function total(frames) {
    const t = {};
    units.forEach(u => {
      const ns = frames.map(f => f.counts[u.key]).filter(n => n != null);
      t[u.key] = ns.length ? ns.reduce((a, b) => a + b, 0) : null;
    });
    return t;
  }
  function redraw() {
    grids.forEach(g => {
      g.frames.forEach(f => { f.src = base + encodeURIComponent(f.file) + query(true, f.v); });
      g.el.querySelectorAll("img.overlay").forEach(img => {
        img.src = img.dataset.src + query(false, img.dataset.v);
      });
      if (g.legend) render(g.legend, total(g.frames), " · counts: the " + g.frames.length +
                           " frames " + g.where);
    });
    const i = window.viewerIndex ? window.viewerIndex() : -1;
    if (open && i >= 0) { render(big, open.frames[i].counts, ""); window.viewerShow(i, true); }
  }
  function addGrid(g) {
    grids.push(g);
    g.el.addEventListener("click", e => {
      const a = e.target instanceof Element && e.target.closest("a[data-file]");
      if (!a || e.metaKey || e.ctrlKey) return;  // Cmd-click still opens a tab
      e.preventDefault();
      const i = g.frames.findIndex(f => f.file === a.dataset.file);
      if (i >= 0) { open = g; window.openViewer(g.frames, i); }
    });
  }
  document.querySelectorAll(".legend").forEach(box => box.addEventListener("click", e => {
    const t = e.target instanceof Element && e.target.closest("[data-act]");
    if (!t || t.dataset.act === "map") return;
    const k = t.dataset.key;
    if (t.dataset.act === "toggle") hidden.has(k) ? hidden.delete(k) : hidden.add(k);
    else if (hidden.size === units.length - 1 && !hidden.has(k)) hidden.clear();  // only → all again
    else { hidden.clear(); units.forEach(u => { if (u.key !== k) hidden.add(u.key); }); }
    redraw();
  }));
  document.querySelectorAll(".legend").forEach(box => box.addEventListener("change", e => {
    if (e.target.dataset && e.target.dataset.act === "map") { map = e.target.checked; redraw(); }
  }));
  document.addEventListener("viewer:show", e => {
    if (open) render(big, open.frames[e.detail.i].counts, "");
  });

  // The preview grid: tiles drawn by the page
  const preview = document.getElementById("preview-grid");
  if (preview) addGrid({el: preview, frames: previewFrames, where: "together",
                        legend: document.getElementById("legend")});

  // "Labeled so far": tiles drawn here, refreshed while results files are written
  const labeledEl = document.getElementById("labeled-list");
  const grid = document.getElementById("labeled-grid");
  if (labeledEl && grid) {
    const {poll, data} = JSON.parse(labeledEl.textContent);
    const g = {el: grid, frames: [], where: "shown together",
               legend: document.getElementById("labeled-legend"), sig: ""};
    const count = document.getElementById("labeled-count");
    const more = document.getElementById("labeled-more");
    function tile(f) {
      const fig = el("figure");
      const a = el("a");
      a.href = base + encodeURIComponent(f.file) + "?size=large";
      a.dataset.file = f.file;
      const img = el("img", "overlay");
      img.dataset.src = base + encodeURIComponent(f.file);
      img.dataset.v = f.v;
      img.loading = "lazy";
      img.alt = f.file;
      a.append(img);
      const cap = el("figcaption", null, f.file);
      cap.append(el("br"));
      units.forEach(u => {
        const n = f.counts[u.key];
        const c = el("span", "count");
        const sw = el("span", "swatch"); sw.style.background = u.colour;
        c.append(sw, " " + u.key + ": " + (n == null ? "not run" : n) +
                 (f.older.includes(u.key) ? " (older settings)" : "") + " ");
        cap.append(c);
      });
      fig.append(a, cap);
      return fig;
    }
    function show(d) {
      g.sig = d.sig;
      g.frames = d.frames.map(f => ({...f}));
      grid.replaceChildren(...g.frames.map(tile));
      count.textContent = d.total + " of " + d.kept + " kept frames labeled" +
        (d.total > d.shown ? "; the newest " + d.shown + " are shown" : "") + ".";
      more.hidden = d.total <= d.shown;
      redraw();
    }
    addGrid(g);
    show(data);
    async function tick() {
      // not while the large viewer shows this grid: new frames would shift its arrows
      const viewing = open === g && window.viewerIndex && window.viewerIndex() >= 0;
      if (!document.hidden && !viewing) {
        try {
          const d = await (await fetch(poll + "&since=" + encodeURIComponent(g.sig),
                                       {cache: "no-store"})).json();
          if (!d.same) show(d);
        } catch (e) { /* the app is restarting: try again */ }
      }
      setTimeout(tick, 3000);
    }
    setTimeout(tick, 3000);
  }
  redraw();
})();
