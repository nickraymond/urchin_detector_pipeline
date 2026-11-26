// Labeling page (S3): a class legend redraws the
// preview overlays and the large viewer (S2's viewer.js) without the hidden classes. The SAM
// job box updates itself (live.js, bite 6b).
(function () {
  "use strict";
  // Class legend (bite 5, SPEC Q16): one state for the preview tiles and the large viewer.
  // Click a class to show or hide its masks; "only" shows that class alone (again: all).
  const listEl = document.getElementById("frame-list");
  if (!listEl) return;
  const {base, units, frames} = JSON.parse(listEl.textContent);
  const hidden = new Set();
  let map = false;
  const legends = document.querySelectorAll(".legend");
  const big = document.querySelector(".v-legend");

  function query(large) {
    const q = new URLSearchParams();  // one hide per unit: a class name may hold a comma
    if (map) q.append("map", "1");    // the class map shows every class
    else hidden.forEach(k => q.append("hide", k));
    if (large) q.append("size", "large");
    return q.toString() ? "?" + q.toString() : "";
  }
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  // counts: {unit key: masks or null (not run)} for one frame, or summed over the preview
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
  function total() {
    const t = {};
    units.forEach(u => {
      const ns = frames.map(f => f.counts[u.key]).filter(n => n != null);
      t[u.key] = ns.length ? ns.reduce((a, b) => a + b, 0) : null;
    });
    return t;
  }
  function redraw() {
    frames.forEach(f => { f.src = base + encodeURIComponent(f.file) + query(true); });
    document.querySelectorAll("img.overlay").forEach(img => { img.src = img.dataset.src + query(false); });
    legends.forEach(box => {
      if (box !== big) render(box, total(), " · counts: the " + frames.length + " frames together");
    });
    const i = window.viewerIndex ? window.viewerIndex() : -1;
    if (i >= 0) { render(big, frames[i].counts, ""); window.viewerShow(i, true); }
  }
  legends.forEach(box => box.addEventListener("click", e => {
    const t = e.target instanceof Element && e.target.closest("[data-act]");
    if (!t || t.dataset.act === "map") return;
    const k = t.dataset.key;
    if (t.dataset.act === "toggle") hidden.has(k) ? hidden.delete(k) : hidden.add(k);
    else if (hidden.size === units.length - 1 && !hidden.has(k)) hidden.clear();  // only → all again
    else { hidden.clear(); units.forEach(u => { if (u.key !== k) hidden.add(u.key); }); }
    redraw();
  }));
  legends.forEach(box => box.addEventListener("change", e => {
    if (e.target.dataset && e.target.dataset.act === "map") { map = e.target.checked; redraw(); }
  }));
  document.addEventListener("viewer:show", e => render(big, frames[e.detail.i].counts, ""));
  const grid = document.getElementById("preview-grid");
  grid.addEventListener("click", e => {
    const a = e.target instanceof Element && e.target.closest("a[data-file]");
    if (!a || e.metaKey || e.ctrlKey) return;  // Cmd-click still opens a tab
    e.preventDefault();
    const i = frames.findIndex(f => f.file === a.dataset.file);
    if (i >= 0) window.openViewer(frames, i);
  });
  redraw();
})();
