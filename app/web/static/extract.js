// Extract frames page (bite 3a): live count while typing, progress while extracting.
(function () {
  "use strict";
  const form = document.getElementById("extract-form");
  if (!form) return;
  const live = document.getElementById("live");
  let timer = null;
  form.querySelectorAll("input").forEach(inp => inp.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const q = new URLSearchParams({every: form.every.value, cap: form.cap.value});
      try {
        const r = await fetch(form.dataset.plan + "?" + q);
        const a = await r.json();
        live.textContent = r.ok ? a.live + " frames" + (a.live > a.planned ? " → " + a.planned +
          " after the cap, spread evenly across all videos" : "") : a.error;
      } catch (e) { /* app restarting */ }
    }, 300);
  }));

  const box = document.getElementById("extract-progress");
  if (box.hidden) return;
  const poll = setInterval(async () => {
    let a;
    try { a = await (await fetch(box.dataset.status)).json(); } catch (e) { return; }
    const j = a.job || {};
    box.querySelector("progress").value = j.percent || 0;
    const eta = j.eta_s != null ? ", about " + Math.ceil(j.eta_s / 60) + " min left" : "";
    box.querySelector(".hint").textContent = (j.message || j.state || "waiting") + eta;
    if (j.state === "done" || j.state === "error") { clearInterval(poll); location.reload(); }
  }, 2000);
})();

// Frame viewer (Nick 2026-11-25): click a frame to see it full screen on this page. Arrows or
// the < > buttons move through every kept frame; Esc or Close leaves. On the image: click zooms
// to actual pixels at that point (click again to fit), scroll or pinch zooms, drag pans,
// + / - / 0 zoom in, out and back to fit.
(function () {
  "use strict";
  const listEl = document.getElementById("frame-list");
  const viewer = document.getElementById("viewer");
  if (!listEl || !viewer) return;
  const {base, frames: kept} = JSON.parse(listEl.textContent);
  let frames = kept;  // the list the arrows move through: the kept frames, or the examples
  const img = viewer.querySelector("img");
  const caption = viewer.querySelector(".v-caption");
  let at = -1, s = 1, fitS = 1, ox = 0, oy = 0, drag = null;

  function fmt(t) {
    const m = Math.floor(t / 60), sec = t - m * 60;
    return m + ":" + (sec < 10 ? "0" : "") + sec.toFixed(1);
  }
  // s = screen pixels per image pixel; (ox, oy) = where the image's top-left corner sits
  function place() {
    const W = viewer.clientWidth, H = viewer.clientHeight;
    const w = img.naturalWidth * s, h = img.naturalHeight * s;
    ox = w <= W ? (W - w) / 2 : Math.min(0, Math.max(W - w, ox));
    oy = h <= H ? (H - h) / 2 : Math.min(0, Math.max(H - h, oy));
    img.style.width = w + "px";
    img.style.transform = "translate(" + ox + "px," + oy + "px)";
    img.classList.toggle("zoomed", s > fitS * 1.01);
  }
  function fit() {
    if (!img.naturalWidth) return;
    fitS = Math.min(viewer.clientWidth / img.naturalWidth, viewer.clientHeight / img.naturalHeight);
    s = fitS;
    place();
  }
  function zoomAt(px, py, next) {
    next = Math.max(fitS, Math.min(Math.max(4, fitS * 8), next));
    ox = px - (px - ox) * next / s;
    oy = py - (py - oy) * next / s;
    s = next;
    place();
  }
  function show(i) {
    at = (i + frames.length) % frames.length;  // wraps around at both ends
    const f = frames[at];
    img.src = base + encodeURIComponent(f.file);
    caption.textContent = (at + 1) + " of " + frames.length + " · " + f.video + " at " + fmt(f.t);
    if (viewer.hidden) {
      viewer.hidden = false;
      document.body.classList.add("viewing");
      if (viewer.requestFullscreen) viewer.requestFullscreen().catch(() => {});
    }
  }
  function close() {
    viewer.hidden = true;
    document.body.classList.remove("viewing");
    at = -1;
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
  }

  img.addEventListener("load", fit);
  window.addEventListener("resize", fit);
  // Esc in full screen is taken by the browser to leave full screen: close the viewer with it
  document.addEventListener("fullscreenchange", () => {
    if (!document.fullscreenElement && !viewer.hidden) close(); else fit();
  });

  // open the viewer on a list of {file, video, t}; the filter examples use this too
  window.openViewer = (list, i) => { frames = list; show(i); };
  const grid = document.getElementById("frame-grid");
  if (grid) grid.addEventListener("click", e => {
    const a = e.target instanceof Element && e.target.closest("a[data-i]");
    if (!a || e.metaKey || e.ctrlKey) return;  // Cmd-click still opens a tab
    e.preventDefault();
    window.openViewer(kept, Number(a.dataset.i));
  });
  viewer.addEventListener("click", e => {
    const v = e.target instanceof Element && e.target.closest("[data-v]");
    if (v) v.dataset.v === "close" ? close() : show(at + (v.dataset.v === "next" ? 1 : -1));
  });

  img.addEventListener("pointerdown", e => {
    e.preventDefault();
    img.setPointerCapture(e.pointerId);
    drag = {x: e.clientX, y: e.clientY, ox, oy, moved: false};
  });
  img.addEventListener("pointermove", e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
    if (drag.moved) { ox = drag.ox + dx; oy = drag.oy + dy; place(); }
  });
  img.addEventListener("pointerup", e => {
    if (drag && !drag.moved) {  // a click, not a drag: actual pixels here, or back to fit
      const r = viewer.getBoundingClientRect();
      s > fitS * 1.01 ? fit() : zoomAt(e.clientX - r.left, e.clientY - r.top, Math.max(1, fitS * 2));
    }
    drag = null;
  });
  viewer.addEventListener("wheel", e => {  // scroll wheel, or pinch on a trackpad
    e.preventDefault();
    const r = viewer.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, s * Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.002)));
  }, {passive: false});

  document.addEventListener("keydown", e => {
    if (viewer.hidden || e.metaKey || e.ctrlKey) return;
    const W = viewer.clientWidth / 2, H = viewer.clientHeight / 2;
    if (e.key === "Escape") close();
    else if (e.key === "ArrowRight") show(at + 1);
    else if (e.key === "ArrowLeft") show(at - 1);
    else if (e.key === "+" || e.key === "=") zoomAt(W, H, s * 1.5);
    else if (e.key === "-") zoomAt(W, H, s / 1.5);
    else if (e.key === "0") fit();
    else return;
    e.preventDefault();
  });
})();

// Filters (bite 3b): while a slider moves, ask the app what Apply would do (reads only) and show
// the counts and, per filter, one row of every frame it would drop. Click one to view it.
(function () {
  "use strict";
  const form = document.getElementById("filter-form");
  if (!form) return;
  const out = document.getElementById("filter-counts");
  const box = document.getElementById("filter-examples");
  const base = JSON.parse(document.getElementById("frame-list").textContent).base;
  let timer = null, list = [];

  function thumb(f, label) {
    const a = document.createElement("a");
    a.href = base + encodeURIComponent(f.file);
    a.title = f.file + " (" + f.video + ")";
    a.dataset.i = list.length;
    list.push(f);
    const img = document.createElement("img");
    img.src = a.href;
    img.alt = "";
    img.loading = "lazy";
    const cap = document.createElement("span");
    cap.textContent = label;
    a.append(img, cap);
    return a;
  }
  // one row per filter: every frame it would drop, in order, scrolling sideways
  function row(title, left, right, frames, label) {
    const h = document.createElement("p");
    h.className = "hint";
    h.textContent = title + " (" + frames.length + ")" +
      (frames.length ? ": " + left + " on the left, " + right + " on the right" : "");
    const strip = document.createElement("div");
    strip.className = "drop-row";
    frames.forEach(f => strip.append(thumb(f, label(f))));
    box.append(h, strip);
  }
  function render(a) {
    const c = a.counts;
    out.textContent = c.kept + " kept + " + c.dropped_blur + " blurry + " + c.dropped_duplicate +
      " duplicate" + (c.rejected ? " + " + c.rejected + " rejected" : "") + " = " + c.total;
    box.replaceChildren();
    list = [];
    row("Dropped as blurry", "least blurry", "most blurry", a.blur_dropped,
        f => "sharpness " + Math.round(f.blur));
    row("Dropped as duplicates", "most unique", "least unique", a.dup_dropped,
        f => f.distance + " of 64 bits differ");
  }
  async function update() {
    form.querySelectorAll("output").forEach(o => { o.value = o.previousElementSibling.value; });
    const q = new URLSearchParams({blur: form.blur.value, dup: form.dup.value});
    try {
      const r = await fetch(form.dataset.preview + "?" + q);
      const a = await r.json();
      if (r.ok) render(a); else out.textContent = a.error;
    } catch (e) { /* app restarting */ }
  }
  form.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(update, 200); });
  box.addEventListener("click", e => {
    const a = e.target instanceof Element && e.target.closest("a[data-i]");
    if (!a || e.metaKey || e.ctrlKey || !window.openViewer) return;
    e.preventDefault();
    window.openViewer(list, Number(a.dataset.i));
  });
  update();
})();
