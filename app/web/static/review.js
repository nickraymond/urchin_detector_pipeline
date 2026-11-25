// Review frames page (bite 4, DESIGN D20): grid with a size slider, arrow keys move the
// selection, D delete (to rejected/), K keep, Z undo, F or click opens the shared viewer
// (viewer.js), where the same keys act on the frame shown. A frame changed here keeps its
// place with its new status until the page is reloaded, so Z puts it back where it was.
(function () {
  "use strict";
  const grid = document.getElementById("review-grid");
  const data = JSON.parse(document.getElementById("frame-list").textContent);
  const msg = document.getElementById("review-msg");
  const counts = document.getElementById("review-counts");
  const slider = document.getElementById("per-row");
  const out = document.getElementById("per-row-out");
  const frames = data.frames;
  let sel = 0, busy = false;

  document.getElementById("show-dropped").addEventListener("change", e => {
    location.href = e.target.dataset.href;
  });
  function perRow(n) {
    slider.value = n;
    out.value = n;
    if (grid) grid.style.gridTemplateColumns = "repeat(" + n + ", 1fr)";
    try { localStorage.setItem("review-per-row", n); } catch (e) { /* private window */ }
  }
  let saved = null;
  try { saved = localStorage.getItem("review-per-row"); } catch (e) { /* private window */ }
  perRow(saved && +saved >= 1 && +saved <= 8 ? +saved : 6);
  slider.addEventListener("input", () => perRow(+slider.value));
  if (!grid) return;

  const cells = frames.map((f, i) => {
    const a = document.createElement("a");
    a.href = data.base + encodeURIComponent(f.file);
    a.title = f.file;
    a.dataset.i = i;
    const img = document.createElement("img");
    img.src = data.thumbs + encodeURIComponent(f.file);
    img.alt = "";
    img.loading = "lazy";
    const cap = document.createElement("span");
    cap.className = "r-label";
    a.append(img, cap);
    grid.append(a);
    return a;
  });
  function paint(i) {
    const f = frames[i], a = cells[i];
    a.className = "st-" + f.status + (i === sel ? " selected" : "");
    a.querySelector(".r-label").textContent = f.status === "kept" ? "" : f.label;
  }
  function select(i) {
    if (!frames.length) return;
    const old = sel;
    sel = Math.max(0, Math.min(frames.length - 1, i));
    paint(old);
    paint(sel);
    cells[sel].scrollIntoView({block: "nearest"});
  }
  frames.forEach((_, i) => paint(i));
  select(0);

  async function send(key, i) {
    if (busy) return;
    busy = true;
    try {
      const r = await fetch(data.act, {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({file: i == null ? "" : frames[i].file, key})});
      const a = await r.json();
      if (a.summary) counts.textContent = "Now: " + a.summary;
      if (a.error) { msg.textContent = a.error; return; }
      const j = frames.findIndex(f => f.file === a.file);
      if (j < 0) {  // undo of a frame not on this page (e.g. hidden by the filter): reload
        msg.textContent = "Undone: " + a.file + " is " + a.label + " again. Reloading…";
        location.reload();
        return;
      }
      frames[j].status = a.status;
      frames[j].label = a.label;
      paint(j);
      msg.textContent = (key === "Z" ? "Undone: " : "") + frames[j].file + ": " + a.label;
      if (key === "Z") select(j);
      else if (a.changed && j === sel && sel < frames.length - 1) select(sel + 1);
      const v = window.viewerIndex ? window.viewerIndex() : -1;
      if (v >= 0) window.viewerShow(key === "Z" ? j : sel);
    } catch (e) {
      msg.textContent = "The app did not answer; is it still running?";
    } finally {
      busy = false;
    }
  }

  // the viewer moved (its arrows): follow it in the grid
  document.addEventListener("viewer:show", e => select(e.detail.i));
  grid.addEventListener("click", e => {
    const a = e.target instanceof Element && e.target.closest("a[data-i]");
    if (!a || e.metaKey || e.ctrlKey) return;  // Cmd-click still opens a tab
    e.preventDefault();
    window.openViewer(frames, Number(a.dataset.i));
  });
  document.addEventListener("keydown", e => {
    if (e.metaKey || e.ctrlKey || e.altKey || e.target instanceof HTMLInputElement) return;
    const open = window.viewerIndex() >= 0;
    const k = e.key.length === 1 ? e.key.toUpperCase() : e.key;
    const cols = +slider.value;
    if (k === "D" || k === "K") send(k, open ? window.viewerIndex() : sel);
    else if (k === "Z") send("Z", null);
    else if (open) return;  // arrows, Esc and zoom keys belong to the viewer
    else if (k === "F") window.openViewer(frames, sel);
    else if (k === "ArrowRight") select(sel + 1);
    else if (k === "ArrowLeft") select(sel - 1);
    else if (k === "ArrowDown") select(sel + cols);
    else if (k === "ArrowUp") select(sel - cols);
    else return;
    e.preventDefault();
  });
})();
