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

// Frame viewer (Nick 2026-11-25): click a frame to see it large on this page; arrows or
// the < > buttons move through every kept frame; Esc, the Close button or the dark
// background closes it.
(function () {
  "use strict";
  const listEl = document.getElementById("frame-list");
  const viewer = document.getElementById("viewer");
  if (!listEl || !viewer) return;
  const {base, frames} = JSON.parse(listEl.textContent);
  const img = viewer.querySelector("img");
  const caption = viewer.querySelector(".v-caption");
  let at = -1;

  function fmt(t) {
    const m = Math.floor(t / 60), s = t - m * 60;
    return m + ":" + (s < 10 ? "0" : "") + s.toFixed(1);
  }
  function show(i) {
    at = (i + frames.length) % frames.length;  // wraps around at both ends
    const f = frames[at];
    img.src = base + encodeURIComponent(f.file);
    caption.textContent = (at + 1) + " of " + frames.length + " · " + f.video + " at " + fmt(f.t);
    viewer.hidden = false;
    document.body.classList.add("viewing");
  }
  function close() {
    viewer.hidden = true;
    document.body.classList.remove("viewing");
    at = -1;
  }

  document.getElementById("frame-grid").addEventListener("click", e => {
    const a = e.target instanceof Element && e.target.closest("a[data-i]");
    if (!a || e.metaKey || e.ctrlKey) return;  // Cmd-click still opens a tab
    e.preventDefault();
    show(Number(a.dataset.i));
  });
  viewer.addEventListener("click", e => {
    const v = e.target instanceof Element && e.target.closest("[data-v]");
    if (v) { v.dataset.v === "close" ? close() : show(at + (v.dataset.v === "next" ? 1 : -1)); }
    else if (e.target === viewer) close();
  });
  document.addEventListener("keydown", e => {
    if (viewer.hidden) return;
    if (e.key === "Escape") close();
    else if (e.key === "ArrowRight") show(at + 1);
    else if (e.key === "ArrowLeft") show(at - 1);
    else return;
    e.preventDefault();
  });
})();
