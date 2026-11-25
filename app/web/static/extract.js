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

// Filters (bite 3b): while a slider moves, ask the app what Apply would do (reads only) and show
// the counts and example frames at each cut-off. Clicking an example opens it in the viewer.
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
    a.title = f.file;
    a.dataset.i = list.length;
    list.push(f);
    const img = document.createElement("img");
    img.src = a.href;
    img.alt = "";
    img.loading = "lazy";
    const cap = document.createElement("span");
    cap.textContent = label;
    const vid = document.createElement("span");  // the blur cut-off is per video
    vid.className = "ex-video";
    vid.textContent = f.video;
    a.append(img, cap, vid);
    return a;
  }
  function row(title, items) {
    const div = document.createElement("div");
    div.className = "example-row";
    const h = document.createElement("p");
    h.className = "hint";
    h.textContent = title + (items.length ? "" : ": none");
    div.append(h);
    items.forEach(nodes => {
      const g = document.createElement("span");
      g.className = "example";
      g.append(...nodes);
      div.append(g);
    });
    box.append(div);
  }
  function render(a) {
    const c = a.counts;
    out.textContent = c.kept + " kept + " + c.dropped_blur + " blurry + " + c.dropped_duplicate +
      " duplicate" + (c.rejected ? " + " + c.rejected + " rejected" : "") + " = " + c.total;
    box.replaceChildren();
    list = [];
    const blur = f => [thumb(f, "sharpness " + Math.round(f.blur))];
    const pair = f => [thumb(f, f.distance + " bits differ"), thumb(f.match, "last kept")];
    row("Blur: the sharpest frames dropped", a.blur_dropped.map(blur));
    row("Blur: the blurriest frames kept", a.blur_kept.map(blur));
    row("Duplicates: dropped, nearest the cut-off (each beside the frame it matched)",
        a.dup_dropped.map(pair));
    row("Duplicates: kept, nearest the cut-off", a.dup_kept.map(pair));
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
