// Label review (S5 bite 3a, spec §6): the frames open in S2's large viewer (viewer.js, reused);
// the picture is drawn by the server from the frame's final masks, with the selected mask in
// white. A click on the picture selects the mask under it (window.viewerPick); the class box,
// X, Enter and Z act on the server, which answers the frame's new state. Nothing is kept in
// the page that the server does not have.
(function () {
  "use strict";
  const data = JSON.parse(document.getElementById("review-data").textContent);
  const base = JSON.parse(document.getElementById("frame-list").textContent).base;
  const viewer = document.getElementById("viewer");
  const img = viewer.querySelector("img");
  const $ = id => document.getElementById(id);
  const byId = Object.fromEntries(data.class_info.map(c => [c.id, c]));
  const frames = data.frames.map(f => ({file: f.file, accepted: f.accepted, src: "", caption: f.file}));
  let at = -1, view = null, sel = null, mode = "all", version = 0;

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function msg(text, ok) { $("rv-msg").textContent = text || ""; $("rv-msg").classList.toggle("ok", !!ok); }
  function picture() {
    const q = new URLSearchParams({mode, v: String(++version)});
    if (sel != null) q.set("sel", String(sel));
    return base + encodeURIComponent(frames[at].file) + "?" + q;
  }
  function selected() { return view && view.masks.find(m => m.id === sel); }

  function render() {
    const m = selected();
    $("rv-selected").textContent = m
      ? "Selected: " + m.name + " (" + m.kind + (m.score != null ? ", score " + m.score.toFixed(2) : "") +
        (m.flag ? ", overlapped another mask" : "") + (m.added ? ", added by hand" : "") + ")"
      : "Click a mask to select it.";
    const box = $("rv-class");
    box.disabled = !m;
    box.value = "";
    const list = $("rv-classes");
    list.replaceChildren(...data.class_info.filter(c => m && c.kind === m.kind)
      .map(c => { const o = el("option"); o.value = c.name; return o; }));
    viewer.querySelector('[data-rv="delete"]').disabled = !m;
    const st = view.state;
    viewer.querySelector('[data-rv="accept"]').hidden = st.accepted;
    viewer.querySelector('[data-rv="reopen"]').hidden = !st.accepted;
    $("rv-status").textContent = (st.accepted ? "Accepted." : "To do.") +
      (view.untagged.length ? " " + view.untagged.length + " untagged urchin" +
       (view.untagged.length > 1 ? "s" : "") + " (grey)." : "");
    // legend: this frame's masks per class
    const counts = {};
    view.masks.forEach(k => { counts[k.name] = counts[k.name] || {n: 0, colour: k.colour, kind: k.kind}; counts[k.name].n++; });
    const lg = $("rv-legend");
    lg.replaceChildren(...Object.entries(counts).map(([name, c]) => {
      const s = el("span", "lg-item");
      const sw = el("span", "swatch"); sw.style.background = c.colour;
      s.append(sw, " " + name + ": " + c.n + (c.kind === "object" ? " ☐" : ""));
      return s;
    }), el("span", "lg-note", "☐ objects are boxed; areas (kelp, rock) are not"));
    const row = document.querySelector('#label-frames tr[data-file="' + CSS.escape(view.frame) + '"]');
    if (row) row.querySelector(".st").textContent = st.accepted ? "accepted" : "to do";
    frames[at].accepted = st.accepted;
    $("todo-count").textContent = frames.filter(f => !f.accepted).length;
  }
  function redraw() {
    frames[at].src = picture();
    window.viewerShow(at, true);
  }
  async function load(i) {
    at = i; sel = null;
    document.querySelectorAll("#label-frames tr").forEach(r =>
      r.classList.toggle("current", r.dataset.file === frames[i].file));
    const r = await fetch(data.project + "/frame/" + encodeURIComponent(frames[i].file), {cache: "no-store"});
    const d = await r.json();
    if (!r.ok) { msg(d.error || "cannot load this frame"); return; }
    view = d;
    msg("");
    render();
  }
  async function act(body) {
    if (at < 0) return;
    const r = await fetch(data.project + "/act/" + encodeURIComponent(frames[at].file), {
      method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) { msg(d.error || "not done"); return; }
    view = d;
    if (!selected()) sel = null;
    msg(d.note, true);
    render();
    redraw();
  }
  function open(i) {
    at = i; sel = null;
    frames.forEach(f => { f.src = ""; });   // a frame not drawn yet is drawn on arrival
    frames[i].src = picture();
    window.openViewer(frames, i);
    load(i);
  }

  // viewer.js tells us which frame it shows; on a new one (arrows, < >) draw it and load its masks
  document.addEventListener("viewer:show", e => {
    if (e.detail.i === at) return;
    at = e.detail.i; sel = null;
    redraw();
    load(at);
  });
  window.viewerPick = (x, y, i) => {
    if (!view || i !== at || !img.naturalWidth) return false;
    const fx = Math.floor(x * view.width / img.naturalWidth), fy = Math.floor(y * view.height / img.naturalHeight);
    fetch(data.project + "/at/" + encodeURIComponent(frames[at].file) + "?x=" + fx + "&y=" + fy, {cache: "no-store"})
      .then(r => r.json()).then(d => {
        sel = d.mask_id == null ? null : d.mask_id;
        render();
        redraw();
      });
    return true;
  };

  document.querySelectorAll('#label-frames button[data-i]').forEach(b =>
    b.addEventListener("click", () => open(Number(b.dataset.i))));
  $("next-todo").addEventListener("click", () => {
    const i = frames.findIndex((f, k) => !f.accepted && k > at);
    const j = i >= 0 ? i : frames.findIndex(f => !f.accepted);
    if (j >= 0) open(j); else alert("Every frame is accepted.");
  });
  viewer.querySelectorAll('input[name="rv-mode"]').forEach(r => r.addEventListener("change", () => {
    mode = r.value; redraw();
  }));
  $("rv-zoom").addEventListener("input", e => window.viewerZoom(Number(e.target.value)));
  $("rv-class").addEventListener("change", e => {
    const c = data.class_info.find(c => c.name.toLowerCase() === e.target.value.trim().toLowerCase());
    if (!c) { msg("No class named " + JSON.stringify(e.target.value) + "."); return; }
    if (sel != null) act({action: "class", mask_ids: [sel], class_id: c.id});
  });
  const panel = viewer.querySelector(".v-review");
  function togglePanel() {
    panel.classList.toggle("collapsed");
    panel.querySelector(".rv-toggle").textContent = panel.classList.contains("collapsed") ? "Review ▸" : "Review ▾";
  }
  panel.querySelector(".rv-toggle").addEventListener("click", togglePanel);
  viewer.querySelector(".v-review").addEventListener("click", e => {
    const b = e.target instanceof Element && e.target.closest("[data-rv]");
    if (!b) return;
    const a = b.dataset.rv;
    act(a === "delete" ? {action: a, mask_ids: [sel]} : {action: a});
  });
  document.addEventListener("keydown", e => {
    if (viewer.hidden || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target instanceof Element && e.target.closest("input, select, textarea")) return;
    const k = e.key.toLowerCase();
    if (k === "x" && sel != null) act({action: "delete", mask_ids: [sel]});
    else if (e.key === "Enter") act({action: "accept"});
    else if (k === "z") act({action: "undo"});
    else if (e.key === "`") togglePanel();   // letters stay free for urchin types (QE PR 51)
    else if (k === "f") document.fullscreenElement ? document.exitFullscreen() : viewer.requestFullscreen();
    else return;
    e.preventDefault();
  });
})();
