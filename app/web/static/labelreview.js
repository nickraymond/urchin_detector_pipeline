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
  function colourOf(id) {
    return data.colours[id] || "#888";
  }
  // Tab (bite 4b): the next untagged urchin after the selected mask in reading order, wrapping
  function nextUntagged() {
    if (!view || !view.untagged.length) return null;
    const order = view.order;
    const start = sel == null ? -1 : order.indexOf(sel);
    for (let k = 1; k <= order.length; k++) {
      const id = order[(start + k + order.length) % order.length];
      if (view.untagged.includes(id)) return id;
    }
    return null;
  }
  function focusOn(id) {
    const m = view.masks.find(k => k.id === id);
    if (!m || !m.focus || !img.naturalWidth) return;
    const f = img.naturalWidth / view.width;
    const r = m.focus;
    // after the new picture loads, so the zoom is not reset by it
    const go = () => window.viewerFocus(r[0] * f, r[1] * f, r[2] * f, r[3] * f);
    img.addEventListener("load", go, {once: true});
  }
  function goNext() {
    const id = nextUntagged();
    if (id == null) { msg("No untagged urchin left on this frame.", true); return; }
    sel = id; render(); focusOn(id); redraw();
  }

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
    const types = view.types || [];
    $("rv-types").replaceChildren(...types.map(t => {
      const b = el("button", "rv-type");
      b.type = "button"; b.dataset.type = t.id; b.disabled = !(m && m.urchin);
      const sw = el("span", "swatch"); sw.style.background = colourOf(t.id);
      b.append(sw, t.name + (t.key ? " (" + t.key + ")" : ""));
      b.title = "Tag the selected urchin " + t.name;
      return b;
    }));
    const all = $("rv-tag-all");
    all.hidden = !types.length;
    if (types.length) {
      all.textContent = "Tag all remaining as " + types[0].name + " (" + view.untagged.length + ")";
      all.dataset.type = types[0].id;
      all.disabled = !view.untagged.length;
    }
    const cu = $("rv-closeup");
    cu.hidden = !m;
    if (m) cu.src = data.project + "/closeup/" + encodeURIComponent(view.frame) + "?mask=" + m.id + "&v=" + version;
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
    viewer.querySelectorAll('input[name="rv-split"]').forEach(r => {
      r.checked = r.value === (st.split_ok == null ? "" : st.split_ok ? "yes" : "no");
    });
    $("rv-checked").replaceChildren(...data.class_info.map(c => {
      const l = el("label");
      const cb = el("input");
      cb.type = "checkbox"; cb.dataset.cid = c.id; cb.checked = st.checked.includes(c.id);
      l.append(cb, " " + c.name);
      return l;
    }));
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
  function showTotals(totals) {
    (totals || []).forEach(t => {
      const row = document.querySelector('#label-totals tr[data-id="' + t.id + '"]');
      if (!row) return;
      row.querySelector(".n-obj").textContent = t.objects;
      row.querySelector(".n-fr").textContent = t.frames;
      row.querySelector("progress").value = Math.min(t.objects, 150);
      row.querySelector(".n-no").textContent = t.split_no;
    });
  }
  async function act(body, advance) {
    if (at < 0) return;
    const r = await fetch(data.project + "/act/" + encodeURIComponent(frames[at].file), {
      method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) { msg(d.error || "not done"); return; }
    view = d;
    if (!selected()) sel = null;
    msg(d.note, true);
    showTotals(d.totals);
    if (advance && view.untagged.length) { sel = nextUntagged(); render(); focusOn(sel); redraw(); return; }
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
  // Add a missed object (bite 5b, D46): in add mode a click or a dragged box goes to SAM as a job
  let adding = false;
  function setAdding(on) {
    adding = on;
    window.viewerBoxMode = on;
    $("rv-add").classList.toggle("on", on);
    $("rv-add").textContent = on ? "Click on it or drag a box… (click here or Esc: cancel)" : "Add a missed object";
  }
  async function addAt(body) {
    setAdding(false);
    const frame = frames[at].file;
    const scale = view.width / img.naturalWidth;
    const px = v => Math.max(0, Math.round(v * scale));
    if (body.point) body.point = body.point.map(px);
    if (body.box) body.box = [px(body.box[0]), px(body.box[1]), Math.min(view.width, px(body.box[2])),
                              Math.min(view.height, px(body.box[3]))];
    body.class_id = Number($("rv-add-class").value);
    const r = await fetch(data.project + "/add/" + encodeURIComponent(frame), {
      method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(body)});
    const d = await r.json();
    if (!r.ok) { msg(d.error || "not added"); return; }
    msg("SAM is looking (about 20 s; it waits for any SAM run first)…", true);
    for (;;) {
      await new Promise(res => setTimeout(res, 1500));
      const j = await (await fetch(data.project + "/job/" + d.job_id, {cache: "no-store"})).json();
      if (["done", "failed", "cancelled"].includes(j.state)) {
        if (frames[at] && frames[at].file === frame) {
          const before = new Set(view.masks.map(m => m.id));
          await load(at);
          const added = view.masks.find(m => !before.has(m.id));
          if (added) { sel = added.id; render(); }
          redraw();
        }
        msg((j.state === "done" ? "" : j.state + ": ") + (j.message || j.line), j.state === "done");
        return;
      }
      if (j.line) msg("SAM: " + j.state + " · " + j.line, true);
    }
  }
  window.viewerBox = (x0, y0, x1, y1, i) => {
    if (adding && view && i === at) addAt({box: [x0, y0, x1, y1]});
  };
  $("rv-add").addEventListener("click", e => { e.currentTarget.blur(); setAdding(!adding); });
  window.addEventListener("keydown", e => {   // before viewer.js, which closes the viewer on Esc
    if (e.key === "Escape" && adding) { setAdding(false); e.stopImmediatePropagation(); e.preventDefault(); }
  }, true);

  window.viewerPick = (x, y, i) => {
    if (!view || i !== at || !img.naturalWidth) return false;
    if (adding) { addAt({point: [x, y]}); return true; }
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
  viewer.querySelectorAll('input[name="rv-split"]').forEach(r => r.addEventListener("change", () =>
    act({action: "split", ok: r.value === "" ? null : r.value === "yes"})));
  $("rv-checked").addEventListener("change", e => {
    const cb = e.target;
    if (cb.dataset && cb.dataset.cid) act({action: "checked", class_id: Number(cb.dataset.cid), checked: cb.checked});
  });
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
  $("rv-types").addEventListener("click", e => {
    const b = e.target instanceof Element && e.target.closest("[data-type]");
    if (b) b.blur();
    if (b && sel != null) act({action: "tag", mask_ids: [sel], class_id: Number(b.dataset.type)}, true);
  });
  viewer.querySelector(".v-review").addEventListener("click", e => {
    const b = e.target instanceof Element && e.target.closest("[data-rv]");
    if (!b) return;
    b.blur();   // Enter / Space must not press it again (Enter is Accept)
    const a = b.dataset.rv;
    if (a === "tag_all") act({action: a, class_id: Number(b.dataset.type)});
    else act(a === "delete" ? {action: a, mask_ids: [sel]} : {action: a});
  });
  document.addEventListener("keydown", e => {
    if (viewer.hidden || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target instanceof Element && e.target.closest("input, select, textarea")) return;
    const k = e.key.toLowerCase();
    const type = view && (view.types || []).find(t => t.key && t.key.toLowerCase() === k);
    const m = selected();
    if (e.key === "Tab") goNext();
    else if (type && m && m.urchin) act({action: "tag", mask_ids: [sel], class_id: type.id}, true);
    else if (type) msg("Select an urchin first (click it, or Tab).");
    else if (k === "x" && sel != null) act({action: "delete", mask_ids: [sel]});
    else if (e.key === "Enter") act({action: "accept"});
    else if (k === "z") act({action: "undo"});
    else if (e.key === "`") togglePanel();   // letters stay free for urchin types (QE PR 51)
    else if (k === "f") document.fullscreenElement ? document.exitFullscreen() : viewer.requestFullscreen();
    else return;
    e.preventDefault();
  });
})();
