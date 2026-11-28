// Train page (S8 bite 5): one small chart per metric of one model (loss, validation scores, ...),
// drawn on canvases from /train/metrics/<model> and polled every 3 s while the tab is visible.
// One chart per metric because their scales differ (a loss of 5 next to a score of 0.6).
// Below the charts: the model's validation frames with boxes drawn, reloaded when a new epoch
// arrives. And the form: input sizes, presets, default epochs and the "why" texts follow the
// model type picked. No library, no assets.
(function () {            // the form follows the model type
  const pick = document.getElementById("model-type"), data = document.getElementById("model-types");
  if (!pick || !data) return;
  const types = JSON.parse(data.textContent), form = pick.form;
  const options = (sel, list, chosen) => sel.replaceChildren(...list.map(v => {
    const o = document.createElement("option"); o.textContent = v; o.selected = v === chosen; return o;
  }));
  pick.addEventListener("change", () => {
    const t = types[pick.value];
    if (!t) return;
    options(form.elements.input, t.inputs, t.defaults.input);
    options(form.elements.augment, t.presets, t.defaults.augment);
    form.elements.epochs.value = t.defaults.epochs;
    form.elements.seed.value = t.defaults.seed;
    if (form.elements.val_every && t.defaults.val_every) form.elements.val_every.value = t.defaults.val_every;
    form.querySelectorAll("[data-why]").forEach(td => { td.textContent = (t.why || {})[td.dataset.why] || ""; });
  });
})();
(function () {
  const box = document.getElementById("curves");
  if (!box) return;
  const url = box.dataset.src, charts = box.querySelector(".charts"), note = box.querySelector(".curve-note");
  let lastCount = -1;

  function metrics(rows) {            // every numeric field except epoch / seconds is a chart
    const keys = [];
    rows.forEach(r => Object.keys(r).forEach(k => {
      if (k !== "epoch" && k !== "seconds" && typeof r[k] === "number" && !keys.includes(k)) keys.push(k);
    }));
    return keys;
  }

  function chart(key, rows, epochs) {
    const canvas = document.createElement("canvas");
    canvas.width = 370; canvas.height = 190; canvas.dataset.metric = key;
    const ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height;
    const L = 48, R = 10, T = 22, B = 24;
    const pts = rows.filter(r => typeof r[key] === "number");
    const vals = pts.map(r => r[key]);
    const lo = Math.min(...vals), hi = Math.max(...vals), span = (hi - lo) || 1;
    const x = e => L + (W - L - R) * (epochs > 1 ? e / (epochs - 1) : 0.5);
    const y = v => T + (H - T - B) * (1 - (v - lo) / span);
    ctx.font = "11px sans-serif"; ctx.fillStyle = "#222"; ctx.strokeStyle = "#999"; ctx.lineWidth = 1;
    ctx.strokeRect(L, T, W - L - R, H - T - B);
    ctx.fillText(key + ": " + vals[vals.length - 1].toPrecision(4) + " at epoch " + (pts[pts.length - 1].epoch + 1), L, 13);
    ctx.fillStyle = "#555";
    ctx.fillText(hi.toPrecision(3), 2, T + 10); ctx.fillText(lo.toPrecision(3), 2, H - B);
    ctx.fillText("1", L, H - 8); ctx.fillText(String(epochs), W - R - 20, H - 8);
    ctx.strokeStyle = "#1565c0"; ctx.lineWidth = 2; ctx.beginPath();
    pts.forEach((r, i) => i ? ctx.lineTo(x(r.epoch), y(r[key])) : ctx.moveTo(x(r.epoch), y(r[key])));
    ctx.stroke();
    if (pts.length === 1) { ctx.fillStyle = "#1565c0"; ctx.fillRect(x(pts[0].epoch) - 2, y(vals[0]) - 2, 4, 4); }
    return canvas;
  }

  const valBox = box.querySelector(".val-frames"), valNote = box.querySelector(".val-note");
  function frames(data) {            // a new address per epoch, or the browser keeps the old picture
    if (!valBox || !data.val_frames || !data.val_frames.length) return;
    valBox.replaceChildren(...data.val_frames.map(i => {
      const img = document.createElement("img");
      img.src = box.dataset.val + i + "?e=" + data.val_epoch; img.alt = "validation frame " + (i + 1);
      img.width = 256; img.className = "val-frame";
      return img;
    }));
    valNote.hidden = false; valNote.querySelector(".val-epoch").textContent = data.val_epoch;
  }

  function draw(data) {
    note.textContent = data.rows.length ? `${data.rows.length} of ${data.epochs} epochs` : "No finished epoch yet.";
    charts.replaceChildren(...metrics(data.rows).map(k => chart(k, data.rows, data.epochs)));
  }

  async function poll() {
    if (document.hidden) return;
    try {
      const data = await (await fetch(url, {cache: "no-store"})).json();
      if (data.rows && data.rows.length !== lastCount) { lastCount = data.rows.length; draw(data); frames(data); }
    } catch (e) { /* the app is restarting: try again on the next tick */ }
  }
  poll();
  setInterval(poll, 3000);
  document.addEventListener("visibilitychange", poll);   // a tab opened in the background: draw when shown
})();
