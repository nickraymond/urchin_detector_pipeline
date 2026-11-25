// Pick clips page (bite 2b): I / O marks, the timeline with clip bands, the clip list.
// Every change is saved at once; the server checks the clips and answers the saved list.
(function () {
  "use strict";
  const dataEl = document.getElementById("clip-data");
  const video = document.getElementById("video");
  if (!dataEl || !video) return;
  const data = JSON.parse(dataEl.textContent);
  const timeline = document.getElementById("timeline");
  const playhead = document.getElementById("playhead");
  const pendingEl = document.getElementById("pending");
  const msg = document.getElementById("clip-msg");
  const body = document.querySelector("#clip-table tbody");
  const noClips = document.getElementById("no-clips");
  let clips = data.clips.map(c => ({start_s: c.start_s, end_s: c.end_s, note: c.note || ""}));
  let pendingIn = null;

  function fmt(t) {
    const m = Math.floor(t / 60), s = t - m * 60;
    return m + ":" + (s < 10 ? "0" : "") + s.toFixed(1);
  }
  const pct = t => (100 * t / data.duration) + "%";

  function show(text) {
    msg.hidden = !text;
    msg.textContent = text || "";
  }

  function draw() {
    timeline.querySelectorAll(".band").forEach(b => b.remove());
    clips.forEach(c => {
      const b = document.createElement("div");
      b.className = "band";
      b.style.left = pct(c.start_s);
      b.style.width = pct(c.end_s - c.start_s);
      b.title = fmt(c.start_s) + "–" + fmt(c.end_s) + (c.note ? " · " + c.note : "");
      timeline.appendChild(b);
    });
    pendingEl.hidden = pendingIn === null;
    if (pendingIn !== null) pendingEl.style.left = pct(pendingIn);
    body.innerHTML = "";
    noClips.hidden = clips.length > 0;
    clips.forEach((c, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = "<td>" + (i + 1) + "</td><td>" + fmt(c.start_s) + "</td><td>" + fmt(c.end_s) +
        "</td><td>" + (c.end_s - c.start_s).toFixed(1) + " s</td><td><input maxlength='200'></td><td>" +
        "<button data-a='go'>Go to</button> <button data-a='start'>Start = now</button> " +
        "<button data-a='end'>End = now</button> <button data-a='del'>Delete</button></td>";
      const note = tr.querySelector("input");
      note.value = c.note;
      note.placeholder = "note, e.g. dense purple urchins";
      note.addEventListener("change", () => { c.note = note.value; save(); });
      tr.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
        const now = Number(video.currentTime.toFixed(2));
        if (btn.dataset.a === "go") { video.currentTime = c.start_s; return; }
        const next = clips.map(x => ({...x}));
        if (btn.dataset.a === "start") next[i].start_s = now;
        if (btn.dataset.a === "end") next[i].end_s = now;
        if (btn.dataset.a === "del") next.splice(i, 1);
        save(next);
      }));
      body.appendChild(tr);
    });
  }

  // Send the whole list for this video; keep the old list if the server refuses it.
  async function save(next) {
    const wanted = next || clips;
    let r, answer;
    try {
      r = await fetch(data.save, {method: "POST", headers: {"Content-Type": "application/json"},
                                  body: JSON.stringify({video: data.video, clips: wanted})});
      answer = await r.json();
    } catch (e) {
      show("Could not save: is the app still running? (" + e + ")");
      return false;
    }
    if (!r.ok) { show(answer.error || "Could not save."); draw(); return false; }
    show("");
    clips = answer.clips;
    updateCounts(answer.summary);
    draw();
    return true;
  }

  function updateCounts(sum) {
    const mmss = s => Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");
    const totals = document.getElementById("totals");
    totals.firstChild.textContent = sum.clips + " clip" + (sum.clips === 1 ? "" : "s") + " · " +
      mmss(sum.marked_s) + " marked · about " + sum.frames + " frames at one every " +
      data.every + " s ";
    document.querySelectorAll(".counts").forEach(el => {
      const v = sum.videos[el.dataset.code];
      if (!v) return;
      el.textContent = (v.clips ? v.clips + " clip" + (v.clips === 1 ? "" : "s") + ", " +
        mmss(v.marked_s) : "whole video") + " · " + v.frames + " frames";
    });
  }

  function mark(which) {
    const now = Number(video.currentTime.toFixed(2));
    if (which === "in") { pendingIn = now; show(""); draw(); return; }
    if (pendingIn === null) { show("Press I first to mark where the clip starts."); return; }
    const start = pendingIn;
    save(clips.concat([{start_s: start, end_s: now, note: ""}])).then(ok => {
      if (ok) pendingIn = null;
      draw();
    });
  }

  document.addEventListener("keydown", e => {
    const typing = e.target instanceof Element && e.target.matches("input, textarea");
    if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === "i") mark("in");
    else if (k === "o") mark("out");
    else if (k === " ") { video.paused ? video.play() : video.pause(); }
    else if (k === "arrowleft" || k === "arrowright") {
      const step = (e.shiftKey ? 5 : 1) * (k === "arrowleft" ? -1 : 1);
      video.currentTime = Math.max(0, Math.min(data.duration, video.currentTime + step));
    } else return;
    e.preventDefault();
  });

  timeline.addEventListener("click", e => {
    const r = timeline.getBoundingClientRect();
    video.currentTime = data.duration * (e.clientX - r.left) / r.width;
  });
  video.addEventListener("timeupdate", () => { playhead.style.left = pct(video.currentTime); });
  draw();
})();
