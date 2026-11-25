// Pick clips page (bite 2a): job progress, and the thumbnail strip next to the player.
// Plain JavaScript, nothing loaded from the internet (DESIGN D12).
(function () {
  "use strict";
  const video = document.getElementById("video");
  const strip = document.getElementById("strip");
  const now = document.getElementById("now");

  function fmt(t) {
    const m = Math.floor(t / 60), s = t - m * 60;
    return m + ":" + (s < 10 ? "0" : "") + s.toFixed(1);
  }

  if (video && strip) {
    const every = Number(strip.dataset.every);
    const thumbs = Array.from(strip.querySelectorAll("img"));
    thumbs.forEach(img => img.addEventListener("click", () => {
      video.currentTime = Number(img.dataset.t);
    }));
    video.addEventListener("timeupdate", () => {
      now.textContent = fmt(video.currentTime);
      const i = Math.min(thumbs.length - 1, Math.floor(video.currentTime / every));
      thumbs.forEach((img, k) => img.classList.toggle("here", k === i));
    });
  }

  // While any copy is being made: poll the status every 2 s, update the list and the bar,
  // and reload once the video being viewed is ready.
  const jobs = document.querySelectorAll(".job");
  const box = document.getElementById("progress");
  if (!box && !Array.from(jobs).some(j => j.textContent.trim() !== "ready")) return;
  const page = location.pathname;
  const timer = setInterval(async () => {
    let status;
    try {
      status = await (await fetch(page + "/status")).json();
    } catch (e) {
      return; // app restarting; try again next tick
    }
    jobs.forEach(j => {
      const st = status[j.dataset.code] || {};
      j.textContent = st.ready ? "ready" : st.state === "error" ? st.message
        : st.state === "running" ? Math.round(st.percent) + " %" : (st.state || "waiting");
    });
    if (box) {
      const st = status[box.dataset.code] || {};
      if (st.ready) { clearInterval(timer); location.reload(); return; }
      box.querySelector("progress").value = st.percent || 0;
      const eta = st.eta_s != null ? ", about " + Math.ceil(st.eta_s / 60) + " min left" : "";
      box.querySelector(".hint").textContent = st.state === "error"
        ? "Error: " + st.message : (st.message || "waiting") + eta;
    } else if (Object.values(status).every(st => st.ready)) {
      clearInterval(timer);
    }
  }, 2000);
})();
