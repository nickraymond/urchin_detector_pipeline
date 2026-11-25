// Labeling page (S3): follow the SAM job and reload when it ends; per-class toggles redraw
// the preview overlays without the hidden classes.
(function () {
  "use strict";
  const job = document.getElementById("sam-job");
  if (job && (job.dataset.state === "queued" || job.dataset.state === "running")) {
    const text = document.getElementById("sam-job-text");
    const tick = () => fetch(job.dataset.status).then(r => r.json()).then(st => {
      text.textContent = st.state + (st.message ? " — " + st.message : "") +
        (st.eta_s ? " · about " + Math.ceil(st.eta_s / 60) + " min left" : "");
      if (st.state === "queued" || st.state === "running") setTimeout(tick, 2000);
      else location.reload();
    }).catch(() => setTimeout(tick, 5000));
    tick();
  }
  const toggles = document.getElementById("toggles");
  if (!toggles) return;
  toggles.addEventListener("change", () => {
    const hide = [...toggles.querySelectorAll("input:not(:checked)")].map(i => i.value);
    document.querySelectorAll("img.overlay").forEach(img => {
      img.src = img.dataset.src + (hide.length ? "?hide=" + encodeURIComponent(hide.join(",")) : "");
    });
  });
})();
