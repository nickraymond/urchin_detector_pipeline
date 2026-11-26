// Live job boxes (S4 bite 6b, Nick: "had to refresh the page"): every element with data-live
// polls its URL (an HTML fragment rendered by the same macros as the page) every 2 s and is
// replaced only when the fragment changed, so nothing flashes and no page reload is needed.
// data-reload-on="done error": reload the page once when a job that was running ends that way
// (the Labeling page then shows the new results and Run plan); its buttons follow the job too.
(function () {
  "use strict";
  const ACTIVE = ["queued", "running", "pausing"];
  let pressed = false;   // never swap a button away between mouse down and click
  document.addEventListener("pointerdown", () => { pressed = true; });
  document.addEventListener("pointerup", () => { setTimeout(() => { pressed = false; }, 300); });

  function states(root) {
    return Array.from(root.querySelectorAll("[data-state]"), e => e.dataset.state);
  }
  // The page's own buttons (Run, Preview, Extract ...) follow its job: data-job-lock lists the
  // states in which a button is off; data-off-anyway keeps it off for another reason.
  function lock(now) {
    document.querySelectorAll("[data-job-lock]").forEach(b => {
      const off = b.dataset.jobLock.split(" ");
      b.disabled = b.hasAttribute("data-off-anyway") || now.some(s => off.includes(s));
    });
  }
  document.querySelectorAll("[data-live]").forEach(box => {
    let last = null;
    const reloadOn = (box.dataset.reloadOn || "").split(" ").filter(Boolean);
    async function tick() {
      if (!document.hidden && !pressed) {
        try {
          const r = await fetch(box.dataset.live, {cache: "no-store"});
          const text = r.ok ? await r.text() : null;
          if (text !== null && text !== last) {
            const before = states(box);
            if (last !== null || text.trim() !== box.innerHTML.trim()) box.innerHTML = text;
            last = text;
            const after = states(box);
            if (reloadOn.length) lock(after);
            if (before.some(s => ACTIVE.includes(s)) && after.some(s => reloadOn.includes(s))) {
              location.reload();
              return;
            }
          }
        } catch (e) { /* the app is restarting: try again */ }
      }
      setTimeout(tick, 2000);
    }
    setTimeout(tick, 2000);
  });
})();
