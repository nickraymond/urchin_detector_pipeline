// Class editor form (S3 bite 6, SPEC Q17): one row per class instead of a YAML text box.
// Ids are given, never typed. A class with SAM results may be renamed: it keeps its id and its
// saved masks (D39, S5; S3 locked the name, D6). After every change
// the server checks the form with the same checks as classes.yaml (app/sam/classes.py), and
// Save stays off while there is a problem; the server checks again before it writes the file.
(function () {
  "use strict";
  const dataEl = document.getElementById("class-form-data");
  const root = document.getElementById("class-form");
  if (!dataEl || !root) return;
  const F = JSON.parse(dataEl.textContent);
  const settings = Object.assign({}, F.settings);
  const rows = F.classes.map(c => Object.assign({}, c, {prompts: c.prompts.slice()}));
  let maxId = Math.max(F.next_id - 1, ...rows.map(c => c.id));
  let errors = [], timer = null, dirty = false, newGroupRow = -1;
  const SETTINGS = [  // [key, label, help, step, min, max]
    ["confidence", "Confidence cut-off", "masks below this score are dropped (0 to 1)", 0.05, 0.01, 1],
    ["tile_px", "Tile size (px)", "SAM looks at big frames in squares of this size", 1, 256, 4096],
    ["tile_overlap", "Tile overlap", "share of a tile that overlaps the next (0 to 0.85)", 0.05, 0, 0.85],
    ["tile_above_px", "Tile frames wider than (px)", "narrower frames go to SAM whole", 1, 0, null],
    ["min_mask_area_px", "Minimum mask area (px)", "smaller masks are dropped; empty = keep all", 1, 0, null],
  ];

  function el(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => {
      if (k === "class") e.className = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (v === true) e.setAttribute(k, "");
      else if (v !== false && v != null) e.setAttribute(k, v);
    });
    kids.flat().forEach(k => k != null && e.append(k));
    return e;
  }
  function select(cls, value, options, onchange, disabled) {  // options: [[value, label]]
    const s = el("select", {class: cls, onchange: e => onchange(e.target.value), disabled});
    options.forEach(([v, label]) => s.append(el("option", {value: v, selected: v === value}, label)));
    return s;
  }
  const colour = id => F.palette[(id - 1) % F.palette.length];
  const groups = () => [...new Set(rows.map(c => c.sam_group).filter(Boolean))];
  const areaNames = () => rows.filter(c => c.kind === "area" && c.name).map(c => c.name);

  function changed(rerender) {
    dirty = true;
    if (rerender) render();
    clearTimeout(timer);
    timer = setTimeout(check, 250);
    status();
  }
  function payload() {  // the keys of classes.yaml, in its order; empty choices left out
    return {settings, classes: rows.map(c => {
      const o = {id: c.id, name: c.name, kind: c.kind, prompts: c.prompts};
      if (c.sam_group) o.sam_group = c.sam_group;
      if (c.seg_as) o.seg_as = c.seg_as;
      return o;
    })};
  }
  function post(url) {
    return fetch(url, {method: "POST", headers: {"Content-Type": "application/json"},
                       body: JSON.stringify(payload())})
      .then(r => r.json().then(j => ({ok: r.ok, errors: j.errors || []})))
      .catch(() => ({ok: false, errors: [{row: null, setting: null,
                                          msg: "the app did not answer; is it still running?"}]}));
  }
  function check() {
    post(F.save + "?check=1").then(r => { errors = r.errors; showErrors(); });
  }
  function save() {
    clearTimeout(timer);
    post(F.save).then(r => {
      if (r.ok) { dirty = false; location.href = location.pathname + "?note=Saved+classes.yaml."; }
      else { errors = r.errors; showErrors(); }
    });
  }

  // a group shares one SAM run: its classes are objects with the same prompts (D18)
  function setGroup(i, g) {
    const c = rows[i];
    newGroupRow = -1;
    if (g === "\u0000new") { newGroupRow = i; return changed(true); }
    c.sam_group = g || undefined;
    const mate = rows.find((m, j) => j !== i && g && m.sam_group === g);
    if (mate) { c.kind = "object"; c.prompts = mate.prompts.slice(); }
    changed(true);
  }
  function setPrompts(i, prompts) {
    const g = rows[i].sam_group;
    rows.forEach((c, j) => { if (j === i || (g && c.sam_group === g)) c.prompts = prompts.slice(); });
    changed(true);
  }
  function rename(i, old, name) {
    rows[i].name = name;
    rows.forEach(c => { if (old && c.seg_as === old) c.seg_as = name; });  // seg_as follows
    changed(true);
  }
  function remove(i) {
    const [c] = rows.splice(i, 1);
    rows.forEach(m => { if (m.seg_as && m.seg_as === c.name && c.kind === "area") m.seg_as = undefined; });
    changed(true);
  }
  function move(i, by) {
    const j = i + by;
    if (j < 0 || j >= rows.length) return;
    [rows[i], rows[j]] = [rows[j], rows[i]];
    changed(true);
  }
  function add() {
    rows.push({id: ++maxId, name: "", kind: "object", prompts: []});
    changed(true);
    const inputs = root.querySelectorAll("input.cf-name");
    inputs[inputs.length - 1].focus();
  }

  function promptCell(c, i) {
    const chips = c.prompts.map((p, k) => el("span", {class: "chip"}, p,
      el("button", {type: "button", title: "Remove the prompt " + p,
                    onclick: () => setPrompts(i, c.prompts.filter((_, n) => n !== k))}, "×")));
    const input = el("input", {type: "text", class: "cf-prompt", placeholder: "add a prompt",
                               size: 12, "aria-label": "New prompt for " + (c.name || "this class")});
    const addIt = () => {
      const v = input.value.trim();
      if (v && !c.prompts.includes(v)) setPrompts(i, c.prompts.concat([v]));
    };
    input.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); addIt(); } });
    input.addEventListener("change", () => setTimeout(addIt));  // after focus moved on
    return el("td", {class: "cf-prompts"}, chips, input,
              c.sam_group && rows.filter(m => m.sam_group === c.sam_group).length > 1
                ? el("small", {class: "hint"}, " shared by group " + c.sam_group) : null);
  }
  function groupCell(c, i) {
    if (newGroupRow === i) {
      const input = el("input", {type: "text", size: 10, placeholder: "group name"});
      input.addEventListener("change", () => setTimeout(() => setGroup(i, input.value.trim())));
      input.addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); input.blur(); } });
      setTimeout(() => input.focus());
      return el("td", null, input);
    }
    return el("td", null, select("cf-group", c.sam_group || "", [["", "—"], ...groups().map(g => [g, g]),
                                                     ["\u0000new", "new group…"]],
                                 v => setGroup(i, v), c.kind !== "object"));
  }
  function row(c, i) {
    const locked = F.used[String(c.id)] !== undefined;  // has results: removing keeps them
    const name = el("input", {type: "text", class: "cf-name", value: c.name, size: 16,
                              placeholder: "class name",
                              title: locked ? "This class has SAM results: a new name keeps its id " + c.id + " and its saved masks (D39). Older exports keep the old name." : null});
    let before = c.name;  // seg_as names that pointed at the old name follow the new one
    name.addEventListener("focus", () => { before = c.name; });
    name.addEventListener("input", () => { c.name = name.value; changed(false); });
    // after the focus moved on, so the redraw keeps the field just clicked (render)
    name.addEventListener("change", () => setTimeout(() => rename(i, before, name.value.trim())));
    return el("tr", {"data-row": i},
      el("td", null, String(c.id)),
      el("td", null, el("span", {class: "swatch", style: "background: " + colour(c.id)})),
      el("td", null, name, locked ? el("small", {class: "lock"}, "has results: a rename keeps id " + c.id) : null),
      el("td", null, select("cf-kind", c.kind, F.kinds.map(k => [k, k]), v => {
        c.kind = v;
        if (v === "area") { c.sam_group = undefined; c.seg_as = undefined; }
        changed(true);
      }, !!c.sam_group)),
      promptCell(c, i),
      groupCell(c, i),
      el("td", null, select("cf-segas", c.seg_as || "", [["", "—"], ...areaNames().filter(n => n !== c.name).map(n => [n, n])],
                            v => { c.seg_as = v || undefined; changed(true); }, c.kind !== "object")),
      el("td", {class: "cf-actions"},
        el("button", {type: "button", title: "Move up (areas higher in the list win)", onclick: () => move(i, -1), disabled: i === 0}, "↑"),
        el("button", {type: "button", title: "Move down", onclick: () => move(i, 1), disabled: i === rows.length - 1}, "↓"),
        el("button", {type: "button", title: "Remove this class" + (locked ? " (its saved results are kept)" : ""),
                      onclick: () => remove(i)}, "Remove")),
    );
  }
  function settingsBlock() {
    const model = select("cf-model", settings.model, F.models.map(m => [m, m]), v => { settings.model = v; changed(false); });
    const fields = SETTINGS.map(([k, label, help, step, min, max]) => {
      const input = el("input", {type: "number", step, min, max, value: settings[k] == null ? "" : settings[k],
                                 "data-setting": k});
      input.addEventListener("input", () => {
        settings[k] = input.value === "" ? null : Number(input.value);
        changed(false);
      });
      return el("label", {"data-setting": k}, el("span", null, label), input, el("small", null, help));
    });
    return el("fieldset", {class: "cf-settings"}, el("legend", null, "SAM settings"),
      el("label", {"data-setting": "model"}, el("span", null, "SAM model"), model,
         el("small", null, "sam3 (default, D10) or sam3.1")), fields);
  }
  function render() {  // redraws everything; the focused field gets the focus back
    const a = document.activeElement;
    const tr = root.contains(a) ? a.closest("tr[data-row]") : null;
    const sel = root.contains(a) && a.classList.length ?
      (tr ? 'tr[data-row="' + tr.dataset.row + '"] ' : "") + "." + a.classList[0] : null;
    root.replaceChildren(
      el("table", {class: "classes cf-table"},
        el("thead", null, el("tr", null, ["Id", "Colour", "Name", "Kind", "SAM prompts", "Group",
                                          "Pixels count as", ""].map(h => el("th", null, h)))),
        el("tbody", null, rows.map(row))),
      el("p", null, el("button", {type: "button", onclick: add}, "+ Add class"),
         el("small", {class: "hint"}, " next id: " + (maxId + 1))),
      settingsBlock(),
      el("ul", {class: "error cf-errors", hidden: true}),
      el("p", null, el("button", {type: "button", class: "cf-save", onclick: save}, "Save"), " ",
         el("span", {class: "cf-status hint"})));
    if (sel) root.querySelector(sel)?.focus();
    showErrors();
  }
  function status() {
    const s = root.querySelector(".cf-status"), b = root.querySelector(".cf-save");
    b.disabled = !dirty || errors.length > 0;
    s.textContent = errors.length ? "Fix the problems above to save." :
      dirty ? "Unsaved changes." : "Saved; matches classes.yaml.";
  }
  function showErrors() {
    root.querySelectorAll(".bad").forEach(e => e.classList.remove("bad"));
    const ul = root.querySelector(".cf-errors");
    ul.replaceChildren(...errors.map(e => {
      let where = "";
      if (e.row != null) {
        root.querySelector('tr[data-row="' + e.row + '"]')?.classList.add("bad");
        where = "Row " + (e.row + 1) + (rows[e.row] && rows[e.row].name ? " (" + rows[e.row].name + ")" : "") + ": ";
      } else if (e.setting) {
        root.querySelector('label[data-setting="' + e.setting + '"]')?.classList.add("bad");
        const s = SETTINGS.find(x => x[0] === e.setting);
        where = (s ? s[1] : e.setting === "model" ? "SAM model" : "Setting " + e.setting) + ": ";
      }
      return el("li", null, where + e.msg);
    }));
    ul.hidden = !errors.length;
    status();
  }
  window.addEventListener("beforeunload", e => { if (dirty) e.preventDefault(); });
  render();
})();
