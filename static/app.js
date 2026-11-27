// Small progressive enhancements. The site works without this file.
(function () {
  // Ask before anything that cannot be undone (delete). Runs before the progress handler below.
  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (e) => {
      if (e.submitter && "noConfirm" in e.submitter.dataset) return;  // e.g. "Move selected" in the same form
      if (!window.confirm(form.dataset.confirm)) { e.preventDefault(); e.stopImmediatePropagation(); }
    });
  });

  // Whole table row is clickable (not only the id). Links, buttons and checkboxes inside keep working;
  // Ctrl/Cmd-click or the middle button opens the invoice in a new tab.
  document.querySelectorAll("select[data-autosubmit]").forEach((sel) => {
    sel.addEventListener("change", () => sel.form.submit());  // dashboard filters apply at once
  });

  document.querySelectorAll("tr[data-href], li[data-href]").forEach((tr) => {
    tr.tabIndex = 0;
    const open = (newTab) => { if (newTab) window.open(tr.dataset.href, "_blank"); else window.location = tr.dataset.href; };
    tr.addEventListener("click", (e) => {
      if (e.target.closest("a, button, input, label, select, textarea")) return;
      if (window.getSelection && String(window.getSelection()).length) return;  // selecting text to copy
      open(e.ctrlKey || e.metaKey);
    });
    tr.addEventListener("auxclick", (e) => { if (e.button === 1 && !e.target.closest("a")) open(true); });
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(false); });
  });

  // Invoices list: select rows, then "Delete selected".
  const pickAll = document.getElementById("pick-all");
  const bulkBtn = document.getElementById("bulk-delete-btn");
  if (bulkBtn) {
    const picks = () => [...document.querySelectorAll('input[name="ids"][form="bulk-delete"]')];
    const refresh = () => {
      const n = picks().filter((c) => c.checked).length;
      bulkBtn.disabled = n === 0;
      bulkBtn.textContent = n ? `Delete selected (${n})` : "Delete selected";
    };
    picks().forEach((c) => c.addEventListener("change", refresh));
    if (pickAll) pickAll.addEventListener("change", () => { picks().forEach((c) => { c.checked = pickAll.checked; }); refresh(); });
    refresh();
  }

  // Show progress while the (OCR) work runs.
  document.querySelectorAll("form").forEach((form) => {
    form.addEventListener("submit", (e) => {
      const btn = e.submitter || form.querySelector('button[type="submit"]');
      if (!btn) return;
      if (btn.name) { btn.classList.add("busy"); return; }  // named buttons (Approve / Reject) must stay enabled to send their value
      btn.disabled = true;
      btn.textContent = form.enctype === "multipart/form-data" ? "Reading invoice..." : "Please wait...";
    });
  });

  const saved = {  // small choices remembered in this browser (rows per page, all columns)
    get(key) { try { return localStorage.getItem(key); } catch (e) { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch (e) { /* private window */ } },
  };

  // Long tables (data-pages): one page of rows at a time, "1–20 of 57", Previous / Next and Rows per page
  // below the table. Rows a filter leaves out have data-out="1".
  const PAGE_SIZES = [10, 20, 50, 100];
  const pagers = new Map();
  document.querySelectorAll("table[data-pages]").forEach((table) => {
    const body = table.tBodies[0];
    if (!body) return;
    const wrap = table.closest(".sheet-wrap, .table-wrap") || table;
    const make = (tag, cls, text) => { const el = document.createElement(tag); el.className = cls; if (text) el.textContent = text; return el; };
    const bar = make("div", "pager");
    const info = make("span", "muted small");
    const nav = make("span", "pager-nav");
    const prev = make("button", "btn secondary", "← Previous");
    const where = make("span", "small");
    const next = make("button", "btn secondary", "Next →");
    prev.type = next.type = "button";
    nav.append(prev, where, next);
    const label = make("label", "small muted", "Rows per page ");
    const sizeBox = document.createElement("select");
    PAGE_SIZES.forEach((n) => sizeBox.add(new Option(String(n), String(n))));
    label.appendChild(sizeBox);
    bar.append(info, nav, label);
    wrap.after(bar);
    let size = Number(saved.get("invoagent.rows"));
    if (!PAGE_SIZES.includes(size)) size = 20;
    sizeBox.value = String(size);
    let page = 1;

    const render = () => {
      const all = [...body.rows];
      const rows = all.filter((r) => r.dataset.out !== "1");
      const pages = Math.max(1, Math.ceil(rows.length / size));
      page = Math.min(Math.max(page, 1), pages);
      const first = (page - 1) * size;
      const shown = new Set(rows.slice(first, first + size));
      let n = 0;
      all.forEach((r) => {
        r.hidden = !shown.has(r);
        if (!r.hidden) r.classList.toggle("alt", n++ % 2 === 1);
      });
      info.textContent = rows.length ? `${first + 1}–${Math.min(first + size, rows.length)} of ${rows.length}` : "No rows";
      where.textContent = `Page ${page} of ${pages}`;
      prev.disabled = page === 1;
      next.disabled = page === pages;
      bar.hidden = wrap.hidden || rows.length <= PAGE_SIZES[0];  // the Data sheet tab not shown: no pager
    };
    const go = (to) => {
      page = to;
      render();
      wrap.scrollTop = 0;
      if (wrap.getBoundingClientRect().top < 0) wrap.scrollIntoView({ block: "start" });
    };
    prev.addEventListener("click", () => go(page - 1));
    next.addEventListener("click", () => go(page + 1));
    sizeBox.addEventListener("change", () => {
      size = Number(sizeBox.value);
      saved.set("invoagent.rows", String(size));
      go(1);
    });
    pagers.set(table, { refresh(fromStart) { if (fromStart) page = 1; render(); } });
    render();
  });

  // Wide tables: the main columns; "Show all columns" adds the others (remembered in this browser).
  document.querySelectorAll("input[data-all-columns]").forEach((box) => {
    const tables = document.querySelectorAll(box.dataset.allColumns);
    const key = "invoagent.allcolumns." + box.dataset.key;
    box.checked = saved.get(key) === "1";
    const apply = () => tables.forEach((t) => t.classList.toggle("main-only", !box.checked));
    box.addEventListener("change", () => { saved.set(key, box.checked ? "1" : "0"); apply(); });
    box.closest("label").hidden = false;
    apply();
  });

  // A wide table also gets a scroll bar above it: it can be scrolled sideways without going down to its end.
  document.querySelectorAll(".sheet-wrap, .table-wrap").forEach((wrap) => {
    const top = document.createElement("div");
    top.className = "top-scroll";
    top.setAttribute("aria-hidden", "true");
    const inner = document.createElement("div");
    top.appendChild(inner);
    wrap.before(top);
    const fit = () => {
      inner.style.width = wrap.scrollWidth + "px";
      top.hidden = wrap.hidden || wrap.scrollWidth <= wrap.clientWidth + 1;
    };
    top.addEventListener("scroll", () => { if (wrap.scrollLeft !== top.scrollLeft) wrap.scrollLeft = top.scrollLeft; });
    wrap.addEventListener("scroll", () => { if (top.scrollLeft !== wrap.scrollLeft) top.scrollLeft = wrap.scrollLeft; });
    if (window.ResizeObserver) {
      const watch = new ResizeObserver(fit);
      watch.observe(wrap);
      const table = wrap.querySelector("table");
      if (table) watch.observe(table);
    }
    window.addEventListener("resize", fit);
    fit();
  });

  // Side panel (.drawer): slides in from the right over the page; ×, Esc or a click beside it closes it.
  const backdrop = document.createElement("div");
  backdrop.className = "drawer-backdrop";
  backdrop.hidden = true;
  document.body.appendChild(backdrop);
  let openDrawer = null, opener = null;
  const closeDrawer = () => {
    if (!openDrawer) return;
    openDrawer.classList.remove("open");
    backdrop.hidden = true;
    openDrawer = null;
    if (opener) opener.focus();
  };
  const showDrawer = (drawer, from) => {
    if (openDrawer && openDrawer !== drawer) closeDrawer();
    drawer.hidden = false;
    void drawer.offsetWidth;  // start the slide from outside the window
    drawer.classList.add("open");
    backdrop.hidden = false;
    openDrawer = drawer;
    opener = from || null;
    drawer.focus();
  };
  backdrop.addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  document.querySelectorAll(".drawer-close").forEach((btn) => { btn.hidden = false; btn.addEventListener("click", closeDrawer); });
  document.querySelectorAll("[data-drawer]").forEach((btn) => {
    const drawer = document.getElementById(btn.dataset.drawer);
    if (!drawer) return;
    btn.hidden = false;
    btn.addEventListener("click", () => showDrawer(drawer, btn));
    if (window.location.hash === "#" + drawer.id) showDrawer(drawer, btn);  // "Items" on the Data sheet without this script
  });

  // Data sheet: "3 items" in an invoice's row shows its line items in the side panel.
  const sheetDrawer = document.getElementById("sheet-drawer");
  if (sheetDrawer) {
    const itemsTable = document.querySelector("#sheet-items table");
    const itemCols = itemsTable ? [...itemsTable.querySelectorAll("thead th")].map((th) => th.dataset.col) : [];
    const invoiceCols = [...document.querySelectorAll("#sheet-invoices thead th")].map((th) => th.dataset.col);
    const SHOW = [["line_no", "#"], ["description", "Description"], ["quantity", "Qty"], ["unit_price", "Rate"], ["amount", "Amount"]];
    document.querySelectorAll("a[data-items]").forEach((link) => {
      link.addEventListener("click", (e) => {
        if (e.ctrlKey || e.metaKey || e.shiftKey || !itemsTable) return;  // the invoice page in a new tab
        e.preventDefault();
        const id = link.dataset.items;
        const row = link.closest("tr");
        const text = (col) => { const i = invoiceCols.indexOf(col); return i < 0 ? "" : (row.children[i].dataset.v || ""); };
        document.getElementById("sheet-drawer-title").textContent =
          `Line items · invoice ${text("invoice_number") || "#" + id}`;
        document.getElementById("sheet-drawer-sub").textContent =
          [text("seller_name"), text("invoice_date"), text("total") && `total ${Number(text("total")).toLocaleString(undefined, { minimumFractionDigits: 2 })} ${text("currency")}`]
            .filter(Boolean).join(" · ");
        const table = document.createElement("table");
        table.className = "items-table";
        const head = table.createTHead().insertRow();
        SHOW.forEach(([, title], i) => { const th = document.createElement("th"); th.textContent = title; if (i > 1) th.className = "num"; head.appendChild(th); });
        const tbody = table.createTBody();
        itemsTable.querySelectorAll(`tbody tr[data-invoice="${CSS.escape(id)}"]`).forEach((src) => {
          const tr = tbody.insertRow();
          SHOW.forEach(([col], i) => {
            const from = src.children[itemCols.indexOf(col)];
            const td = tr.insertCell();
            td.innerHTML = from ? from.innerHTML : "";
            if (i > 1) td.className = "num";
          });
        });
        const body = document.getElementById("sheet-drawer-body");
        body.textContent = "";
        const open = document.createElement("a");
        open.href = link.href.split("#")[0];
        open.className = "btn secondary";
        open.textContent = "Open the invoice";
        const wrap = document.createElement("div");
        wrap.className = "table-wrap";
        wrap.appendChild(table);
        body.append(wrap, open);
        showDrawer(sheetDrawer, link);
      });
    });
  }

  // Data sheet: sheet tabs, filter, sort by column, totals per currency.
  const sheetFilter = document.getElementById("sheet-filter");
  if (sheetFilter) {
    const reviewBox = document.getElementById("sheet-review");
    const summary = document.getElementById("sheet-summary");
    const tabs = document.querySelectorAll(".sheet-tab");
    let active = document.querySelector(".sheet-wrap:not([hidden])") || document.getElementById("sheet-invoices");
    const money = (n) => n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

    const update = () => {
      const q = sheetFilter.value.trim().toLowerCase();
      const onlyReview = reviewBox.checked && active.id === "sheet-invoices";
      const table = active.querySelector("table");
      const cols = [...table.querySelectorAll("thead th")].map((th) => th.dataset.col);
      const moneyCols = (table.dataset.money || "").split(",").filter((c) => cols.includes(c));
      const sums = {};
      let shown = 0;
      table.querySelectorAll("tbody tr").forEach((tr) => {
        const visible = (!q || tr.textContent.toLowerCase().includes(q)) && (!onlyReview || tr.dataset.review === "1");
        tr.dataset.out = visible ? "" : "1";  // the pager shows one page of the rows left
        if (!visible) return;
        shown += 1;
        const cur = tr.dataset.currency || "no currency";
        sums[cur] = sums[cur] || {};
        moneyCols.forEach((col) => {
          const v = parseFloat(tr.children[cols.indexOf(col)].dataset.v);
          if (!isNaN(v)) sums[cur][col] = (sums[cur][col] || 0) + v;
        });
      });
      const parts = Object.entries(sums)
        .map(([cur, s]) => `${cur}: ` + moneyCols.filter((c) => c in s).map((c) => `${c} ${money(s[c])}`).join(", "))
        .filter((p) => p.includes(" "));
      summary.textContent = `${shown} row${shown === 1 ? "" : "s"}` + (parts.length ? `  ·  Totals by currency → ${parts.join("  |  ")}` : "");
      const pager = pagers.get(table);
      if (pager) pager.refresh(true);
    };

    tabs.forEach((tab) => tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.toggle("active", t === tab));
      document.querySelectorAll(".sheet-wrap").forEach((w) => { w.hidden = w.id !== tab.dataset.sheet; });
      active = document.getElementById(tab.dataset.sheet);
      pagers.forEach((pager) => pager.refresh());
      update();
    }));

    const isNumber = (s) => /^-?\d+(\.\d+)?$/.test(s);
    document.querySelectorAll("table.sheet").forEach((table) => {
      table.querySelectorAll("thead th").forEach((th, idx) => {
        th.title = "Click to sort";
        th.addEventListener("click", () => {
          const asc = th.dataset.dir !== "asc";
          table.querySelectorAll("thead th").forEach((h) => { delete h.dataset.dir; });
          th.dataset.dir = asc ? "asc" : "desc";
          const body = table.tBodies[0];
          const rows = [...body.rows];
          rows.sort((a, b) => {
            const x = a.children[idx].dataset.v, y = b.children[idx].dataset.v;
            if (x === "" || y === "") return x === y ? 0 : x === "" ? 1 : -1;  // nulls always last
            const cmp = isNumber(x) && isNumber(y) ? parseFloat(x) - parseFloat(y)
              : x.localeCompare(y, undefined, { numeric: true, sensitivity: "base" });
            return asc ? cmp : -cmp;
          });
          rows.forEach((r) => body.appendChild(r));
          const pager = pagers.get(table);
          if (pager) pager.refresh(true);
        });
      });
    });

    sheetFilter.addEventListener("input", update);
    reviewBox.addEventListener("change", update);
    update();
  }

  // View JSON: the invoice's data shown on the page first; Download JSON inside it saves the file
  const viewJson = document.getElementById("view-json");
  const jsonView = document.getElementById("json-view");
  if (viewJson && jsonView) {
    viewJson.addEventListener("click", () => {
      const pre = document.getElementById("json-text");
      if (!pre.textContent) {
        const text = document.getElementById("result-json").textContent;
        try {
          pre.textContent = JSON.stringify(JSON.parse(text), null, 2);
        } catch (e) {
          pre.textContent = text;
        }
      }
      jsonView.hidden = !jsonView.hidden;
      viewJson.textContent = jsonView.hidden ? "View JSON" : "Hide JSON";
      viewJson.setAttribute("aria-expanded", String(!jsonView.hidden));
      if (!jsonView.hidden) jsonView.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  const downloadBtn = document.getElementById("download-json");
  if (downloadBtn) {
    downloadBtn.addEventListener("click", () => {
      const text = document.getElementById("result-json").textContent;
      const blob = new Blob([text], { type: "application/json" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "invoice.json";
      link.click();
    });
  }
})();

// Photos are made smaller in the browser before they are sent: at most 3000 px, a little more than the app reads
// them at, so nothing is lost for the extraction. A 3-5 MB phone photo becomes about 0.5 MB, which
// uploads several times faster on a mobile connection. PDFs, ZIPs, TIFFs and images that are already
// small are sent as they are; if anything goes wrong the original files are sent.
(function () {
  const MAX_SIDE = 3000, QUALITY = 0.92, SMALL_BYTES = 1200 * 1024;

  async function shrink(file) {
    if (!/^image\/(jpeg|png|webp|bmp)$/i.test(file.type) || !window.createImageBitmap) return file;
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });  // upright, as the phone shows it
    const scale = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
    if (scale === 1 && file.size <= SMALL_BYTES) { if (bmp.close) bmp.close(); return file; }
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bmp.width * scale);
    canvas.height = Math.round(bmp.height * scale);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff";  // a transparent screenshot gets a white page, not a black one
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
    if (bmp.close) bmp.close();
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", QUALITY));
    if (!blob || blob.size >= file.size) return file;
    return new File([blob], file.name.replace(/\.[^./]+$/, "") + ".jpg", { type: "image/jpeg", lastModified: file.lastModified });
  }
  window.invoagentShrinkPhoto = shrink;  // used by the tests in the browser

  document.addEventListener("submit", async (e) => {
    const form = e.target;
    const inputs = form.querySelectorAll('input[type="file"]');
    if (!inputs.length || form.dataset.shrunk || form.id === "uploader" || typeof DataTransfer === "undefined") return;
    e.preventDefault();
    try {
      for (const input of inputs) {
        const dt = new DataTransfer();
        for (const f of input.files) dt.items.add(await shrink(f).catch(() => f));
        input.files = dt.files;
      }
    } catch (err) { /* send the files as they are */ }
    form.dataset.shrunk = "1";
    form.submit();  // does not fire 'submit' again
  }, true);
})();

// Invoices list: "Move selected" sends the picked rows to a company, "Set project" puts them in a project
// (no delete question for either).
(function () {
  const buttons = [[document.getElementById("bulk-move-btn"), "Move selected"],
                   [document.getElementById("bulk-project-btn"), "Set project"]].filter(([btn]) => btn);
  if (!buttons.length) return;
  const picks = () => [...document.querySelectorAll('input[name="ids"][form="bulk-delete"]')];
  const refresh = () => {
    const n = picks().filter((c) => c.checked).length;
    for (const [btn, text] of buttons) {
      btn.disabled = n === 0;
      btn.textContent = n ? `${text} (${n})` : text;
    }
  };
  document.addEventListener("change", (e) => { if (e.target.matches('input[name="ids"], #pick-all')) refresh(); });
  refresh();
})();

// Add invoices page: one place for files, a whole folder, phone photos and pasted pictures. Everything is
// listed first (thumbnail, size, remove), photos are made smaller, then it is all sent with a progress bar;
// a pasted picture is sent straight away.
// Without this script the form still works: the chosen files are sent as they are.
(function () {
  const form = document.getElementById("uploader");
  if (!form) return;
  const $ = (id) => document.getElementById(id);
  const zone = $("drop-zone"), list = $("upload-list"), box = $("upload-selected"), count = $("upload-count");
  const send = $("upload-send"), msg = $("upload-msg"), bar = $("upload-progress"), company = $("upload-company");
  const fill = bar.querySelector("span");
  const maxMb = Number(form.dataset.maxMb || 300);
  const SUPPORTED = /\.(pdf|zip|png|jpe?g|jpe|jfif|tiff?|bmp|webp)$/i;
  const PASTE_TYPES = { "image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/bmp": "bmp", "image/tiff": "tif" };
  const EXT_OF_TYPE = { ...PASTE_TYPES, "image/pjpeg": "jpg", "application/pdf": "pdf", "application/zip": "zip",
                        "application/x-zip-compressed": "zip" };
  // A picture dragged from a web page (WhatsApp Web) has no extension in its name: it is still a JPEG or a PNG,
  // so it gets the extension of its type. null: a file that cannot be read.
  const readable = (file) => {
    if (SUPPORTED.test(file.name)) return file;
    const ext = EXT_OF_TYPE[file.type];
    if (!ext) return null;
    const base = (file.name || "").replace(/\.[^./\\]*$/, "").trim() || `picture-${Date.now()}`;
    return new File([file], `${base}.${ext}`, { type: file.type, lastModified: file.lastModified });
  };
  const PREVIEW = /^image\/(png|jpeg|webp|bmp|gif)$/i;
  let items = [], busy = false;

  const size = (n) => (n >= 1048576 ? (n / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1024)) + " KB");
  const say = (text, bad) => { msg.textContent = text || ""; msg.className = "upload-msg" + (bad ? " bad" : ""); };
  const total = () => items.reduce((sum, it) => sum + it.file.size, 0);

  function render() {
    list.textContent = "";
    items.forEach((it) => {
      const li = document.createElement("li");
      let thumb;
      if (it.url) {
        thumb = document.createElement("img");
        thumb.src = it.url;
        thumb.alt = "";
      } else {
        thumb = document.createElement("span");
        thumb.textContent = (it.file.name.split(".").pop() || "file").toUpperCase().slice(0, 4);
      }
      thumb.className = "upload-thumb";
      const name = document.createElement("span");
      name.className = "upload-name";
      name.textContent = it.label;
      name.title = it.label;
      const bytes = document.createElement("span");
      bytes.className = "upload-size";
      bytes.textContent = size(it.file.size);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "upload-remove";
      remove.textContent = "×";
      remove.setAttribute("aria-label", "Remove " + it.label);
      remove.disabled = busy;
      remove.addEventListener("click", () => {
        if (it.url) URL.revokeObjectURL(it.url);
        items = items.filter((x) => x !== it);
        say("");
        render();
      });
      li.append(thumb, name, bytes, remove);
      list.appendChild(li);
    });
    box.hidden = !items.length;
    count.textContent = items.length ? `${items.length} file${items.length > 1 ? "s" : ""} · ${size(total())}` : "";
    send.disabled = busy || !items.length || total() > maxMb * 1048576;
    if (!busy) send.textContent = items.length > 1 ? `Upload and extract ${items.length} files` : "Upload and extract";
    if (total() > maxMb * 1048576) say(`Together these files are larger than ${maxMb} MB: upload them in parts.`, true);
  }

  function add(files) {
    let added = 0;
    const skipped = [];
    for (const original of files) {
      const file = readable(original);
      if (!file) { skipped.push(original.name || "(no name)"); continue; }
      const key = [original.webkitRelativePath || original.name, original.size, original.lastModified].join("|");
      if (items.some((it) => it.key === key)) continue;
      items.push({ file, key, label: original.webkitRelativePath || file.name,
                   url: PREVIEW.test(file.type) ? URL.createObjectURL(file) : null });
      added += 1;
    }
    if (skipped.length) {
      const names = skipped.slice(0, 3).join(", ") + (skipped.length > 3 ? " ..." : "");
      say(`${skipped.length} file${skipped.length > 1 ? "s" : ""} skipped (${names}): only PDF, photos (JPG, PNG, ` +
          "WEBP, BMP, TIFF) and ZIP files can be read.", true);
    }
    else if (added) say("");
    render();
  }

  function companyFromFolder(folder) {  // a folder named after the company fills an empty Company box
    if (company && folder && !company.value.trim()) {
      company.value = folder;
      say(`Company set to "${folder}" from the folder name. Change it if needed.`);
    }
  }

  const pick = (id, onPick) => $(id).addEventListener("change", (e) => {
    const files = [...e.target.files];
    if (onPick) onPick(files);
    add(files);
    e.target.value = "";  // the same file can be chosen again
  });
  pick("pick-files");
  pick("pick-photo", () => { $("photo-label").textContent = "Take another photo"; });
  pick("pick-folder", (files) => { if (files.length) companyFromFolder(files[0].webkitRelativePath.split("/")[0]); });
  if (window.matchMedia && matchMedia("(pointer: coarse)").matches) {  // a phone: the camera comes first
    $("folder-label").hidden = true;
    $("photo-label").className = "btn";
    zone.querySelector('label[for="pick-files"]').className = "btn secondary";
    zone.parentNode.insertBefore($("phone-camera"), zone);
    form.querySelectorAll("[data-phone]").forEach((el) => { el.textContent = el.dataset.phone; });
  }

  // drag and drop anywhere on the page, folders included
  async function readEntry(entry, out) {
    if (entry.isFile) {
      out.push(await new Promise((ok, fail) => entry.file(ok, fail)));
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (;;) {
        const batch = await new Promise((ok, fail) => reader.readEntries(ok, fail));
        if (!batch.length) break;
        for (const child of batch) await readEntry(child, out);
      }
    }
  }
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  let depth = 0;
  document.addEventListener("dragenter", (e) => { if (hasFiles(e)) { depth += 1; zone.classList.add("over"); } });
  document.addEventListener("dragleave", (e) => { if (hasFiles(e) && --depth <= 0) { depth = 0; zone.classList.remove("over"); } });
  document.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  document.addEventListener("drop", async (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    zone.classList.remove("over");
    if (busy) return;
    const entries = [...(e.dataTransfer.items || [])].map((it) => it.webkitGetAsEntry && it.webkitGetAsEntry()).filter(Boolean);
    const folder = entries.find((en) => en.isDirectory);
    if (!folder) { add(e.dataTransfer.files); return; }
    say("Reading the folder...");
    const out = [];
    try {
      for (const en of entries) await readEntry(en, out);
    } catch (err) {
      say("Could not read the folder: " + err, true);
    }
    companyFromFolder(folder.name);
    add(out.map((f) => f));
  });

  // a copied picture (WhatsApp, Gallery, a screenshot): Ctrl+V anywhere, or the Paste button
  const stamp = () => new Date().toISOString().replace(/\D/g, "").slice(0, 14);
  const pasted = (blob, n) => new File([blob], `pasted-${stamp()}${n ? "-" + n : ""}.${PASTE_TYPES[blob.type]}`, { type: blob.type });
  function sendPasted(files) {  // a pasted picture is read straight away (with anything already listed)
    if (busy) return;
    add(files);
    say("Picture pasted. Reading it now...");
    if (form.requestSubmit) form.requestSubmit(); else send.click();
  }
  document.addEventListener("paste", (e) => {
    const blobs = [...((e.clipboardData && e.clipboardData.items) || [])]
      .filter((it) => it.kind === "file" && PASTE_TYPES[it.type]).map((it) => it.getAsFile()).filter(Boolean);
    if (!blobs.length) return;  // text into the Company box stays text
    e.preventDefault();
    sendPasted(blobs.map((b, n) => pasted(b, n)));
  });
  const pasteBtn = $("paste-btn");
  if (navigator.clipboard && navigator.clipboard.read) {  // HTTPS pages and localhost only
    pasteBtn.hidden = false;
    pasteBtn.addEventListener("click", async () => {
      try {
        const found = [];
        for (const item of await navigator.clipboard.read()) {
          const type = item.types.find((t) => PASTE_TYPES[t]);
          if (type) found.push(pasted(await item.getType(type), found.length));
        }
        if (found.length) sendPasted(found);
        else say("No picture copied. Copy an invoice picture first.", true);
      } catch (err) {
        say("The browser did not allow reading the clipboard: press Ctrl+V instead.", true);
      }
    });
  }

  $("upload-clear").addEventListener("click", () => {
    items.forEach((it) => { if (it.url) URL.revokeObjectURL(it.url); });
    items = [];
    say("");
    render();
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (busy || !items.length || total() > maxMb * 1048576) return;
    busy = true;
    render();
    send.textContent = "Preparing photos...";
    const data = new FormData();
    data.append("company", company ? company.value : "");
    const project = $("upload-project");
    if (project) data.append("project", project.value);
    for (const it of items) {
      let file = it.file;
      if (window.invoagentShrinkPhoto) {
        try { file = await window.invoagentShrinkPhoto(file); } catch (err) { /* send it as it is */ }
      }
      data.append("files", file, file.name);
    }
    const many = items.length > 1;
    const xhr = new XMLHttpRequest();
    xhr.open("POST", form.action);
    xhr.setRequestHeader("X-Upload", "1");  // the answer is JSON {next: url}, so its message is kept for that page
    bar.hidden = false;
    fill.style.width = "0%";
    xhr.upload.onprogress = (ev) => {
      if (!ev.lengthComputable) return;
      const pct = Math.round((100 * ev.loaded) / ev.total);
      fill.style.width = pct + "%";
      send.textContent = `Uploading... ${pct}%`;
    };
    xhr.upload.onload = () => {
      bar.classList.add("working");
      send.textContent = many ? "Adding to the queue..." : "Reading invoice...";
      say(many ? "" : "Reading the invoice. A handwritten bill can take up to a minute.");
    };
    xhr.onload = () => {
      let next = null;
      try { next = JSON.parse(xhr.responseText).next; } catch (err) { /* not JSON, e.g. the sign-in page */ }
      window.location.href = next || xhr.responseURL || form.action;
    };
    xhr.onerror = () => {
      busy = false;
      bar.hidden = true;
      bar.classList.remove("working");
      render();
      say("The upload did not reach the app (network problem). Check the connection and press Upload again.", true);
    };
    xhr.send(data);
  });

  render();
})();
