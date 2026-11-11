// Small progressive enhancements. The site works without this file.
(function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file");
  const fileName = document.getElementById("filename");

  if (dropzone && fileInput) {
    const show = (files) => {
      if (!fileName) return;
      if (!files || !files.length) fileName.textContent = "";
      else if (files.length === 1) fileName.textContent = files[0].name;
      else fileName.textContent = files.length + " files selected (they will be processed in the background)";
    };
    fileInput.addEventListener("change", () => show(fileInput.files));
    ["dragenter", "dragover"].forEach((ev) =>
      dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) =>
      dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("over"); }));
    dropzone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        show(e.dataTransfer.files);
      }
    });
  }

  // Ask before anything that cannot be undone (delete). Runs before the progress handler below.
  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (e) => {
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

  // Phone camera over Wi-Fi (DroidCam / IP Webcam): connect, live preview, find phones.
  const phoneForm = document.getElementById("phone-form");
  if (phoneForm) {
    const byId = (id) => document.getElementById(id);
    const msg = byId("phone-msg"), box = byId("phone-preview-box"), img = byId("phone-preview");
    const overlay = byId("camera-overlay"), big = byId("phone-big"), note = byId("camera-note");
    const found = byId("phone-found"), connectBtn = byId("phone-connect");
    const streamUrl = () => "/camera/phone/stream?t=" + Date.now();
    const pcMsg = byId("pc-msg");
    const say = (text, bad, target) => {
      [msg, pcMsg].forEach((el) => { if (el) el.textContent = ""; });
      const el = target || msg;
      el.textContent = text;
      el.className = "small " + (bad ? "missing" : "muted");
      note.textContent = text;
      note.className = "camera-note" + (bad ? " warn" : "");
    };
    // Only one <img> streams at a time, so the PC relays a single video feed.
    const showPreview = () => { big.src = ""; img.src = streamUrl(); box.hidden = false; };
    const openLarge = () => {
      img.src = "";
      big.src = streamUrl();
      overlay.hidden = false;
      document.body.classList.add("no-scroll");
    };
    const closeLarge = () => {
      if (document.fullscreenElement) document.exitFullscreen();
      overlay.hidden = true;
      document.body.classList.remove("no-scroll");
      showPreview();
    };
    const disconnect = async () => {
      if (document.fullscreenElement) document.exitFullscreen();
      img.src = "";
      big.src = "";
      overlay.hidden = true;
      box.hidden = true;
      document.body.classList.remove("no-scroll");
      await fetch("/camera/phone/disconnect", { method: "POST" });
      say("Disconnected. The phone is free for other apps.");
    };
    byId("phone-enlarge").addEventListener("click", openLarge);
    byId("camera-close").addEventListener("click", closeLarge);
    byId("camera-disconnect").addEventListener("click", disconnect);
    byId("camera-fullscreen").addEventListener("click", () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (overlay.requestFullscreen) overlay.requestFullscreen();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !overlay.hidden && !document.fullscreenElement) closeLarge();
    });

    const connect = async (button = connectBtn, body = null, target = msg) => {
      const label = button.textContent;
      button.disabled = true;
      button.textContent = "Connecting...";
      say("Connecting to the camera...", false, target);
      try {
        const resp = await fetch("/camera/phone/connect", { method: "POST", body: body || new FormData(phoneForm) });
        const data = await resp.json();
        if (data.ok) {
          const size = `${data.width}×${data.height} video`;
          if (data.scrambled) say(data.warning, true, target);
          else if (data.source === "pc") say(`Connected to ${data.app} on this PC (${size}).`, false, target);
          else say(`Connected to ${data.app} at ${data.host}:${data.port} (${size}).`, false, target);
          box.hidden = false;
          openLarge();  // show the camera big right away
        } else {
          say(data.error, true, target);
          box.hidden = true;
        }
      } catch (err) {
        say("Could not reach this app: " + err, true, target);
      }
      button.disabled = false;
      button.textContent = label;
    };

    byId("pc-connect").addEventListener("click", (e) => {
      const body = new FormData();
      body.append("source", "pc");
      const index = byId("pc-index").value;
      if (index !== "") body.append("index", index);
      connect(e.currentTarget, body, pcMsg);
    });

    phoneForm.addEventListener("submit", (e) => { e.preventDefault(); connect(); });

    byId("phone-find").addEventListener("click", async (e) => {
      const btn = e.currentTarget;
      btn.disabled = true;
      say("Looking for DroidCam / IP Webcam on your network (a few seconds)...");
      found.innerHTML = "";
      try {
        const data = await (await fetch("/camera/phone/find")).json();
        if (!data.phones.length) {
          say("No phone found. Open DroidCam (or start the IP Webcam server) on the phone, check it is on the same Wi-Fi, and try again.", true);
        } else {
          say(`Found ${data.phones.length}. Click one to connect.`);
          data.phones.forEach((p) => {
            const li = document.createElement("li");
            const b = document.createElement("button");
            b.type = "button";
            b.className = "chip";
            b.textContent = `${p.app}: ${p.host}:${p.port}`;
            b.addEventListener("click", () => { byId("phone-host").value = p.host; byId("phone-port").value = p.port; connect(); });
            li.appendChild(b);
            found.appendChild(li);
          });
        }
      } catch (err) {
        say("Search failed: " + err, true);
      }
      btn.disabled = false;
    });

    byId("phone-disconnect").addEventListener("click", disconnect);

    if (phoneForm.dataset.connected === "1") showPreview();  // still connected from before
  }

  // Data sheet: sheet tabs, filter, sort by column, totals per currency.
  const sheetFilter = document.getElementById("sheet-filter");
  if (sheetFilter) {
    const reviewBox = document.getElementById("sheet-review");
    const summary = document.getElementById("sheet-summary");
    const tabs = document.querySelectorAll(".sheet-tab");
    let active = document.getElementById("sheet-invoices");
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
        tr.hidden = !visible;
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
      summary.textContent = `${shown} rows shown` + (parts.length ? `  ·  Totals by currency → ${parts.join("  |  ")}` : "");
    };

    tabs.forEach((tab) => tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.toggle("active", t === tab));
      document.querySelectorAll(".sheet-wrap").forEach((w) => { w.hidden = w.id !== tab.dataset.sheet; });
      active = document.getElementById(tab.dataset.sheet);
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
        });
      });
    });

    sheetFilter.addEventListener("input", update);
    reviewBox.addEventListener("change", update);
    update();
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
