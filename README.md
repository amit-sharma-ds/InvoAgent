<img src="static/logo.svg" width="72" alt="InvoAgent logo">

# InvoAgent

**Live:** https://invoagent.onrender.com (Render free plan: it sleeps after 15 minutes without visits and wakes
in about a minute; stored invoices are reset when it restarts or is redeployed; OCR of photos is slow on its 0.1 CPU).

Invoice data from PDFs, scans and phone photos. Extract structured data from invoices in any layout (text PDFs, scanned PDFs, scans, phone photos),
store every result in a database, review it in a table, and export it to Excel / CSV.
Built for volume: hundreds or thousands of invoices are processed in parallel in the background.

Labels are matched by **meaning**, not by position: `Seller`, `Vendor`, `Supplier`, `Sold by`, `Bill from`,
`From` all map to the same field (synonyms + typo-tolerant matching + sentence embeddings).

Works from:
- the web UI: one file (result shown immediately), many files or a ZIP (background batch),
- the **folder import command** for thousands of files,
- your phone's camera (browser capture, no app needed), an Android phone as IP camera, or a webcam/DroidCam,
- a JSON API for other company systems.

No model training is required. See [Do I need a training dataset?](#do-i-need-a-training-dataset).

### How fast (measured on the development PC)

The development PC is a 2-core / 4-thread Intel i3-3240 that also runs other programs. Time from upload
to finished result, one invoice at a time:

| What you upload | Time per invoice |
|---|---|
| **Digital PDF** (Tally or other accounting export, an emailed PDF: it has a text layer) | **under 1 s** (about 0.03 s each in bulk) |
| **Scanned PDF** (office scanner, a phone's *scan document* PDF: no text layer) | about **13 s** per page |
| **Image file** PNG / JPG / JPEG / TIFF / BMP / WEBP (a 300-DPI scan) | about **18 s** per page |
| **Phone photo** (browser camera on the *Phone camera* tab, or a photo from the gallery) | about **4-7 s** (a SuperStore-style page ~4 s, a dense Tally GST page ~5-7 s); sideways the same; upside down about 10 s |

- Phone photos are read by a small neural OCR model (PaddleOCR PP-OCRv6 tiny on ONNX Runtime) instead of
  Tesseract: on the same photos and the same busy PC, 22-32 s became 5-10 s, with more fields right (40 of 40
  instead of 35 of 40). A photo counts as a phone photo when it carries the camera's make/model (every phone
  camera adds it) or comes from the app's camera page. Scans and other images keep Tesseract.
- The file type does not change the speed: a PNG and a JPG of the same page take the same time. The number of
  pixels and the amount of text do.
- All times above were measured while another program on this PC used 65-83% of the CPU; on a free PC they
  are lower.
- The first invoice after starting the app takes 1-2 minutes longer while the label model loads (not on the
  server image, which has no model).
- A faster CPU or a server with more cores is the way to go much faster; see [Deploying online](#deploying-online-github--render).

---

## Using the app (left sidebar)

Every page has a **left sidebar** with one tab per job, so new features always get their own tab:

| Tab | What it is for |
|---|---|
| **Live / Demo switch** | Two separate workspaces, each with its own database and files (see below) |
| **Overview** | Counts (invoices, need review, approved, rejected, possible duplicates, failed), amounts per currency, this month, recent invoices |
| **Dashboard** | Invoice value by month (taxable value + GST stacked), top 10 vendors, totals; per currency, for the last 12 months, this / last financial year (Apr-Mar) or all years. Hover a bar for the numbers; "Show as a table" lists them |
| **Upload** | Drop one PDF / photo (result opens at once) or many files / a ZIP (background batch) |
| **Invoices** | Search and filter all invoices (status, review status, needs review), Excel / CSV export |
| **Review queue** | Only the invoices a person must check: low confidence, failed checks, missing fields or a possible duplicate |
| **Data sheet** | Spreadsheet view of all invoices and line items, sort, filter, totals per currency, Excel / CSV |
| **Vendors** | Every seller with GSTIN, invoice count, total billed and last invoice date |
| **Phone camera** | DroidCam / IP Webcam over Wi-Fi, or the phone browser's camera |

On a narrow screen (phone) the sidebar becomes a scrollable tab bar at the top.

In every table a click anywhere on a row opens that invoice (Ctrl-click: new tab). Every invoice page has a
**Back** button that returns to the list you came from, with its search and filters.

**Deleting:** the invoice page has a **Delete** button, and the Invoices list has check boxes with
**Delete selected**. Both ask for confirmation. Deleting removes the database row, its line items and the
stored original file; it cannot be undone (back up the `data` folder first if unsure). An invoice that is
being processed at that moment is not deleted.

### Live and Demo workspaces

- **Live** is for real company invoices. It starts empty. Upload one or two of your own PDFs/photos here and only
  your results are shown.
- **Demo** holds the sample and public-dataset invoices used during development (about 100), for showing the app
  to someone without exposing company data. Sample invoices can only be added in Demo.
- Each workspace has its own database and file folder: `data/live/` and `data/demo/`. Nothing is shared between
  them, so switching never mixes data. The switch is remembered per browser.
- The folder import takes `--workspace live|demo` (default `live`).

### What an accounts team does with an invoice here (review workflow)

1. **Capture**: upload, ZIP, folder import or phone camera.
2. **Extract**: number, dates, PO number, seller and buyer (name, address, GSTIN), line items, tax, totals, currency.
3. **Check** automatically:
   - subtotal + tax + shipping/freight − discount = total, line items add up to the subtotal,
   - total in figures = **amount in words** ("Two Lakh Ninety Five Thousand Only"),
   - **OCR digit errors are repaired from the invoice's own arithmetic**: on a phone photo SGST was read as
     22,600 instead of 22,500. CGST must equal SGST, and total (confirmed by the amount in words) − taxable
     value = 45,000, so it is corrected and a warning says so. A value is only changed when the rest of the
     invoice proves it **and** the difference is a single digit; a real mistake on the invoice still fails its check,
   - **GSTIN check digit** for seller and buyer (OCR mistakes such as `O`/`0` are repaired only when the check digit then fits),
   - **CGST = SGST**, and CGST+SGST for the same state vs IGST for different states (first two GSTIN digits),
   - **duplicate invoice**: the same invoice number from the same seller (same GSTIN, or same name) already
     exists, even as a different file or photo (`DPNP/2026-27/007` and `DPNP 2026 27 007` count as equal).
     The exact same file is never processed twice (SHA-256).
4. **Review**: anything doubtful goes to the **Review queue**. On the invoice page a reviewer can **correct any
   value** (the change is saved to the database, the Data sheet and the exports, marked "corrected by reviewer"),
   then **Approve** or **Reject** with a note. "Then open the next invoice to review" walks through the queue.
5. **Export**: Excel / CSV include `review_status`, `review_note`, `reviewed_at`, `edited` and `duplicate_of`,
   so accounting can import only approved invoices.

---

## Tech stack (what is used and why)

| Part | Tool | Why |
|---|---|---|
| Web app and UI | **Flask** + Jinja2 templates, plain CSS and JS | Simple, easy to use, no React build step |
| Web server | **waitress** | Production-grade WSGI server that runs on Windows |
| Database | **SQLite** (WAL mode) | One file, no server to install, handles many thousands of invoices; safe for concurrent readers |
| Background processing | Worker **threads** with a database-backed queue | Tesseract runs as a separate program, so threads OCR several files at once while the model is loaded once. Crash-safe: unfinished files are retried |
| Digital PDF text | **PyMuPDF** | Reads the text layer with exact word positions (milliseconds per file) |
| Image cleanup | **OpenCV** | Finds page edges in phone photos and flattens perspective, evens out shadows, straightens skew |
| OCR | **Tesseract 5** via `pytesseract` | Free, accurate on printed text, gives word boxes and confidence. Used for scanned PDFs and image files |
| Phone-photo OCR | **RapidOCR** with PaddleOCR's **PP-OCRv6 tiny** models on **ONNX Runtime** (CPU) | Neural text detection and recognition, 6 MB of models shipped in `invoice_parser/models` (Apache-2.0, checked by SHA-256, nothing downloaded at run time). About 5x faster than Tesseract on phone photos and reads tilted, shadowed photos cleanly |
| Label matching | Synonyms + **rapidfuzz** + **sentence-transformers** `all-MiniLM-L6-v2` | "Vendor", "Seller", "Supplier", "Billed by" get the same meaning; unknown wording still works. Decisions are cached on disk |
| Field parsing | Regular expressions, **python-dateutil**, custom amount parser | `1,234.50`, `1.234,50`, `1,23,456.00`, `05/03/2024`, `3 Mar 2024`, GSTIN / VAT IDs, e-mail, phone |
| Table parsing | Row heuristics with a `qty x rate = amount` check | Line items are verified arithmetically, which catches OCR digit errors |
| Exports | **openpyxl** (Excel), csv | Excel file with an *Invoices* sheet and a *Line items* sheet |
| Tests and sample data | **pytest**, **ReportLab**, PyMuPDF, OpenCV | Synthetic invoices, scans, scanned PDFs and phone photos with ground truth |

How a file moves through the system:

```
upload / folder import / camera / API
  -> storage.py   save the original (SHA-256 name, duplicates skipped), row status = queued
  -> worker.py    N threads take queued rows
       ingest.py      PDF text layer? use it. Otherwise render + OCR (preprocess.py, ocr.py)
       layout.py      words -> lines -> cells (two-column headers stay separate)
       extract.py     labels -> matcher.py -> canonical fields, party blocks, line items, cross-checks
  -> storage.py   fixed columns (NULL when missing) + line items; needs_review flag
  -> table view, Excel / CSV export, JSON API
```

---

## Setup (Windows)

1. **Python 3.12** (3.11 or 3.12 recommended for the OpenCV and torch wheels).
2. **Tesseract OCR**, needed for scanned PDFs and photos: `winget install UB-Mannheim.TesseractOCR`.
   The default location is `C:\Program Files\Tesseract-OCR\tesseract.exe`; set `TESSERACT_CMD` if it is elsewhere.
3. Install the packages:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

4. Start the app:

```bash
python app.py
```

Open http://localhost:5077 and sign in.

### Sign-in (users and passwords)

Every page, download, phone-camera stream and API call needs a signed-in user. Manage users on the PC that runs
the app:

```bash
python scripts/set_password.py admin
```

It asks for the password twice (at least 8 characters). The same command changes an existing password;
`--list` shows the users and `--delete NAME` removes one. No restart is needed.

- Passwords are stored only as salted **scrypt hashes** in `data/users.json`, never as text.
- The session key is random and kept in `data/secret_key`, so a sign-in cookie cannot be forged. Sign-ins last
  12 hours; **Log out** is at the bottom of the sidebar.
- After 5 wrong passwords from one device, that device waits 15 minutes.
- The cookie is `HttpOnly` and `SameSite=Lax`: another web site cannot submit forms (such as Delete) with it.
- The JSON API also accepts the same user name and password as HTTP Basic auth, for other company systems.
- `data/users.json`, `data/secret_key`, `data/live/` and `data/demo/` are in `.gitignore`.

**Port 5077, not 5000, on purpose:** Flask's default port 5000 is often taken by another local project
(on this PC another Flask service uses it). The app checks the port at startup and refuses to start if
another program is already listening, so two projects can never share it. It also uses its own cookie name
(`invoice_parser_session`), because browsers share cookies across all `localhost` ports.
To use a different port, set `INVOICE_PARSER_PORT` (for example `$env:INVOICE_PARSER_PORT="5078"` in PowerShell).

The first start downloads the embedding model (about 90 MB, once). If that is not possible, the app still runs
with lexical matching; the result page shows which matcher was used.

---

## Handling hundreds or thousands of invoices

**From the browser:** choose many files at once, or upload a ZIP. They are saved and queued, and the batch page
shows live progress (it refreshes every 3 seconds). You can close the page; processing continues in the background.

**From a folder (recommended for large volumes):**

```bash
python scripts/import_folder.py "D:\Invoices\2024" --export "D:\Invoices\2024.xlsx"
```

- Finds PDFs, images and ZIPs recursively.
- Files already in the database (same content, even with a different name) are not processed again.
- Uses all CPU cores minus one (`--workers N` to change). One bad file is marked *failed*; the rest continue.
- Safe to stop with Ctrl+C: run the same command again and it continues.
- The batch also appears in the web app, so you can review and export it there.

**Speed measured on this PC (4 CPU cores, 3 workers):**

| Input | Throughput |
|---|---|
| Digital PDF (has a text layer) | about 0.03 s per file |
| 300-DPI scanned invoice images (public dataset, 50 files) | 175 s in total, about 3.5 s per page |

| Phone photo (12-50 MP, one at a time) | about 4-7 s (fast neural OCR); upside down about 10 s |

This PC has a 2-core / 4-thread CPU (Intel i3-3240), and while these were measured another Python program on
port 8000 used 65-83% of the CPU; closing it, or a PC with more cores, makes photos noticeably faster.
Most of a photo's time is the OCR model reading every text line (about 75 lines on a Tally invoice).

So 1,000 digital PDFs take under a minute, and 1,000 scanned 300-DPI pages take about an hour on this 4-core PC.
More cores scale it almost linearly (`INVOICE_WORKERS`); an 8-core server roughly halves the time.

What was made faster:
- **Phone photos: 22-32 s → 5-10 s** (same photos, same busy PC). Tesseract was replaced for camera photos by
  PaddleOCR's PP-OCRv6 *tiny* text detector and recogniser (6 MB, ONNX Runtime on the CPU, via RapidOCR). Text
  lines are found on a 1280 px copy of the page and read from a 2000 px copy, 16 lines per model call. No light
  flattening, deskewing or line removal is needed: tilted rows are levelled from the slope of the nearby text
  lines (a bent page tilts differently at the top and bottom). A sideways page is turned and a quick check of its
  8 widest lines decides which way up it is. Accuracy went up: real Tally photos 90 of 90 fields, SuperStore
  photos 132 of 132. Tried and not used: finding lines on a 960 px copy (missed the Tally subtotal on 5 of 9
  photos), an int8 version of the model (no faster on this CPU, worse text), 4 threads (slower than 2 here).
- The embedding model used to load at every start (72 s here). It now loads only when a never-seen label appears,
  and every label decision is saved in `data/label_cache.json`, so repeated labels never need the model again.
- Preprocessing was simplified. It is faster, and accuracy on scans went from 34% to 100% (see below).
- Each page is scaled so its letters are about 18 px tall before OCR (the median letter height is measured).
  On 300-DPI scans this cut OCR time by 37% with no change in accuracy.
- Deskew uses a quick coarse-to-fine search; files are processed in parallel; duplicates are skipped instantly.
- Doubtful words (low OCR confidence) are re-read together in one OCR call instead of one call per word;
  on a busy PC this saved up to 8 s per photo (a real Tally photo went from 63 s to about 36 s).
- The re-reading of doubtful words and the letterhead check run as two Tesseract processes at the same time
  (same words as before, checked word by word). The sideways-page check uses OpenCV's own nearest-neighbour
  search instead of importing scipy (that import alone took up to 13 s on a busy PC for the first photo).
- Tried and **not** used because they lowered accuracy on the real Tally photos (54 of 60 fields): a coarser tilt
  search (47/60), reading photos at 18 px instead of 24 px (48/60), and cutting the page in two halves (lower).
  The tilt angle must be precise, and Tesseract's time is set mostly by the amount of text on the page.

---

## The invoice table and exports

Every invoice gets the **same columns**. When a field is not on an invoice, its value is `NULL`. The table shows
it as a grey *null*; in Excel and CSV it is an empty cell, so sums and filters work. Columns:

`invoice_number, invoice_date, due_date, po_number, currency, seller_name, seller_address, seller_tax_id,
seller_email, seller_phone, buyer_name, buyer_address, buyer_tax_id, buyer_email, buyer_phone, subtotal, tax,
discount, shipping, total, line_item_count, overall_confidence, needs_review, status, file_name, ...`

Line items go in a separate table (and Excel sheet): `invoice_id, invoice_number, seller_name, line_no,
description, quantity, unit_price, amount`.

- **Search and filters:** invoice no, seller, buyer, GSTIN or file name; status; "needs review only"; batch.
- **Needs review:** an invoice is flagged when overall confidence is below 0.8, a required field
  (number, date, seller, total) is missing, or the totals or line items do not add up. Flagged rows are highlighted.
- **Exports:** *Export Excel* (two sheets, real date cells), *Invoices CSV* and *Line items CSV* (UTF-8, opens
  correctly in Excel). Exports use the current filters.
- **Each invoice** has its own page with all fields, confidence, the source label each value came from,
  the original file, and a *Re-process* button.
- **Data sheet tab** (`/data`): every invoice in one spreadsheet-style table with all export columns, a second
  sheet with every line item, click-to-sort columns, a filter box, "needs review only", and live totals per
  currency (USD, INR, EUR ... are never added together). Download buttons give the same data as Excel or CSV.

**Where the data is stored:** Live: `data/live/invoices.db` (SQLite) and the original files in `data/live/files/`.
Demo: the same under `data/demo/`. Back up by copying the `data` folder. `INVOICE_DATA_DIR` moves everything to
another drive or folder.

> The sample invoices imported during development (`classic_01.pdf`, `vendor_02_scan.png`, ...) are in the
> **Demo** workspace only. The Live workspace is for real invoices.

### JSON API (for other systems)

| Method and path | What it does |
|---|---|
| `POST /api/parse` (form field `file`) | Saves and parses one invoice, returns all fields as JSON (`invoice_id`, `duplicate`) |
| `GET /api/invoices?q=&status=&review=1&batch=&limit=&offset=` | Table rows as JSON |
| `GET /api/batches/<id>` | Batch progress (queued, processing, done, failed) |
| `GET /export/invoices.xlsx`, `/export/invoices.csv`, `/export/line_items.csv` | Downloads (same filters) |
| `GET /health` | Matcher in use, worker count, queue counts |

---

## Opening it from other devices (network link)

The app listens on the local network, so other PCs and phones on the same Wi-Fi/LAN can open
`http://<this-PC-IP>:5077` (on this PC: `http://192.168.1.180:5077`; `ipconfig` shows the address).

Windows blocks this by default when the network is marked **Public**. Do one of these yourself (it is a security
setting, so the app does not change it):
1. Settings → Network & Internet → your connection → Network profile type → **Private**, then click *Allow* when
   Windows asks the first time the app starts; or
2. In an **Administrator** PowerShell, allow only this port and only devices on your local network:

```bash
New-NetFirewallRule -DisplayName "Invoice Parser (port 5077, LAN only)" -Direction Inbound -Protocol TCP -LocalPort 5077 -RemoteAddress LocalSubnet -Action Allow
```

## Using a phone

- **Phone browser photo (best accuracy):** open `http://<PC-IP>:5077/camera` on the phone and tap *Take photo*.
  This uses the full camera resolution. It needs the network link above (firewall allowed).
- **DroidCam or IP Webcam over Wi-Fi (one click, no PC software):** open **DroidCam** on the phone (port 4747),
  or **IP Webcam** and press *Start server* (port 8080). On the PC's `/camera` page, enter the phone's Wi-Fi IP once
  (it is remembered) and press **Connect**. The live picture opens in a large view (with a *Full screen* button;
  `Esc` returns to the small view); press **Capture and extract**.
  - DroidCam "busy" (another viewer still connected) is handled with DroidCam's documented `force` option
    (`/video/force/1280x720`), and a scrambled HD picture is retried at DroidCam's standard 640×480.
  - **Find phones** looks for DroidCam / IP Webcam on this PC's local network (ports 4747 and 8080), so any phone
    can be connected without typing its IP.
  - The PC connects *to* the phone, so the Windows firewall setting above is not needed for this option.
  - DroidCam allows one viewer at a time. The app keeps a single connection and shares it between the preview
    and the capture; close the DroidCam Windows client or other viewers first. After 60 seconds without a viewer
    the phone is released.
  - On capture the phone is asked to autofocus and the sharpest of the latest frames is used. With IP Webcam the
    app takes a real full-resolution photo instead of a video frame (better for reading small text).
  - **Tested with a real phone (DroidCam on 192.168.1.5):** the connection worked (1280×720 video), but DroidCam sent
    a scrambled picture (green scan lines with "using droidcam.app") to this direct connection. That looks like a
    restriction of the free app on its browser / IP-camera feed. The app detects this, refuses to save the garbage
    picture and explains the options. **Use the free IP Webcam app** for this button, the DroidCam Windows client
    with "Webcam on this PC", or a phone browser photo.
  - "DroidCam is busy" means another viewer (DroidCam Windows client, OBS, a browser tab) is connected. Close it or
    restart DroidCam on the phone. The app retries a busy phone automatically twice.
  - Video frames are often smaller than photos (DroidCam's free video is commonly 640×480), so let the invoice fill
    the picture. The page shows the actual video size after connecting.
  - Only local-network addresses are accepted.
- Phone and PC must be on the **same network** (Wi-Fi and the PC's wired LAN on the same router is fine).
- **DroidCam with its Windows client (recommended for DroidCam, no webcam needed):** the free DroidCam client
  on the PC receives the phone's normal picture (no green lines, no "busy" conflicts) and offers it as the
  "DroidCam Video" camera. Connect the client to the phone in its own window, then press **Connect PC camera**
  (card 3), or simply **Connect** in card 2: when the DroidCam client is running, Connect uses its camera
  automatically instead of connecting to the phone directly. Card 3 also works with any webcam
  (camera "Auto" picks the first one that gives a picture), with the same live large view.

Photos are flattened (page edges found, perspective corrected) and shadows are evened out before OCR.
A page photographed **sideways or upside down** is
turned upright automatically: phones often store a photo of paper lying flat on a desk without a rotation
tag. For best results, lay the paper flat, fill the frame with it, and avoid strong glare and shadows.

**Tested on real phone photos of a Tally GST invoice** (6 photos taken with a phone, plus one of them turned
sideways both ways and upside down = 9 pictures, 10 fields each): **90 of 90 fields** right with the fast neural
OCR, including the PO number and both GSTINs (Tesseract got 81 of 90: the PO number and seller GSTIN, printed
small, were often unreadable). Five more photos of the same invoice in the Live workspace: all fields right.

---

## Accuracy on the synthetic set

```bash
python scripts/make_sample_invoices.py --count 4 --scans --photos
python scripts/evaluate.py
pytest -q
```

The generator makes 5 different layouts (Seller / Vendor-without-label / From-To with EU numbers /
Indian GST with CGST+SGST / label-only headings). Each layout is produced as a digital PDF, a noisy scan image,
an image-only scanned PDF, and a phone photo (tilted page on a table, uneven light, JPEG).

Latest result: **672 / 672 fields correct (100%) on 80 documents**. That covers every input type
(digital PDF, scanned PDF, scan image, phone photo) and every layout.

| Change | Effect (measured) |
|---|---|
| Removed custom thresholding before OCR; Tesseract binarises | Scans: 34% → 100% |
| Light flattening for photos | Phone photos: 20% → 100% |
| `To:` added as a buyer label (only as a real label, not inside sentences) | EU layout buyer: 0% → 100% |

**Be careful with these numbers.** Synthetic invoices are cleaner and more regular than real ones. They prove the
pipeline works end to end; they do not predict accuracy on your suppliers' invoices. Measure that on 100 to 200 of
your own invoices: put each file next to a JSON file in the same format as `data/samples/*.json` and run
`python scripts/evaluate.py --dir path\to\folder`. The report lists every wrong field.

---

## Accuracy on a public invoice dataset

Dataset: [katanaml invoices-donut-data-v1](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1)
(Hugging Face, MIT license). These are 300-DPI invoice images with labels for invoice number, date, seller and client
(name + address), both tax IDs, net / VAT / gross totals and line items. They are generated but realistic: seller and
client in two columns, tax IDs after a blank line, US dates, item tables with wrapped descriptions, and totals inside
a summary table.

| Run | Fields correct |
|---|---|
| Before tuning, validation split (50 invoices) | 46.8% |
| After tuning, validation split (50 invoices) | 99.6% (526 / 528) |
| **Held-out test split (26 invoices, never used while tuning)** | **99.3% (282 / 284)** |

The remaining misses were checked by hand, and the ones checked are errors in the dataset's labels:
`invoice_110` has 6 line items but is labelled with 0; `invoice_464`'s seller label is a tax ID; `invoice_366` shows a
total of $ 157 929,37 but is labelled 1,579,929.37. (`invoice_351`, labelled with 0 items, was not checked visually.)

What this dataset taught the parser (general rules, not tied to this dataset):
- Day/month order is decided per invoice: an unambiguous date first, then hints (GSTIN / INR → day first;
  USD or a US "STATE 12345" address → month first; EUR / GBP → day first), then the `DAY_FIRST` setting.
- Party blocks continue past a blank line for tax-ID, IBAN, e-mail and phone lines.
- Summary tables: each amount is matched to the column header above it (Net worth / VAT / Gross worth).
- Line items are read column by column (Qty / Price / Amount); wrapped descriptions are joined to their item,
  and quantities with units ("2,00 each", "5 pcs") are read.
- "Tax Id", "VAT No" and "Tax rate" are never taken as tax amounts.

Caveat: the invoices in this set share one visual layout, so it shows robustness on real-looking scans,
not on many different layouts. Your own suppliers' invoices are still the real test.

Reproduce (about 1 minute for the 26 test invoices):

```bash
curl -L --create-dirs -o datasets/katanaml/raw/test.parquet https://huggingface.co/api/datasets/katanaml-org/invoices-donut-data-v1/parquet/default/test/0.parquet
```

```bash
python -I scripts/convert_katanaml.py datasets/katanaml/raw/test.parquet datasets/katanaml/test
```

```bash
python scripts/evaluate.py --dir datasets/katanaml/test
```

---

## Datasets for testing

**Used for the accuracy numbers in this README:**

| Set | What it is | Result |
|---|---|---|
| **katanaml invoices-donut-data-v1** (Hugging Face, **MIT license**, no login): https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1 | 501 realistic invoice images (seller / client columns, tax IDs, item and summary tables) with labelled fields. The 26 test and 50 validation invoices are in `datasets/katanaml/` after the download command below | 99.3% (test), 99.6% (validation) |
| Synthetic set made by `scripts/make_sample_invoices.py` | 20 invoices in 5 layouts (GST India, EU, US classic, label-only, vendor) × 4 forms (digital PDF, scan, scanned PDF, phone photo) | 100% |
| SuperStore layout | 200 PDFs + 40 scans / photos + 12 phone-style photos | 100% / 39 of 40 / 12 of 12 |
| Real phone photos of a Tally GST invoice | 6 photos + 3 turned copies | 90 of 90 fields |

**Best single public dataset for wider testing: DocILE** (Rossum), https://github.com/rossumai/docile ·
https://docile.rossum.ai. About 6,700 real business documents (mostly invoices, scanned and digital, many
vendors and layouts) with labels for 55 field types including line items. It needs a registration and you
must accept its terms; check that they allow your company's use. It has no Indian GST / Tally invoices and no
phone photos, and its answer format needs a small converter for `scripts/evaluate.py` (like
`scripts/convert_katanaml.py`).

Other sets:

| Dataset | What it is | Link |
|---|---|---|
| **FATURA** | 10,000 synthetic invoice images in 50 layouts, with JSON labels. Check the license on the page | https://zenodo.org/records/10371464 |
| **High-Quality Invoice Images for OCR** (Kaggle) | 8,000+ synthetic invoice images, about 1,500 with labels. Kaggle login needed | https://www.kaggle.com/datasets/osamahosamabdellatif/high-quality-invoice-images-for-ocr |
| **INV-CDIP** (Salesforce) | 350 labelled real scanned invoices. Labels are CC BY-NC 4.0 (non-commercial), so not used here | https://github.com/salesforce/inv-cdip |
| **SROIE** (ICDAR 2019) | Scanned receipts (company, date, total). Tests OCR more than invoice layout | https://www.kaggle.com/datasets/urbikn/sroie-datasetv2 |

No good public dataset of Indian GST / Tally invoices exists. The most important test set is **100-200 of your
own invoices** (Tally PDFs, scans and phone photos). Upload test files in the **Demo** workspace, not Live.

---

## Do I need a training dataset?

**Not for this version.** The parser is rule-based and uses pretrained embeddings, so it works on new layouts without
training. You need labelled examples only to **measure** accuracy (the ground-truth JSON files above).

Train a model only if measured accuracy on your real invoices is not good enough. Then:
- Fine-tuning a layout model such as LayoutLMv3 or Donut needs labelled invoices: a few hundred to a few thousand,
  such as DocILE, INV-CDIP, or your own annotated set.
- Start from the evaluation report's mistakes, so you label the cases that actually fail.
- Often a few synonyms in `invoice_parser/schema.py` fix a supplier's unusual label without any training.

---

## Deploying online (GitHub + Render)

The repository has a `Dockerfile`: Python 3.12, the Tesseract OCR program with the same English model as the
tested Windows install (checked by hash), and the web app. Render (or any Docker host) builds and runs it.

1. Render → **New → Web Service** → connect this GitHub repository (or paste its public URL). Render detects the
   Dockerfile; leave the start command empty.
2. **Instance type.** OCR needs CPU. A phone photo needs about 5-8 s of CPU time, so Render's free instance
   (0.1 CPU) should take roughly a minute per photo (estimated, not yet measured there); scans read by Tesseract
   take several minutes there. A 1-CPU instance is about as fast as the development PC.
3. **Environment variables:**

   | Name | Value |
   |---|---|
   | `INVOICE_ADMIN_USER` | the first user name, e.g. `admin` |
   | `INVOICE_ADMIN_PASSWORD` | its password (8+ characters). Used only when no user exists yet |
   | `FLASK_SECRET_KEY` | optional: a long random string, so sign-ins survive restarts without a disk |

4. **Persistent disk** (paid instances): mount path `/var/data`, 1 GB or more. Without a disk every restart or
   deploy **erases** the database, the stored invoices and the users. Render's free instances have no disk and
   go to sleep after 15 minutes without visits (the next visit takes about a minute to wake up).
5. Health check path: `/health`.

Differences on the server: the sentence-transformers model is not installed (`USE_EMBEDDINGS=0`; it needs over
1 GB of RAM). Labels are then matched with the synonym list and rapidfuzz; on every test set above the results
were the same (synthetic 672/672, katanaml 282/284). Connecting to a phone by IP (DroidCam / IP Webcam) only
works on the local network; on the server use the browser camera on the *Phone camera* tab (works over HTTPS).

Run the same image on your own PC or server:

```
docker build -t invoagent .
docker run -p 5077:10000 -e INVOICE_ADMIN_USER=admin -e INVOICE_ADMIN_PASSWORD=change-me -v invoagent-data:/var/data invoagent
```

---

## Configuration

Set these as environment variables (PowerShell: `$env:NAME="value"`):

| Variable | Default | Meaning |
|---|---|---|
| `INVOICE_PARSER_PORT` | `5077` | Web server port (kept away from Flask's common 5000). `PORT` (set by Render) is used when this is not set |
| `INVOICE_ADMIN_USER` / `INVOICE_ADMIN_PASSWORD` | not set | Servers: create the first sign-in user at start when none exists |
| `INVOICE_DATA_DIR` | `data` | Folder for the database, stored files and label cache |
| `INVOICE_DB_PATH` | `data/live/invoices.db` | Live database file (Demo is always `data/demo/invoices.db`) |
| `INVOICE_WORKERS` | CPU cores − 1 (max 8) | Parallel worker threads |
| `REVIEW_CONFIDENCE` | `0.8` | Below this overall confidence an invoice is flagged for review |
| `TESSERACT_CMD` | `C:\Program Files\Tesseract-OCR\tesseract.exe` | Path to Tesseract |
| `OCR_LANG` | `eng` | Tesseract language(s), for example `eng+hin` |
| `OCR_CONFIG` | `--oem 3 --psm 6` | Tesseract page-segmentation options |
| `RENDER_DPI` | `300` | DPI used when OCR-ing PDF pages |
| `MAX_OCR_SIDE` | `4000` | Larger photos are scaled down to this before OCR |
| `USE_EMBEDDINGS` | `1` | `0` turns off the embedding model (lexical matching only) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any sentence-transformers model |
| `DAY_FIRST` | `1` | Fallback for ambiguous dates such as `05/03/2024` when the invoice gives no clue (see below) |
| `OCR_TEXT_PX` | `18` | Pages are scaled so letters are about this many pixels tall before OCR |
| `FAST_PHOTO_OCR` | `1` | `0` reads phone photos with Tesseract instead of the fast neural OCR |
| `PHOTO_OCR_THREADS` | all cores (`1` in the Docker image) | CPU threads for the phone-photo OCR |
| `MAX_UPLOAD_MB` | `300` | Browser upload limit per request (use the folder import for more) |
| `FLASK_SECRET_KEY` | random, in `data/secret_key` | Session signing key (set it only to share one key between servers) |

---

## Project layout

```
app.py                      Web app (pages, exports, JSON API); starts the worker threads
invoice_parser/
  pipeline.py               InvoiceParser: parse one file -> result dict
  storage.py                SQLite per workspace: invoices, line items, batches, review, corrections,
                            duplicate detection (SHA-256 file + seller/invoice number)
  worker.py                 background worker threads (queue in the database)
  exporting.py              CSV and Excel exports
  ingest.py                 PDF / image / multi-page TIFF loading, text layer vs OCR
  preprocess.py             page-edge flattening, light flattening, deskew
  ocr.py                    Tesseract wrapper (word boxes + confidence)
  fast_ocr.py               phone-photo OCR: PP-OCRv6 tiny models on ONNX Runtime (RapidOCR)
  models/                   the two PP-OCRv6 tiny model files (6 MB)
  layout.py                 words -> lines -> cells
  matcher.py                label -> field: exact, fuzzy, embedding (+ label cache)
  schema.py                 canonical fields, synonyms, guard words
  extract.py                fields, party blocks, line items, cross-checks, GST checks
  normalize.py              amounts, amount in words, dates, GSTIN check digit + OCR repair, currency
  camera.py                 phone (IP Webcam) snapshot and webcam capture
templates/, static/         Plain HTML, CSS and JS pages
scripts/import_folder.py          bulk import of a folder (thousands of files)
scripts/make_sample_invoices.py   synthetic invoices, scans, scanned PDFs, photos + ground truth
scripts/evaluate.py               accuracy report (per field, input type, layout)
scripts/convert_katanaml.py       turns the public katanaml dataset into the evaluation format
datasets/                         downloaded public datasets (not committed)
tests/                            pytest suite (parser, storage, workers, exports, web app)
data/live/, data/demo/             one database + stored files per workspace (created at runtime)
data/                             label cache, samples, phone settings
```

---

## Layouts handled without labels

Real invoices often leave out the labels a parser would look for. These cases are handled:

| On the invoice | How it is read |
|---|---|
| Seller printed only as a name at the top (`SuperStore`), no "Seller:" label | Letterhead rule: a short name above the first label on page 1 (flagged "please check") |
| Seller only in `for Deepnap Softech / Authorised Signatory` | Signature-block rule |
| Seller only identified by the first GSTIN block | Tax-ID block rule |
| `INVOICE` with `# 5873` under it | A bare `#` near the top is the invoice number |
| `Bill To` / `Ship To` side by side, `Ship Mode: Standard Class` on the right at the same height | Each party block is read by x-position under its own heading, so neighbouring columns never mix |
| Labels in boxes with the value *below* (Tally) | Value-below rule |
| White text on a dark table header bar (scans / photos) | Dark bars are inverted before OCR |
| A small word on the same line as a big title (OCR misreads it) | Words with very low OCR confidence are read again on their own (at most 8 per page) |
| `Shipping`, `Freight`, `Delivery charges`, `Packing and forwarding` | Own `shipping` column, included in the totals check |
| `Order ID : MX-2012-...` | Stored as the order / PO number |
| `Total  1 NOS  2,95,000.00` (Tally prints the total quantity in the Total row) | A number with a unit (`NOS`, `pcs`, `kg` ...) is never taken as money; on a tilted photo the amount at the same height further right is used |
| Total read wrongly, but the amount in words and a figure printed on the invoice agree | The total is corrected to that amount, with a warning |
| No readable line items and no printed subtotal | Subtotal = total - tax, only when the total is confirmed by the amount in words and that subtotal is printed on the invoice |

Tested on 200 invoices in the SuperStore layout (all fields 200/200) and 40 scan / phone-photo versions of them.

## Limitations

- **OCR quality limits scans and photos.** Blur, folds, glare, handwriting and very small print reduce accuracy.
  The confidence score and the *needs review* flag show which invoices to check.
- **Line items are heuristic.** Rows are read as `qty x rate = amount`; merged cells and multi-line descriptions
  can be missed. Use the line-item cross-check to decide what to review.
- **Non-English labels** need the matching Tesseract language pack (`OCR_LANG`) and label synonyms in that language.
- **SQLite fits one machine.** For many users on a network, the same storage layer can move to PostgreSQL later.
- **Security:** the server listens on the local network so phones can reach it, and it needs a sign-in. It
  serves plain HTTP (no TLS), so use it on the company's trusted network; for access over the internet put it
  behind HTTPS (a reverse proxy) or a VPN.
