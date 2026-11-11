#!/usr/bin/env python
"""Measure field-level accuracy against ground-truth JSON files.

Each ground-truth JSON has a "file" key pointing to the invoice in the same folder.
Works with the synthetic set from make_sample_invoices.py and with any dataset you
convert to the same JSON format (for example, annotated invoices from DocILE or INV-CDIP).

Usage:
    python scripts/evaluate.py                      # uses data/samples
    python scripts/evaluate.py --dir path/to/set --workers 6 --report report.json
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from invoice_parser import config  # noqa: E402
from invoice_parser.pipeline import InvoiceParser  # noqa: E402

AMOUNT_FIELDS = {"subtotal", "tax", "discount", "shipping", "total"}
PARTY_FIELDS = {"seller", "buyer"}
BLOCK_FIELDS = {"seller_block": "seller", "buyer_block": "buyer"}  # ground truth = name + address in one string
TAX_ID_FIELDS = {"seller_tax_id": "seller", "buyer_tax_id": "buyer"}
FIELDS = ["invoice_number", "invoice_date", "due_date", "seller", "buyer", "seller_block", "buyer_block",
          "seller_tax_id", "buyer_tax_id", "currency", "subtotal", "tax", "discount", "total", "line_item_count"]


def _predicted(result, field):
    if field == "line_item_count":
        return len(result.get("line_items") or [])
    party = BLOCK_FIELDS.get(field) or TAX_ID_FIELDS.get(field) or (field if field in PARTY_FIELDS else None)
    f = result["fields"].get(party or field)
    if not f:
        return None
    if party:
        return (f["value"] or {}).get("tax_id" if field in TAX_ID_FIELDS else "name")
    return f["value"]


def _correct(field, expected, predicted):
    if predicted is None:
        return False
    if field in AMOUNT_FIELDS:
        return abs(float(expected) - float(predicted)) <= 0.01
    if field in PARTY_FIELDS:
        return fuzz.token_sort_ratio(str(expected).lower(), str(predicted).lower()) >= 90
    if field in BLOCK_FIELDS:  # the predicted name must be the start of "name + address"
        pred, block = str(predicted).lower().strip(), str(expected).lower()
        return len(pred) >= 3 and fuzz.ratio(pred, block[:len(pred)]) >= 90
    if field in TAX_ID_FIELDS:
        return re.sub(r"\W", "", str(expected)).upper() == re.sub(r"\W", "", str(predicted)).upper()
    if field == "line_item_count":
        return int(expected) == int(predicted)
    return str(expected).strip().upper() == str(predicted).strip().upper()


def _kind(gt):
    if gt.get("kind"):
        return gt["kind"]
    name = gt["file"].lower()
    return "digital pdf" if name.endswith(".pdf") else "image"


def _table(title, counts):
    print(title)
    for key, (ok, n) in sorted(counts.items()):
        print(f"  {key:<16}{f'{ok}/{n}':>10}{ok / n:>11.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data" / "samples"))
    ap.add_argument("--workers", type=int, default=config.WORKERS, help="parallel threads")
    ap.add_argument("--report", default=None, help="write a JSON report with every mismatch")
    args = ap.parse_args()
    if args.workers > 1:
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")

    folder = Path(args.dir)
    cases = []
    for gt_path in sorted(folder.glob("*.json")):
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        if "file" in gt and (folder / gt["file"]).exists():
            cases.append(gt)

    parser = InvoiceParser()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda gt: parser.parse_path(folder / gt["file"]), cases))
    elapsed = time.perf_counter() - started
    parser.matcher.flush()

    per_field = defaultdict(lambda: [0, 0])
    per_kind = defaultdict(lambda: [0, 0])
    per_template = defaultdict(lambda: [0, 0])
    mismatches = []
    for gt, result in zip(cases, results):
        for field in FIELDS:
            expected = gt.get(field)
            if expected is None:
                continue  # field not present on this invoice
            pred = _predicted(result, field)
            ok = _correct(field, expected, pred)
            for bucket, key in ((per_field, field), (per_kind, _kind(gt)), (per_template, gt.get("template", "?"))):
                bucket[key][0] += ok
                bucket[key][1] += 1
            if not ok:
                mismatches.append({"file": gt["file"], "field": field, "expected": expected, "predicted": pred})

    total_ok = sum(v[0] for v in per_field.values())
    total_n = sum(v[1] for v in per_field.values())
    print(f"Documents: {len(cases)}   time: {elapsed:.1f}s ({len(cases) / elapsed:.2f} docs/s with "
          f"{args.workers} workers)   matcher: {parser.matcher.backend}\n")
    _table("Per field:", per_field)
    _table("\nPer input type:", per_kind)
    _table("\nPer layout:", per_template)
    if total_n:
        print(f"\nOVERALL  {total_ok}/{total_n} fields correct = {total_ok / total_n:.1%}")
    if mismatches:
        print(f"\nFirst mismatches ({len(mismatches)} total):")
        for m in mismatches[:15]:
            print(f"  {m['file']:<26} {m['field']:<15} expected={m['expected']!r}  got={m['predicted']!r}")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "documents": len(cases), "seconds": round(elapsed, 1),
            "overall": total_ok / total_n if total_n else None,
            "per_field": {k: v[0] / v[1] for k, v in per_field.items()},
            "per_input_type": {k: v[0] / v[1] for k, v in per_kind.items()},
            "mismatches": mismatches}, indent=2), encoding="utf-8")
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
