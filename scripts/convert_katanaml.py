#!/usr/bin/env python
"""Convert the Hugging Face dataset katanaml-org/invoices-donut-data-v1 (MIT license) into the
ground-truth format used by scripts/evaluate.py.

Download a split (no login needed), then convert it:
    curl -L -o datasets/katanaml/raw/test.parquet ^
      https://huggingface.co/api/datasets/katanaml-org/invoices-donut-data-v1/parquet/default/test/0.parquet
    python -I scripts/convert_katanaml.py datasets/katanaml/raw/test.parquet datasets/katanaml/test
    python scripts/evaluate.py --dir datasets/katanaml/test
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq


def _amount(text):
    """'$ 1 234,56' -> 1234.56 (this dataset writes decimals with a comma)."""
    if not text:
        return None
    s = re.sub(r"[^\d,.-]", "", text)
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def _iso_date(text):
    try:
        return datetime.strptime(text.strip(), "%m/%d/%Y").date().isoformat()  # dataset uses US dates
    except (AttributeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet", help="a downloaded split, e.g. datasets/katanaml/raw/test.parquet")
    ap.add_argument("out_dir", help="folder for the images and ground-truth JSON files")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(args.parquet).to_pylist()
    for i, row in enumerate(rows):
        image = row["image"]
        name = Path(image.get("path") or "").name
        if not re.fullmatch(r"[\w.-]+\.(jpe?g|png)", name, flags=re.I):  # never trust names from downloads
            name = f"invoice_{i:04d}.jpg"
        (out / name).write_bytes(image["bytes"])

        gt = json.loads(row["ground_truth"])["gt_parse"]
        head, summary = gt.get("header", {}), gt.get("summary", {})
        truth = {
            "file": name,
            "template": "katanaml",
            "kind": "katanaml invoice image",
            "invoice_number": head.get("invoice_no"),
            "invoice_date": _iso_date(head.get("invoice_date")),
            "seller_block": head.get("seller"),     # name + address as one string
            "buyer_block": head.get("client"),
            "seller_tax_id": head.get("seller_tax_id"),
            "buyer_tax_id": head.get("client_tax_id"),
            "currency": "USD" if "$" in (summary.get("total_gross_worth") or "") else None,
            "subtotal": _amount(summary.get("total_net_worth")),
            "tax": _amount(summary.get("total_vat")),
            "total": _amount(summary.get("total_gross_worth")),
            "line_item_count": len(gt.get("items") or []),
        }
        (out / f"{Path(name).stem}.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"Converted {len(rows)} invoices into {out}")


if __name__ == "__main__":
    main()
