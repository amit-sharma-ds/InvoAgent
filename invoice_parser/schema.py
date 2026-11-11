"""Canonical invoice fields, the label phrases that mean each one, and guard words.

The phrase lists are only a seed. The matcher also uses sentence embeddings, so a
label that is merely *similar* in meaning ("Name of supplier company") still maps to
the right field. "seller", "vendor" and "supplier" all land in the same field because
they are listed here as synonyms and embed close together.
"""

FIELD_SYNONYMS = {
    "invoice_number": [
        "invoice number", "invoice no", "invoice num", "invoice id", "invoice ref",
        "inv no", "inv number", "bill number", "bill no", "tax invoice no",
        "tax invoice number", "document number", "document no",
    ],
    "invoice_date": [
        "invoice date", "date", "date of invoice", "invoice dated", "dated",
        "issue date", "date of issue", "bill date", "billing date", "date issued",
    ],
    "due_date": [
        "due date", "payment due", "payment due date", "due on", "due by",
        "date due", "pay by", "payment due on",
    ],
    "po_number": [
        "po number", "po no", "po", "purchase order", "purchase order number",
        "purchase order no", "order number", "order no", "p o number",
        "buyer s order no", "buyers order no", "buyer order no", "buyer s order number",
        "order id", "order ref", "order reference", "sales order no", "so number",
    ],
    "seller": [
        "seller", "seller name", "vendor", "vendor name", "supplier", "supplier name",
        "sold by", "billed by", "bill from", "billed from", "from", "issued by",
        "issuer", "service provider", "merchant", "merchant name", "company",
        "company name", "seller details", "vendor details", "supplier details",
        "payee", "sender", "provider", "from company",
    ],
    "buyer": [
        "buyer", "buyer name", "customer", "customer name", "client", "client name",
        "bill to", "billed to", "bill to party", "sold to", "sold to party", "ship to",
        "recipient", "purchaser", "consignee", "customer details", "buyer details",
        "bill to details", "invoice to", "to",
    ],
    "subtotal": [
        "subtotal", "sub total", "net amount", "taxable amount", "taxable value",
        "amount before tax", "total before tax", "net total", "total excluding tax",
        "base amount", "net worth", "total net worth", "net value", "total net",
    ],
    "tax": [
        "tax", "tax amount", "total tax", "vat", "vat amount", "gst", "gst amount",
        "total gst", "total vat", "sales tax", "tax total", "igst", "cgst", "sgst",
        "service tax",
    ],
    "discount": ["discount", "discount amount", "less discount", "total discount", "discount total"],
    # charges added on top of the goods: subtotal + tax + shipping - discount = total
    "shipping": [
        "shipping", "shipping charges", "shipping charge", "shipping cost", "shipping fee",
        "shipping and handling", "shipping handling", "freight", "freight charges", "freight charge",
        "delivery charges", "delivery charge", "delivery fee", "postage", "postage and packing",
        "carriage", "courier charges", "transport charges", "transportation charges",
        "packing and forwarding", "packing forwarding charges", "handling charges",
    ],
    "total": [
        "total", "grand total", "total amount", "amount due", "balance due", "total due",
        "net payable", "amount payable", "total payable", "invoice total",
        "total amount due", "amount to pay", "net amount payable", "total amount payable",
        "final amount", "total invoice amount", "grand total amount",
        "gross worth", "total gross worth", "gross amount", "gross total", "total gross",
        "gross value", "gross",
    ],
    "currency": ["currency", "currency code"],
}

# If a label contains one of these words, the field is ruled out. This stops
# "Subtotal" from being read as "Total", and "Ship to" from being read as the seller.
FIELD_BLOCKLIST = {
    "invoice_number": {"date", "due", "pay", "payment"},
    "invoice_date": {"due", "pay", "payment", "ship", "delivery", "expiry", "valid", "order"},
    "due_date": set(),
    "po_number": {"date"},
    "seller": {"to", "ship", "shipping", "deliver", "delivery", "customer", "buyer",
               "client", "recipient", "consignee", "purchaser"},
    "buyer": {"by", "from", "seller", "vendor", "supplier", "issued", "issuer",
              "merchant", "provider", "payee", "sender"},
    "subtotal": {"discount", "grand", "payable", "due", "balance", "pay", "gross", "shipping", "freight"},
    # "Tax Id", "VAT No", "GST Number", "Tax rate" are identifiers or percentages, not tax amounts
    "tax": {"id", "no", "number", "reg", "registration", "identification", "code", "rate"},
    "discount": set(),
    "shipping": {"to", "address", "mode", "method", "date", "via", "terms", "from", "no", "number"},
    "total": {"sub", "subtotal", "tax", "vat", "gst", "igst", "cgst", "sgst", "discount", "shipping", "freight",
              "unit", "rate", "qty", "quantity", "price", "before", "excluding"},
    "currency": set(),
}

PARTY_FIELDS = {"seller", "buyer"}
AMOUNT_FIELDS = {"subtotal", "tax", "discount", "shipping", "total"}
DATE_FIELDS = {"invoice_date", "due_date"}
IDENT_FIELDS = {"invoice_number", "po_number"}
