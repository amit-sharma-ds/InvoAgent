import pytest

from invoice_parser.matcher import LabelMatcher


@pytest.fixture(scope="module")
def lexical():
    return LabelMatcher(use_embeddings=False)


@pytest.mark.parametrize("label, field", [
    ("Seller", "seller"), ("Vendor", "seller"), ("Vendor Name", "seller"), ("Supplier", "seller"),
    ("Sold By", "seller"), ("Billed by", "seller"), ("Bill From", "seller"), ("From", "seller"),
    ("Bill To", "buyer"), ("Customer", "buyer"), ("Ship To", "buyer"), ("Sold to", "buyer"),
    ("Invoice No.", "invoice_number"), ("Invoice #", "invoice_number"),
    ("Invoice Date", "invoice_date"), ("Date", "invoice_date"), ("Due Date", "due_date"),
    ("Subtotal", "subtotal"), ("Sub Total", "subtotal"), ("Sub-total", "subtotal"),
    ("Tax", "tax"), ("GST", "tax"), ("Discount", "discount"),
    ("Grand Total", "total"), ("Total", "total"), ("Amount Due", "total"), ("Balance due", "total"),
    ("Currency", "currency"),
])
def test_known_labels(lexical, label, field):
    m = lexical.match(label)
    assert m is not None and m.field == field, f"{label!r} -> {m}"


@pytest.mark.parametrize("label", ["Thank you for your business", "Terms and conditions", "Bank details"])
def test_non_field_labels_are_ignored(lexical, label):
    assert lexical.match(label) is None


def test_subtotal_is_never_total(lexical):
    assert lexical.match("Subtotal").field == "subtotal"
    assert lexical.match("Sub Total").field == "subtotal"


@pytest.fixture(scope="module")
def embedded(tmp_path_factory):
    m = LabelMatcher(use_embeddings=True, cache_path=tmp_path_factory.mktemp("cache") / "labels.json").warm_up()
    if m.backend.startswith("lexical"):
        pytest.skip("sentence-transformers model not available offline")
    return m


def test_semantic_decisions_are_reused_from_disk_without_loading_the_model(embedded, tmp_path):
    path = tmp_path / "labels.json"
    first = LabelMatcher(use_embeddings=True, cache_path=path)
    first._model, first._syn_vecs = embedded._model, embedded._syn_vecs  # reuse the loaded model
    decided = first.match("Name of the supplier company")
    first.flush()

    second = LabelMatcher(use_embeddings=True, cache_path=path)
    assert second.match("Name of the supplier company") == decided
    assert second._model is None  # answered from the label cache; the model was never loaded


@pytest.mark.parametrize("label, field", [
    ("Name of the supplier company", "seller"),
    ("Merchant business name", "seller"),
    ("Amount you have to pay", "total"),
])
def test_semantic_labels(embedded, label, field):
    m = embedded.match(label)
    assert m is not None and m.field == field, f"{label!r} -> {m}"
