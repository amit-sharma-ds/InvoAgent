"""Common spellings for party names and addresses read from handwriting (invoice_parser/spelling.py).
Made-up firms and addresses only."""
from types import SimpleNamespace

from invoice_parser import spelling
from invoice_parser.models import Page, Word


def unsure(_token):
    return True


def sure(_token):
    return False


def test_misread_business_words_in_a_name_get_their_common_spelling():
    name, changes = spelling.fix_text("Kavya Lovistics Pvt. L+d", spelling._NAME_WORDS, unsure)
    assert name == "Kavya Logistics Pvt. Ltd"
    assert changes == [("Lovistics", "Logistics"), ("L+d", "Ltd")]
    assert spelling.fix_text("OM TRADRS", spelling._NAME_WORDS, unsure)[0] == "OM TRADERS"  # capitals kept


def test_misread_address_words_and_places_get_their_common_spelling():
    address, _ = spelling.fix_text("Plot.Ho.12 Shanti Bulding, Naggr, Andheri(e) maombai",
                                   spelling._ADDRESS_WORDS, unsure)
    assert address == "Plot No. 12 Shanti Building, Nagar, Andheri (E) Mumbai"
    # a word cut in two at a wide gap of the handwriting, and a place written in small letters
    assert spelling.fix_text("+d - Ram Nagar, Kurla m umbai", spelling._ADDRESS_WORDS, unsure)[0] == \
        "Ltd - Ram Nagar, Kurla Mumbai"
    assert spelling.fix_text("Sector 5, gurugram", spelling._ADDRESS_WORDS, sure)[0] == "Sector 5, Gurugram"
    assert spelling.fix_text("Shanti cHS", spelling._ADDRESS_WORDS, sure)[0] == "Shanti CHS"


def test_words_read_surely_are_kept():
    name, changes = spelling.fix_text("Kavya Lovistics Pvt Ltd", spelling._NAME_WORDS, sure)
    assert name == "Kavya Lovistics Pvt Ltd" and changes == []
    # fixed patterns need no doubt: a plot number's 'No.', 'Pvt' before 'Ltd'
    assert spelling.fix_text("Rahul Electronics Put Ltd", spelling._NAME_WORDS, sure)[0] == "Rahul Electronics Pvt Ltd"
    assert spelling.fix_text("Plot No. 7, Sector 4", spelling._ADDRESS_WORDS, sure) == ("Plot No. 7, Sector 4", [])


def test_names_short_words_and_unclear_cases_are_left_as_read():
    # place words are not used in names: a person's name can be close to a place
    assert spelling.fix_text("Karnail Singh", spelling._NAME_WORDS, unsure)[0] == "Karnail Singh"
    assert spelling.fix_text("Naggr Traders", spelling._NAME_WORDS, unsure)[0] == "Naggr Traders"
    # short words of letters only are often real words ('Tram'), and 'Put' not before 'Ltd' stays
    assert spelling.fix_text("Tron Put", spelling._ADDRESS_WORDS, unsure)[0] == "Tron Put"
    # far from every common word, or two parts that are common words themselves
    assert spelling.fix_text("Shanti, Ram Nagar", spelling._ADDRESS_WORDS, unsure)[0] == "Shanti, Ram Nagar"
    assert spelling.closest_word("Gala", spelling._ADDRESS_WORDS) is None  # a common word already
    assert spelling.closest_word("Steele", spelling._NAME_WORDS) is None


def test_party_fields_of_a_photo_are_fixed_and_the_changes_listed():
    words = [Word("M/s.", 0.1, 0.1, 0.15, 0.12, 0.99), Word("Kavya", 0.16, 0.1, 0.25, 0.12, 0.99),
             Word("Lovistics", 0.26, 0.1, 0.4, 0.12, 0.97), Word("Pvt.", 0.41, 0.1, 0.45, 0.12, 0.99),
             Word("L+d", 0.46, 0.1, 0.5, 0.12, 0.97), Word("Naggr", 0.1, 0.13, 0.2, 0.15, 0.97)]
    # the fast reader's letter confidences: the 'v' of 'Lovistics' and a 'g' of 'Naggr' were doubtful
    read = [("M/s. Kavya Lovistics Pvt. L+d", 0.9,
             SimpleNamespace(confs=[0.99] * 13 + [0.7] + [0.99] * 15)),
            ("Naggr", 0.9, SimpleNamespace(confs=[0.99, 0.99, 0.99, 0.6, 0.99]))]
    page = Page(0, words, "ocr", 0.9, SimpleNamespace(read=read))
    result = {"fields": {"buyer": {"value": {"name": "Kavya Lovistics Pvt. L+d", "address": "Naggr"}, "page": 1},
                         "seller": None}}
    spelling.fix_parties(result, [page])
    assert result["fields"]["buyer"]["value"] == {"name": "Kavya Logistics Pvt. Ltd", "address": "Nagar"}
    assert "Lovistics → Logistics" in result["warnings"][0] and "Naggr → Nagar" in result["warnings"][0]

    # every letter read surely: kept as read (only the fixed patterns apply)
    read[0] = ("M/s. Kavya Lovistics Pvt. L+d", 0.99, SimpleNamespace(confs=[0.99] * 29))
    result["fields"]["buyer"]["value"] = {"name": "Kavya Lovistics Pvt. L+d", "address": None}
    spelling.fix_parties(result, [page])
    assert result["fields"]["buyer"]["value"]["name"] == "Kavya Lovistics Pvt. Ltd"


def test_text_from_a_pdf_text_layer_is_never_changed():
    page = Page(0, [Word("Lovistics", 0.1, 0.1, 0.2, 0.12, 1.0)], "digital", 1.0)
    result = {"fields": {"seller": {"value": {"name": "Kavya Lovistics", "address": "maombai"}, "page": 1}}}
    spelling.fix_parties(result, [page])
    assert result["fields"]["seller"]["value"] == {"name": "Kavya Lovistics", "address": "maombai"}
    assert "warnings" not in result
