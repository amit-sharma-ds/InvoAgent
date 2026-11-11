import os
import sys
import tempfile
from pathlib import Path

# Tests use a throw-away data folder (database, stored files, label cache), never the real data/.
os.environ["INVOICE_DATA_DIR"] = tempfile.mkdtemp(prefix="invoice_parser_tests_")

# Make the `invoice_parser` package and app.py importable when running `pytest` from the project folder.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# A throw-away user for the web tests (lives in the temporary data folder above).
TEST_USER, TEST_PASSWORD = "tester", "test-password-1"


def login(client):
    """Sign a Flask test client in, creating the test user on first use."""
    from invoice_parser import auth

    if not auth.verify(TEST_USER, TEST_PASSWORD):
        auth.set_password(TEST_USER, TEST_PASSWORD)
    resp = client.post("/login", data={"username": TEST_USER, "password": TEST_PASSWORD})
    assert resp.status_code == 302, resp.data[:300]
    return client
