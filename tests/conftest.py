import os
import sys
import tempfile
from pathlib import Path

# Tests use a throw-away data folder (database, stored files, label cache), never the real data/.
os.environ["INVOICE_DATA_DIR"] = tempfile.mkdtemp(prefix="invoice_parser_tests_")
# ... and never the real MongoDB: SQLite, unless TEST_MONGODB_URI names a throw-away MongoDB server,
# in which case the app and the storage tests use a fresh database there (dropped at the end).
TEST_MONGODB_URI = os.environ.get("TEST_MONGODB_URI", "")
if TEST_MONGODB_URI:
    os.environ["MONGODB_URI"] = TEST_MONGODB_URI
    os.environ["MONGODB_DB"] = f"invoagent_test_{os.getpid()}"
else:
    os.environ.pop("MONGODB_URI", None)


def pytest_sessionfinish(session, exitstatus):
    if TEST_MONGODB_URI:
        from pymongo import MongoClient

        client = MongoClient(TEST_MONGODB_URI, serverSelectionTimeoutMS=5000)
        for name in client.list_database_names():
            if name.startswith("invoagent_test_") or name.startswith("invoagent_t_"):
                client.drop_database(name)

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
