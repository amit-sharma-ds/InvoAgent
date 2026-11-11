"""Users and sign-in.

Users live in data/users.json as {"username": {"password_hash": ..., "role": ...}}. Only salted
password hashes are stored (werkzeug / scrypt), never the passwords. Manage users with

    python scripts/set_password.py admin          # asks for the new password twice

The Flask session secret is random and kept in data/secret_key (or FLASK_SECRET_KEY), so a session
cookie cannot be forged and sign-ins survive a restart.
"""
import json
import secrets
import threading
import time
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

from . import config

USERS_PATH = config.DATA_DIR / "users.json"
SECRET_PATH = config.DATA_DIR / "secret_key"
MIN_PASSWORD_LENGTH = 8

# brute-force guard: after this many wrong passwords from one address, wait LOCK_SECONDS
MAX_FAILURES = 5
LOCK_SECONDS = 15 * 60

_lock = threading.Lock()
_failures = {}  # address -> [count, first failure time]


def load_users(path: Path = None):
    path = path or USERS_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def has_users(path: Path = None):
    return bool(load_users(path))


def set_password(username, password, role="admin", path: Path = None):
    """Create the user or change its password."""
    username = (username or "").strip()
    if not username:
        raise ValueError("Username is empty.")
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must have at least {MIN_PASSWORD_LENGTH} characters.")
    path = path or USERS_PATH
    with _lock:
        users = load_users(path)
        users[username] = {"password_hash": generate_password_hash(password), "role": role}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(users, indent=2), encoding="utf-8")
        tmp.replace(path)


def bootstrap_admin_from_env(path: Path = None):
    """On a server without a console (Render, Docker): create the first user from the environment
    variables INVOICE_ADMIN_USER and INVOICE_ADMIN_PASSWORD. Only when no user exists yet, so a
    password changed later in the app's data folder is never overwritten. Returns True if created."""
    import os

    username = os.environ.get("INVOICE_ADMIN_USER", "").strip()
    password = os.environ.get("INVOICE_ADMIN_PASSWORD", "")
    if not username or not password or has_users(path):
        return False
    set_password(username, password, path=path)
    return True


def verify(username, password, path: Path = None):
    """The user record when the password is right, else None."""
    user = load_users(path).get((username or "").strip())
    if user and check_password_hash(user["password_hash"], password or ""):
        return user
    return None


def secret_key():
    """FLASK_SECRET_KEY, else a random key created once and kept in data/secret_key."""
    import os

    if os.environ.get("FLASK_SECRET_KEY"):
        return os.environ["FLASK_SECRET_KEY"]
    try:
        return SECRET_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        key = secrets.token_hex(32)
        SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
        SECRET_PATH.write_text(key, encoding="utf-8")
        return key


def locked_out(address):
    """Seconds left before this address may try again (0 = allowed)."""
    with _lock:
        count, since = _failures.get(address, (0, 0.0))
        if count >= MAX_FAILURES:
            left = LOCK_SECONDS - (time.time() - since)
            if left > 0:
                return int(left) + 1
            _failures.pop(address, None)
    return 0


def record_failure(address):
    with _lock:
        count, since = _failures.get(address, (0, time.time()))
        if time.time() - since > LOCK_SECONDS:
            count, since = 0, time.time()
        _failures[address] = (count + 1, since)


def record_success(address):
    with _lock:
        _failures.pop(address, None)
