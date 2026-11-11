"""Create a user for the web app or change a user's password.

Usage (from the project folder, with the venv active):
    python scripts/set_password.py admin           # asks for the password twice (not shown on screen)
    python scripts/set_password.py --list          # shows the user names
    python scripts/set_password.py --delete olduser

Passwords are stored only as salted hashes in data/users.json. The running app picks up a change
at the next sign-in; no restart is needed.
"""
import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invoice_parser import auth  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("username", nargs="?")
    ap.add_argument("--list", action="store_true", help="show the user names")
    ap.add_argument("--delete", metavar="USERNAME", help="remove a user")
    ap.add_argument("--stdin", action="store_true", help="read the password from standard input (for scripts)")
    args = ap.parse_args()

    if args.list:
        for name, user in sorted(auth.load_users().items()):
            print(f"{name}  ({user.get('role', 'admin')})")
        return
    if args.delete:
        users = auth.load_users()
        if args.delete not in users:
            sys.exit(f"No user called {args.delete!r}.")
        if len(users) == 1:
            sys.exit("This is the only user: create another one first, or nobody can sign in.")
        del users[args.delete]
        import json
        auth.USERS_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")
        print(f"Removed {args.delete}.")
        return
    if not args.username:
        ap.error("give a username, e.g.  python scripts/set_password.py admin")

    if args.stdin:
        password = sys.stdin.readline().rstrip("\r\n")
    else:
        password = getpass.getpass(f"New password for {args.username}: ")
        if getpass.getpass("Repeat it: ") != password:
            sys.exit("The two passwords differ. Nothing was changed.")
    try:
        existed = args.username in auth.load_users()
        auth.set_password(args.username, password)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"Password {'changed' if existed else 'set'} for {args.username}.")


if __name__ == "__main__":
    main()
