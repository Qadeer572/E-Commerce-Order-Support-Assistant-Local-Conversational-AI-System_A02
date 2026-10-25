"""SQLite-backed customer memory (bonus CRM): profile, preferences and past interactions.

All values are fictional demo data. Real PII would need authentication and encryption.
"""
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

from backend.config import CRM_DB_FILE

PREFERENCE_KEYS = ("preferred_carrier", "favorite_category", "preferred_color",
                   "preferred_size", "shipping_speed", "language")
SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    user_id TEXT PRIMARY KEY, name TEXT, email TEXT, phone TEXT,
    preferences_json TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
    ts REAL NOT NULL, intent TEXT, order_id TEXT, sku TEXT, summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id, id);
"""
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_NAME = re.compile(r"^[^\W\d_]+(?:[ '\-][^\W\d_]+)*$")


class CRMValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def clean(value: str) -> str:
    return _CONTROL.sub("", value).strip()


def reject_sensitive(value: str) -> None:
    """Refuse payment-card-like digit runs and anything that mentions a password."""
    if re.search(r"\d{13,19}", re.sub(r"[ -]", "", value)):
        raise CRMValidationError("sensitive_data", "I can't store card numbers. Please don't share payment details.")
    if re.search(r"\b(?:password|passcode|pin)\b", value, re.I):
        raise CRMValidationError("sensitive_data", "I can't store passwords or PINs.")


def validate_fields(name=None, email=None, phone=None, preference_key=None, preference_value=None) -> dict:
    """Return cleaned, validated fields (only the ones supplied)."""
    fields: dict = {}
    for value in (name, email, phone, preference_value):
        if value is not None:
            reject_sensitive(value)
    if name is not None:
        name = clean(name)
        if not name or len(name) > 60 or not _NAME.match(name):
            raise CRMValidationError("invalid_name", "That doesn't look like a valid name.")
        fields["name"] = name
    if email is not None:
        email = clean(email)
        if not _EMAIL.match(email) or len(email) > 100:
            raise CRMValidationError("invalid_email", "That email address doesn't look valid.")
        fields["email"] = email
    if phone is not None:
        phone = clean(phone)
        digits = re.sub(r"\D", "", phone)
        if not 7 <= len(digits) <= 15 or not re.fullmatch(r"\+?[\d ()-]+", phone):
            raise CRMValidationError("invalid_phone", "That phone number doesn't look valid.")
        fields["phone"] = phone
    if preference_key is not None or preference_value is not None:
        if preference_key not in PREFERENCE_KEYS:
            raise CRMValidationError("invalid_preference", "I can only remember carrier, category, color, size, "
                                                           "shipping speed or language preferences.")
        value = clean(preference_value or "")
        if not value or len(value) > 60:
            raise CRMValidationError("invalid_preference", "That preference value is empty or too long.")
        fields["preference"] = (preference_key, value)
    return fields


class CRMStore:
    def __init__(self, db_path: Path = CRM_DB_FILE):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        return conn

    def get(self, user_id: str) -> Optional[dict]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM customers WHERE user_id=?", (user_id,)).fetchone()
            last = conn.execute("SELECT intent, order_id, sku, summary FROM interactions "
                                "WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        profile = {"name": row["name"], "email": row["email"],
                   "phone": row["phone"],
                   "preferences": json.loads(row["preferences_json"]),
                   "last_interaction": dict(last) if last else None}
        return profile

    def update(self, user_id: str, fields: dict) -> dict:
        """Upsert validated fields (from validate_fields) and return the saved changes."""
        now = time.time()
        conn = self._connect()
        try:
            with conn:
                row = conn.execute("SELECT preferences_json FROM customers WHERE user_id=?", (user_id,)).fetchone()
                prefs = json.loads(row["preferences_json"]) if row else {}
                if "preference" in fields:
                    prefs[fields["preference"][0]] = fields["preference"][1]
                if row is None:
                    conn.execute("INSERT INTO customers(user_id, preferences_json, created_at, updated_at) "
                                 "VALUES (?,?,?,?)", (user_id, "{}", now, now))
                for column in ("name", "email", "phone"):
                    if column in fields:
                        conn.execute(f"UPDATE customers SET {column}=? WHERE user_id=?", (fields[column], user_id))
                conn.execute("UPDATE customers SET preferences_json=?, updated_at=? WHERE user_id=?",
                             (json.dumps(prefs), now, user_id))
        finally:
            conn.close()
        saved = {key: value for key, value in fields.items() if key != "preference"}
        if "preference" in fields:
            saved[fields["preference"][0]] = fields["preference"][1]
        return saved

    def add_interaction(self, user_id: str, session_id: str, intent: str, order_id: Optional[str],
                        sku: Optional[str], summary: str) -> None:
        conn = self._connect()
        try:
            with conn:
                row = conn.execute("SELECT 1 FROM customers WHERE user_id=?", (user_id,)).fetchone()
                if row is None:
                    return
                conn.execute("INSERT INTO interactions(user_id, session_id, ts, intent, order_id, sku, summary) "
                             "VALUES (?,?,?,?,?,?,?)",
                             (user_id, session_id, time.time(), intent, order_id, sku, clean(summary)[:160]))
        finally:
            conn.close()

    def delete(self, user_id: str) -> bool:
        conn = self._connect()
        try:
            with conn:
                gone = conn.execute("DELETE FROM customers WHERE user_id=?", (user_id,)).rowcount
                gone += conn.execute("DELETE FROM interactions WHERE user_id=?", (user_id,)).rowcount
        finally:
            conn.close()
        return bool(gone)


def welcome_line(profile: Optional[dict]) -> str:
    """Deterministic greeting from stored data; a 0.5B model cannot be trusted to recall a name."""
    if not profile or not profile.get("name"):
        return ""
    line = f"Welcome back, {profile['name']}!"
    last = profile.get("last_interaction")
    if last and last.get("order_id"):
        line += f" Last time you asked about order {last['order_id']}."
    elif last and last.get("sku"):
        line += f" Last time you asked about product {last['sku']}."
    elif last and last.get("summary"):
        line += f' Last time you asked: "{last["summary"]}"'
    return line
