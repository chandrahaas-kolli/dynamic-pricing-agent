"""SQLite persistence for the pricing agent: decision history, per-product
monthly state, and open escalations. All SQL for the staging layer lives
here; callers never write SQL directly.

Every function takes an open sqlite3 connection and never opens one itself,
so callers control the connection's lifetime (and can point it at a real
file or an in-memory DB for tests).
"""

import json

CREATE_DECISIONS = """
    CREATE TABLE IF NOT EXISTS decisions (
        id                INTEGER PRIMARY KEY,
        product_id        TEXT    NOT NULL,
        observed_at       TEXT    NOT NULL,
        outcome           TEXT    NOT NULL,
        price             REAL,
        escalation_cause  TEXT,
        tier              TEXT,
        reason            TEXT    NOT NULL,
        brief             TEXT,
        created_at        TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (product_id, observed_at)
    )
"""

CREATE_MONTHLY_STATE = """
    CREATE TABLE IF NOT EXISTS monthly_state (
        product_id     TEXT NOT NULL,
        month          TEXT NOT NULL,
        anchor_price   REAL NOT NULL,
        current_price  REAL NOT NULL,
        PRIMARY KEY (product_id, month)
    )
"""

CREATE_ESCALATIONS = """
    CREATE TABLE IF NOT EXISTS escalations (
        id          INTEGER PRIMARY KEY,
        product_id  TEXT NOT NULL,
        cause       TEXT NOT NULL,
        status      TEXT NOT NULL DEFAULT 'open',
        created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
"""


def init_db(conn):
    """Create the decisions, monthly_state and escalations tables if they don't already exist."""
    with conn:
        conn.execute(CREATE_DECISIONS)
        conn.execute(CREATE_MONTHLY_STATE)
        conn.execute(CREATE_ESCALATIONS)


def record_decision(conn, product_id, observed_at, outcome, price, escalation_cause, tier, reason, brief):
    """Insert one row into decisions.

    brief is stored as its JSON text (json.dumps), or NULL when brief is None.
    (product_id, observed_at) is unique: inserting a duplicate raises
    sqlite3.IntegrityError.
    """
    brief_json = json.dumps(brief) if brief is not None else None
    with conn:
        conn.execute(
            """
            INSERT INTO decisions
                (product_id, observed_at, outcome, price, escalation_cause, tier, reason, brief)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (product_id, observed_at, outcome, price, escalation_cause, tier, reason, brief_json),
        )


def find_decision(conn, product_id, observed_at):
    """Return the decisions row for product_id/observed_at as a dict, or None if it doesn't exist.

    brief is returned as its parsed dict (json.loads), or None when NULL.
    """
    cursor = conn.execute(
        """
        SELECT id, product_id, observed_at, outcome, price, escalation_cause, tier, reason, brief, created_at
        FROM decisions
        WHERE product_id = ? AND observed_at = ?
        """,
        (product_id, observed_at),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [description[0] for description in cursor.description]
    decision = dict(zip(columns, row))
    decision["brief"] = json.loads(decision["brief"]) if decision["brief"] is not None else None
    return decision


def upsert_monthly_state(conn, product_id, month, anchor_price, current_price):
    """Insert the monthly state row for product_id/month, or update it if it already exists.

    anchor_price is frozen for the month once set: on conflict, only
    current_price is updated.
    """
    with conn:
        conn.execute(
            """
            INSERT INTO monthly_state
                (product_id, month, anchor_price, current_price)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (product_id, month) DO UPDATE SET
                current_price = excluded.current_price
            """,
            (product_id, month, anchor_price, current_price),
        )


def get_monthly_state(conn, product_id, month):
    """Return the monthly_state row for product_id/month as a dict, or None if it doesn't exist."""
    cursor = conn.execute(
        """
        SELECT product_id, month, anchor_price, current_price
        FROM monthly_state
        WHERE product_id = ? AND month = ?
        """,
        (product_id, month),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [description[0] for description in cursor.description]
    return dict(zip(columns, row))


def open_escalation(conn, product_id, cause):
    """Insert a new open escalation for product_id/cause and return its id."""
    with conn:
        cursor = conn.execute(
            "INSERT INTO escalations (product_id, cause) VALUES (?, ?)",
            (product_id, cause),
        )
        return cursor.lastrowid


def resolve_escalation(conn, escalation_id):
    """Mark the escalation with escalation_id as resolved."""
    with conn:
        conn.execute(
            "UPDATE escalations SET status = 'resolved' WHERE id = ?",
            (escalation_id,),
        )


def has_open_escalation(conn, product_id):
    """Return True if product_id has any escalation with status 'open'."""
    row = conn.execute(
        "SELECT 1 FROM escalations WHERE product_id = ? AND status = 'open' LIMIT 1",
        (product_id,),
    ).fetchone()
    return row is not None
