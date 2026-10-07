"""SQLite persistence for the pricing agent: decision history and per-product
monthly state. All SQL for the staging layer lives here; callers never write
SQL directly.

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
        created_at        TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
"""

CREATE_MONTHLY_STATE = """
    CREATE TABLE IF NOT EXISTS monthly_state (
        product_id          TEXT    NOT NULL,
        month               TEXT    NOT NULL,
        anchor_price        REAL    NOT NULL,
        current_price       REAL    NOT NULL,
        pending_escalation  INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (product_id, month)
    )
"""


def init_db(conn):
    """Create the decisions and monthly_state tables if they don't already exist."""
    with conn:
        conn.execute(CREATE_DECISIONS)
        conn.execute(CREATE_MONTHLY_STATE)


def record_decision(conn, product_id, observed_at, outcome, price, escalation_cause, tier, reason, brief):
    """Insert one row into decisions.

    brief is stored as its JSON text (json.dumps), or NULL when brief is None.
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


def upsert_monthly_state(conn, product_id, month, anchor_price, current_price, pending_escalation):
    """Insert the monthly state row for product_id/month, or update it if it already exists.

    anchor_price is frozen for the month once set: on conflict, only
    current_price and pending_escalation are updated. pending_escalation is
    set to MAX(existing, new) on conflict, so a later run can never clear a
    pending escalation on its own; only a human resolving it can (step 12).
    """
    with conn:
        conn.execute(
            """
            INSERT INTO monthly_state
                (product_id, month, anchor_price, current_price, pending_escalation)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (product_id, month) DO UPDATE SET
                current_price = excluded.current_price,
                pending_escalation = MAX(monthly_state.pending_escalation, excluded.pending_escalation)
            """,
            (product_id, month, anchor_price, current_price, pending_escalation),
        )


def get_monthly_state(conn, product_id, month):
    """Return the monthly_state row for product_id/month as a dict, or None if it doesn't exist.

    pending_escalation is returned as a bool (stored as INTEGER 0/1).
    """
    cursor = conn.execute(
        """
        SELECT product_id, month, anchor_price, current_price, pending_escalation
        FROM monthly_state
        WHERE product_id = ? AND month = ?
        """,
        (product_id, month),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [description[0] for description in cursor.description]
    state = dict(zip(columns, row))
    state["pending_escalation"] = bool(state["pending_escalation"])
    return state
