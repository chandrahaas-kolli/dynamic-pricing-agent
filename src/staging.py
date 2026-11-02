"""SQLite persistence for the pricing agent: decision history, per-product
monthly state, open escalations, and per-attempt LLM call usage/cost. All
SQL for the staging layer lives here; callers never write SQL directly.

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
        input_hash        TEXT    NOT NULL,
        inputs            TEXT    NOT NULL,
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
        detail      TEXT,
        status      TEXT NOT NULL DEFAULT 'open',
        created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
"""

# One row per LLM attempt that got an answer (invoke_with_retries' llm_usage
# records); an attempt that raised (ClientError/timeout) is never recorded.
# input_tokens/output_tokens/estimated_cost_usd are nullable -- meaning
# "unknown", never 0. Prices are stored per row, not looked up from current
# settings, so a later price change never rewrites the cost of past rows.
CREATE_LLM_CALLS = """
    CREATE TABLE IF NOT EXISTS llm_calls (
        id                     INTEGER PRIMARY KEY,
        run_id                 TEXT    NOT NULL,
        product_id             TEXT    NOT NULL,
        observed_at            TEXT    NOT NULL,
        call                   TEXT    NOT NULL,
        attempt                INTEGER NOT NULL,
        input_tokens           INTEGER,
        output_tokens          INTEGER,
        outcome                TEXT    NOT NULL,
        request_id             TEXT,
        input_price_per_mtok   REAL    NOT NULL,
        output_price_per_mtok  REAL    NOT NULL,
        estimated_cost_usd     REAL,
        created_at             TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
"""


def init_db(conn):
    """Create the decisions, monthly_state, escalations and llm_calls tables if they don't already exist."""
    with conn:
        conn.execute(CREATE_DECISIONS)
        conn.execute(CREATE_MONTHLY_STATE)
        conn.execute(CREATE_ESCALATIONS)
        conn.execute(CREATE_LLM_CALLS)


def record_llm_calls(conn, rows):
    """Insert rows into llm_calls, one per answered LLM attempt.

    Each row is a tuple in this column order: run_id, product_id,
    observed_at, call, attempt, input_tokens, output_tokens, outcome,
    request_id, input_price_per_mtok, output_price_per_mtok,
    estimated_cost_usd. id and created_at are filled by SQLite. None for
    input_tokens, output_tokens or estimated_cost_usd means unknown, never 0.

    rows is empty when no LLM call was made (e.g. the direct path): this
    returns without touching the DB. Otherwise all rows insert in one
    transaction, so a row that violates a NOT NULL constraint rolls back
    the whole batch instead of leaving a partial write.
    """
    if not rows:
        return
    with conn:
        conn.executemany(
            """
            INSERT INTO llm_calls
                (run_id, product_id, observed_at, call, attempt, input_tokens,
                 output_tokens, outcome, request_id, input_price_per_mtok,
                 output_price_per_mtok, estimated_cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def run_cost(conn, run_id):
    """Return this run's total LLM cost: {"estimated_cost_usd": float, "unknown_cost_calls": int}.

    estimated_cost_usd is the sum, over every llm_calls row for run_id, of
    estimated_cost_usd (an estimate at list price, not billing truth).
    unknown_cost_calls is the count of those rows whose cost is NULL
    (unknown, from a missing token count -- never 0). A run with no rows
    returns 0.0 and 0.
    """
    # SQL SUM skips NULL costs silently, so it alone would under-report the
    # true total without saying so; count the NULL rows separately instead
    # of hiding them.
    cursor = conn.execute(
        "SELECT COALESCE(SUM(estimated_cost_usd), 0.0), "
        "COUNT(*) - COUNT(estimated_cost_usd) "
        "FROM llm_calls WHERE run_id = ?",
        (run_id,),
    )
    row = cursor.fetchone()
    return {"estimated_cost_usd": row[0], "unknown_cost_calls": row[1]}


def decision_cost(conn, run_id, product_id, observed_at):
    """Return one observation's LLM cost: {"estimated_cost_usd": float, "unknown_cost_calls": int}.

    Same as run_cost, but filtered to the llm_calls rows for this exact
    run_id/product_id/observed_at -- one observation's attempts across
    every call it made, not the whole run.
    """
    # Same NULL handling as run_cost: SUM skips NULL costs silently, so
    # unknown_cost_calls counts those rows instead of letting them vanish.
    cursor = conn.execute(
        "SELECT COALESCE(SUM(estimated_cost_usd), 0.0), "
        "COUNT(*) - COUNT(estimated_cost_usd) "
        "FROM llm_calls WHERE run_id = ? AND product_id = ? AND observed_at = ?",
        (run_id, product_id, observed_at),
    )
    row = cursor.fetchone()
    return {"estimated_cost_usd": row[0], "unknown_cost_calls": row[1]}


def record_decision(
    conn,
    product_id,
    observed_at,
    outcome,
    price,
    escalation_cause,
    tier,
    reason,
    brief,
    input_hash,
    inputs,
):
    """Insert one row into decisions.

    observed_at stores the canonical ISO 8601 UTC key (canonical_timestamp's
    output), not the caller's raw timestamp text; it is the idempotency key.
    brief is stored as its JSON text (json.dumps), or NULL when brief is None.
    input_hash is the observation_fingerprint of this run's inputs, used by
    the idempotency gate to detect a resend vs a genuine data conflict.
    inputs is the dict those inputs were computed from (comp_prices,
    comp_ratings, our_rating), stored as its JSON text, kept alongside the
    hash so a conflict can be explained without recomputing anything.
    (product_id, observed_at) is unique: inserting a duplicate raises
    sqlite3.IntegrityError.
    """
    brief_json = json.dumps(brief) if brief is not None else None
    inputs_json = json.dumps(inputs)
    with conn:
        conn.execute(
            """
            INSERT INTO decisions
                (product_id, observed_at, outcome, price, escalation_cause, tier, reason, brief, input_hash, inputs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                product_id,
                observed_at,
                outcome,
                price,
                escalation_cause,
                tier,
                reason,
                brief_json,
                input_hash,
                inputs_json,
            ),
        )


def find_decision(conn, product_id, observed_at):
    """Return the decisions row for product_id/observed_at as a dict, or None if it doesn't exist.

    observed_at is the canonical ISO 8601 UTC key, not raw caller text; pass
    the same canonical_timestamp output used when the row was recorded.
    brief and inputs are returned as their parsed dicts (json.loads); brief
    is None when NULL.
    """
    cursor = conn.execute(
        """
        SELECT id, product_id, observed_at, outcome, price, escalation_cause, tier, reason, brief,
               input_hash, inputs, created_at
        FROM decisions
        WHERE product_id = ? AND observed_at = ?
        """,
        (product_id, observed_at),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [description[0] for description in cursor.description]
    decision = dict(zip(columns, row, strict=True))
    decision["brief"] = (
        json.loads(decision["brief"]) if decision["brief"] is not None else None
    )
    decision["inputs"] = json.loads(decision["inputs"])
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
    return dict(zip(columns, row, strict=True))


def open_escalation(conn, product_id, cause, detail=None):
    """Insert a new open escalation for product_id/cause and return its id.

    detail is stored as its JSON text (json.dumps), or NULL when detail is None.
    """
    detail_json = json.dumps(detail) if detail is not None else None
    with conn:
        cursor = conn.execute(
            "INSERT INTO escalations (product_id, cause, detail) VALUES (?, ?, ?)",
            (product_id, cause, detail_json),
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
