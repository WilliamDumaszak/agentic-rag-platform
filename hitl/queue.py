"""
Human-in-the-Loop (HITL) review queue — SQLite-backed.

Why HITL?
  LLMs are probabilistic. Some responses have low confidence — the model itself
  signals uncertainty. Rather than silently returning a low-quality answer,
  we route uncertain responses to a human review queue.
  A reviewer can approve the answer (it gets served), correct it, or reject it.

Architecture:
  Local: SQLite file (hitl_queue.db) — zero infrastructure, works offline.
  Production: swap get_connection() to return a psycopg2/asyncpg connection
              pointing at the same PostgreSQL used by llm-serving-monitoring.

Table schema:
  id              — auto-increment primary key
  query           — original user question
  answer          — LLM-generated answer pending review
  confidence      — confidence score that triggered the routing
  source          — retrieval source (chroma, web, etc.)
  prompt_hash     — SHA-256 of the prompt template version used
  status          — "pending" | "approved" | "rejected"
  reviewed_by     — reviewer identifier (optional)
  review_note     — reviewer free-text note (optional)
  created_at      — UTC timestamp
  reviewed_at     — UTC timestamp (null until reviewed)
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

_DB_PATH = CONFIG.get("hitl", {}).get("db_path", "hitl_queue.db")


# ── Connection ────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row  # return dicts instead of tuples
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────

def setup_table() -> None:
    """Create hitl_queue table if it does not exist."""
    conn = get_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hitl_queue (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                query        TEXT    NOT NULL,
                answer       TEXT    NOT NULL,
                confidence   REAL    NOT NULL,
                source       TEXT    DEFAULT 'unknown',
                prompt_hash  TEXT    DEFAULT '',
                status       TEXT    NOT NULL DEFAULT 'pending'
                             CHECK (status IN ('pending', 'approved', 'rejected')),
                reviewed_by  TEXT,
                review_note  TEXT,
                created_at   TEXT    NOT NULL,
                reviewed_at  TEXT
            )
        """)
        conn.commit()
        logger.info("HITL queue table ready.")
    except Exception as exc:
        logger.error(f"HITL table setup failed: {exc}")
        conn.rollback()
    finally:
        conn.close()


# ── Write ─────────────────────────────────────────────────────────────────────

def enqueue(
    query: str,
    answer: str,
    confidence: float,
    source: str = "unknown",
    prompt_hash: str = "",
) -> int:
    """
    Insert a low-confidence response into the review queue.

    Returns:
        The new row id (used in approve/reject endpoints).
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            """INSERT INTO hitl_queue (query, answer, confidence, source, prompt_hash, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (query, answer, confidence, source, prompt_hash,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        row_id = cur.lastrowid
        logger.info(f"Enqueued HITL item id={row_id}, confidence={confidence:.3f}")
        return row_id
    except Exception as exc:
        conn.rollback()
        logger.error(f"Failed to enqueue HITL item: {exc}")
        return -1
    finally:
        conn.close()


# ── Read ──────────────────────────────────────────────────────────────────────

def get_pending() -> list[dict]:
    """Return all items with status='pending', oldest first."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT * FROM hitl_queue WHERE status = 'pending' ORDER BY created_at ASC"
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_item(item_id: int) -> dict | None:
    """Fetch a single queue item by id."""
    conn = get_connection()
    try:
        cur = conn.execute("SELECT * FROM hitl_queue WHERE id = ?", (item_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Review actions ────────────────────────────────────────────────────────────

def approve(item_id: int, reviewed_by: str = "", review_note: str = "") -> bool:
    """Mark a pending item as approved."""
    return _update_status(item_id, "approved", reviewed_by, review_note)


def reject(item_id: int, reviewed_by: str = "", review_note: str = "") -> bool:
    """Mark a pending item as rejected."""
    return _update_status(item_id, "rejected", reviewed_by, review_note)


def _update_status(
    item_id: int,
    status: str,
    reviewed_by: str,
    review_note: str,
) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            """UPDATE hitl_queue
               SET status = ?, reviewed_by = ?, review_note = ?, reviewed_at = ?
               WHERE id = ? AND status = 'pending'""",
            (status, reviewed_by, review_note,
             datetime.now(timezone.utc).isoformat(), item_id),
        )
        conn.commit()
        updated = cur.rowcount > 0
        if updated:
            logger.info(f"HITL item id={item_id} → {status}")
        else:
            logger.warning(f"HITL item id={item_id} not found or already reviewed.")
        return updated
    except Exception as exc:
        conn.rollback()
        logger.error(f"Failed to update HITL item {item_id}: {exc}")
        return False
    finally:
        conn.close()
