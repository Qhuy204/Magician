"""Lưu job / request gốc / event log / webhook outbox vào SQLite để restart không mất việc."""
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    state TEXT NOT NULL,
    idem_key TEXT,
    created_at REAL NOT NULL,
    finished_at REAL,
    data TEXT NOT NULL,
    request TEXT
);
CREATE INDEX IF NOT EXISTS jobs_client ON jobs(client_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS jobs_idem ON jobs(client_id, idem_key) WHERE idem_key IS NOT NULL;
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    smart_ad_id TEXT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    message TEXT,
    data TEXT
);
CREATE INDEX IF NOT EXISTS events_job ON events(job_id, id);
CREATE TABLE IF NOT EXISTS webhooks (
    event_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    url TEXT NOT NULL,
    body BLOB NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_at REAL NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    last_status TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS webhooks_due ON webhooks(state, next_at);
"""


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, args)

    def _all(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    # --- jobs ---
    def save_job(self, job_id: str, client_id: str, state: str, created_at: float, finished_at: float | None,
                 data: str, idem_key: str | None = None, request: str | None = None) -> None:
        self._exec(
            "INSERT INTO jobs(job_id, client_id, state, idem_key, created_at, finished_at, data, request) "
            "VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET state=excluded.state, "
            "finished_at=excluded.finished_at, data=excluded.data, request=COALESCE(excluded.request, jobs.request)",
            (job_id, client_id, state, idem_key, created_at, finished_at, data, request),
        )

    def load_jobs(self, since: float) -> list[sqlite3.Row]:
        return self._all("SELECT * FROM jobs WHERE created_at >= ? OR state IN ('queued','running') ORDER BY created_at", (since,))

    def has_job(self, job_id: str) -> bool:
        return bool(self._all("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)))

    def purge_jobs(self, before: float) -> list[str]:
        rows = self._all("SELECT job_id FROM jobs WHERE finished_at IS NOT NULL AND finished_at < ?", (before,))
        ids = [r["job_id"] for r in rows]
        for jid in ids:
            self._exec("DELETE FROM events WHERE job_id=?", (jid,))
            self._exec("DELETE FROM webhooks WHERE job_id=?", (jid,))
            self._exec("DELETE FROM jobs WHERE job_id=?", (jid,))
        return ids

    # --- events ---
    def add_event(self, job_id: str, smart_ad_id: str | None, kind: str, message: str = "", data: dict | None = None) -> None:
        self._exec("INSERT INTO events(job_id, smart_ad_id, ts, kind, message, data) VALUES(?,?,?,?,?,?)",
                   (job_id, smart_ad_id, time.time(), kind, message[:2000],
                    json.dumps(data, ensure_ascii=False) if data else None))

    def events(self, job_id: str, after: int = 0, limit: int = 500) -> list[dict]:
        rows = self._all("SELECT * FROM events WHERE job_id=? AND id>? ORDER BY id LIMIT ?", (job_id, after, limit))
        return [{"id": r["id"], "ts": r["ts"], "smart_ad_id": r["smart_ad_id"], "kind": r["kind"],
                 "message": r["message"], "data": json.loads(r["data"]) if r["data"] else None} for r in rows]

    # --- webhook outbox ---
    def add_webhook(self, event_id: str, job_id: str, kind: str, url: str, body: bytes) -> None:
        now = time.time()
        self._exec("INSERT OR IGNORE INTO webhooks(event_id, job_id, kind, url, body, next_at, created_at) VALUES(?,?,?,?,?,?,?)",
                   (event_id, job_id, kind, url, body, now, now))

    def due_webhooks(self, now: float, limit: int = 20) -> list[sqlite3.Row]:
        # Mỗi job gửi tuần tự theo thứ tự tạo: event sau chờ event trước của cùng job gửi xong (hoặc bỏ hẳn)
        return self._all(
            "SELECT * FROM webhooks w WHERE state='pending' AND next_at<=? AND NOT EXISTS ("
            "SELECT 1 FROM webhooks e WHERE e.job_id=w.job_id AND e.state='pending' AND e.rowid<w.rowid) "
            "ORDER BY next_at LIMIT ?", (now, limit))

    def update_webhook(self, event_id: str, attempts: int, state: str, next_at: float, last_status: str) -> None:
        self._exec("UPDATE webhooks SET attempts=?, state=?, next_at=?, last_status=? WHERE event_id=?",
                   (attempts, state, next_at, last_status, event_id))

    def webhooks(self, job_id: str) -> list[dict]:
        rows = self._all("SELECT event_id, kind, attempts, state, last_status, created_at, next_at FROM webhooks "
                         "WHERE job_id=? ORDER BY created_at", (job_id,))
        return [dict(r) for r in rows]

    def requeue_webhooks(self, job_id: str, kind: str | None = None) -> int:
        sql, args = "UPDATE webhooks SET state='pending', attempts=0, next_at=? WHERE job_id=?", [time.time(), job_id]
        if kind:
            sql += " AND kind=?"
            args.append(kind)
        return self._exec(sql, tuple(args)).rowcount

    def webhook_stats(self) -> dict[str, int]:
        return {r["state"]: r["n"] for r in self._all("SELECT state, COUNT(*) n FROM webhooks GROUP BY state")}
