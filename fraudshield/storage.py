"""SQLite prediction history and human review state."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

ReviewStatus = Literal["Pending review", "Under investigation", "Confirmed fraud", "Dismissed"]
STATUSES = ["Pending review", "Under investigation", "Confirmed fraud", "Dismissed"]


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY, transaction_id TEXT NOT NULL, model_version TEXT NOT NULL,
                    suspected INTEGER NOT NULL, risk REAL NOT NULL, threshold REAL NOT NULL,
                    payload TEXT NOT NULL, provider TEXT NOT NULL,
                    analyzed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE TABLE IF NOT EXISTS investigations (
                    id INTEGER PRIMARY KEY, transaction_id TEXT NOT NULL, model_version TEXT NOT NULL,
                    prediction_id INTEGER NOT NULL REFERENCES predictions(id),
                    UNIQUE(transaction_id, model_version)
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    case_id INTEGER PRIMARY KEY REFERENCES investigations(id),
                    status TEXT NOT NULL CHECK (status IN
                      ('Pending review','Under investigation','Confirmed fraud','Dismissed')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    id INTEGER PRIMARY KEY, case_id INTEGER NOT NULL REFERENCES investigations(id),
                    prior_status TEXT, status TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '',
                    changed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE INDEX IF NOT EXISTS reviews_status_idx ON reviews(status);
                CREATE INDEX IF NOT EXISTS review_events_case_idx ON review_events(case_id, id DESC);
                CREATE INDEX IF NOT EXISTS predictions_date_idx ON predictions(analyzed_at);
                INSERT INTO review_events(case_id,status,changed_at)
                    SELECT case_id,status,updated_at FROM reviews
                    WHERE NOT EXISTS (SELECT 1 FROM review_events e WHERE e.case_id=reviews.case_id);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, transaction_id, result):
        with self.connect() as db:
            row = db.execute(
                "INSERT INTO predictions(transaction_id,model_version,suspected,risk,threshold,payload,provider)"
                " VALUES(?,?,?,?,?,?,?)",
                (transaction_id, result["model_version"], result["suspected_fraud"], result["risk"],
                 result["threshold"], json.dumps(result), result["explanation"]["provider"]),
            )
            if result["suspected_fraud"]:
                db.execute(
                    "INSERT INTO investigations(transaction_id,model_version,prediction_id) VALUES(?,?,?) "
                    "ON CONFLICT(transaction_id,model_version) DO UPDATE SET prediction_id=excluded.prediction_id",
                    (transaction_id, result["model_version"], row.lastrowid),
                )
                inserted = db.execute(
                    "INSERT OR IGNORE INTO reviews(case_id,status) SELECT id,'Pending review' FROM investigations "
                    "WHERE transaction_id=? AND model_version=?", (transaction_id, result["model_version"]),
                )
                if inserted.rowcount:
                    db.execute("INSERT INTO review_events(case_id,status) SELECT id,'Pending review' "
                               "FROM investigations WHERE transaction_id=? AND model_version=?",
                               (transaction_id, result["model_version"]))

    def records(self, search="", status=None, limit=100, offset=0, history=False, sort="recent"):
        order = {"recent": "i.id DESC", "high": "p.risk DESC, i.id DESC",
                 "low": "p.risk ASC, i.id DESC"}[sort]
        with self.connect() as db:
            if history:
                rows = db.execute(
                    "SELECT * FROM predictions WHERE instr(lower(transaction_id),lower(?))>0 "
                    "ORDER BY id DESC LIMIT ? OFFSET ?", (search, limit, offset),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT i.id,p.transaction_id,p.model_version,p.risk,p.threshold,p.payload,p.provider,"
                    "p.analyzed_at,r.status,r.updated_at,"
                    "(SELECT e.notes FROM review_events e WHERE e.case_id=i.id ORDER BY e.id DESC LIMIT 1) AS notes "
                    "FROM investigations i "
                    "JOIN predictions p ON p.id=i.prediction_id JOIN reviews r ON r.case_id=i.id "
                    "WHERE instr(lower(p.transaction_id),lower(?))>0 AND (? IS NULL OR r.status=?) "
                    f"ORDER BY {order} LIMIT ? OFFSET ?", (search, status, status, limit, offset),
                ).fetchall()
        return [dict(row) for row in rows]

    def activity(self):
        with self.connect() as db:
            analyzed = db.execute("SELECT COUNT(DISTINCT transaction_id) FROM predictions").fetchone()[0]
            statuses = dict(db.execute("SELECT status, COUNT(*) FROM reviews GROUP BY status").fetchall())
            recent = db.execute(
                "SELECT substr(analyzed_at,1,10) AS day, COUNT(*) AS analyses, SUM(suspected) AS flagged "
                "FROM predictions GROUP BY day ORDER BY day DESC LIMIT 14"
            ).fetchall()
        return {"analyzed": analyzed, "cases": sum(statuses.values()),
                "pending": statuses.get("Pending review", 0), "confirmed": statuses.get("Confirmed fraud", 0),
                "dismissed": statuses.get("Dismissed", 0)}, [dict(row) for row in reversed(recent)]

    def review_history(self, case_id):
        with self.connect() as db:
            events = db.execute("SELECT prior_status,status,notes,changed_at FROM review_events "
                                "WHERE case_id=? ORDER BY id DESC", (case_id,)).fetchall()
        return [dict(event) for event in events]

    def review_status(self, transaction_id, model_version):
        with self.connect() as db:
            row = db.execute(
                "SELECT r.status FROM reviews r JOIN investigations i ON i.id=r.case_id "
                "WHERE i.transaction_id=? AND i.model_version=?", (transaction_id, model_version)
            ).fetchone()
        return row["status"] if row else None

    def review(self, case_id, status, notes=None):
        if status not in STATUSES:
            raise ValueError("Invalid review status")
        if notes is not None and len(notes) > 1500:
            raise ValueError("Review notes exceed 1,500 characters")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT status FROM reviews WHERE case_id=?", (case_id,)).fetchone()
            if not old:
                return False
            last = db.execute("SELECT notes FROM review_events WHERE case_id=? ORDER BY id DESC LIMIT 1",
                              (case_id,)).fetchone()
            old_note = last["notes"] if last else ""
            notes = old_note if notes is None else notes.strip()
            if old["status"] == status and old_note == notes:
                return True
            db.execute(
                "UPDATE reviews SET status=?,updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE case_id=?",
                (status, case_id),
            )
            db.execute("INSERT INTO review_events(case_id,prior_status,status,notes) VALUES(?,?,?,?)",
                       (case_id, old["status"], status, notes))
            return True
