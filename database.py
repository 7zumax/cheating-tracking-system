"""SQLite storage for alerts (alerts.db)."""
import sqlite3

DB_PATH = "alerts.db"


def get_connection(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def create_table(db_path=DB_PATH):
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime        TEXT    NOT NULL,
                video_name      TEXT,
                student_id      INTEGER,
                behavior        TEXT    NOT NULL,
                confidence      REAL,
                video_time      REAL,
                screenshot_path TEXT
            )
        """)


def save_alert(event, video_name, screenshot_path, db_path=DB_PATH):
    """Insert an alert and return its id."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO alerts
               (datetime, video_name, student_id, behavior,
                confidence, video_time, screenshot_path)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (event["datetime"], video_name, event["student_id"], event["behavior"],
             event["confidence"], event["video_time"], screenshot_path),
        )
        return cur.lastrowid


def get_all_alerts(db_path=DB_PATH):
    """All alerts, newest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


def get_alert(alert_id, db_path=DB_PATH):
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    return dict(row) if row else None


def clear_alerts(db_path=DB_PATH):
    """Delete all alerts (screenshot files are kept)."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM alerts")
