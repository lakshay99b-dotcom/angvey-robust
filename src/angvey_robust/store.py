"""Persistent storage for reflections and suggestions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any

from .models import ReflectionRecord, ImprovementSuggestion, FailureCategory


class ReflectionStore:
    """
    Simple local store backed by SQLite.

    Designed to be lightweight and dependency-free (stdlib only).
    """

    def __init__(self, db_path: str | Path = "angvey_robust.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reflections (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    task TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    confidence REAL,
                    what_worked TEXT,
                    what_failed TEXT,
                    root_cause TEXT,
                    lessons TEXT,
                    suggested_improvement TEXT,
                    tools_used TEXT,
                    steps INTEGER,
                    duration_seconds REAL,
                    raw_output TEXT,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS suggestions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    title TEXT,
                    description TEXT,
                    category TEXT,
                    priority TEXT,
                    based_on_reflections TEXT,
                    status TEXT,
                    metadata TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reflections_task ON reflections(task)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reflections_success ON reflections(success)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_reflections_root_cause ON reflections(root_cause)"
            )
            conn.commit()

    def add_reflection(self, record: ReflectionRecord) -> str:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reflections (
                    id, timestamp, task, success, confidence,
                    what_worked, what_failed, root_cause, lessons,
                    suggested_improvement, tools_used, steps,
                    duration_seconds, raw_output, tags, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.timestamp,
                    record.task,
                    int(record.success),
                    record.confidence,
                    record.what_worked,
                    record.what_failed,
                    record.root_cause.value,
                    record.lessons,
                    record.suggested_improvement,
                    json.dumps(record.tools_used),
                    record.steps,
                    record.duration_seconds,
                    record.raw_output,
                    json.dumps(record.tags),
                    json.dumps(record.metadata),
                ),
            )
            conn.commit()
        return record.id

    def get_reflection(self, reflection_id: str) -> Optional[ReflectionRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reflections WHERE id = ?", (reflection_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_reflection(row)

    def list_reflections(
        self,
        limit: int = 50,
        success: Optional[bool] = None,
        root_cause: Optional[FailureCategory] = None,
    ) -> List[ReflectionRecord]:
        query = "SELECT * FROM reflections WHERE 1=1"
        params: list[Any] = []

        if success is not None:
            query += " AND success = ?"
            params.append(int(success))
        if root_cause is not None:
            query += " AND root_cause = ?"
            params.append(root_cause.value)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_reflection(r) for r in rows]

    def search_similar(self, task: str, limit: int = 5) -> List[ReflectionRecord]:
        """
        Very simple similarity: keyword overlap on the task text.
        Good enough for v0.1; can be replaced later with embeddings.
        """
        keywords = set(task.lower().split())
        if not keywords:
            return []

        all_refs = self.list_reflections(limit=200)
        scored = []
        for ref in all_refs:
            ref_words = set(ref.task.lower().split())
            overlap = len(keywords & ref_words)
            if overlap > 0:
                scored.append((overlap, ref))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ref for _, ref in scored[:limit]]

    def add_suggestion(self, suggestion: ImprovementSuggestion) -> str:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO suggestions (
                    id, created_at, title, description, category,
                    priority, based_on_reflections, status, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    suggestion.id,
                    suggestion.created_at,
                    suggestion.title,
                    suggestion.description,
                    suggestion.category,
                    suggestion.priority,
                    json.dumps(suggestion.based_on_reflections),
                    suggestion.status,
                    json.dumps(suggestion.metadata),
                ),
            )
            conn.commit()
        return suggestion.id

    def list_suggestions(
        self, status: Optional[str] = None, limit: int = 20
    ) -> List[ImprovementSuggestion]:
        query = "SELECT * FROM suggestions WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_suggestion(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]
            successes = conn.execute(
                "SELECT COUNT(*) FROM reflections WHERE success = 1"
            ).fetchone()[0]
            failures = total - successes
            by_cause = conn.execute(
                """
                SELECT root_cause, COUNT(*) as cnt
                FROM reflections
                WHERE success = 0
                GROUP BY root_cause
                ORDER BY cnt DESC
                """
            ).fetchall()

        return {
            "total_reflections": total,
            "successes": successes,
            "failures": failures,
            "success_rate": round(successes / total, 3) if total else 0.0,
            "failure_breakdown": {row[0]: row[1] for row in by_cause},
        }

    def _row_to_reflection(self, row: sqlite3.Row) -> ReflectionRecord:
        return ReflectionRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            task=row["task"],
            success=bool(row["success"]),
            confidence=row["confidence"] or 0.5,
            what_worked=row["what_worked"] or "",
            what_failed=row["what_failed"] or "",
            root_cause=FailureCategory(row["root_cause"] or "unknown"),
            lessons=row["lessons"] or "",
            suggested_improvement=row["suggested_improvement"] or "",
            tools_used=json.loads(row["tools_used"] or "[]"),
            steps=row["steps"] or 0,
            duration_seconds=row["duration_seconds"],
            raw_output=row["raw_output"] or "",
            tags=json.loads(row["tags"] or "[]"),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def _row_to_suggestion(self, row: sqlite3.Row) -> ImprovementSuggestion:
        return ImprovementSuggestion(
            id=row["id"],
            created_at=row["created_at"],
            title=row["title"] or "",
            description=row["description"] or "",
            category=row["category"] or "prompt",
            priority=row["priority"] or "medium",
            based_on_reflections=json.loads(row["based_on_reflections"] or "[]"),
            status=row["status"] or "proposed",
            metadata=json.loads(row["metadata"] or "{}"),
        )
