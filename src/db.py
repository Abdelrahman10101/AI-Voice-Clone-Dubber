import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.config import DB_PATH

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

# Alias for backward compatibility
get_db = get_connection

def get_latest_job_for_file(input_file: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE input_file = ? ORDER BY created_at DESC LIMIT 1", (input_file,))
        row = cursor.fetchone()
        return dict(row) if row else None

def init_db():
    """Initialize database tables for jobs and chunk tracking."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Jobs table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            input_file TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            total_chunks INTEGER DEFAULT 0,
            model_stt TEXT,
            model_llm TEXT,
            model_tts TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Chunks table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            duration REAL NOT NULL,
            arabic_text TEXT DEFAULT '',
            english_text TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            chunk_audio_path TEXT,
            cloned_audio_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE,
            UNIQUE(job_id, chunk_index)
        );
        """)
        conn.commit()

def create_job(job_id: str, input_file: str, output_dir: str, model_stt: str, model_llm: str, model_tts: str) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO jobs (id, input_file, output_dir, status, model_stt, model_llm, model_tts)
        VALUES (?, ?, ?, 'pending', ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            status = 'pending',
            updated_at = CURRENT_TIMESTAMP
        """, (job_id, input_file, output_dir, model_stt, model_llm, model_tts))
        conn.commit()

def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def update_job_status(job_id: str, status: str, total_chunks: Optional[int] = None) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        if total_chunks is not None:
            cursor.execute("""
            UPDATE jobs SET status = ?, total_chunks = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
            """, (status, total_chunks, job_id))
        else:
            cursor.execute("""
            UPDATE jobs SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
            """, (status, job_id))
        conn.commit()

def save_chunks_metadata(job_id: str, segments: List[Dict[str, Any]]) -> None:
    """Save or update audio chunk time ranges."""
    with get_connection() as conn:
        cursor = conn.cursor()
        for seg in segments:
            cursor.execute("""
            INSERT INTO chunks (job_id, chunk_index, start_time, end_time, duration, chunk_audio_path, status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending')
            ON CONFLICT(job_id, chunk_index) DO UPDATE SET
                start_time = excluded.start_time,
                end_time = excluded.end_time,
                duration = excluded.duration,
                chunk_audio_path = excluded.chunk_audio_path
            """, (
                job_id,
                seg["index"],
                seg["start"],
                seg["end"],
                seg["end"] - seg["start"],
                seg.get("audio_path", "")
            ))
        conn.commit()

def get_chunks(job_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chunks WHERE job_id = ? ORDER BY chunk_index ASC", (job_id,))
        return [dict(row) for row in cursor.fetchall()]

def update_chunk_stt(job_id: str, chunk_index: int, arabic_text: str) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE chunks
        SET arabic_text = ?, status = 'transcribed'
        WHERE job_id = ? AND chunk_index = ?
        """, (arabic_text, job_id, chunk_index))
        conn.commit()

def update_chunk_translation(job_id: str, chunk_index: int, english_text: str) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE chunks
        SET english_text = ?, status = 'translated'
        WHERE job_id = ? AND chunk_index = ?
        """, (english_text, job_id, chunk_index))
        conn.commit()

def update_chunk_tts(job_id: str, chunk_index: int, cloned_audio_path: str) -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE chunks
        SET cloned_audio_path = ?, status = 'synthesized'
        WHERE job_id = ? AND chunk_index = ?
        """, (cloned_audio_path, job_id, chunk_index))
        conn.commit()
