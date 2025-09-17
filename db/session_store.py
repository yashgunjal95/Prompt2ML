import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path
from config.settings import DB_PATH

logger = logging.getLogger(__name__)

class SessionStore:
    """Handle database operations for session management."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self) -> None:
        """Initialize the database with proper schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prompt TEXT NOT NULL,
                        dataset_summary TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        target_column TEXT NOT NULL,
                        features TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index for better query performance
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created_at 
                    ON sessions(created_at DESC)
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def save_session(self, prompt: str, summary: str, task_type: str, 
                    target: str, features: List[str]) -> int:
        """Save a new session to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sessions (prompt, dataset_summary, task_type, target_column, features)
                    VALUES (?, ?, ?, ?, ?)
                """, (prompt, summary, task_type, target, json.dumps(features)))
                
                session_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Session {session_id} saved successfully")
                return session_id
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            raise
    
    def fetch_sessions(self, limit: int = 50) -> List[Tuple]:
        """Fetch recent sessions from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, prompt, dataset_summary, task_type, target_column, features, created_at
                    FROM sessions 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
                
                sessions = cursor.fetchall()
                logger.info(f"Fetched {len(sessions)} sessions")
                return sessions
        except Exception as e:
            logger.error(f"Failed to fetch sessions: {e}")
            return []
    
    def get_session_by_id(self, session_id: int) -> Optional[Tuple]:
        """Get specific session by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, prompt, dataset_summary, task_type, target_column, features, created_at
                    FROM sessions 
                    WHERE id = ?
                """, (session_id,))
                
                session = cursor.fetchone()
                return session
        except Exception as e:
            logger.error(f"Failed to fetch session {session_id}: {e}")
            return None