import sqlite3
import os
import json
import logging
import threading
from datetime import datetime
from contextlib import contextmanager
from typing import Generator, Optional

# Setup logging
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), 'research_projects.db')

# Thread-local storage for connections
_local = threading.local()

def _get_thread_connection() -> sqlite3.Connection:
    """Get a connection for the current thread."""
    if not hasattr(_local, 'conn') or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, timeout=5.0, isolation_level=None)
        _local.conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA busy_timeout=30000")
        logger.debug(f"Created new database connection for thread {threading.current_thread().name}")
    return _local.conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database connections using thread-local storage.

    Usage:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects")
            ...
    """
    conn = _get_thread_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db():
    """Initialize the SQLite database with required tables."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Create projects table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            completed_at TEXT,
            sources_count INTEGER DEFAULT 0,
            charts_count INTEGER DEFAULT 0,
            processing_time_seconds INTEGER DEFAULT 0
        )
        ''')

        # Create project_files table to track files associated with each project
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS project_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (project_id) REFERENCES projects (id)
        )
        ''')

        # Create project_metadata table for additional project data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS project_metadata (
            project_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            PRIMARY KEY (project_id, key),
            FOREIGN KEY (project_id) REFERENCES projects (id)
        )
        ''')

        logger.info("Database initialized successfully")

def create_project(project_id, query):
    """Create a new project record in the database."""
    with get_db() as conn:
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO projects (id, query, created_at, status) VALUES (?, ?, ?, ?)",
            (project_id, query, now, "in_progress")
        )

        logger.info(f"Created project record: {project_id}")
        return True

def update_project_status(project_id, status, metadata=None):
    """Update a project's status and optionally add metadata."""
    try:
        conn = _get_thread_connection()

        now = datetime.now().isoformat()
        if status == "completed":
            conn.execute(
                "UPDATE projects SET status = ?, completed_at = ? WHERE id = ?",
                (status, now, project_id)
            )
        else:
            conn.execute(
                "UPDATE projects SET status = ? WHERE id = ?",
                (status, project_id)
            )

        # Add metadata if provided
        if metadata and isinstance(metadata, dict):
            for key, value in metadata.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                conn.execute(
                    "INSERT OR REPLACE INTO project_metadata (project_id, key, value) VALUES (?, ?, ?)",
                    (project_id, key, str(value))
                )

        logger.info(f"Updated project {project_id} status to {status}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Error updating project {project_id}: {e}")
        return False

def register_project_file(project_id, file_path, file_type):
    """Register a file associated with a project."""
    try:
        conn = _get_thread_connection()

        now = datetime.now().isoformat()
        conn.execute(
            "INSERT INTO project_files (project_id, file_path, file_type, created_at) VALUES (?, ?, ?, ?)",
            (project_id, file_path, file_type, now)
        )

        # Update counts based on file type
        if file_type == "chart":
            conn.execute(
                "UPDATE projects SET charts_count = charts_count + 1 WHERE id = ?",
                (project_id,)
            )

        logger.info(f"Registered {file_type} file for project {project_id}: {file_path}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Error registering file for project {project_id}: {e}")
        return False

def update_project_metrics(project_id, sources_count=None, processing_time=None):
    """Update project metrics like sources count and processing time."""
    try:
        conn = _get_thread_connection()

        updates = []
        params = []

        if sources_count is not None:
            updates.append("sources_count = ?")
            params.append(sources_count)

        if processing_time is not None:
            updates.append("processing_time_seconds = ?")
            params.append(processing_time)

        if updates:
            query = f"UPDATE projects SET {', '.join(updates)} WHERE id = ?"
            params.append(project_id)
            conn.execute(query, params)

            logger.info(f"Updated metrics for project {project_id}")
            return True
        return False
    except sqlite3.Error as e:
        logger.error(f"Error updating metrics for project {project_id}: {e}")
        return False

def get_project(project_id):
    """Get a project by ID with its metadata."""
    try:
        conn = _get_thread_connection()
        conn.row_factory = sqlite3.Row

        # Get project data
        cursor = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
        project = cursor.fetchone()

        if not project:
            return None

        # Convert to dict
        project_dict = dict(project)

        # Get metadata
        cursor = conn.execute("SELECT key, value FROM project_metadata WHERE project_id = ?", (project_id,))
        metadata = cursor.fetchall()

        # Add metadata to project dict
        project_dict['metadata'] = {row['key']: row['value'] for row in metadata}

        # Get files
        cursor = conn.execute("SELECT file_path, file_type FROM project_files WHERE project_id = ?", (project_id,))
        files = cursor.fetchall()

        # Group files by type
        project_dict['files'] = {}
        for row in files:
            file_type = row['file_type']
            if file_type not in project_dict['files']:
                project_dict['files'][file_type] = []
            project_dict['files'][file_type].append(row['file_path'])

        return project_dict
    except sqlite3.Error as e:
        logger.error(f"Error getting project {project_id}: {e}")
        return None

def get_all_projects(limit=50, offset=0, status=None):
    """Get all projects with optional filtering by status."""
    try:
        conn = _get_thread_connection()
        conn.row_factory = sqlite3.Row

        query = "SELECT * FROM projects"
        params = []

        if status:
            query += " WHERE status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.execute(query, params)
        projects = [dict(row) for row in cursor.fetchall()]

        return projects
    except sqlite3.Error as e:
        logger.error(f"Error getting projects: {e}")
        return []

def get_project_stats():
    """Get statistics about projects."""
    try:
        conn = _get_thread_connection()

        # Get total counts
        cursor = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
            SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress,
            SUM(sources_count) as total_sources,
            SUM(charts_count) as total_charts,
            AVG(processing_time_seconds) as avg_processing_time
        FROM projects
        """)

        result = cursor.fetchone()

        stats = {
            "total_projects": result[0] or 0,
            "completed_projects": result[1] or 0,
            "failed_projects": result[2] or 0,
            "in_progress_projects": result[3] or 0,
            "total_sources": result[4] or 0,
            "total_charts": result[5] or 0,
            "avg_processing_time": result[6] or 0
        }

        # Get recent projects
        cursor = conn.execute("""
        SELECT id, query, created_at, status
        FROM projects
        ORDER BY created_at DESC
        LIMIT 5
        """)

        recent_projects = []
        for row in cursor.fetchall():
            recent_projects.append({
                "id": row[0],
                "query": row[1],
                "created_at": row[2],
                "status": row[3]
            })

        stats["recent_projects"] = recent_projects

        return stats
    except sqlite3.Error as e:
        logger.error(f"Error getting project stats: {e}")
        return {
            "total_projects": 0,
            "completed_projects": 0,
            "failed_projects": 0,
            "in_progress_projects": 0,
            "total_sources": 0,
            "total_charts": 0,
            "avg_processing_time": 0,
            "recent_projects": []
        }

def delete_project(project_id):
    """Delete a project and all associated data."""
    try:
        conn = _get_thread_connection()

        # Delete project files
        conn.execute("DELETE FROM project_files WHERE project_id = ?", (project_id,))

        # Delete project metadata
        conn.execute("DELETE FROM project_metadata WHERE project_id = ?", (project_id,))

        # Delete project
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

        logger.info(f"Deleted project {project_id}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Error deleting project {project_id}: {e}")
        return False

# Initialize the database when this module is imported
init_db()