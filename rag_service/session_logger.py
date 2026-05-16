import sqlite3
import os

class SessionLogger:
    def __init__(self, db_filename="session_data.sqlite3"):
        self.db_path = os.path.join(os.path.dirname(__file__), db_filename)
        self._initialize_db()

    def _initialize_db(self):
        """Create the schema if it doesn't already exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_engagement_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_timestamp REAL NOT NULL,
                emotion_state TEXT NOT NULL,
                transcript_segment TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def log_engagement(self, video_timestamp:float, emotion_state: str, 
                  transcript_segment: str):
        """Insert a new synchronized data point into the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO video_engagement_logs (timestamp, emotion_state,
                transcript_segment) VALUES (?, ?, ?, ?)
            )
        ''', (video_timestamp, emotion_state, transcript_segment))
        
        conn.commit()
        conn.close()
