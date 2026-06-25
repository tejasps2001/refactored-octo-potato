import sqlite3
import os
import queue
import threading

class SessionLogger:
    def __init__(self, db_filename="session_data.sqlite3"):
        self.db_path = os.path.join(os.path.dirname(__file__), db_filename)
        self._initialize_db()
        
        # Asynchronous queue and worker thread for thread-isolated sequential database writes
        self.write_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._write_worker, daemon=True)
        self.worker_thread.start()

    def _initialize_db(self):
        """Create schemas and migrate if necessary"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if video_engagement_logs exists and has session_id column
        cursor.execute("PRAGMA table_info(video_engagement_logs)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if columns and "session_id" not in columns:
            # Schema migration: Drop table to recreate with session_id
            cursor.execute("DROP TABLE video_engagement_logs")
            
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_engagement_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                video_timestamp REAL NOT NULL,
                emotion_state TEXT NOT NULL,
                transcript_segment TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_navigation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp_from REAL NOT NULL,
                timestamp_to REAL NOT NULL,
                event_type TEXT NOT NULL,
                transcript_segment TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                student_emotion TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS struggle_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                struggle_reason TEXT NOT NULL,
                transcript_segment TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS post_video_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                questions_json TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()

    def _write_worker(self):
        """Background thread worker to process database writes sequentially"""
        while True:
            func, args = self.write_queue.get()
            try:
                func(*args)
            except Exception as e:
                print(f"[ERROR] Database async write worker exception: {e}")
            finally:
                self.write_queue.task_done()
                
    def _submit_write(self, func, *args):
        """Queue a write operation to execute sequentially on the background thread"""
        self.write_queue.put((func, args))

    def log_engagement(self, session_id: str, video_timestamp: float, emotion_state: str, 
                      transcript_segment: str):
        self._submit_write(self._sync_log_engagement, session_id, video_timestamp, emotion_state, transcript_segment)

    def _sync_log_engagement(self, session_id: str, video_timestamp: float, emotion_state: str, 
                            transcript_segment: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO video_engagement_logs (session_id, video_timestamp, emotion_state,
                transcript_segment) VALUES (?, ?, ?, ?)
        ''', (session_id, video_timestamp, emotion_state, transcript_segment))
        conn.commit()
        conn.close()

    def log_navigation(self, session_id: str, timestamp_from: float, timestamp_to: float, 
                       event_type: str, transcript_segment: str):
        self._submit_write(self._sync_log_navigation, session_id, timestamp_from, timestamp_to, event_type, transcript_segment)

    def _sync_log_navigation(self, session_id: str, timestamp_from: float, timestamp_to: float, 
                            event_type: str, transcript_segment: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO video_navigation_logs (session_id, timestamp_from, timestamp_to,
                event_type, transcript_segment) VALUES (?, ?, ?, ?, ?)
        ''', (session_id, timestamp_from, timestamp_to, event_type, transcript_segment))
        conn.commit()
        conn.close()

    def log_chat(self, session_id: str, question: str, answer: str, student_emotion: str):
        self._submit_write(self._sync_log_chat, session_id, question, answer, student_emotion)

    def _sync_log_chat(self, session_id: str, question: str, answer: str, student_emotion: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chat_logs (session_id, question, answer, student_emotion)
            VALUES (?, ?, ?, ?)
        ''', (session_id, question, answer, student_emotion))
        conn.commit()
        conn.close()

    def log_struggle(self, session_id: str, struggle_reason: str, transcript_segment: str):
        self._submit_write(self._sync_log_struggle, session_id, struggle_reason, transcript_segment)

    def _sync_log_struggle(self, session_id: str, struggle_reason: str, transcript_segment: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO struggle_logs (session_id, struggle_reason, transcript_segment)
            VALUES (?, ?, ?)
        ''', (session_id, struggle_reason, transcript_segment))
        conn.commit()
        conn.close()

    def save_post_video_questions(self, session_id: str, questions_json: str):
        self._submit_write(self._sync_save_post_video_questions, session_id, questions_json)

    def _sync_save_post_video_questions(self, session_id: str, questions_json: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO post_video_questions (session_id, questions_json)
            VALUES (?, ?)
        ''', (session_id, questions_json))
        conn.commit()
        conn.close()

    def get_engagement_spikes(self, session_id: str) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT transcript_segment, COUNT(id) as count
            FROM video_engagement_logs
            WHERE session_id = ? AND emotion_state IN ('Frustration', 'Confusion')
            GROUP BY transcript_segment
        ''', (session_id,))
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_navigation_rewinds(self, session_id: str) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT transcript_segment, COUNT(id) as count
            FROM video_navigation_logs
            WHERE session_id = ? AND timestamp_to < timestamp_from
            GROUP BY transcript_segment
        ''', (session_id,))
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_chat_history(self, session_id: str) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT question, answer, student_emotion
            FROM chat_logs
            WHERE session_id = ?
        ''', (session_id,))
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_struggle_logs(self, session_id: str) -> list:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT struggle_reason, transcript_segment
            FROM struggle_logs
            WHERE session_id = ?
        ''', (session_id,))
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_post_video_questions(self, session_id: str) -> str | None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT questions_json
            FROM post_video_questions
            WHERE session_id = ?
            ORDER BY id DESC LIMIT 1
        ''', (session_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
