"""MySQL persistence helpers for application data and prediction history."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import mysql.connector
from mysql.connector import Error

from backend.config import DB_CONFIG

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Open short-lived MySQL connections and manage prediction history."""

    def _connect(self):
        """Return a new connection so requests do not retain stale connections."""
        if not all(DB_CONFIG[key] for key in ("host", "user", "database")):
            raise Error("Database configuration is incomplete")
        return mysql.connector.connect(**DB_CONFIG)

    def initialize(self) -> bool:
        """Create application tables when the configured database is reachable."""
        connection = None
        try:
            connection = self._connect()
            self.create_tables(connection)
            return True
        except Error:
            logger.exception("Unable to initialize MySQL tables")
            return False
        finally:
            if connection and connection.is_connected():
                connection.close()

    @staticmethod
    def create_tables(connection) -> None:
        """Create tables without deleting or recreating existing data."""
        cursor = connection.cursor()
        try:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(120) NOT NULL,
                    email VARCHAR(254) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    token_hash CHAR(64) NOT NULL UNIQUE,
                    expires_at DATETIME NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_auth_sessions_user
                        FOREIGN KEY (user_id) REFERENCES users(id)
                        ON DELETE CASCADE,
                    INDEX idx_auth_sessions_expiry (expires_at),
                    INDEX idx_auth_sessions_user (user_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    customer_id VARCHAR(50) NULL,
                    model_used VARCHAR(50) NOT NULL,
                    prediction TINYINT NOT NULL,
                    probability DECIMAL(7,6) NOT NULL,
                    risk_level VARCHAR(20) NOT NULL,
                    features JSON NOT NULL,
                    migration_key CHAR(64) NULL UNIQUE,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_predictions_created_at (created_at)
                )
                """
            )
            connection.commit()
        finally:
            cursor.close()

    def create_user(self, name: str, email: str, password_hash: str) -> Dict[str, Any]:
        connection = cursor = None
        try:
            connection = self._connect()
            self.create_tables(connection)
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s)",
                (name, email, password_hash),
            )
            connection.commit()
            cursor.execute(
                "SELECT id, name, email FROM users WHERE id = %s",
                (cursor.lastrowid,),
            )
            return cursor.fetchone()
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        return self._get_user("SELECT id, name, email, password_hash FROM users WHERE email = %s", (email,))

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        return self._get_user("SELECT id, name, email FROM users WHERE id = %s", (user_id,))

    def _get_user(self, query: str, params: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
        connection = cursor = None
        try:
            connection = self._connect()
            self.create_tables(connection)
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, params)
            return cursor.fetchone()
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def create_session(self, user_id: int, token_hash: str, expires_at: datetime) -> None:
        connection = cursor = None
        try:
            connection = self._connect()
            self.create_tables(connection)
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO auth_sessions (user_id, token_hash, expires_at) VALUES (%s, %s, %s)",
                (user_id, token_hash, expires_at),
            )
            connection.commit()
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def get_user_by_session(self, token_hash: str) -> Optional[Dict[str, Any]]:
        connection = cursor = None
        try:
            connection = self._connect()
            self.create_tables(connection)
            cursor = connection.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT users.id, users.name, users.email
                FROM auth_sessions
                INNER JOIN users ON users.id = auth_sessions.user_id
                WHERE auth_sessions.token_hash = %s
                  AND auth_sessions.expires_at > UTC_TIMESTAMP()
                """,
                (token_hash,),
            )
            return cursor.fetchone()
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def delete_session(self, token_hash: str) -> None:
        connection = cursor = None
        try:
            connection = self._connect()
            self.create_tables(connection)
            cursor = connection.cursor()
            cursor.execute("DELETE FROM auth_sessions WHERE token_hash = %s", (token_hash,))
            connection.commit()
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def save_prediction(
        self,
        customer_id: Optional[str],
        model_used: str,
        prediction: int,
        probability: float,
        risk_level: str,
        features: Dict[str, Any],
        *,
        created_at: Optional[datetime] = None,
        migration_key: Optional[str] = None,
    ) -> bool:
        """Persist a prediction; return whether the row was newly inserted."""
        connection = cursor = None
        try:
            connection = self._connect()
            self.create_tables(connection)
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO predictions
                    (customer_id, model_used, prediction, probability, risk_level,
                     features, migration_key, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, COALESCE(%s, CURRENT_TIMESTAMP))
                ON DUPLICATE KEY UPDATE id = id
                """,
                (customer_id, model_used, int(prediction), float(probability), risk_level,
                 json.dumps(features), migration_key, created_at),
            )
            inserted = cursor.rowcount == 1
            connection.commit()
            return inserted
        except Error:
            if connection:
                connection.rollback()
            logger.exception("Unable to save prediction history")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def get_prediction_history(
        self, page: int = 1, page_size: int = 20
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Return newest-first history in the legacy UI's column format."""
        page, page_size = max(1, page), min(max(1, page_size), 100)
        connection = cursor = None
        try:
            connection = self._connect()
            self.create_tables(connection)
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT COUNT(*) AS count FROM predictions")
            total = cursor.fetchone()["count"]
            cursor.execute(
                """
                SELECT customer_id, model_used, prediction, probability, risk_level,
                       features, created_at AS timestamp
                FROM predictions
                ORDER BY created_at DESC, id DESC
                LIMIT %s OFFSET %s
                """,
                (page_size, (page - 1) * page_size),
            )
            history = cursor.fetchall()
            for row in history:
                if isinstance(row.get("features"), (dict, list)):
                    row["features"] = json.dumps(row["features"])
            return history, total
        except Error:
            logger.exception("Unable to load prediction history")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()

    def count_predictions_today(self) -> int:
        """Return the number of predictions recorded since the DB server's midnight."""
        connection = cursor = None
        try:
            connection = self._connect()
            self.create_tables(connection)
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE created_at >= CURDATE()")
            return int(cursor.fetchone()[0])
        except Error:
            logger.exception("Unable to count today's predictions")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection and connection.is_connected():
                connection.close()
