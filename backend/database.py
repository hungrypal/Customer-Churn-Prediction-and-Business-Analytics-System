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
