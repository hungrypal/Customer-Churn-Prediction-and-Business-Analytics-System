"""One-time migration of legacy ``user_predictions.csv`` records to MySQL.

Run manually after setting DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, and DB_NAME:
    python scripts/migrate_prediction_history_csv.py
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from mysql.connector import Error

from backend.database import DatabaseManager


REQUIRED_COLUMNS = {
    "customer_id", "model_used", "prediction", "probability", "risk_level", "timestamp"
}


def parse_timestamp(value: str) -> datetime:
    """Parse pandas' usual CSV timestamp formats without changing its value."""
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def migrate(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not REQUIRED_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
            raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

        database = DatabaseManager()
        if not database.initialize():
            raise RuntimeError("Could not connect to or initialize the configured MySQL database")

        inserted = failed = skipped = 0
        for line_number, row in enumerate(reader, start=2):
            try:
                raw_features = row.get("features") or "{}"
                features = json.loads(raw_features)
                fingerprint = hashlib.sha256(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                was_inserted = database.save_prediction(
                    customer_id=row["customer_id"],
                    model_used=row["model_used"],
                    prediction=int(row["prediction"]),
                    probability=float(row["probability"]),
                    risk_level=row["risk_level"],
                    features=features,
                    created_at=parse_timestamp(row["timestamp"]),
                    migration_key=fingerprint,
                )
                if was_inserted:
                    inserted += 1
                else:
                    skipped += 1
            except (ValueError, TypeError, KeyError, json.JSONDecodeError, Error) as error:
                failed += 1
                print(f"Row {line_number} was not migrated: {error}", file=sys.stderr)

    print(f"Migration complete: {inserted} inserted, {skipped} already present, {failed} failed.")
    return failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate legacy prediction history to MySQL")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "user_predictions.csv",
        help="Path to the legacy prediction-history CSV",
    )
    arguments = parser.parse_args()
    if not arguments.csv.is_file():
        print(f"Legacy CSV not found: {arguments.csv}", file=sys.stderr)
        sys.exit(1)
    try:
        sys.exit(1 if migrate(arguments.csv) else 0)
    except (ValueError, RuntimeError) as error:
        print(f"Migration failed: {error}", file=sys.stderr)
        sys.exit(1)
