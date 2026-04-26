"""
api/data_pipeline.py
Data ingestion, transformation, and retrieval pipeline.
Data flows: ingest -> validate -> transform -> store -> fetch -> process -> return
"""

import json
import uuid
from datetime import datetime
from db.database import Database


class DataPipeline:
    """
    Orchestrates data flow from ingestion to retrieval.
    Applies validation and transformation steps before storage.
    """

    REQUIRED_FIELDS = ["name", "value", "category"]

    def __init__(self, db: Database):
        self.db = db
        self._transform_registry: dict[str, callable] = {
            "normalize": self._normalize,
            "enrich": self._enrich,
            "sanitize": self._sanitize,
        }

    def ingest(self, raw_payload: dict) -> str:
        """
        Full ingestion pipeline: validate -> transform -> store.
        Returns the new record's ID.
        """
        self._validate(raw_payload)
        transformed = self._run_transforms(raw_payload)
        record_id = str(uuid.uuid4())
        transformed["id"] = record_id
        transformed["created_at"] = datetime.utcnow().isoformat()
        self.db.insert_record(transformed)
        return record_id

    def fetch_and_process(self, limit: int = 100) -> list[dict]:
        """
        Fetch records from DB and apply post-processing.
        """
        raw_records = self.db.fetch_records(limit=limit)
        return [self._post_process(r) for r in raw_records]

    def _validate(self, payload: dict):
        for field in self.REQUIRED_FIELDS:
            if field not in payload:
                raise ValueError(f"Missing required field: {field}")
        if not isinstance(payload["value"], (int, float)):
            raise TypeError("Field 'value' must be numeric")

    def _run_transforms(self, payload: dict) -> dict:
        result = dict(payload)
        for step in ["sanitize", "normalize", "enrich"]:
            fn = self._transform_registry[step]
            result = fn(result)
        return result

    def _normalize(self, data: dict) -> dict:
        data["name"] = data["name"].strip().lower()
        data["category"] = data["category"].strip().upper()
        return data

    def _enrich(self, data: dict) -> dict:
        data["_enriched"] = True
        data["value_squared"] = data["value"] ** 2
        return data

    def _sanitize(self, data: dict) -> dict:
        data["name"] = data["name"].replace("<", "").replace(">", "")
        return data

    def _post_process(self, record: dict) -> dict:
        record.pop("_enriched", None)
        return record