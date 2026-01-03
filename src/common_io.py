from __future__ import annotations

import csv
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def default_csv_path() -> str:
    return os.path.join(_project_root(), "data", "DatenBank.csv")


def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return None
    if text in {"yes", "y", "true", "t", "1"}:
        return True
    if text in {"no", "n", "false", "f", "0"}:
        return False
    return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _detect_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";"])
        if dialect.delimiter in {",", ";"}:
            return dialect.delimiter
    except csv.Error:
        pass
    comma = sample.count(",")
    semicolon = sample.count(";")
    return ";" if semicolon > comma else ","


def load_bank_data(csv_path: Optional[str] = None) -> List[Dict[str, Any]]:
    path = csv_path or default_csv_path()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = _detect_delimiter(sample)
        reader = csv.DictReader(f, delimiter=delimiter)
        rows: List[Dict[str, Any]] = []
        for raw in reader:
            row: Dict[str, Any] = dict(raw)

            row["age"] = _to_int(row.get("age"))
            row["balance"] = _to_float(row.get("balance"))
            row["duration"] = _to_int(row.get("duration"))
            row["pdays"] = _to_int(row.get("pdays"))

            row["housing"] = _to_bool(row.get("housing"))
            row["loan"] = _to_bool(row.get("loan"))

            complete = row.get("complete")
            complete_bool = _to_bool(complete)
            row["complete"] = bool(complete_bool) if complete_bool is not None else False

            for key in ("job", "marital", "education"):
                value = row.get(key)
                row[key] = "" if value is None else str(value).strip()

            rows.append(row)

    return rows


def normalize_choice_yes_no(value: str) -> Optional[bool]:
    text = (value or "").strip().lower()
    if text in {"", "skip", "-"}:
        return None
    if text in {"yes", "y", "true", "t", "1"}:
        return True
    if text in {"no", "n", "false", "f", "0"}:
        return False
    return None


def parse_balance_gt(value: str) -> Optional[float]:
    text = (value or "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None
