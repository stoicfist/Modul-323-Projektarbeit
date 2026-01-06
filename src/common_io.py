from __future__ import annotations

"""Common I/O Utilities for Bank Marketing Data Analysis

Authors: Peter Ngo, Alex Uscata
Class: INA 23A
Module: M323 - Functional Programming
Date: 2026-01-06

This module provides shared input/output functionality for loading and parsing
bank marketing data. It handles CSV file reading with automatic delimiter detection,
type conversion for numeric and boolean fields, and data normalization.

Key Features:
    - Automatic CSV delimiter detection (comma vs semicolon)
    - Type-safe conversion functions for bool, int, float
    - Robust handling of missing/malformed values
    - UTF-8 BOM support for international character sets
    - Centralized data loading logic shared by both imperative and functional versions

Dataset Schema:
    - age: int (customer age)
    - balance: float (account balance in euros)
    - duration: int (last contact duration in seconds)
    - pdays: int (days since previous contact)
    - housing: bool (has housing loan)
    - loan: bool (has personal loan)
    - complete: bool (subscribed to term deposit - target variable)
    - job, marital, education: str (categorical fields)

This module uses a pragmatic mix of styles - the type conversion functions
are functional (pure functions), while the CSV loading uses imperative iteration
for clarity and performance.
"""

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
    """Load and parse bank marketing data from CSV file.
    
    Reads CSV file with automatic delimiter detection, performs type conversions
    for numeric and boolean fields, and normalizes string fields. Handles missing
    values gracefully by converting them to None (numeric/bool) or empty strings.
    
    The function applies domain-specific transformations:
    - Converts age, duration, pdays to integers (None if missing)
    - Converts balance to float (None if missing)
    - Parses housing/loan as booleans (yes/no → True/False)
    - Ensures 'complete' (target variable) is always a boolean (defaults to False)
    - Strips whitespace from categorical fields (job, marital, education)
    
    Args:
        csv_path: Path to CSV file. If None, uses default data/DatenBank.csv
    
    Returns:
        List of dictionaries, one per CSV row, with properly typed values
    
    Example:
        >>> data = load_bank_data()
        >>> data[0]
        {'age': 58, 'balance': 2143.0, 'duration': 261, 'housing': True, 
         'loan': False, 'complete': False, 'job': 'management', ...}
    """
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
    """Parse user input for yes/no questions into boolean or None.
    
    Accepts various common representations of yes/no answers and normalizes
    them to True/False/None. Case-insensitive and whitespace-tolerant.
    
    Args:
        value: User input string (e.g., 'yes', 'no', 'Y', 'n', '1', '0', 'skip')
    
    Returns:
        True for affirmative responses, False for negative, None for skip/empty
    
    Example:
        >>> normalize_choice_yes_no('YES')
        True
        >>> normalize_choice_yes_no('n')
        False
        >>> normalize_choice_yes_no('')
        None
    """
    text = (value or "").strip().lower()
    if text in {"", "skip", "-"}:
        return None
    if text in {"yes", "y", "true", "t", "1"}:
        return True
    if text in {"no", "n", "false", "f", "0"}:
        return False
    return None


def parse_balance_gt(value: str) -> Optional[float]:
    """Parse user input for balance threshold into float or None.
    
    Converts string input to float for balance filtering. Returns None for
    empty input (meaning "no filter") or invalid values.
    
    Args:
        value: User input string (e.g., '1000', '2500.50', '')
    
    Returns:
        Float value if valid number, None if empty or invalid
    
    Example:
        >>> parse_balance_gt('1000')
        1000.0
        >>> parse_balance_gt('')
        None
        >>> parse_balance_gt('invalid')
        None
    """
    text = (value or "").strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None
