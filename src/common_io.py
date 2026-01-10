from __future__ import annotations

"""Common I/O Utilities for Bank Marketing Data Analysis

Authors: Peter Ngo, Alex Uscata
Class: INA 23A
Module: M323 - Functional Programming
Date: 2026-01-06
Version: 1.2.1

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

# ==============================================================================
# Pure Functions - Type Conversion Utilities
# ==============================================================================
# These functions are pure (no side effects, referentially transparent) and
# safe for use with map(), filter(), reduce() in functional pipelines.
# ==============================================================================

import csv
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _project_root() -> str:
    """Return the absolute path to the project root directory (parent of src/)."""
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def default_csv_path() -> str:
    """Return the default CSV path (data/DatenBank.csv) relative to project root."""
    return os.path.join(_project_root(), "data", "DatenBank.csv")


def _to_bool(value: Any) -> Optional[bool]:
    """Convert any value to boolean with graceful None handling.
    
    Pure function: Same input always produces same output, no side effects.
    Ideal for normalizing yes/no survey responses in functional pipelines.
    
    Args:
        value: Any value to convert (str, bool, int, None)
    
    Returns:
        True for 'yes'/'y'/'true'/'1', False for 'no'/'n'/'false'/'0',
        None for empty/invalid/null values
    
    Example:
        >>> _to_bool("yes")
        True
        >>> list(map(_to_bool, ["yes", "no", None, "invalid"]))
        [True, False, None, None]
    """
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
    """Convert any value to integer with graceful None handling.
    
    Pure function: Deterministic conversion with no side effects.
    Handles string representations of floats (e.g., "42.7" → 42).
    
    Args:
        value: Any value to convert (str, int, float, None)
    
    Returns:
        Integer value if conversion succeeds, None for empty/invalid values
    
    Example:
        >>> _to_int("42")
        42
        >>> list(map(_to_int, ["10", "20.5", "", None]))
        [10, 20, None, None]
    """
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
    """Convert any value to float with graceful None handling.
    
    Pure function: Consistent behavior for numeric conversions.
    Perfect for parsing financial data (balances, amounts) in pipelines.
    
    Args:
        value: Any value to convert (str, int, float, None)
    
    Returns:
        Float value if conversion succeeds, None for empty/invalid values
    
    Example:
        >>> _to_float("3.14")
        3.14
        >>> list(map(_to_float, ["100", "2.5", "", "invalid"]))
        [100.0, 2.5, None, None]
    """
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
    """Detect CSV delimiter (comma or semicolon) using csv.Sniffer or character counting.
    
    Returns ',' or ';' based on the sample content.
    """
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
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


# ==============================================================================
# Formatting and Display Utilities
# ==============================================================================
# Shared formatting functions for consistent output display across both
# imperative and functional implementations.
# ==============================================================================

def _header(title: str) -> str:
    """Create a formatted section header with centered title.
    
    Args:
        title: Text to display centered in the header.
    
    Returns:
        Formatted string with 72-char separator lines and centered title.
    """
    line = "=" * 72
    return f"{line}\n{title.center(72)}\n{line}"


def _fmt_pct(rate: float) -> str:
    """Format a decimal rate as a percentage string.
    
    Args:
        rate: Decimal rate between 0.0 and 1.0 (e.g., 0.123 = 12.3%).
    
    Returns:
        Right-aligned percentage string with 1 decimal place (e.g., " 12.3%").
    
    Example:
        >>> _fmt_pct(0.123)
        ' 12.3%'
    """
    return f"{rate * 100:5.1f}%"


def _fmt_num(value: Optional[float], decimals: int = 2) -> str:
    """Format a numeric value with specified decimal places.
    
    Args:
        value: Number to format, or None for missing values.
        decimals: Number of decimal places to display (default: 2).
    
    Returns:
        Formatted number string, or "-" if value is None.
    
    Example:
        >>> _fmt_num(3.14159, 2)
        '3.14'
        >>> _fmt_num(None)
        '-'
    """
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def _fmt_int(value: Optional[int]) -> str:
    """Format an integer value as string.
    
    Args:
        value: Integer to format, or None for missing values.
    
    Returns:
        String representation of the integer, or "-" if value is None.
    
    Example:
        >>> _fmt_int(42)
        '42'
        >>> _fmt_int(None)
        '-'
    """
    if value is None:
        return "-"
    return str(value)


def _table(headers: List[str], rows: List[List[str]], aligns: Optional[List[str]] = None) -> str:
    """Generate a formatted ASCII table with headers and aligned columns.
    
    Args:
        headers: Column header strings.
        rows: 2D sequence of cell values (list of lists/tuples).
        aligns: Optional alignment specs ("<" left, ">" right, default: all left).
    
    Returns:
        Formatted ASCII table string with separator line between header and rows.
    
    Example:
        >>> _table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
        'Name  | Age\\n------+----\\nAlice | 30\\nBob   | 25'
    """
    aligns = aligns or ["<"] * len(headers)
    widths = [len(h) for h in headers]
    
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(items: List[str]) -> str:
        parts = []
        for i, cell in enumerate(items):
            spec = aligns[i]
            parts.append(f"{cell:{spec}{widths[i]}}")
        return " | ".join(parts)

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_row(headers), sep]
    for row in rows:
        out.append(fmt_row(row))
    return "\n".join(out)
