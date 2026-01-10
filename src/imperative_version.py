from __future__ import annotations

"""Bank Marketing Analysis - Imperative Implementation

Authors: Peter Ngo, Alex Uscata
Class: INA 23A
Module: M323 - Functional Programming
Date: 2026-01-06
Version: 2.2.1

This module implements a bank marketing data analysis tool using an imperative
programming approach. The implementation emphasizes explicit control flow through
loops, mutable state management, and step-by-step iteration over data structures.

Key Techniques:
    - for-loops for explicit iteration over collections
    - continue statements for early loop termination
    - Manual accumulation with mutable variables
    - Explicit state management and variable updates
    - Index-based access and manipulation

Dataset Information:
    Portuguese bank marketing campaign data containing:
    - Demographics: age, marital status, education, job
    - Financial: balance (account balance in euros)
    - Campaign: duration (last contact duration in seconds), pdays
    - Target: complete (whether client subscribed to term deposit)

The imperative approach makes the execution flow explicit and easy to follow
for developers familiar with traditional procedural programming.
"""

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common_io import (
    _fmt_int,
    _fmt_num,
    _fmt_pct,
    _header,
    _table,
    load_bank_data,
    normalize_choice_yes_no,
    parse_balance_gt,
)


# Public API
__all__ = [
    # Core analysis functions
    'apply_filters',
    'success_overall',
    'duration_stats',
    'duration_buckets',
    'group_metrics',
    'marital_compare',
    'compare_two_groups',
    'anova_f_balance',
    # Statistical functions
    'mean',
    'variance_population',
    # Grouping functions
    'group_by_key',
]


def apply_filters(data: List[Dict[str, Any]], housing: Optional[bool], loan: Optional[bool], balance_gt: Optional[float]) -> List[Dict[str, Any]]:
    """Filter bank records by housing, loan and balance criteria.
    
    Args:
        data: Records to filter.
        housing/loan: If set, keep only records matching the boolean value.
        balance_gt: If set, keep only records with balance > balance_gt.
    
    Returns:
        New list with matching records.
    """
    out: List[Dict[str, Any]] = []
    for row in data:
        if housing is not None and row.get("housing") is not housing:
            continue
        if loan is not None and row.get("loan") is not loan:
            continue
        bal = row.get("balance")
        if balance_gt is not None:
            if bal is None or bal <= balance_gt:
                continue
        out.append(row)
    return out


def success_overall(data: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """Calculate overall campaign success metrics.
    
    Args:
        data: List of bank marketing records (dicts with 'complete' field).
    
    Returns:
        Tuple of (total_count, success_count, success_rate) where success_rate
        is a float between 0.0 and 1.0. Success is complete=True.
    
    Example:
        >>> success_overall([{'complete': True}, {'complete': False}])
        (2, 1, 0.5)
    """
    total = 0
    yes_count = 0
    for row in data:
        total += 1
        if row.get("complete") is True:
            yes_count += 1
    quote = (yes_count / total) if total else 0.0
    return total, yes_count, quote


def mean(values: List[float]) -> Optional[float]:
    """Calculate arithmetic mean using explicit accumulation.
    
    Args:
        values: List of numeric values to average.
    
    Returns:
        Mean value, or None if list is empty.
    """
    if not values:
        return None
    s = 0.0
    n = 0
    for v in values:
        s += float(v)
        n += 1
    if n == 0:
        return None
    return s / float(n)


def variance_population(values: List[float]) -> Optional[float]:
    """Compute population variance (divide by N) using explicit accumulation.
    
    Returns None for an empty list or if mean is None.
    """
    if not values:
        return None
    mu = mean(values)
    if mu is None:
        return None
    ss = 0.0
    for v in values:
        d = float(v) - mu
        ss += d * d
    return ss / float(len(values))


def duration_stats(data: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
    """Compute min, max, mean and population variance for call durations.
    
    Ignores missing/non-integer 'duration' values. Returns all None if no valid values exist.
    """
    durations: List[int] = []
    for row in data:
        d = row.get("duration")
        if isinstance(d, int):
            durations.append(d)
    if not durations:
        return None, None, None, None
    mn = durations[0]
    mx = durations[0]
    for d in durations[1:]:
        if d < mn:
            mn = d
        if d > mx:
            mx = d
    durations_f: List[float] = []
    for d in durations:
        durations_f.append(float(d))
    return mn, mx, mean(durations_f), variance_population(durations_f)


def duration_buckets(data: List[Dict[str, Any]], bucket_size: int) -> List[Tuple[str, int, float]]:
    """Bucketize call durations and compute success rate per bucket.
    
    Ignores missing/non-integer 'duration'. Success is row['complete'] is True.
    
    Returns:
        List of (bucket_label, count, success_rate), sorted by bucket start.
    """
    if bucket_size <= 0:
        bucket_size = 60

    max_d = 0
    has_any = False
    for row in data:
        d = row.get("duration")
        if isinstance(d, int):
            has_any = True
            if d > max_d:
                max_d = d
    if not has_any:
        return []

    buckets: List[Tuple[int, int, int]] = []  # (start, total, yes)
    start = 0
    while start <= max_d:
        buckets.append((start, 0, 0))
        start += bucket_size

    mutable: List[List[int]] = []
    for b in buckets:
        mutable.append([b[0], b[1], b[2]])

    for row in data:
        d = row.get("duration")
        if not isinstance(d, int):
            continue
        idx = d // bucket_size
        if idx < 0:
            continue
        if idx >= len(mutable):
            idx = len(mutable) - 1
        mutable[idx][1] += 1
        if row.get("complete") is True:
            mutable[idx][2] += 1

    out: List[Tuple[str, int, float]] = []
    for start, total, yes in mutable:
        end = start + bucket_size - 1
        label = f"{start:>4d}-{end:<4d}"
        rate = (yes / total) if total else 0.0
        out.append((label, total, rate))
    return out


def group_by_key(data: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    """Group records by a field value using imperative dictionary building.
    
    Imperative approach: Explicitly checks if each key exists in the dictionary,
    creates empty lists when needed, then appends records one by one. The
    grouping logic is transparent with clear conditional checks and mutations.
    
    Args:
        data: List of records to group
        key: Field name to group by (e.g., 'education', 'marital', 'job')
    
    Returns:
        Dictionary mapping each unique key value to list of matching records
    
    Example:
        >>> records = [{'job': 'admin'}, {'job': 'tech'}, {'job': 'admin'}]
        >>> group_by_key(records, 'job')
        {'admin': [{'job': 'admin'}, {'job': 'admin'}], 'tech': [{'job': 'tech'}]}
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in data:
        k = row.get(key)
        k_str = "" if k is None else str(k)
        if k_str not in groups:
            groups[k_str] = []
        groups[k_str].append(row)
    return groups


def group_metrics(data: List[Dict[str, Any]], key: str) -> List[Tuple[str, int, Optional[float], Optional[float], float]]:
    """Group records by `key` and compute count, avg(age), avg(balance) and success rate.
    
    Args:
        data: List of bank records to group.
        key: Field to group by ('education', 'marital', 'job').
    
    Returns:
        List of (group_name, count, avg_age, avg_balance, success_rate) tuples,
        sorted alphabetically. Missing values excluded from averages.
    """
    groups = group_by_key(data, key)
    out: List[Tuple[str, int, Optional[float], Optional[float], float]] = []
    for group_name in sorted(groups.keys()):
        rows = groups[group_name]
        count = len(rows)
        ages: List[float] = []
        balances: List[float] = []
        yes = 0
        for r in rows:
            a = r.get("age")
            b = r.get("balance")
            if isinstance(a, int):
                ages.append(float(a))
            if isinstance(b, (int, float)):
                balances.append(float(b))
            if r.get("complete") is True:
                yes += 1
        rate = (yes / count) if count else 0.0
        out.append((group_name, count, mean(ages), mean(balances), rate))
    return out


def marital_compare(data: List[Dict[str, Any]]) -> List[Tuple[str, int, Optional[float], Optional[float], float]]:
    """Compare marital status groups (single, married, divorced).
    
    Args:
        data: List of bank records.
    
    Returns:
        List of (status, count, avg_balance, avg_duration, success_rate) tuples
        in order: single, married, divorced.
    """
    order = ["single", "married", "divorced"]
    groups = group_by_key(data, "marital")
    out: List[Tuple[str, int, Optional[float], Optional[float], float]] = []
    for name in order:
        rows = groups.get(name, [])
        count = len(rows)
        balances: List[float] = []
        durations: List[float] = []
        yes = 0
        for r in rows:
            b = r.get("balance")
            d = r.get("duration")
            if isinstance(b, (int, float)):
                balances.append(float(b))
            if isinstance(d, int):
                durations.append(float(d))
            if r.get("complete") is True:
                yes += 1
        rate = (yes / count) if count else 0.0
        out.append((name, count, mean(balances), mean(durations), rate))
    return out


def compare_two_groups(data: List[Dict[str, Any]], key: str, g1: str, g2: str) -> Tuple[Tuple[str, int, Optional[float], Optional[float], Optional[float], float], Tuple[str, int, Optional[float], Optional[float], Optional[float], float]]:
    """Compute detailed metrics for two groups (A/B) within a categorical field.
    
    Args:
        data: List of bank records.
        key: Field to compare ('education', 'marital', 'job').
        g1, g2: Names of the two groups to compare.
    
    Returns:
        Tuple of (group1_metrics, group2_metrics) with counts, averages, success rate.
    """
    groups = group_by_key(data, key)

    def metrics(name: str) -> Tuple[str, int, Optional[float], Optional[float], Optional[float], float]:
        rows = groups.get(name, [])
        count = len(rows)
        ages: List[float] = []
        balances: List[float] = []
        durations: List[float] = []
        yes = 0
        for r in rows:
            a = r.get("age")
            b = r.get("balance")
            d = r.get("duration")
            if isinstance(a, int):
                ages.append(float(a))
            if isinstance(b, (int, float)):
                balances.append(float(b))
            if isinstance(d, int):
                durations.append(float(d))
            if r.get("complete") is True:
                yes += 1
        rate = (yes / count) if count else 0.0
        return name, count, mean(ages), mean(balances), mean(durations), rate

    return metrics(g1), metrics(g2)


def anova_f_balance(data: List[Dict[str, Any]], key: str) -> Optional[Tuple[float, int, int]]:
    """Compute one-way ANOVA F-statistic for balance grouped by `key`.
    
    Returns:
        (F, df_between, df_within) or None if insufficient data.
    """
    groups = group_by_key(data, key)

    group_values: List[Tuple[str, List[float]]] = []
    all_values: List[float] = []
    for name in sorted(groups.keys()):
        vals: List[float] = []
        for r in groups[name]:
            b = r.get("balance")
            if isinstance(b, (int, float)):
                vals.append(float(b))
        if vals:
            group_values.append((name, vals))
            all_values.extend(vals)

    k = len(group_values)
    n = len(all_values)
    if k < 2 or n <= k:
        return None

    overall = mean(all_values)
    if overall is None:
        return None

    ss_between = 0.0
    ss_within = 0.0

    for _, vals in group_values:
        mu = mean(vals)
        if mu is None:
            continue
        ss_between += len(vals) * (mu - overall) * (mu - overall)
        for v in vals:
            d = v - mu
            ss_within += d * d

    df_between = k - 1
    df_within = n - k
    if df_between <= 0 or df_within <= 0:
        return None

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    if ms_within == 0.0:
        return float("inf"), df_between, df_within
    return ms_between / ms_within, df_between, df_within


def _prompt_filters() -> Tuple[Optional[bool], Optional[bool], Optional[float]]:
    """Prompt user for filter criteria (housing, loan, balance threshold).
    
    Returns:
        Tuple of (housing, loan, balance_gt) where None means no filter.
    """
    print(_header("FILTER"))
    print("Leerlassen = kein Filter / unverändert")
    housing = normalize_choice_yes_no(input("housing (yes/no): "))
    loan = normalize_choice_yes_no(input("loan (yes/no): "))
    balance_gt = parse_balance_gt(input("balance > X (z.B. 1000): "))
    return housing, loan, balance_gt


def _prompt_bucket_size() -> int:
    """Prompt user for duration bucket size in seconds.
    
    Returns:
        Bucket size in seconds (default: 60 if invalid input).
    """
    raw = input("Bucket-Größe für duration (Sekunden, Default 60): ").strip()
    if raw == "":
        return 60
    try:
        v = int(float(raw))
        return v if v > 0 else 60
    except ValueError:
        return 60


def _prompt_group_field(default: str) -> str:
    """Prompt user to select grouping field (education/marital/job).
    
    Returns:
        Selected field name or default if invalid input.
    """
    raw = input(f"Gruppierungsfeld (education/marital/job) [Default: {default}]: ").strip().lower()
    if raw in {"education", "marital", "job"}:
        return raw
    return default


def _prompt_group_names(available: List[str]) -> Tuple[str, str]:
    """Prompt user to select two groups for comparison.
    
    Returns:
        Tuple of (group_a, group_b) names as entered by user.
    """
    print("Verfügbare Gruppen:")
    print(", ".join(available) if available else "-")
    g1 = input("Gruppe A: ").strip()
    g2 = input("Gruppe B: ").strip()
    return g1, g2


def _menu() -> str:
    """Display main menu and return user's choice.
    
    Returns:
        User's menu selection as string ("1"-"8" or "q").
    """
    print(_header("MENU"))
    print("1) Erfolgsquote gesamt")
    print("2) Filter setzen (housing/loan/balance>X)")
    print("3) Transformationen (log(balance), balance^2+1)")
    print("4) duration Analyse + optional Buckets")
    print("5) Group by education")
    print("6) Group by marital")
    print("7) Vergleich zweier Gruppen")
    print("8) ANOVA-ähnlicher F-Wert (balance)")
    print("q) Quit")
    return input("Auswahl: ").strip().lower()


def main() -> None:
    """Run the interactive CLI for the bank marketing analysis.
    
    Loads the dataset, maintains an active working subset ("current"),
    and dispatches menu actions (imperative version uses while-loop and if/elif).
    """
    data = load_bank_data()
    current = list(data)

    print(_header("BANK MARKETING – CLI"))
    print(f"Datensätze geladen: {len(data)}")

    while True:
        choice = _menu()
        if choice == "q":
            print("Bye")
            return

        if choice == "1":
            print(_header("ERFOLGSQUOTE"))
            total, yes_count, quote = success_overall(current)
            rows = [
                ["Total", str(total)],
                ["Yes", str(yes_count)],
                ["Quote", _fmt_pct(quote)],
            ]
            print(_table(["Metric", "Value"], rows, aligns=["<", ">"]))

        elif choice == "2":
            housing, loan, balance_gt = _prompt_filters()
            current = apply_filters(data, housing, loan, balance_gt)
            print(_header("FILTER RESULT"))
            print(f"Aktueller Datenbestand: {len(current)} / {len(data)}")

        elif choice == "3":
            print(_header("TRANSFORMATIONEN"))
            logs: List[float] = []
            sq1: List[float] = []
            for row in current:
                bal = row.get("balance")
                if isinstance(bal, (int, float)):
                    b = float(bal)
                    sq1.append(b * b + 1.0)
                    if b > 0.0:
                        logs.append(math.log(b))

            def stats_line(name: str, values: List[float]) -> List[str]:
                mu = mean(values)
                var = variance_population(values)
                mn: Optional[float] = min(values) if values else None
                mx: Optional[float] = max(values) if values else None
                return [name, str(len(values)), _fmt_num(mn), _fmt_num(mx), _fmt_num(mu), _fmt_num(var)]

            rows = [
                stats_line("log(balance)", logs),
                stats_line("balance^2+1", sq1),
            ]
            print(_table(["Transform", "n", "min", "max", "mean", "var"], rows, aligns=["<", ">", ">", ">", ">", ">"]))

        elif choice == "4":
            print(_header("DURATION ANALYSE"))
            mn, mx, mu, var = duration_stats(current)
            rows = [[_fmt_int(mn), _fmt_int(mx), _fmt_num(mu), _fmt_num(var)]]
            print(_table(["min", "max", "mean", "var"], rows, aligns=[">", ">", ">", ">"]))

            do_buckets = input("Buckets anzeigen? (y/n): ").strip().lower() == "y"
            if do_buckets:
                bucket_size = _prompt_bucket_size()
                buckets = duration_buckets(current, bucket_size)
                b_rows: List[List[str]] = []
                for label, cnt, rate in buckets:
                    b_rows.append([label, str(cnt), _fmt_pct(rate)])
                print(_header("DURATION BUCKETS"))
                print(_table(["bucket", "count", "success"], b_rows, aligns=["<", ">", ">"]))

        elif choice == "5":
            print(_header("GROUP BY EDUCATION"))
            metrics = group_metrics(current, "education")
            rows = []
            for name, cnt, avg_age, avg_bal, rate in metrics:
                rows.append([name or "(blank)", str(cnt), _fmt_num(avg_age, 1), _fmt_num(avg_bal, 2), _fmt_pct(rate)])
            print(_table(["education", "count", "avg(age)", "avg(balance)", "success"], rows, aligns=["<", ">", ">", ">", ">"]))

        elif choice == "6":
            print(_header("GROUP BY MARITAL"))
            metrics = marital_compare(current)
            rows = []
            for name, cnt, avg_bal, avg_dur, rate in metrics:
                rows.append([name, str(cnt), _fmt_num(avg_bal, 2), _fmt_num(avg_dur, 1), _fmt_pct(rate)])
            print(_table(["marital", "count", "avg(balance)", "avg(duration)", "success"], rows, aligns=["<", ">", ">", ">", ">"]))

        elif choice == "7":
            print(_header("VERGLEICH ZWEIER GRUPPEN"))
            field = _prompt_group_field("education")
            available_set: set[str] = set()
            for r in current:
                val = r.get(field) or ""
                available_set.add(val)
            available = sorted(available_set)
            g1, g2 = _prompt_group_names(available)
            m1, m2 = compare_two_groups(current, field, g1, g2)

            def row(m: Tuple[str, int, Optional[float], Optional[float], Optional[float], float]) -> List[str]:
                name = m[0]
                cnt = m[1]
                avg_age = m[2]
                avg_bal = m[3]
                avg_dur = m[4]
                rate = m[5]
                result: List[str] = []
                result.append(name or "(blank)")
                result.append(str(cnt))
                result.append(_fmt_num(avg_age, 1))
                result.append(_fmt_num(avg_bal, 2))
                result.append(_fmt_num(avg_dur, 1))
                result.append(_fmt_pct(rate))
                
                return result

            rows = [row(m1), row(m2)]
            print(_table([field, "count", "avg(age)", "avg(balance)", "avg(duration)", "success"], rows, aligns=["<", ">", ">", ">", ">", ">"]))
            delta = m1[-1] - m2[-1]
            sign = "+" if delta >= 0 else ""
            print(f"Δ Erfolgsquote (A-B): {sign}{delta * 100:0.1f}%")

        elif choice == "8":
            print(_header("ANOVA-ÄHNLICHER F-WERT (BALANCE)"))
            field = _prompt_group_field("education")
            result = anova_f_balance(current, field)
            if result is None:
                print("Nicht genug Daten für F-Berechnung (mind. 2 Gruppen, ausreichend Beobachtungen).")
            else:
                f_value, dfb, dfw = result
                f_text = "inf" if math.isinf(f_value) else f"{f_value:0.3f}"
                print(f"F({dfb}, {dfw}) = {f_text}")
                if math.isinf(f_value):
                    print("Interpretation: Innerhalb-Varianz ist 0; Gruppenmittelwerte unterscheiden sich stark oder Werte sind konstant pro Gruppe.")
                elif f_value < 1.5:
                    print("Interpretation: Eher geringe Unterschiede der Mittelwerte zwischen Gruppen (relativ zur Streuung).")
                elif f_value < 5.0:
                    print("Interpretation: Moderate Unterschiede der Mittelwerte zwischen Gruppen.")
                else:
                    print("Interpretation: Deutliche Unterschiede der Mittelwerte zwischen Gruppen möglich (hoher F-Wert).")

        else:
            print("Ungültige Auswahl")


if __name__ == "__main__":
    main()
