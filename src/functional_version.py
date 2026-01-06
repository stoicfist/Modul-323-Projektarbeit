from __future__ import annotations

"""Bank Marketing Analysis - Functional Implementation (V2.0)

Authors: Peter Ngo, Alex Uscata
Class: INA 23A
Module: M323 - Functional Programming
Date: 2026-01-06

This module implements a bank marketing data analysis tool using a functional
programming approach. The implementation emphasizes declarative transformations,
immutability, and composition of higher-order functions.

Key Techniques:
    - map() for element-wise transformations
    - filter() for declarative selection
    - reduce() for accumulation without mutation
    - Lambda functions for inline operations
    - List comprehensions for concise data processing
    - Function composition and chaining

Dataset Information:
    Portuguese bank marketing campaign data containing:
    - Demographics: age, marital status, education, job
    - Financial: balance (account balance in euros)
    - Campaign: duration (last contact duration in seconds), pdays
    - Target: complete (whether client subscribed to term deposit)

Paradigm:
    The functional approach treats data transformations as a series of
    immutable operations, making the code more declarative and often more
    concise. Functions are pure when possible, avoiding side effects and
    making the code easier to test and reason about.
"""

import math
from functools import reduce
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from common_io import load_bank_data, normalize_choice_yes_no, parse_balance_gt


def _header(title: str) -> str:
    line = "=" * 72
    return f"{line}\n{title.center(72)}\n{line}"


def _fmt_pct(rate: float) -> str:
    return f"{rate * 100:5.1f}%"


def _fmt_num(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"


def _fmt_int(value: Optional[int]) -> str:
    if value is None:
        return "-"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], aligns: Optional[Sequence[str]] = None) -> str:
    aligns = aligns or ["<"] * len(headers)
    # Calculate column widths using map (transform headers to lengths)
    widths = list(map(len, headers))

    # Use reduce to fold over rows and accumulate maximum widths.
    # The accumulator pattern: acc = [w1, w2, ...], updated for each row.
    def upd(acc: List[int], row: Sequence[str]) -> List[int]:
        return [max(acc[i], len(cell)) for i, cell in enumerate(row)]

    widths = reduce(upd, rows, widths)

    def fmt_row(items: Sequence[str]) -> str:
        return " | ".join(f"{cell:{aligns[i]}{widths[i]}}" for i, cell in enumerate(items))

    sep = "-+-".join("-" * w for w in widths)
    out = [fmt_row(headers), sep]
    out.extend(map(fmt_row, rows))
    return "\n".join(out)


def _apply_filters(data: List[Dict[str, Any]], housing: Optional[bool], loan: Optional[bool], balance_gt: Optional[float]) -> List[Dict[str, Any]]:
    """Filter bank records using functional filter with predicate function.
    
    Creates a predicate function that encapsulates all filter logic, then
    applies it declaratively using filter(). No explicit loops or mutable
    state - the filtering is expressed as "what to keep" rather than "how to iterate".
    
    Functional approach: Uses filter() higher-order function with a locally-defined
    predicate. The predicate is a pure function that returns True/False based solely
    on its input, making the logic easy to test and compose.
    
    Args:
        data: List of bank records (dictionaries) to filter
        housing: If provided, keep only records matching this housing loan status
        loan: If provided, keep only records matching this personal loan status
        balance_gt: If provided, keep only records with balance > this value
    
    Returns:
        New list containing only records that pass all specified filters
    
    Example:
        >>> records = [{'housing': True, 'balance': 1500}, {'housing': False, 'balance': 500}]
        >>> _apply_filters(records, housing=True, loan=None, balance_gt=1000)
        [{'housing': True, 'balance': 1500}]
    """
    # Functional approach: Define predicate function, then filter entire dataset.
    # No mutable state, no continue statements - pure boolean logic.
    def pred(row: Dict[str, Any]) -> bool:
        if housing is not None and row.get("housing") is not housing:
            return False
        if loan is not None and row.get("loan") is not loan:
            return False
        if balance_gt is not None:
            bal = row.get("balance")
            if bal is None or bal <= balance_gt:
                return False
        return True

    return list(filter(pred, data))


def create_balance_filter(threshold: float) -> Callable[[Dict[str, Any]], bool]:
    """Higher-order function: Returns a filter function for balance threshold.
    
    This demonstrates functional composition - a function that returns a function.
    Useful for creating multiple filters with different thresholds.
    
    Args:
        threshold: Minimum balance value
        
    Returns:
        Filter function that checks if balance > threshold
        
    Example:
        >>> high_balance_filter = create_balance_filter(1000.0)
        >>> filtered = list(filter(high_balance_filter, data))
    """
    def balance_predicate(row: Dict[str, Any]) -> bool:
        bal = row.get("balance")
        if bal is None:
            return False
        return float(bal) > threshold
    return balance_predicate


def _success_overall(data: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    total = len(data)
    # Count successes using generator expression - memory efficient, no intermediate list.
    yes_count = sum(1 for r in data if r.get("complete") is True)
    quote = (yes_count / total) if total else 0.0
    return total, yes_count, quote


def _mean(values: List[float]) -> Optional[float]:
    """Calculate arithmetic mean using functional composition.
    
    Functional approach: Uses map() to ensure float conversion and sum() as a
    reduction operation. No mutable accumulator - the sum is computed declaratively
    in a single expression using built-in higher-order functions.
    
    Args:
        values: List of numeric values to average
    
    Returns:
        Mean value, or None if list is empty
    
    Example:
        >>> _mean([10.0, 20.0, 30.0])
        20.0
    """
    return (sum(map(float, values)) / float(len(values))) if values else None


def _variance_population(values: List[float]) -> Optional[float]:
    if not values:
        return None
    mu = _mean(values)
    if mu is None:
        return None
    return sum((float(v) - mu) ** 2 for v in values) / float(len(values))


def _duration_stats(data: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
    # Functional pipeline:
    # 1. Generator: extract 'duration' field from each record
    # 2. Filter: keep only integer values
    # 3. Map: convert to int type
    # 4. List: materialize filtered results
    # This replaces a loop with continue statements.
    durations = list(map(int, filter(lambda d: isinstance(d, int), (r.get("duration") for r in data))))
    if not durations:
        return None, None, None, None
    durations_f = list(map(float, durations))
    return min(durations), max(durations), _mean(durations_f), _variance_population(durations_f)


def _duration_buckets(data: List[Dict[str, Any]], bucket_size: int) -> List[Tuple[str, int, float]]:
    """Distribute call durations into buckets using functional transformations.
    
    Creates histogram-like buckets for call durations (e.g., 0-59s, 60-119s, etc.)
    and computes the success rate (complete=True) for each bucket.
    
    Functional approach: Chains multiple functional operations:
    1. Generator expression + filter() to extract valid durations
    2. max(map()) to find maximum duration declaratively
    3. range() to generate bucket start points
    4. Tuple-based immutable accumulator initialization
    5. reduce() with step function that returns NEW tuples (truly immutable)
    6. map() to transform raw counts into formatted output tuples
    
    Pure functional accumulator pattern: Unlike the previous implementation that
    mutated list elements in-place (acc[i][0] += 1), this version creates entirely
    new tuples at each reduction step. The step function never modifies existing
    data structures - it constructs a fresh tuple by copying unchanged buckets and
    creating a new tuple for the updated bucket. This is more purely functional
    because it maintains referential transparency: the same input always produces
    the same output without side effects. While less efficient for large datasets,
    it better demonstrates immutability principles central to functional programming.
    
    No explicit loop indices or manual counter increments - each transformation
    is expressed as a data pipeline operation.
    
    Args:
        data: List of bank records with 'duration' and 'complete' fields
        bucket_size: Size of each bucket in seconds (default: 60)
    
    Returns:
        List of tuples: (bucket_label, total_count, success_rate)
        Sorted by bucket start time
    
    Example:
        >>> records = [{'duration': 45, 'complete': True}, {'duration': 120, 'complete': False}]
        >>> _duration_buckets(records, 60)
        [('   0-  59', 1, 1.0), ('  60- 119', 0, 0.0), (' 120- 179', 1, 0.0)]
    """
    bucket_size = bucket_size if bucket_size > 0 else 60
    durations = list(filter(lambda d: isinstance(d, int), (r.get("duration") for r in data)))
    if not durations:
        return []
    max_d = max(map(int, durations))
    starts = list(range(0, max_d + 1, bucket_size))

    def idx_of(d: int) -> int:
        i = d // bucket_size
        return min(max(i, 0), len(starts) - 1)

    # Use reduce to accumulate bucket counts across all records.
    # Accumulator: Tuple[Tuple[int, int], ...] where each inner tuple = (total, yes_count)
    # This uses truly immutable data structures - no in-place mutations.
    def step(acc: Tuple[Tuple[int, int], ...], row: Dict[str, Any]) -> Tuple[Tuple[int, int], ...]:
        """Pure reducer function that returns a NEW tuple instead of mutating.
        
        This is more purely functional than the list-based approach because:
        - No side effects: acc is never modified, only read
        - Referential transparency: same inputs always produce same output
        - Immutability: tuples cannot be changed after creation
        
        Creates new tuple by concatenating:
        1. All buckets before the target index (unchanged)
        2. Updated bucket at target index (new tuple with incremented counts)
        3. All buckets after the target index (unchanged)
        """
        d = row.get("duration")
        if not isinstance(d, int):
            return acc
        i = idx_of(d)
        # Extract current counts for the target bucket
        total, yes = acc[i]
        # Increment counts based on record data
        new_total = total + 1
        new_yes = yes + (1 if row.get("complete") is True else 0)
        # Return NEW tuple: buckets before + updated bucket + buckets after
        return acc[:i] + ((new_total, new_yes),) + acc[i+1:]

    init = tuple((0, 0) for _ in starts)  # Tuple of tuples: ((total, yes), ...)
    counts = reduce(step, data, init)

    def mk(i: int) -> Tuple[str, int, float]:
        start = starts[i]
        end = start + bucket_size - 1
        total, yes = counts[i]
        label = f"{start:>4d}-{end:<4d}"
        rate = (yes / total) if total else 0.0
        return label, total, rate

    return list(map(mk, range(len(starts))))


def _group_by_key(data: List[Dict[str, Any]], key: str) -> Dict[str, Tuple[Dict[str, Any], ...]]:
    """Group records by a field value using functional reduce with immutable structures.
    
    Pure functional approach: Uses reduce() to build the grouping dictionary by
    accumulating records through a pure reducer function. Unlike the previous
    implementation that mutated the accumulator with setdefault().append(), this
    version creates a completely NEW dictionary on each iteration using dict
    unpacking {**acc, key: value}.
    
    Immutability benefits:
    - No side effects: The accumulator is never modified, only replaced
    - Referential transparency: Same input always produces same output
    - Thread-safe: No shared mutable state to cause race conditions
    - Easier debugging: Each reduction step creates new state, preserving history
    - True functional purity: Aligns with functional programming principles
    
    Groups are stored as tuples (immutable) rather than lists, reinforcing the
    immutable nature of the data structure throughout. The tuple concatenation
    (existing_group + (row,)) creates a new tuple rather than mutating a list.
    
    The reducer function 'add' takes an accumulator and one row, creates a new
    dictionary with the updated group, and returns it for the next iteration - a
    truly pure functional pattern for building complex data structures.
    
    Args:
        data: List of records to group
        key: Field name to group by (e.g., 'education', 'marital', 'job')
    
    Returns:
        Dictionary mapping each unique key value to tuple of matching records
    
    Example:
        >>> records = [{'job': 'admin'}, {'job': 'tech'}, {'job': 'admin'}]
        >>> _group_by_key(records, 'job')
        {'admin': ({'job': 'admin'}, {'job': 'admin'}), 'tech': ({'job': 'tech'},)}
    """
    def add(acc: Dict[str, Tuple[Dict[str, Any], ...]], row: Dict[str, Any]) -> Dict[str, Tuple[Dict[str, Any], ...]]:
        """Pure reducer that creates new dictionary instead of mutating accumulator.
        
        Uses dict unpacking {**acc, key: value} to create a new dictionary with
        all previous groups plus the updated group. The updated group is created
        by tuple concatenation (old_tuple + (new_element,)) rather than list
        mutation. This ensures complete immutability throughout the reduction.
        """
        k = row.get(key)
        k_str = "" if k is None else str(k)
        # Get existing group as tuple, or empty tuple if key doesn't exist
        existing_group = acc.get(k_str, ())
        # Create new tuple by concatenating existing group with new row
        new_group = existing_group + (row,)
        # Return NEW dictionary with all previous groups plus updated group
        return {**acc, k_str: new_group}

    return reduce(add, data, {})


def _group_metrics(data: List[Dict[str, Any]], key: str) -> List[Tuple[str, int, Optional[float], Optional[float], float]]:
    groups = _group_by_key(data, key)

    def metrics(name: str) -> Tuple[str, int, Optional[float], Optional[float], float]:
        rows = list(groups[name])  # Convert tuple to list for processing
        count = len(rows)
        ages = [float(r["age"]) for r in rows if isinstance(r.get("age"), int)]
        balances = [float(r["balance"]) for r in rows if isinstance(r.get("balance"), (int, float))]
        yes = sum(1 for r in rows if r.get("complete") is True)
        rate = (yes / count) if count else 0.0
        return name, count, _mean(ages), _mean(balances), rate

    return list(map(metrics, sorted(groups.keys())))


def _marital_compare(data: List[Dict[str, Any]]) -> List[Tuple[str, int, Optional[float], Optional[float], float]]:
    groups = _group_by_key(data, "marital")
    order = ["single", "married", "divorced"]

    def metrics(name: str) -> Tuple[str, int, Optional[float], Optional[float], float]:
        rows = list(groups.get(name, ()))  # Convert tuple to list, default to empty tuple
        count = len(rows)
        balances = [float(r["balance"]) for r in rows if isinstance(r.get("balance"), (int, float))]
        durations = [float(r["duration"]) for r in rows if isinstance(r.get("duration"), int)]
        yes = sum(1 for r in rows if r.get("complete") is True)
        rate = (yes / count) if count else 0.0
        return name, count, _mean(balances), _mean(durations), rate

    return list(map(metrics, order))


def _compare_two_groups(data: List[Dict[str, Any]], key: str, g1: str, g2: str) -> Tuple[Tuple[str, int, Optional[float], Optional[float], Optional[float], float], Tuple[str, int, Optional[float], Optional[float], Optional[float], float]]:
    groups = _group_by_key(data, key)

    def metrics(name: str) -> Tuple[str, int, Optional[float], Optional[float], Optional[float], float]:
        rows = list(groups.get(name, ()))  # Convert tuple to list, default to empty tuple
        count = len(rows)
        ages = [float(r["age"]) for r in rows if isinstance(r.get("age"), int)]
        balances = [float(r["balance"]) for r in rows if isinstance(r.get("balance"), (int, float))]
        durations = [float(r["duration"]) for r in rows if isinstance(r.get("duration"), int)]
        yes = sum(1 for r in rows if r.get("complete") is True)
        rate = (yes / count) if count else 0.0
        return name, count, _mean(ages), _mean(balances), _mean(durations), rate

    return metrics(g1), metrics(g2)


def _anova_f_balance(data: List[Dict[str, Any]], key: str) -> Optional[Tuple[float, int, int]]:
    """Calculate ANOVA F-statistic for balance across groups using functional style.
    
    Performs one-way ANOVA to test if mean balance differs significantly across
    groups (e.g., education levels, job types). The F-statistic compares variance
    between groups to variance within groups.
    
    Functional approach: Uses declarative transformations:
    1. List comprehension to extract balance values per group
    2. filter() to remove empty groups
    3. Nested list comprehension to flatten all values
    4. sum() with generator expression for ss_between (no accumulator variable)
    5. Nested function ss_within() with sum() for within-group variance
    6. sum() over map-like generator to total ss_within
    
    All variance components are computed through expressions rather than loops,
    making the mathematical formula more visible in the code.
    
    Args:
        data: List of bank records
        key: Field to group by (e.g., 'education', 'marital', 'job')
    
    Returns:
        Tuple of (F-statistic, df_between, df_within), or None if insufficient data
        F > 1 suggests between-group variance exceeds within-group variance
        Higher F values indicate more significant group differences
    
    Example:
        >>> records = [{'education': 'primary', 'balance': 1000}, 
        ...            {'education': 'tertiary', 'balance': 5000}]
        >>> _anova_f_balance(records, 'education')
        (8.0, 1, 0)  # F-statistic, degrees of freedom between, within
    """
    groups = _group_by_key(data, key)
    group_values = [
        (name, [float(r["balance"]) for r in rows if isinstance(r.get("balance"), (int, float))])
        for name, rows in sorted(groups.items(), key=lambda x: x[0])
    ]
    group_values = list(filter(lambda t: len(t[1]) > 0, group_values))

    k = len(group_values)
    all_values = [v for _, vals in group_values for v in vals]
    n = len(all_values)
    if k < 2 or n <= k:
        return None

    overall = _mean(all_values)
    if overall is None:
        return None

    # Calculate between-group sum of squares using sum() with generator expression.
    # Equivalent to imperative loop but more declarative.
    ss_between = sum(len(vals) * (_mean(vals) - overall) ** 2 for _, vals in group_values if _mean(vals) is not None)

    def ss_within(vals: List[float]) -> float:
        mu = _mean(vals)
        return sum((v - mu) ** 2 for v in vals) if mu is not None else 0.0

    ss_within_total = sum(ss_within(vals) for _, vals in group_values)

    df_between = k - 1
    df_within = n - k
    if df_between <= 0 or df_within <= 0:
        return None

    ms_between = ss_between / df_between
    ms_within = ss_within_total / df_within
    if ms_within == 0.0:
        return float("inf"), df_between, df_within
    return ms_between / ms_within, df_between, df_within


def _prompt_filters() -> Tuple[Optional[bool], Optional[bool], Optional[float]]:
    print(_header("FILTER"))
    print("Leerlassen = kein Filter / unverändert")
    housing = normalize_choice_yes_no(input("housing (yes/no): "))
    loan = normalize_choice_yes_no(input("loan (yes/no): "))
    balance_gt = parse_balance_gt(input("balance > X (z.B. 1000): "))
    return housing, loan, balance_gt


def _prompt_bucket_size() -> int:
    raw = input("Bucket-Größe für duration (Sekunden, Default 60): ").strip()
    if raw == "":
        return 60
    try:
        v = int(float(raw))
        return v if v > 0 else 60
    except ValueError:
        return 60


def _prompt_group_field(default: str) -> str:
    raw = input(f"Gruppierungsfeld (education/marital/job) [Default: {default}]: ").strip().lower()
    return raw if raw in {"education", "marital", "job"} else default


def _prompt_group_names(available: List[str]) -> Tuple[str, str]:
    print("Verfügbare Gruppen:")
    print(", ".join(available) if available else "-")
    g1 = input("Gruppe A: ").strip()
    g2 = input("Gruppe B: ").strip()
    return g1, g2


def _menu() -> str:
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
    data = load_bank_data()
    current = list(data)

    # Example of higher-order function usage (not used in main flow):
    # high_balance_filter = create_balance_filter(1000.0)
    # high_balance_customers = list(filter(high_balance_filter, current))

    print(_header("BANK MARKETING – CLI"))
    print(f"Datensätze geladen: {len(data)}")

    while True:
        choice = _menu()
        if choice == "q":
            print("Bye")
            return

        if choice == "1":
            print(_header("ERFOLGSQUOTE"))
            total, yes_count, quote = _success_overall(current)
            rows = [["Total", str(total)], ["Yes", str(yes_count)], ["Quote", _fmt_pct(quote)]]
            print(_table(["Metric", "Value"], rows, aligns=["<", ">"]))

        elif choice == "2":
            housing, loan, balance_gt = _prompt_filters()
            current = _apply_filters(data, housing, loan, balance_gt)
            print(_header("FILTER RESULT"))
            print(f"Aktueller Datenbestand: {len(current)} / {len(data)}")

        elif choice == "3":
            print(_header("TRANSFORMATIONEN"))
            balances = [float(r["balance"]) for r in current if isinstance(r.get("balance"), (int, float))]
            logs = list(filter(lambda x: x is not None, map(lambda b: math.log(b) if b > 0.0 else None, balances)))
            sq1 = list(map(lambda b: b * b + 1.0, balances))

            def stats_line(name: str, values: List[float]) -> List[str]:
                mu = _mean(values)
                var = _variance_population(values)
                mn = min(values) if values else None
                mx = max(values) if values else None
                return [name, str(len(values)), _fmt_num(mn), _fmt_num(mx), _fmt_num(mu), _fmt_num(var)]

            rows = [stats_line("log(balance)", list(map(float, logs))), stats_line("balance^2+1", sq1)]
            print(_table(["Transform", "n", "min", "max", "mean", "var"], rows, aligns=["<", ">", ">", ">", ">", ">"]))

        elif choice == "4":
            print(_header("DURATION ANALYSE"))
            mn, mx, mu, var = _duration_stats(current)
            rows = [[_fmt_int(mn), _fmt_int(mx), _fmt_num(mu), _fmt_num(var)]]
            print(_table(["min", "max", "mean", "var"], rows, aligns=[">", ">", ">", ">"]))

            do_buckets = input("Buckets anzeigen? (y/n): ").strip().lower() == "y"
            if do_buckets:
                bucket_size = _prompt_bucket_size()
                buckets = _duration_buckets(current, bucket_size)
                b_rows = list(map(lambda t: [t[0], str(t[1]), _fmt_pct(t[2])], buckets))
                print(_header("DURATION BUCKETS"))
                print(_table(["bucket", "count", "success"], b_rows, aligns=["<", ">", ">"]))

        elif choice == "5":
            print(_header("GROUP BY EDUCATION"))
            metrics = _group_metrics(current, "education")
            rows = list(
                map(
                    lambda t: [t[0] or "(blank)", str(t[1]), _fmt_num(t[2], 1), _fmt_num(t[3], 2), _fmt_pct(t[4])],
                    metrics,
                )
            )
            print(_table(["education", "count", "avg(age)", "avg(balance)", "success"], rows, aligns=["<", ">", ">", ">", ">"]))

        elif choice == "6":
            print(_header("GROUP BY MARITAL"))
            metrics = _marital_compare(current)
            rows = list(map(lambda t: [t[0], str(t[1]), _fmt_num(t[2], 2), _fmt_num(t[3], 1), _fmt_pct(t[4])], metrics))
            print(_table(["marital", "count", "avg(balance)", "avg(duration)", "success"], rows, aligns=["<", ">", ">", ">", ">"]))

        elif choice == "7":
            print(_header("VERGLEICH ZWEIER GRUPPEN"))
            field = _prompt_group_field("education")
            available = sorted({(r.get(field) or "") for r in current})
            g1, g2 = _prompt_group_names(available)
            m1, m2 = _compare_two_groups(current, field, g1, g2)

            def row(m: Tuple[str, int, Optional[float], Optional[float], Optional[float], float]) -> List[str]:
                name, cnt, avg_age, avg_bal, avg_dur, rate = m
                return [name or "(blank)", str(cnt), _fmt_num(avg_age, 1), _fmt_num(avg_bal, 2), _fmt_num(avg_dur, 1), _fmt_pct(rate)]

            rows = [row(m1), row(m2)]
            print(_table([field, "count", "avg(age)", "avg(balance)", "avg(duration)", "success"], rows, aligns=["<", ">", ">", ">", ">", ">"]))
            delta = m1[-1] - m2[-1]
            sign = "+" if delta >= 0 else ""
            print(f"Δ Erfolgsquote (A-B): {sign}{delta * 100:0.1f}%")

        elif choice == "8":
            print(_header("ANOVA-ÄHNLICHER F-WERT (BALANCE)"))
            field = _prompt_group_field("education")
            result = _anova_f_balance(current, field)
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
