from __future__ import annotations

"""Bank Marketing Analysis - Functional Implementation

Authors: Peter Ngo, Alex Uscata
Class: INA 23A
Module: M323 - Functional Programming
Date: 2026-01-06
Version: 2.0.1

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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

from common_io import load_bank_data, normalize_choice_yes_no, parse_balance_gt


# Type variables for generic function composition
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')


def compose(*fns: Callable) -> Callable:
    """Right-to-left function composition (mathematical convention).
    
    Composes multiple functions into a single function, applying them from right
    to left. This follows mathematical notation: (f ∘ g)(x) = f(g(x)).
    
    Functional approach: Uses reduce() to fold functions together, creating a new
    function that applies each transformation in reverse order. This enables
    declarative pipeline construction where operations read in mathematical order.
    
    The composition is lazy - no computation happens until the resulting function
    is called with an argument. This allows for efficient function reuse and
    clear separation between pipeline definition and execution.
    
    Args:
        *fns: Variable number of functions to compose. Each function should take
              one argument (the output of the next function in the chain).
    
    Returns:
        A new function that applies all input functions in right-to-left order.
        If no functions provided, returns identity function.
    
    Examples:
        >>> # Simple numeric transformations
        >>> add_10 = lambda x: x + 10
        >>> multiply_2 = lambda x: x * 2
        >>> square = lambda x: x ** 2
        >>> 
        >>> # compose applies right-to-left: square(multiply_2(add_10(5)))
        >>> transform = compose(square, multiply_2, add_10)
        >>> transform(5)  # (5 + 10) * 2 = 30, then 30^2 = 900
        900
        
        >>> # Data pipeline: filter then map
        >>> positive = lambda nums: filter(lambda x: x > 0, nums)
        >>> doubled = lambda nums: map(lambda x: x * 2, nums)
        >>> to_list = lambda it: list(it)
        >>> 
        >>> # Reads mathematically: to_list(doubled(positive(data)))
        >>> process = compose(to_list, doubled, positive)
        >>> process([-2, -1, 0, 1, 2, 3])
        [2, 4, 6]
        
        >>> # Bank data transformation pipeline
        >>> extract_balance = lambda data: map(lambda r: r.get('balance', 0), data)
        >>> filter_positive = lambda vals: filter(lambda x: x > 0, vals)
        >>> compute_total = lambda vals: sum(vals)
        >>> 
        >>> total_positive_balance = compose(compute_total, filter_positive, extract_balance)
        >>> records = [{'balance': 1000}, {'balance': -500}, {'balance': 2000}]
        >>> total_positive_balance(records)
        3000
    
    Note:
        For left-to-right composition (Unix pipe style), use pipe() instead.
    """
    def composed(arg: Any) -> Any:
        # reduce applies functions right-to-left by using reversed()
        # Starting value is the input arg, each function transforms the accumulator
        return reduce(lambda acc, fn: fn(acc), reversed(fns), arg)
    
    # Handle edge case: no functions provided
    return composed if fns else lambda x: x


def pipe(*fns: Callable) -> Callable:
    """Left-to-right function composition (Unix pipe style).
    
    Composes multiple functions into a single function, applying them from left
    to right. This follows Unix pipe notation: data | fn1 | fn2 | fn3.
    
    Functional approach: Uses reduce() to fold functions together, creating a new
    function that applies each transformation in the order they appear. This enables
    intuitive pipeline construction where operations read like natural language or
    Unix commands.
    
    The composition is lazy - no computation happens until the resulting function
    is called with an argument. This is more intuitive than compose() for many
    developers as it reads in execution order.
    
    Args:
        *fns: Variable number of functions to compose. Each function should take
              one argument (the output of the previous function in the chain).
    
    Returns:
        A new function that applies all input functions in left-to-right order.
        If no functions provided, returns identity function.
    
    Examples:
        >>> # Simple numeric transformations
        >>> add_10 = lambda x: x + 10
        >>> multiply_2 = lambda x: x * 2
        >>> square = lambda x: x ** 2
        >>> 
        >>> # pipe applies left-to-right: add_10(5), then multiply_2, then square
        >>> transform = pipe(add_10, multiply_2, square)
        >>> transform(5)  # (5 + 10) = 15, then * 2 = 30, then 30^2 = 900
        900
        
        >>> # Data pipeline: reads like English "filter, then map, then collect"
        >>> positive = lambda nums: filter(lambda x: x > 0, nums)
        >>> doubled = lambda nums: map(lambda x: x * 2, nums)
        >>> to_list = lambda it: list(it)
        >>> 
        >>> # Reads naturally: positive, then doubled, then to_list
        >>> process = pipe(positive, doubled, to_list)
        >>> process([-2, -1, 0, 1, 2, 3])
        [2, 4, 6]
        
        >>> # Bank data ETL pipeline (Extract, Transform, Load)
        >>> extract_balance = lambda data: map(lambda r: r.get('balance', 0), data)
        >>> filter_positive = lambda vals: filter(lambda x: x > 0, vals)
        >>> compute_total = lambda vals: sum(vals)
        >>> 
        >>> # Reads like steps: extract, filter, compute
        >>> total_positive_balance = pipe(extract_balance, filter_positive, compute_total)
        >>> records = [{'balance': 1000}, {'balance': -500}, {'balance': 2000}]
        >>> total_positive_balance(records)
        3000
        
        >>> # Complex bank marketing analysis
        >>> has_housing_loan = lambda r: r.get('housing') is True
        >>> high_balance = lambda r: r.get('balance', 0) > 1000
        >>> extract_age = lambda r: r.get('age', 0)
        >>> 
        >>> analyze_customers = pipe(
        ...     lambda data: filter(has_housing_loan, data),
        ...     lambda data: filter(high_balance, data),
        ...     lambda data: map(extract_age, data),
        ...     lambda ages: list(ages),
        ...     lambda ages: sum(ages) / len(ages) if ages else 0
        ... )
        >>> customers = [
        ...     {'housing': True, 'balance': 1500, 'age': 30},
        ...     {'housing': True, 'balance': 500, 'age': 25},
        ...     {'housing': False, 'balance': 2000, 'age': 40}
        ... ]
        >>> analyze_customers(customers)  # Average age of housing loan customers with balance > 1000
        30.0
    
    Note:
        For right-to-left composition (mathematical style), use compose() instead.
    """
    def piped(arg: Any) -> Any:
        # reduce applies functions left-to-right (no reversed() needed)
        # Starting value is the input arg, each function transforms the accumulator
        return reduce(lambda acc, fn: fn(acc), fns, arg)
    
    # Handle edge case: no functions provided
    return piped if fns else lambda x: x


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


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], aligns: Optional[Sequence[str]] = None) -> str:
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
    widths = list(map(len, headers))

    # Use reduce to calculate maximum column widths across all rows
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

def combine_predicates(*predicates: Callable[[Dict[str, Any]], bool]) -> Callable[[Dict[str, Any]], bool]:
    """Combine multiple predicate functions into a single predicate using AND logic.
    
    Higher-order function that takes multiple predicate functions and returns a new
    predicate that returns True only if ALL input predicates return True. This enables
    declarative composition of filter conditions without nested if statements.
    
    Functional approach: Uses all() with a generator expression to short-circuit
    evaluation. The combined predicate maintains purity - no side effects, returns
    same output for same input.
    
    This is particularly useful for composing multiple filter conditions in a
    declarative way, allowing individual predicates to be defined separately,
    tested independently, and combined as needed.
    
    Args:
        *predicates: Variable number of predicate functions. Each predicate should
                    take a record (dict) and return bool.
    
    Returns:
        A new predicate function that returns True if all predicates return True.
        If no predicates provided, returns a predicate that always returns True.
    
    Examples:
        >>> # Individual predicates
        >>> has_housing = lambda r: r.get('housing') is True
        >>> has_loan = lambda r: r.get('loan') is True
        >>> high_balance = lambda r: r.get('balance', 0) > 1000
        >>> 
        >>> # Combine predicates
        >>> combined = combine_predicates(has_housing, has_loan, high_balance)
        >>> 
        >>> # Test with records
        >>> record1 = {'housing': True, 'loan': True, 'balance': 1500}
        >>> combined(record1)
        True
        >>> 
        >>> record2 = {'housing': True, 'loan': False, 'balance': 1500}
        >>> combined(record2)
        False
        
        >>> # Use with filter()
        >>> records = [
        ...     {'housing': True, 'loan': True, 'balance': 1500},
        ...     {'housing': True, 'loan': False, 'balance': 2000},
        ...     {'housing': False, 'loan': True, 'balance': 500}
        ... ]
        >>> list(filter(combined, records))
        [{'housing': True, 'loan': True, 'balance': 1500}]
        
        >>> # Dynamically build predicates
        >>> predicates_list = []
        >>> if True:  # Some condition
        ...     predicates_list.append(has_housing)
        >>> if True:  # Another condition
        ...     predicates_list.append(high_balance)
        >>> dynamic_filter = combine_predicates(*predicates_list)
        >>> len(list(filter(dynamic_filter, records)))
        2
    
    Note:
        Uses short-circuit evaluation - stops checking predicates as soon as
        one returns False, improving performance.
    """
    def combined_predicate(row: Dict[str, Any]) -> bool:
        # Short-circuit evaluation: returns True only if all predicates return True
        return all(pred(row) for pred in predicates)
    
    return combined_predicate


def _success_overall(data: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """Calculate overall campaign success metrics.
    
    Args:
        data: List of bank marketing records (dicts with 'complete' field).
    
    Returns:
        Tuple of (total_count, success_count, success_rate) where success_rate
        is a float between 0.0 and 1.0. Success is complete=True.
    
    Example:
        >>> _success_overall([{'complete': True}, {'complete': False}])
        (2, 1, 0.5)
    """
    total = len(data)
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
    # Extract and filter valid duration values using functional pipeline
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
    
    No explicit loop indices or manual counter increments - each transformation
    is expressed as a data pipeline operation.
    
    Performance Note:
    The immutable tuple accumulator creates O(n) copies per reduction step
    (where n is the number of buckets). With m records in the dataset, the total
    time complexity is O(n × m) = O(n²) when n and m are of similar magnitude.
    This is intentional to demonstrate pure functional immutability principles:
    the step function never mutates existing data structures, instead returning
    entirely new tuples. For production use with large datasets, a hybrid approach
    using list mutation would be more efficient (O(m) time with in-place updates).
    This implementation prioritizes educational clarity and functional purity over
    runtime performance.
    
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

    # Immutable accumulator: Tuple[Tuple[int, int], ...] where each tuple = (total, yes_count)
    def step(acc: Tuple[Tuple[int, int], ...], row: Dict[str, Any]) -> Tuple[Tuple[int, int], ...]:
        """Pure reducer: returns new tuple instead of mutating.
        
        Creates new tuple by concatenating: buckets_before + updated_bucket + buckets_after
        This maintains referential transparency and immutability.
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
        return acc[:i] + ((new_total, new_yes),) + acc[i+1:]

    init = tuple((0, 0) for _ in starts)
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
    
    Performance Considerations:
    The dict unpacking {**acc, key: value} creates a new dictionary on each
    iteration, copying all existing key-value pairs. For n records, this results
    in O(n²) time complexity due to repeated copying of the growing dictionary.
    Additionally, the tuple concatenation (existing_group + (row,)) creates a new
    tuple for each group on every update. This demonstrates pure functional
    immutability principles but is not optimal for large datasets. Python's
    dict.setdefault() or collections.defaultdict with list.append() would achieve
    O(n) time complexity but would mutate state, sacrificing functional purity.
    In languages like Haskell, persistent data structures (using structural
    sharing) would make this pattern efficient without sacrificing immutability.
    
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
        """Pure reducer: creates new dict via unpacking instead of mutation."""
        k = row.get(key)
        k_str = "" if k is None else str(k)
        existing_group = acc.get(k_str, ())
        new_group = existing_group + (row,)
        return {**acc, k_str: new_group}

    return reduce(add, data, {})


def _group_metrics(data: List[Dict[str, Any]], key: str) -> List[Tuple[str, int, Optional[float], Optional[float], float]]:
    groups = _group_by_key(data, key)

    def metrics(name: str) -> Tuple[str, int, Optional[float], Optional[float], float]:
        rows = list(groups[name])
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
        rows = list(groups.get(name, ()))
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
        rows = list(groups.get(name, ()))
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
    return raw if raw in {"education", "marital", "job"} else default


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
    data = load_bank_data()
    current = list(data)

    print(_header("BANK MARKETING – CLI"))
    print(f"Datensätze geladen: {len(data)}")

    while True:
        choice = _menu()
        if choice == "q":
            print("Bye")
            return

        # Pattern matching with match/case (requires Python 3.10+)
        # More structured and readable than if/elif chains for menu handling
        match choice:
            case "1":
                print(_header("ERFOLGSQUOTE"))
                total, yes_count, quote = _success_overall(current)
                rows = [["Total", str(total)], ["Yes", str(yes_count)], ["Quote", _fmt_pct(quote)]]
                print(_table(["Metric", "Value"], rows, aligns=["<", ">"]))

            case "2":
                housing, loan, balance_gt = _prompt_filters()
                
                # Build predicates dynamically using functional composition
                predicates = []
                
                if housing is not None:
                    predicates.append(lambda r, h=housing: r.get("housing") is h)
                
                if loan is not None:
                    predicates.append(lambda r, l=loan: r.get("loan") is l)
                
                if balance_gt is not None:
                    predicates.append(create_balance_filter(balance_gt))
                
                # Compose all predicates with AND logic
                if predicates:
                    combined_filter = combine_predicates(*predicates)
                    current = list(filter(combined_filter, data))
                else:
                    current = list(data)
                
                print(_header("FILTER RESULT"))
                print(f"Aktueller Datenbestand: {len(current)} / {len(data)}")

            case "3":
                print(_header("TRANSFORMATIONEN"))
                
                # Declarative Pipeline Composition Demonstration:
                # Each analysis is expressed as "WHAT to compute" rather than "HOW to compute it".
                # The pipe() function chains transformations: data -> filter -> extract -> transform -> analyze -> format
                # This makes the data flow explicit and eliminates intermediate variables.
                
                # Base pipeline: Extract valid balance values from records
                balance_pipeline = pipe(
                    lambda records: filter(  # Keep only records with numeric balance
                        lambda r: isinstance(r.get("balance"), (int, float)), 
                        records
                    ),
                    lambda records: map(lambda r: float(r["balance"]), records),  # Extract as floats
                    lambda it: list(it)  # Materialize lazy iterators
                )
                
                # Transformation pipeline: Apply logarithmic transformation (handles zeros/negatives)
                log_transform_pipeline = pipe(
                    lambda vals: map(lambda b: math.log(b) if b > 0.0 else None, vals),  # log(x) for x > 0
                    lambda vals: filter(lambda x: x is not None, vals),  # Remove invalid results
                    lambda it: list(it)
                )
                
                # Transformation pipeline: Apply polynomial transformation (always valid)
                square_plus_one_pipeline = pipe(
                    lambda vals: map(lambda b: b * b + 1.0, vals),  # x² + 1 for all values
                    lambda it: list(it)
                )
                
                # Higher-order function: Creates a stats computation pipeline for a named transformation
                def create_stats_pipeline(name: str):
                    """Returns pipeline: values -> statistics dict -> formatted table row"""
                    return pipe(
                        # Compute all statistics declaratively (no intermediate variables)
                        lambda vals: {
                            'name': name,
                            'count': len(vals),
                            'min': min(vals) if vals else None,
                            'max': max(vals) if vals else None,
                            'mean': _mean(vals),
                            'var': _variance_population(vals)
                        },
                        # Transform stats dict into formatted strings for table display
                        lambda stats: [
                            stats['name'],
                            str(stats['count']),
                            _fmt_num(stats['min']),
                            _fmt_num(stats['max']),
                            _fmt_num(stats['mean']),
                            _fmt_num(stats['var'])
                        ]
                    )
                
                # Compose complete analysis pipelines by chaining: extract -> transform -> analyze -> format
                log_analysis_pipeline = pipe(
                    balance_pipeline,              # records -> balances
                    log_transform_pipeline,        # balances -> log(balances)
                    create_stats_pipeline("log(balance)")  # log values -> formatted row
                )
                
                square_analysis_pipeline = pipe(
                    balance_pipeline,              # records -> balances
                    square_plus_one_pipeline,      # balances -> balance^2 + 1
                    create_stats_pipeline("balance^2+1")  # squared values -> formatted row
                )
                
                # Execute both analysis pipelines in parallel (no shared state or side effects)
                rows = [
                    log_analysis_pipeline(current),
                    square_analysis_pipeline(current)
                ]
                
                print(_table(["Transform", "n", "min", "max", "mean", "var"], rows, aligns=["<", ">", ">", ">", ">", ">"]))

            case "4":
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

            case "5":
                print(_header("GROUP BY EDUCATION"))
                metrics = _group_metrics(current, "education")
                rows = list(
                    map(
                        lambda t: [t[0] or "(blank)", str(t[1]), _fmt_num(t[2], 1), _fmt_num(t[3], 2), _fmt_pct(t[4])],
                        metrics,
                    )
                )
                print(_table(["education", "count", "avg(age)", "avg(balance)", "success"], rows, aligns=["<", ">", ">", ">", ">"]))

            case "6":
                print(_header("GROUP BY MARITAL"))
                metrics = _marital_compare(current)
                rows = list(map(lambda t: [t[0], str(t[1]), _fmt_num(t[2], 2), _fmt_num(t[3], 1), _fmt_pct(t[4])], metrics))
                print(_table(["marital", "count", "avg(balance)", "avg(duration)", "success"], rows, aligns=["<", ">", ">", ">", ">"]))

            case "7":
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

            case "8":
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

            case _:
                print("Ungültige Auswahl")


if __name__ == "__main__":
    main()
