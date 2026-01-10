from __future__ import annotations

"""
Automated Validation Tests – Functional vs Imperative Implementation

Authors: Peter Ngo, Alex Uscata
Class: INA 23A
Module: M323 – Functional Programming
Date: 2026-01-10

This test module validates that the functional and imperative implementations
of the bank marketing analysis produce identical results for the same input data.

The goal is not to test performance, but to verify semantic equivalence between
both programming paradigms.

Tested aspects:
    1. Group-by education metrics:
        - group names
        - group sizes (counts)
        - average age
        - average balance
        - success rate
    2. ANOVA F-statistic for balance:
        - identical F-value
        - identical degrees of freedom (df_between, df_within)

These tests demonstrate that both implementations compute the same analytical
results despite using fundamentally different programming styles.
"""

import math
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

# Imports
from common_io import load_bank_data
import functional_version as func
import imperative_version as imp


def round_or_none(x, digits=4):
    if x is None:
        return None
    return round(x, digits)


def test_group_by_education_metrics():
    """
    Test: The functional and imperative implementations produce identical results
    for the "Group by Education" analysis.

    Compared aspects:
       - Group names
       - Group sizes (counts)
       - Average age (rounded)
       - Average balance (rounded)
       - Success rate (rounded)
    """
    data = load_bank_data()

    f_metrics = func.group_metrics(data, "education")
    i_metrics = imp.group_metrics(data, "education")

    assert len(f_metrics) == len(i_metrics), "Unterschiedliche Anzahl Gruppen"

    for f, i in zip(f_metrics, i_metrics):
        f_name, f_cnt, f_age, f_bal, f_rate = f
        i_name, i_cnt, i_age, i_bal, i_rate = i

        assert f_name == i_name, f"Gruppenname verschieden: {f_name} vs {i_name}"
        assert f_cnt == i_cnt, f"Count verschieden in Gruppe {f_name}"

        assert round_or_none(f_age, 4) == round_or_none(i_age, 4), f"avg(age) verschieden in {f_name}"
        assert round_or_none(f_bal, 4) == round_or_none(i_bal, 4), f"avg(balance) verschieden in {f_name}"
        assert round_or_none(f_rate, 6) == round_or_none(i_rate, 6), f"success rate verschieden in {f_name}"


def test_anova_f_value_education():
    """
    Test: The ANOVA F-statistic for the "education" grouping is identical in the
    functional and imperative implementations.

    Compared aspects:
        - F-value (rounded)
        - df_between (degrees of freedom between groups)
        - df_within (degrees of freedom within groups)
    """
    data = load_bank_data()

    f_res = func.anova_f_balance(data, "education")
    i_res = imp.anova_f_balance(data, "education")

    assert f_res is not None and i_res is not None, "Eine der Versionen liefert None"

    f_f, f_dfb, f_dfw = f_res
    i_f, i_dfb, i_dfw = i_res

    assert f_dfb == i_dfb, "df_between verschieden"
    assert f_dfw == i_dfw, "df_within verschieden"

    if math.isinf(f_f) or math.isinf(i_f):
        assert math.isinf(f_f) and math.isinf(i_f), "Nur eine Version liefert inf"
    else:
        assert round(f_f, 6) == round(i_f, 6), "F-Wert verschieden"


def test_group_counts_sum_to_total():
    """
    Invariant test: The sum of group counts must equal the number of records.
    Also checks basic bounds for success rates.
    """
    data = load_bank_data()

    f_metrics = func.group_metrics(data, "education")
    i_metrics = imp.group_metrics(data, "education")

    f_total = sum(cnt for _, cnt, *_ in f_metrics)
    i_total = sum(cnt for _, cnt, *_ in i_metrics)

    assert f_total == len(data), "Functional: group counts do not sum to total records"
    assert i_total == len(data), "Imperative: group counts do not sum to total records"

    for name, cnt, avg_age, avg_bal, rate in f_metrics:
        assert cnt >= 0
        assert 0.0 <= rate <= 1.0, f"Functional: invalid success rate in {name}"

    for name, cnt, avg_age, avg_bal, rate in i_metrics:
        assert cnt >= 0
        assert 0.0 <= rate <= 1.0, f"Imperative: invalid success rate in {name}"


def test_deterministic_results_repeated_runs():
    """
    Determinism test: Running the same computation multiple times must produce
    identical results (including ordering).
    """
    data = load_bank_data()

    f_first = func.group_metrics(data, "education")
    i_first = imp.group_metrics(data, "education")

    for _ in range(30):
        assert func.group_metrics(data, "education") == f_first, "Functional results are not deterministic"
        assert imp.group_metrics(data, "education") == i_first, "Imperative results are not deterministic"


def test_anova_returns_none_on_insufficient_data():
    """
    Edge-case test: ANOVA should return None if there is not enough data
    (e.g., empty dataset or fewer than 2 non-empty groups).
    """
    assert func.anova_f_balance([], "education") is None
    assert imp.anova_f_balance([], "education") is None

    # Dataset with only one group -> df_between would be 0 -> should return None
    one_group = [
        {"education": "primary", "balance": 100.0, "age": 30, "complete": False},
        {"education": "primary", "balance": 200.0, "age": 40, "complete": True},
        {"education": "primary", "balance": 300.0, "age": 50, "complete": False},
    ]
    assert func.anova_f_balance(one_group, "education") is None
    assert imp.anova_f_balance(one_group, "education") is None




if __name__ == "__main__":
    print("Running functional vs imperative comparison tests...")
    test_group_by_education_metrics()
    print("✓ Group by education metrics identical")

    test_anova_f_value_education()
    print("✓ ANOVA F-value identical")

    print("\nAll tests passed successfully.")
    
    test_group_counts_sum_to_total()
    print("✓ Group count invariant holds")

    test_deterministic_results_repeated_runs()
    print("✓ Determinism verified (repeated runs)")

    test_anova_returns_none_on_insufficient_data()
    print("✓ ANOVA edge cases handled correctly")
