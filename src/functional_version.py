from __future__ import annotations

import math
from functools import reduce
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    widths = list(map(len, headers))

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


def _success_overall(data: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    total = len(data)
    yes_count = sum(1 for r in data if r.get("complete") is True)
    quote = (yes_count / total) if total else 0.0
    return total, yes_count, quote


def _mean(values: List[float]) -> Optional[float]:
    return (sum(map(float, values)) / float(len(values))) if values else None


def _variance_population(values: List[float]) -> Optional[float]:
    if not values:
        return None
    mu = _mean(values)
    if mu is None:
        return None
    return sum((float(v) - mu) ** 2 for v in values) / float(len(values))


def _duration_stats(data: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
    durations = list(map(int, filter(lambda d: isinstance(d, int), (r.get("duration") for r in data))))
    if not durations:
        return None, None, None, None
    durations_f = list(map(float, durations))
    return min(durations), max(durations), _mean(durations_f), _variance_population(durations_f)


def _duration_buckets(data: List[Dict[str, Any]], bucket_size: int) -> List[Tuple[str, int, float]]:
    bucket_size = bucket_size if bucket_size > 0 else 60
    durations = list(filter(lambda d: isinstance(d, int), (r.get("duration") for r in data)))
    if not durations:
        return []
    max_d = max(map(int, durations))
    starts = list(range(0, max_d + 1, bucket_size))

    def idx_of(d: int) -> int:
        i = d // bucket_size
        return min(max(i, 0), len(starts) - 1)

    def step(acc: List[List[int]], row: Dict[str, Any]) -> List[List[int]]:
        d = row.get("duration")
        if not isinstance(d, int):
            return acc
        i = idx_of(d)
        acc[i][0] += 1
        acc[i][1] += 1 if row.get("complete") is True else 0
        return acc

    init = [[0, 0] for _ in starts]  # [total, yes]
    counts = reduce(step, data, init)

    def mk(i: int) -> Tuple[str, int, float]:
        start = starts[i]
        end = start + bucket_size - 1
        total, yes = counts[i]
        label = f"{start:>4d}-{end:<4d}"
        rate = (yes / total) if total else 0.0
        return label, total, rate

    return list(map(mk, range(len(starts))))


def _group_by_key(data: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    def add(acc: Dict[str, List[Dict[str, Any]]], row: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        k = row.get(key)
        k_str = "" if k is None else str(k)
        acc.setdefault(k_str, []).append(row)
        return acc

    return reduce(add, data, {})


def _group_metrics(data: List[Dict[str, Any]], key: str) -> List[Tuple[str, int, Optional[float], Optional[float], float]]:
    groups = _group_by_key(data, key)

    def metrics(name: str) -> Tuple[str, int, Optional[float], Optional[float], float]:
        rows = groups[name]
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
        rows = groups.get(name, [])
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
        rows = groups.get(name, [])
        count = len(rows)
        ages = [float(r["age"]) for r in rows if isinstance(r.get("age"), int)]
        balances = [float(r["balance"]) for r in rows if isinstance(r.get("balance"), (int, float))]
        durations = [float(r["duration"]) for r in rows if isinstance(r.get("duration"), int)]
        yes = sum(1 for r in rows if r.get("complete") is True)
        rate = (yes / count) if count else 0.0
        return name, count, _mean(ages), _mean(balances), _mean(durations), rate

    return metrics(g1), metrics(g2)


def _anova_f_balance(data: List[Dict[str, Any]], key: str) -> Optional[Tuple[float, int, int]]:
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
