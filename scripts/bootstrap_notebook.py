from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "01_start.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    md = (
        "# Start – Bank Marketing Analyse (Stdlib)\n\n"
        "Dieses Notebook lädt `data/DatenBank.csv` (Semikolon-getrennt) und erzeugt Start-Auswertungen – "
        "ohne externe Libraries (nur Python-Standardbibliothek).\n\n"
        "In VS Code: oben rechts den Kernel aus `.venv` auswählen.\n"
    )

    def code_cell(src: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"language": "python"},
            "outputs": [],
            "source": src.splitlines(True),
        }

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"language": "markdown"},
                "source": md.splitlines(True),
            },
            code_cell(
                "from __future__ import annotations\n\n"
                "import csv\n"
                "import statistics\n"
                "from collections import defaultdict\n"
                "from pathlib import Path\n\n"
                "DATA_PATH = Path('..') / 'data' / 'DatenBank.csv'\n"
                "print('CSV exists:', DATA_PATH.exists())\n"
                "print('CSV path  :', DATA_PATH.resolve())\n"
            ),
            code_cell(
                "NUM_FIELDS = {'id', 'age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous'}\n\n"
                "def load_rows(path: Path) -> list[dict[str, object]]:\n"
                "    rows: list[dict[str, object]] = []\n"
                "    with path.open('r', encoding='utf-8', newline='') as f:\n"
                "        reader = csv.DictReader(f, delimiter=';', quotechar='\"')\n"
                "        for raw in reader:\n"
                "            row: dict[str, object] = {}\n"
                "            for key, value in raw.items():\n"
                "                if key in NUM_FIELDS:\n"
                "                    row[key] = int(value)\n"
                "                else:\n"
                "                    row[key] = value\n"
                "            rows.append(row)\n"
                "    return rows\n\n"
                "rows = load_rows(DATA_PATH)\n"
                "print('rows:', len(rows))\n"
                "rows[0]\n"
            ),
            code_cell(
                "# 1) Erfolgsquote der Kampagne\n\n"
                "total = len(rows)\n"
                "yes = sum(1 for r in rows if r['complete'] == 'yes')\n"
                "rate = (yes / total) * 100 if total else 0\n"
                "print(f'Anzahl Kunden: {total}')\n"
                "print(f'Abschlüsse (yes): {yes}')\n"
                "print(f'Erfolgsquote: {rate:.1f} %')\n"
            ),
            code_cell(
                "# 2) Gesprächsdauer-Statistiken (duration)\n\n"
                "durations = [r['duration'] for r in rows]\n"
                "print('duration min     :', min(durations))\n"
                "print('duration max     :', max(durations))\n"
                "print('duration mean    :', round(statistics.mean(durations), 2))\n"
                "print('duration variance:', round(statistics.pvariance(durations), 2))\n"
            ),
            code_cell(
                "def group_summary(group_field: str):\n"
                "    groups: dict[str, list[dict[str, object]]] = defaultdict(list)\n"
                "    for r in rows:\n"
                "        groups[str(r[group_field])].append(r)\n\n"
                "    out = []\n"
                "    for key, items in sorted(groups.items(), key=lambda kv: kv[0]):\n"
                "        n = len(items)\n"
                "        avg_age = statistics.mean([i['age'] for i in items])\n"
                "        avg_balance = statistics.mean([i['balance'] for i in items])\n"
                "        yes = sum(1 for i in items if i['complete'] == 'yes')\n"
                "        rate = (yes / n) * 100 if n else 0\n"
                "        out.append((key, n, avg_age, avg_balance, rate))\n"
                "    return out\n\n"
                "def print_table(title: str, items):\n"
                "    print(title)\n"
                "    print('Kategorie | n | avg_age | avg_balance | success_rate')\n"
                "    for key, n, avg_age, avg_balance, rate in items:\n"
                "        print(f'{key:>10} | {n:>4} | {avg_age:>7.2f} | {avg_balance:>11.2f} | {rate:>11.2f}%')\n\n"
                "print_table('Education', group_summary('education'))\n"
                "print()\n"
                "print_table('Marital', group_summary('marital'))\n"
            ),
            code_cell(
                "# 3) Beispiel-Filter: balance > 1000 und kein loan\n\n"
                "filtered = [r for r in rows if (r['balance'] > 1000 and r['loan'] == 'no')]\n"
                "print('Treffer:', len(filtered))\n"
                "print('Beispiel:', filtered[0] if filtered else None)\n"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    notebook_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {notebook_path} ({notebook_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
