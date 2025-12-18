# Bank Marketing Data – Kundenanalyse & Abschlussprognose

## Projektübersicht

Dieses Projekt analysiert einen Bank-Marketing-Datensatz aus Portugal. Ziel ist es, Kundendaten aus einer Marketingkampagne auszuwerten, um Erfolgsquoten zu berechnen, Kundengruppen zu vergleichen und statistische Zusammenhänge zu untersuchen.

Das Projekt wird im Rahmen des Moduls **M323 – Funktionales Programmieren** umgesetzt und besteht aus zwei Versionen:

- **Version 1.0:** Imperative Umsetzung
- **Version 2.0:** Funktional refactored (map, filter, reduce)

---

## Ausgangssituation / Problembeschreibung

Während einer Marketingkampagne hat eine portugiesische Bank verschiedene Kundendaten gesammelt, darunter Alter, Beruf, Familienstand, Kontostand und Gesprächsdauer.  
Diese Informationen liegen aktuell als unstrukturierter Datensatz vor, was es erschwert, schnell wichtige Muster zu erkennen – zum Beispiel, welche Kundentypen häufiger ein Produkt abschließen oder welche Faktoren besonders stark mit dem Kampagnenerfolg zusammenhängen.

Das Programm analysiert diese Daten und bereitet sie verständlich auf, sodass Trends sichtbar werden, Kundengruppen miteinander verglichen werden können und die Erfolgswahrscheinlichkeit eines Produktabschlusses (`complete = yes`) besser eingeschätzt werden kann.

---

## Datengrundlage

### Hauptdimensionen des Datensatzes

1. **Demografie:** Alter (`age`), Bildungsstand (`education`), Familienstand (`marital`)
2. **Finanzen:** Kontostand (`balance`)
3. **Gesprächsdauer:** Dauer des letzten Telefonkontakts (`duration`)
4. **Ergebnisvariable:** Produktabschluss (`complete`)

### Verwendete Variablen

- `age` (numerisch)
- `job` (kategorisch)
- `marital` (kategorisch)
- `education` (kategorisch)
- `balance` (numerisch)
- `housing` (binär)
- `loan` (binär)
- `duration` (numerisch)
- `pdays` (numerisch)
- `complete` (binär, Zielvariable)

Der Datensatz besitzt mehrere Dimensionen und eignet sich sehr gut für Filter-, Transformations- und Reduktionsoperationen.

---

## Produktfunktionen

Das Programm stellt folgende Auswertungen bereit:

1. **Erfolgsquote der Kampagne**  
   Berechnung der Gesamtanzahl Kunden, der Anzahl erfolgreicher Abschlüsse (`complete = yes`) sowie der Erfolgsquote in Prozent.

2. **Filterung bestimmter Kundengruppen**  
   Filter nach Kriterien wie Immobilienkredit, kein Privatkredit oder `balance > 1000`.

3. **Transformation numerischer Variablen**  
   Transformationen wie `log(balance)` oder `balance² + 1`.

4. **Analyse der Gesprächsdauer**  
   Berechnung von Minimum, Maximum, Durchschnitt und Varianz der Gesprächsdauer sowie optional Untersuchung des Zusammenhangs zwischen Gesprächsdauer und Kampagnenerfolg.

5. **Gruppierung nach Bildungsstand (education)**  
   Anzahl Kunden, Durchschnittsalter, Durchschnittsbalance und Erfolgsquote pro Kategorie.

6. **Gruppierung nach Familienstand (marital)**  
   Vergleich von `single`, `married` und `divorced` hinsichtlich Balance, Gesprächsdauer und Erfolgsquote.

7. **Direkter Vergleich zweier Gruppen**  
   Vergleich zweier frei wählbarer Gruppen (z. B. `married` vs. `single`) bezüglich Alter, Balance, Gesprächsdauer, Abschlussquote und Differenz der Erfolgsquoten.

8. **Statistische Analyse (ANOVA-ähnlich)**  
   Berechnung eines F-Werts zur Analyse der Balance zwischen Gruppen, basierend auf Varianz innerhalb und zwischen Gruppen, inklusive optionaler Interpretation.

---

## Technologien

Das Projekt wird in **Python** umgesetzt und in einem **Jupyter Notebook** ausgeführt.  
Es werden ausschließlich Funktionen der Python-Standardbibliothek verwendet, ohne externe Frameworks oder Libraries.  
Die Benutzeroberfläche besteht aus einer textbasierten Konsolenausgabe.

In Version 1.0 kommen imperative Sprachmittel wie Schleifen, if-Anweisungen, Variablen und manuelle Aggregationen zum Einsatz.  
Version 2.0 nutzt funktionale Sprachmittel wie `map`, `filter`, `reduce` (aus `functools`), Lambda-Funktionen sowie Listen-Comprehensions.

Die Projektdokumentation wird vollständig in **LaTeX (Overleaf)** erstellt, um eine saubere, reproduzierbare und kollaborative Zusammenarbeit im Team zu ermöglichen.

---

## Jupyter + Git in VS Code (Empfehlung)

Damit ihr Notebooks sauber versionieren könnt, ist wichtig:

- Virtuelle Umgebungen und Cache-Dateien **nicht** committen (ist über `.gitignore` bereits abgedeckt).
- Abhängigkeiten **festhalten** (hier: `requirements.txt`).
- Notebook-Outputs möglichst **nicht** als „Rauschen“ im Git-Review lassen (siehe optional unten).

### 1) VS Code Extensions

- Installiere die Extensions **Python** (ms-python.python) und **Jupyter** (ms-toolsai.jupyter).

### 2) Setup: Virtuelle Umgebung + Kernel

#### Windows (PowerShell)

Python installieren (falls noch nicht vorhanden):

```powershell
winget install -e --id Python.Python.3.12
```

Projekt-Umgebung erstellen:

```powershell
cd c:\GIT\Modul-323-Projektarbeit

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Falls `python` den Microsoft Store öffnet oder "Python was not found" erscheint: VS Code/Terminal neu starten und optional in Windows die **App execution aliases** für `python.exe`/`python3.exe` deaktivieren.

#### Ubuntu (bash)

Python + venv installieren:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```

Projekt-Umgebung erstellen:

```bash
cd /path/to/Modul-323-Projektarbeit

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### Manjaro (bash)

Python installieren:

```bash
sudo pacman -Syu --needed python python-pip
```

Projekt-Umgebung erstellen:

```bash
cd /path/to/Modul-323-Projektarbeit

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dann in VS Code:

- `Ctrl+Shift+P` → **Python: Select Interpreter** → `.venv` wählen.
- Notebook öffnen/erstellen (`.ipynb`) → oben rechts **Select Kernel** → den `.venv`-Kernel wählen.

### 3) Notebook-Struktur (praktisch)

- Daten bleiben in `data/` (wie jetzt).
- Notebooks z. B. in `notebooks/` ablegen (Ordner optional).

### 4) Optional: Saubere Git-Diffs für Notebooks

Notebooks sind JSON-Dateien; Git-Diffs werden schnell unübersichtlich, besonders wenn Outputs mit-committed werden.

Option A (Team-freundlich): **Outputs vor dem Commit löschen**

- In VS Code: `Kernel` → **Clear All Outputs**.

Option B (automatisch): **nbstripout** (entfernt Outputs beim Commit)

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install nbstripout
nbstripout --install
```

```bash
source .venv/bin/activate
python -m pip install nbstripout
nbstripout --install
```

Hinweis: Das setzt Git-Filter lokal; jedes Teammitglied sollte das einmal ausführen.

---

## Output (Beispiel)

Die Applikation gibt die Resultate als formatierten Text auf der Konsole aus.

### Beispielhafter Konsolen-Output

```
===========================
Bankkampagne – Erfolgsquote

Anzahl Kunden: 4521
Abschlüsse (yes): 512
Erfolgsquote: 11.3 %

===========================
Vergleich nach Education
Kategorie | Durchschnitt Balance | Erfolgsquote

primary | 850 | 8.5 %
secondary | 1200 | 10.1 %
tertiary | 1800 | 14.3 %

ANOVA F-Wert: 5.21
```

---

## Hinweise

- Version 1.0 und Version 2.0 erzeugen denselben Output
- Der Fokus liegt auf dem Vergleich imperativer und funktionaler Programmieransätze
- Das Projekt ist für Partnerarbeit konzipiert
