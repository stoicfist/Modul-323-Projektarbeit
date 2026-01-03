# Projektübersicht

**Problem**
Wir analysieren den „Bank Marketing“-Datensatz einer portugiesischen Bank, um Muster in Kundendaten zu erkennen und die Abschlusswahrscheinlichkeit einer Kampagne verständlich auszuwerten (Zielvariable: `complete=yes`).

**Datensatz**
Relevante Merkmale sind u.a. `age`, `job`, `marital`, `education`, `balance`, `housing`, `loan`, `duration`, `pdays` und `complete`.

**Zwei Implementationen**
Dieses Projekt enthält zwei gleichwertige Konsolen-Tools mit identischer Funktionalität und identischem Output:

- **Imperativ**: klassische Schleifen (`for`/`while`), mutable Zustände, manuelle Aggregationen.
- **Funktional**: `map`/`filter`/`reduce` (aus `functools`), Comprehensions und Lambdas, Mutation minimiert.

Beide Varianten laden die CSV, normalisieren Werte (z.B. `complete`), erlauben kombinierbare Filter und liefern Kennzahlen, Gruppierungen sowie eine ANOVA-ähnliche F-Auswertung für `balance`.
