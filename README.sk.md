# NN · Názov kroku

🇬🇧 [English version](README.md)

## TL;DR
Kľúčové čísla v 3–5 bodoch, každé z `results.json`. Baseline vs najlepší výsledok.

## Problém
Akú otázku tento krok rieši a prečo na nej záleží (odkaz na cieľ z roadmapy).

## Setup
Dáta (zdroj, veľkosť, rozdelenie), hardvér (Mac MPS / cloudové CPU / Colab), knižnice, seedy.

## Ako to funguje
Prechod kódom, ktorý vysvetľuje *prečo*, nielen *čo*. Krátke ukážky kódu, jeden graf na jednu myšlienku.

## Výsledky
| Metóda | Metrika | … |
|---|---|---|
| baseline | | |

![Výsledky](assets/results.png)

## Analýza chýb
Kde to zlyháva a prečo. Ktoré rozdiely sú v šume.

## Čo som sa naučil
- …

## Spustenie
```bash
uv sync
uv run python experiment.py   # → results.json
uv run python plot.py         # → assets/*.png
```

## Obmedzenia
- …
