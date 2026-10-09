# Golden Dataset Changelog

## [0.1.0] — 2026-10-08

### Added
- Initial 15 seed entries across 5 categories (definition, calculation, regulatory, nuanced, out_of_scope).
- Difficulty distribution: 7 easy, 6 medium, 2 hard.
- Dataset schema documented in `README.md`.
- Validation script `validate_dataset.py`.

### Methodology
- Questions drafted from real advisor use cases.
- Expected answers verified against:
  - Code de la consommation (France)
  - Haut Conseil de Stabilité Financière publications
  - Loi Lemoine (2022)
- `must_contain` / `must_not_contain` lists tuned to catch common hallucinations observed in Week 2 model comparison.

### Known gaps
- Moroccan regulatory content (Bank Al-Maghrib directives) not yet included.
- Only French regulatory context so far.
- Hard-difficulty entries under-represented (2/15).

## [0.2.0] — planned 2026-10-12

### To add
- 15-25 additional entries, matching target distribution.
- Moroccan regulatory content (minimum 5 entries).
- Prompt injection attempts in out_of_scope.