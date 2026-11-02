# CreditDoc AI — Corpus de référence

Corpus de 7 documents Markdown construit pour servir de base de connaissances à l'assistant RAG CreditDoc AI, et comme référence (grounding) pour le golden dataset d'évaluation.

## Composition

| Fichier | Thème | Source | Articles / Sections | Taille |
| --- | --- | --- | --- | --- |
| `01-credit-conso-definitions.md` | Définitions, champ d'application | Code de la conso (Legifrance) | L311-1, L312-1 à L312-4 | ~10 KB |
| `02-credit-conso-publicite-info.md` | Publicité, information précontractuelle, FISE, FICP | Code de la conso | L312-5 à L312-16 | ~10 KB |
| `03-credit-conso-formation-retractation.md` | Formation du contrat, rétractation 14 jours, délai 7 jours | Code de la conso | L312-18 à L312-28 | ~6 KB |
| `04-credit-renouvelable.md` | Crédit renouvelable (durée, remboursement, information) | Code de la conso | L312-57 à L312-73 | ~10 KB |
| `05-taeg-taux-usure.md` | TAEG, taux effectif global, taux d'usure | Code de la conso | L314-1 à L314-11 | ~6 KB |
| `06-credit-immobilier.md` | Crédit immobilier (champ, FISE, délai de réflexion, remboursement anticipé) | Code de la conso | L313-1 à L313-41 | ~14 KB |
| `07-hcsf-credit-immobilier.md` | **HCSF : règles 35% / 25 ans** | HCSF / economie.gouv.fr | Décisions D-HCSF-2021-7, 2023-2, 2023-6 | ~4 KB |

**Total** : ~60 KB, 7 documents.

## Couverture par rapport au golden dataset

Chaque document a été sélectionné pour couvrir les catégories de questions du golden dataset :

- **definition** → 01, 05 (TAEG), 07 (taux d'effort)
- **calculation** → 05 (TAEG, taux d'usure), 07 (35%)
- **regulatory** → 02 (FICP, FISE), 06 (délai de réflexion 10 jours), 07 (contraintes HCSF)
- **nuanced** → 03 (rétractation vs annulation), 04 (crédit renouvelable), 06 (remboursement anticipé immobilier)
- **out_of_scope** → aucun contenu dédié (c'est voulu : le modèle doit refuser)

## Chemin cible et ingestion

Placer ces 8 fichiers (00-README.md inclus) dans :

```
C:\repos\creditdoc-ai\data\corpus\
```

Puis lancer l'ingestion :

```powershell
uv run python src\ingestion\ingest.py --source data\corpus
```

Ou, si ton script `ingest.py` prend un pattern :

```powershell
uv run python src\ingestion\ingest.py --pattern "data/corpus/*.md"
```

## Notes

- Les textes légaux ont été extraits de Legifrance le 2 novembre 2026. Pour un déploiement production, re-scraper trimestriellement pour capter les mises à jour (le Code de la conso bouge, notamment via ordonnances).
- Les en-têtes `## Article LXXX-YY` servent de **points d'ancrage pour le chunking** : paramétrer le splitter sur `markdown_header_text_splitter` ou équivalent pour conserver un article par chunk quand c'est possible.
- Chaque article conserve sa **source URL en bas** — utile pour le `expected_sources` du golden dataset et pour l'affichage des citations dans la réponse de l'assistant.
- Le fichier HCSF (07) est volontairement **synthétique** plutôt que l'intégralité du texte de la décision PDF, car la décision officielle est très courte (3 pages) et peu informative hors contexte ; la synthèse officielle du site HCSF est pédagogiquement plus utile pour le RAG.

## Sources

- **Legifrance — Code de la consommation, Livre III** : https://www.legifrance.gouv.fr/codes/section_lc/LEGITEXT000006069565/LEGISCTA000032221961/
- **HCSF — Mesure sur les crédits immobiliers** : https://www.economie.gouv.fr/hcsf/mesures/mesure-relative-loctroi-de-credits-immobiliers
