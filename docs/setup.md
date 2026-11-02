# CreditDoc AI — Commandes utiles

Référence rapide pour tous les workflows courants. Toutes les commandes supposent que tu es dans `C:\repos\creditdoc-ai\` dans un terminal PowerShell.

---

## 0. Démarrage de la stack

### Lancer les services (Qdrant + Langfuse + Postgres)

```powershell
docker compose up -d
```

### Vérifier que tout tourne

```powershell
docker compose ps
```

Attendu : 3 services `Up` — `qdrant`, `langfuse-server`, `langfuse-db`.

### Logs d'un service

```powershell
docker compose logs -f langfuse-server
docker compose logs -f qdrant
```

### Arrêter la stack (données préservées)

```powershell
docker compose down
```

### Reset complet (⚠️ efface les traces Langfuse et Qdrant)

```powershell
docker compose down -v
Remove-Item -Recurse -Force data\langfuse-db, data\qdrant-storage
docker compose up -d
```

### URLs

| Service | URL | Login |
| --- | --- | --- |
| Langfuse UI | http://localhost:3000 | `amine@local.dev` / (mot de passe défini dans `.env`) |
| Qdrant UI | http://localhost:6333/dashboard | (pas d'auth) |
| Ollama | http://localhost:11434 | (pas d'auth) |

---

## 1. Ollama — modèles locaux

### Lister les modèles installés

```powershell
ollama list
```

### Tirer un modèle (si absent)

```powershell
ollama pull phi4
ollama pull llama3:8b
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
ollama pull llama-guard3
```

### Lancer un modèle en interactif (REPL)

```powershell
ollama run phi4
# puis tape ta question, Entrée. /bye pour quitter.
```

### Question one-shot

```powershell
ollama run phi4 "Qu'est-ce que le TAEG ?"
```

### Via l'API HTTP (utile pour debug)

```powershell
curl http://localhost:11434/api/generate -d '{\"model\": \"phi4\", \"prompt\": \"Qu''est-ce que le HCSF ?\", \"stream\": false}'
```

### Supprimer un modèle (libérer disque)

```powershell
ollama rm llama2
```

---

## 2. Ingestion du corpus

### Ingestion standard

```powershell
uv run python src\ingestion\ingest.py --source data\corpus
```

### Reset de la collection Qdrant avant ingestion (si schéma change)

```powershell
uv run python src\ingestion\reset_collection.py
uv run python src\ingestion\ingest.py --source data\corpus
```

### Vérifier le nombre de chunks ingérés

```powershell
curl http://localhost:6333/collections/creditdoc
```

Chercher `"points_count"` dans la réponse.

### Inspecter un chunk précis

```powershell
curl http://localhost:6333/collections/creditdoc/points/1
```

---

## 3. Smoke tests des pipelines (une seule question)

Avant chaque éval complète (30+ min), toujours lancer un smoke test (quelques secondes) pour vérifier qu'il n'y a pas d'erreur triviale.

### Baseline (LLM seul, sans RAG)

```powershell
uv run python src\pipelines\baseline.py
```

### Naive RAG (vector search → top-K → LLM)

```powershell
uv run python src\pipelines\naive_rag.py
```

### Advanced RAG (hybrid BM25+vector → RRF → rerank → LLM)

```powershell
uv run python src\pipelines\advanced_rag.py
```

### Vérifications post-smoke test

Pour chaque pipeline, ouvrir **http://localhost:3000** → **Traces** et vérifier qu'une nouvelle trace du bon nom apparaît avec les spans attendus :

| Pipeline | Spans attendus |
| --- | --- |
| `baseline` | `prompt_build`, `llm_answer` |
| `naive_rag` | `vector_search`, `prompt_build`, `llm_answer` |
| `advanced_rag` | `vector_search`, `bm25_search`, `rrf_fusion`, `rerank`, `prompt_build`, `llm_answer` |

Si un span manque ou est mal nommé, c'est à corriger dans le code avant de lancer l'éval.

---

## 4. Évaluation (Ragas sur golden dataset)

### Variables d'environnement à vérifier dans `.env`

```ini
LITELLM_MODEL=ollama/phi4          # modèle sous test
RAGAS_JUDGE_MODEL=ollama/llama3:8b # juge indépendant (ne pas confondre)
OLLAMA_BASE=http://localhost:11434
QDRANT_URL=http://localhost:6333
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
```

### Éval d'un pipeline seul

```powershell
uv run python eval\ragas_eval.py --pipeline baseline
uv run python eval\ragas_eval.py --pipeline naive_rag
uv run python eval\ragas_eval.py --pipeline advanced_rag
```

Chaque run : 15 questions × ~1-2 min = **~15-30 min** de runtime unattended.

### Éval des 3 pipelines en séquence (une seule commande)

```powershell
uv run python eval\ragas_eval.py --pipeline baseline; `
uv run python eval\ragas_eval.py --pipeline naive_rag; `
uv run python eval\ragas_eval.py --pipeline advanced_rag
```

**Total : ~60-90 min**. Lance avant d'aller dormir, résultats au réveil.

### Résultats

Les rapports atterrissent dans :

```
eval\results\<pipeline>_<DATE>.md
```

### Comparer deux runs côte à côte (manuellement)

```powershell
# Ouvre les deux rapports dans VS Code en split pane
code eval\results\naive_rag_2026-11-02.md eval\results\naive_rag_2026-11-15.md
```

### Debug une question spécifique via Langfuse

1. Ouvrir http://localhost:3000 → **Traces**
2. Filtrer par tag : `naive_rag` ou `advanced_rag`
3. Cliquer sur la trace de la question qui a échoué (par ex. `q4`)
4. Inspecter chaque span : chunks retrieved, prompt complet, réponse brute

---

## 5. Gestion des dépendances (uv)

### Ajouter un package

```powershell
uv add rank-bm25
uv add "langfuse>=2.60,<3"
```

### Retirer un package

```powershell
uv remove some-package
```

### Synchroniser l'environnement avec `pyproject.toml`

```powershell
uv sync
```

### Lister ce qui est installé

```powershell
uv pip list
uv pip show langfuse    # version précise d'un package
```

### Reset complet du venv (en cas de drift)

```powershell
Remove-Item -Recurse -Force .venv
uv sync
```

---

## 6. Observabilité Langfuse

### Flush manuel des traces (CLI quick-check)

```powershell
uv run python -c "from src.observability.langfuse_helper import flush; flush(); print('flushed')"
```

### Vérifier la santé du serveur Langfuse

```powershell
curl http://localhost:3000/api/public/health
```

### Export des traces d'une période (via SDK Python)

```powershell
uv run python scripts\export_traces.py --from "2026-11-01" --to "2026-11-02" --out traces.jsonl
```
*(à créer si pas encore présent)*

---

## 7. Dépannage — erreurs fréquentes

### `ModuleNotFoundError: No module named 'src'`

Ajouter en tête de `eval\ragas_eval.py` et des pipelines si run en standalone :

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Vérifier aussi la présence des fichiers vides :
`src\__init__.py`, `src\pipelines\__init__.py`, `src\observability\__init__.py`, `eval\__init__.py`.

### `AttributeError: 'QdrantClient' object has no attribute 'search'`

Qdrant 1.x a déprécié `search()`. Utiliser :

```python
hits = client.query_points(
    collection_name="creditdoc",
    query=query_vector,
    limit=k,
    with_payload=True,
).points
```

### `OSError: [WinError 1114] c10.dll` (PyTorch)

Installer Visual C++ Redistributable (x64), puis :

```powershell
uv remove torch sentence-transformers
uv add torch --index-url https://download.pytorch.org/whl/cpu
uv add sentence-transformers
```

### Langfuse `AttributeError: 'Langfuse' object has no attribute 'trace'`

SDK v3 contre serveur v2. Pinner v2 :

```powershell
uv add "langfuse>=2.60,<3"
```

### Ragas timeouts sur Ollama local

Dans `eval\ragas_eval.py`, forcer :

```python
run_config = RunConfig(max_workers=1, timeout=600, max_retries=2, max_wait=120)
# et sur le judge LLM :
judge = ChatOllama(model="llama3:8b", timeout=600, num_ctx=4096)
```

### Email validation error au démarrage de Langfuse

Dans `docker-compose.yml`, `LANGFUSE_INIT_USER_EMAIL` doit être une adresse valide **et** la DB doit être vide pour que la variable soit lue :

```powershell
Remove-Item -Recurse -Force data\langfuse-db
docker compose up -d
```

---

## 8. Workflow type d'une itération complète

```powershell
# 1. Modifier le code (pipeline, prompt, chunking…)

# 2. Smoke test rapide
uv run python src\pipelines\naive_rag.py

# 3. Vérifier la trace dans Langfuse
# → http://localhost:3000

# 4. Si OK, lancer l'éval complète
uv run python eval\ragas_eval.py --pipeline naive_rag

# 5. Comparer le rapport résultat au précédent
code eval\results\naive_rag_<DATE_PRECEDENTE>.md eval\results\naive_rag_<DATE_NOUVELLE>.md

# 6. Si delta intéressant, drafter le post LinkedIn dans drafts/
```

---

## 9. Chemins importants

| Rôle | Chemin |
| --- | --- |
| Corpus | `C:\repos\creditdoc-ai\data\corpus\` |
| Golden dataset | `C:\repos\creditdoc-ai\eval\golden_dataset.jsonl` |
| Résultats d'évals | `C:\repos\creditdoc-ai\eval\results\` |
| Drafts LinkedIn | `C:\repos\creditdoc-ai\drafts\` |
| Pipelines | `C:\repos\creditdoc-ai\src\pipelines\` |
| Helper Langfuse | `C:\repos\creditdoc-ai\src\observability\langfuse_helper.py` |
| Config Docker | `C:\repos\creditdoc-ai\docker-compose.yml` |
| Config env | `C:\repos\creditdoc-ai\.env` |
