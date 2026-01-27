# Distributed Job Recommendation System (Polyglot Persistence)

A hybrid job recommender that combines:
- **Semantic similarity** (Sentence-Transformers embeddings + **FAISS** vector index)
- **Skill-graph reasoning** (**Neo4j** property graph with weighted skill co-occurrence)
- **Popularity / analytics** (**MongoDB** event log + view counters)
- **Explicit skill overlap** (set-intersection signal)

The project includes an offline ETL pipeline to prepare data and embeddings, plus a **Gradio** UI to explore semantic search, graph search, hybrid recommendations, popularity analytics, and optional LLM/RAG explanations.

> **Dataset link:** the dataset source URL is reported in the project report under **Reference [4]**.

---

## Project Structure (typical)

```
.
├─ src/
│  ├─ ui.py                 # Gradio app (main entry)
│  ├─ core.py               # Hybrid ranking, explanation hooks, scoring
│  ├─ features.py           # ETL / embeddings / artifact generation
│  ├─ stores.py             # MongoDB + FAISS wrappers, popularity logging
│  ├─ neo4j_store.py        # Neo4j schema/ingest/query utilities
│  └─ config.py             # Central configuration (paths, weights, flags)
├─ data/                    # Raw + processed artifacts (created by UI pipeline)
├─ logs/
├─ docker-compose.yml       # (if using Docker choice)
└─ README.md
.env
```

---

## Quick Start (Choose ONE option)

You have **two choices** to run the system.

### ✅ Choice A — Run with Docker (recommended)
Use this for reproducibility (recommended for submission).

**1) Start services**
```bash
docker compose up -d --build
```

**2) Open the UI**
```bash
python -m src.ui
```

Open the link shown in the terminal (typically `http://127.0.0.1:7860`).

---

### ✅ Choice B — Run Locally (no Docker)
Use this if you already have **Neo4j** and **MongoDB** running on your machine.

**1) Start Neo4j and MongoDB**
- Neo4j Browser: `http://localhost:7474`
- Neo4j Bolt: `bolt://localhost:7687`
- MongoDB: `mongodb://localhost:27017`

**2) Configure connection settings**
Edit `src/config.py` (or `.env`, if used) to match your local URIs and credentials.

**3) Run the UI**
```bash
python -m src.ui
```

---

## Generate Raw and Processed Data (from the UI)

All artifacts can be created directly from the interface.

### Step 1 — Generate **Raw Data**
1. Open the UI in your browser.
2. Go to **“Pipeline & Health”** tab.
3. Click **“⚙️ Run ETL”**.

**What it produces**
- A cleaned *raw* jobs table and a derived *raw* users table (synthetic users are built from job skills/locations if the dataset has no user profiles).
- Files are written under the project’s `data/raw/` directory (jobs and users tables).

**Expected UI confirmation**
- “✅ ETL Completed”
- Counts for jobs/users
- Empty-skill ratio statistics

---

### Step 2 — Generate **Processed Data**
1. In the same **“Pipeline & Health”** tab, click **“🧮 Compute Embeddings”**.

**What it produces**
- Vector embeddings for jobs and users (Sentence-Transformers)
- Processed artifacts under `data/processed/` (embedding arrays and ID mappings)
- A FAISS-ready representation used by semantic retrieval

**Expected UI confirmation**
- “✅ Embeddings completed / saved” (or similar success message)

---

### Step 3 — Populate Databases (optional but recommended)
After raw + processed artifacts exist:

- Click **“🍃 Sync Mongo”**  
  Populates MongoDB structures used for event logging and popularity analytics.

- Click **“🔷 Sync Neo4j”**  
  Builds/updates the skill graph (User/Job/Skill nodes + weighted co-occurrence edges).

> If Neo4j/MongoDB are unavailable, the system is designed to degrade gracefully (e.g., graph score falls back to zero).

---

## Using the UI (tabs)
- **Hybrid Recommendations**: final ranking with adjustable weights (semantic/graph/popularity/overlap)
- **Semantic Search**: FAISS-based k-NN search over embeddings
- **Skill Graph Analysis (Neo4j)**: graph evidence (shared skills + co-occurrence support)
- **Popularity Analytics**: event logs and top-viewed jobs
- **AI Explanations**: global summaries and per-result rationale; optional LLM/RAG mode with fallback templates
- **Pipeline & Health**: generate artifacts and validate component availability
- **Data Browser**: inspect current jobs/users tables

---

## Configuration Notes
Key knobs usually live in `src/config.py`:
- Dataset limits (e.g., max jobs, number of users)
- Embedding model name
- Hybrid weights `(alpha, beta, gamma, delta)`
- Oversampling factor for candidate generation
- Flags to enable/disable Neo4j and LLM/RAG explanations

---

## Troubleshooting (common)

**Docker shows containers healthy but UI is empty**
- Run **“⚙️ Run ETL”** then **“🧮 Compute Embeddings”** from the UI.

**Neo4j Browser access**
- Open `http://localhost:7474` and log in using the configured credentials.

**Mongo replica-set init container is “Exited”**
- This is expected; it runs once to initialize the replica set.

**LLM/RAG not working**
- Disable the advanced mode or set the required API key/model in configuration.
- The system falls back to deterministic template explanations when the LLM is unavailable.

---

## Course Positioning
This system illustrates **polyglot persistence**: using specialized datastores for distinct access patterns (vector similarity search, graph traversal, and event/log analytics), consistent with Big Data architecture and NoSQL design principles.
