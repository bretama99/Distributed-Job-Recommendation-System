# Distributed Job Recommendation System

Hybrid, explainable job recommender with:

- Spark-based ETL
- SentenceTransformer embeddings + FAISS vector search
- Skill graph (NetworkX)
- NoSQL storage (Mongo + Redis)
- FastAPI backend
- Gradio demo UI
- LLM-based explanations (Groq/OpenAI-compatible)

## Architecture

1. **ETL (Spark)**  
   - Ingest raw job data (`data/external/*`).  
   - Clean and normalize into `data/raw/jobs/*.csv` and `data/raw/users/*.csv`.

2. **Embeddings**  
   - Build text for jobs/users.  
   - Compute SentenceTransformer embeddings and save to `data/processed/*.npy`.

3. **Stores**  
   - FAISS job vector index.  
   - NetworkX skill graph (user–skill–job).  
   - MongoDB for job documents.  
   - Redis for popularity counters and recommendation cache.

4. **Recommender**  
   - Hybrid score = alpha * embedding + beta * graph + gamma * popularity.  
   - Business rules: location, remote-only, company caps.

5. **LLM Explanations (RAG)**  
   - LLM client uses Groq/OpenAI-compatible API.  
   - Context built from user profile, job docs, and scores.  
   - Generates explanations per job.

6. **Serving**  
   - FastAPI for REST endpoints.  
   - Gradio UI for demos (local exploration).  
   - Docker and docker-compose for full stack.

## Quickstart

1. Create and activate virtualenv:

for the dataset we have to create data/raw/and here the datasets

python -m venv venv
source venv/bin/activate  # or venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
PS A:\Courses\bigdata\job-rec-system> py -3.11 -m venv .venv
PS A:\Courses\bigdata\job-rec-system> .\.venv\Scripts\Activate.ps1
(.venv) PS A:\Courses\bigdata\job-rec-system> python --version
# should show Python 3.11.x

(.venv) PS A:\Courses\bigdata\job-rec-system> python -m pip install --upgrade pip wheel setuptools
(.venv) PS A:\Courses\bigdata\job-rec-system> pip install -r requirements.txt

python -m src.etl.spark_jobs
python -m src.features.embeddings
python -m src.ui.gradio_app
=======
# Distributed-Job-Recommendation-System
