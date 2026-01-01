# src/config.py
from __future__ import annotations

from pathlib import Path
import os

from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Load environment variables from .env (project root)
# ---------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"

# Ensure base dirs exist (safe for local development)
for _p in (DATA_DIR, DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL):
    _p.mkdir(parents=True, exist_ok=True)

# CSVs produced by ETL
JOBS_CSV = DATA_RAW / "jobs.csv"
USERS_CSV = DATA_RAW / "users.csv"

# Numpy files produced by embeddings pipeline
JOB_EMB_NPY = DATA_PROCESSED / "job_embeddings.npy"
JOB_IDS_NPY = DATA_PROCESSED / "job_ids.npy"
USER_EMB_NPY = DATA_PROCESSED / "user_embeddings.npy"
USER_IDS_NPY = DATA_PROCESSED / "user_ids.npy"

TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "15"))

# ---------------------------------------------------------------------
# Hybrid recommender weights
# ---------------------------------------------------------------------
ALPHA = float(os.getenv("HYBRID_ALPHA", "0.7"))   # embeddings
BETA = float(os.getenv("HYBRID_BETA", "0.25"))    # skill graph
GAMMA = float(os.getenv("HYBRID_GAMMA", "0.05"))  # popularity

# ---------------------------------------------------------------------
# Embeddings / models
# ---------------------------------------------------------------------
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    os.getenv("MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
)

# ---------------------------------------------------------------------
# MongoDB (NoSQL store)
# ---------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "jobrec")
MONGO_JOBS_COLLECTION = os.getenv("MONGO_JOBS_COLLECTION", "jobs")

# ---------------------------------------------------------------------
# Redis (cache / popularity) – kept for future extensions if you want
# ---------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB_CACHE = int(os.getenv("REDIS_DB_CACHE", "0"))
REDIS_DB_POPULARITY = int(os.getenv("REDIS_DB_POPULARITY", "1"))

# ---------------------------------------------------------------------
# LLM / Groq (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.groq.com/openai/v1")

# Prefer explicit LLM_API_KEY, otherwise fall back to GROQ_API_KEY
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")

# Model name, temperature, token limit, timeout
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
