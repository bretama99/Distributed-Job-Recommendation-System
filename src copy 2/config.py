from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _b(x: Optional[str], default: bool) -> bool:
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _i(x: Optional[str], default: int) -> int:
    try:
        return int(x) if x is not None else default
    except Exception:
        return default


def _f(x: Optional[str], default: float) -> float:
    try:
        return float(x) if x is not None else default
    except Exception:
        return default


DEBUG = _b(os.getenv("DEBUG"), False)
STRICT_CONFIG = _b(os.getenv("STRICT_CONFIG"), True)

REQUIRE_MONGO = _b(os.getenv("REQUIRE_MONGO"), True)
REQUIRE_NEO4J = _b(os.getenv("REQUIRE_NEO4J"), True)
REQUIRE_FAISS = _b(os.getenv("REQUIRE_FAISS"), True)

DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"
for p in (DATA_DIR, DATA_RAW, DATA_PROCESSED, DATA_EXTERNAL):
    p.mkdir(parents=True, exist_ok=True)

LINKEDIN_DIR = Path(os.getenv("LINKEDIN_DIR", str(DATA_EXTERNAL / "linkedin")))

def _pick_existing(*candidates: Path) -> Path:
    for p in candidates:
        try:
            if p and p.exists():
                return p
        except Exception:
            pass
    for p in candidates:
        if p is not None:
            return p
    return DATA_EXTERNAL

LINKEDIN_POSTS = _pick_existing(
    Path(os.getenv("LINKEDIN_POSTS")) if os.getenv("LINKEDIN_POSTS") else None,
    LINKEDIN_DIR / "linkedin_job_postings.csv",
    DATA_EXTERNAL / "linkedin_job_postings.csv",
)

LINKEDIN_SUMMARY = _pick_existing(
    Path(os.getenv("LINKEDIN_SUMMARY")) if os.getenv("LINKEDIN_SUMMARY") else None,
    LINKEDIN_DIR / "job_summary.csv",
    DATA_EXTERNAL / "job_summary.csv",
)

LINKEDIN_SKILLS = _pick_existing(
    Path(os.getenv("LINKEDIN_SKILLS")) if os.getenv("LINKEDIN_SKILLS") else None,
    LINKEDIN_DIR / "job_skills.csv",
    DATA_EXTERNAL / "job_skills.csv",
)

JOBS_CSV = Path(os.getenv("JOBS_CSV", str(DATA_RAW / "jobs.csv")))
USERS_CSV = Path(os.getenv("USERS_CSV", str(DATA_RAW / "users.csv")))

PIPELINE_META_JSON = Path(os.getenv("PIPELINE_META_JSON", str(DATA_PROCESSED / "pipeline_meta.json")))

JOB_EMB_NPY = Path(os.getenv("JOB_EMB_NPY", str(DATA_PROCESSED / "job_embeddings.npy")))
JOB_IDS_NPY = Path(os.getenv("JOB_IDS_NPY", str(DATA_PROCESSED / "job_ids.npy")))
USER_EMB_NPY = Path(os.getenv("USER_EMB_NPY", str(DATA_PROCESSED / "user_embeddings.npy")))
USER_IDS_NPY = Path(os.getenv("USER_IDS_NPY", str(DATA_PROCESSED / "user_ids.npy")))

JOB_FAISS_INDEX = Path(os.getenv("JOB_FAISS_INDEX", str(DATA_PROCESSED / "job_faiss.index")))

APP_TITLE = os.getenv("APP_TITLE", "Distributed Job Recommendation System")
APP_INSTANCE = os.getenv("APP_INSTANCE", "app1")

TOP_K_DEFAULT = _i(os.getenv("TOP_K_DEFAULT"), 15)
MAX_JOBS = _i(os.getenv("MAX_JOBS"), 80000)
N_USERS = _i(os.getenv("N_USERS"), 20000)
OVERSAMPLE = _i(os.getenv("OVERSAMPLE"), 6)

HYBRID_ALPHA = _f(os.getenv("HYBRID_ALPHA"), 0.62)
HYBRID_BETA = _f(os.getenv("HYBRID_BETA"), 0.23)
HYBRID_GAMMA = _f(os.getenv("HYBRID_GAMMA"), 0.06)
HYBRID_DELTA = _f(os.getenv("HYBRID_DELTA"), 0.09)

MMR_LAMBDA = _f(os.getenv("MMR_LAMBDA"), 0.82)
MMR_MAX_CANDIDATES = _i(os.getenv("MMR_MAX_CANDIDATES"), 120)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = _f(os.getenv("LLM_TEMPERATURE"), 0.2)
LLM_MAX_TOKENS = _i(os.getenv("LLM_MAX_TOKENS"), 700)
LLM_TIMEOUT = _i(os.getenv("LLM_TIMEOUT"), 30)
LLM_RETRIES = _i(os.getenv("LLM_RETRIES"), 2)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "jobrec")
MONGO_JOBS_COLLECTION = os.getenv("MONGO_JOBS_COLLECTION", "jobs")
MONGO_EVENTS_COLLECTION = os.getenv("MONGO_EVENTS_COLLECTION", "events")
MONGO_CONNECT_TIMEOUT_MS = _i(os.getenv("MONGO_CONNECT_TIMEOUT_MS"), 3000)
MONGO_SOCKET_TIMEOUT_MS = _i(os.getenv("MONGO_SOCKET_TIMEOUT_MS"), 8000)
MONGO_SERVER_SELECTION_TIMEOUT_MS = _i(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS"), 3000)
MONGO_WRITE_CONCERN = os.getenv("MONGO_WRITE_CONCERN", "majority")
MONGO_READ_CONCERN = os.getenv("MONGO_READ_CONCERN", "majority")
MONGO_READ_PREFERENCE = os.getenv("MONGO_READ_PREFERENCE", "primaryPreferred")

ENABLE_NEO4J_GRAPH = _b(os.getenv("ENABLE_NEO4J_GRAPH"), True)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
NEO4J_MAX_CONNECTION_POOL_SIZE = _i(os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE"), 20)
NEO4J_CONNECTION_TIMEOUT = _i(os.getenv("NEO4J_CONNECTION_TIMEOUT"), 30)

ENABLE_REDIS_CACHE = _b(os.getenv("ENABLE_REDIS_CACHE"), False)
ENABLE_CASSANDRA_EVENTS = _b(os.getenv("ENABLE_CASSANDRA_EVENTS"), False)
ENABLE_KAFKA_STREAMING = _b(os.getenv("ENABLE_KAFKA_STREAMING"), False)
ELASTICSEARCH_ENABLED = _b(os.getenv("ELASTICSEARCH_ENABLED"), False)
ENABLE_ADVANCED_RAG = _b(os.getenv("ENABLE_ADVANCED_RAG"), False)

PROMETHEUS_ENABLED = _b(os.getenv("PROMETHEUS_ENABLED"), False)
PROMETHEUS_PORT = _i(os.getenv("PROMETHEUS_PORT"), 8000)

VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss")
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chromadb")))

LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("GROQ_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)
