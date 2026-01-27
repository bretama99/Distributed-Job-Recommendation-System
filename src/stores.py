from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

import numpy as np
from pymongo import MongoClient, ReturnDocument, UpdateOne
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import ReadPreference
from pymongo.write_concern import WriteConcern

import src.config as cfg
from src.config import (
    ENABLE_NEO4J_GRAPH,
    JOB_EMB_NPY,
    JOB_IDS_NPY,
    MONGO_CONNECT_TIMEOUT_MS,
    MONGO_DB,
    MONGO_EVENTS_COLLECTION,
    MONGO_JOBS_COLLECTION,
    MONGO_SERVER_SELECTION_TIMEOUT_MS,
    MONGO_SOCKET_TIMEOUT_MS,
    MONGO_URI,
    MONGO_WRITE_CONCERN,
    REQUIRE_FAISS,
)
from src.features import load_jobs_df, load_users_df, parse_skills
from src.neo4j_store import get_neo4j_graph

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False


_mongo: Optional[MongoClient] = None
_backends_ready: bool = False


def _mongo_client() -> MongoClient:
    global _mongo
    if _mongo is not None:
        return _mongo
    c = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=int(MONGO_SERVER_SELECTION_TIMEOUT_MS),
        connectTimeoutMS=int(MONGO_CONNECT_TIMEOUT_MS),
        socketTimeoutMS=int(MONGO_SOCKET_TIMEOUT_MS),
        retryWrites=True,
    )
    c.admin.command("ping")
    _mongo = c
    return _mongo


def close_mongo() -> None:
    global _mongo
    if _mongo is not None:
        try:
            _mongo.close()
        except Exception:
            pass
    _mongo = None


def _db():
    c = _mongo_client()
    wc = WriteConcern(w=MONGO_WRITE_CONCERN)
    rc = ReadConcern("majority")
    rp = ReadPreference.PRIMARY_PREFERRED
    return c.get_database(MONGO_DB, write_concern=wc, read_concern=rc, read_preference=rp)


def jobs_coll():
    return _db().get_collection(MONGO_JOBS_COLLECTION)


def events_coll():
    return _db().get_collection(MONGO_EVENTS_COLLECTION)


@lru_cache(maxsize=1)
def ensure_indexes() -> None:
    jc = jobs_coll()
    ec = events_coll()
    jc.create_index("job_id", unique=True)
    jc.create_index([("views", -1)])
    jc.create_index([("company_name", 1)])
    jc.create_index([("location", 1)])
    ec.create_index([("type", 1), ("ts", -1)])
    ec.create_index([("payload.user_id", 1), ("ts", -1)])
    ec.create_index([("payload.job_id", 1), ("ts", -1)])


def mongo_status() -> Dict[str, Any]:
    c = _mongo_client()
    jc = c[MONGO_DB][MONGO_JOBS_COLLECTION]
    cnt = int(jc.count_documents({}))
    return {"enabled": True, "connected": True, "error": None, "jobs_count": cnt}


def neo4j_status() -> Dict[str, Any]:
    if not ENABLE_NEO4J_GRAPH:
        return {"enabled": False, "connected": False, "error": None, "stats": {}}

    g = get_neo4j_graph()
    if g is None:
        return {"enabled": True, "connected": False, "error": "neo4j unavailable", "stats": {}}

    try:
        g.ping()
        s = g.stats()
        return {"enabled": True, "connected": True, "error": None, "stats": s}
    except Exception as e:
        return {"enabled": True, "connected": False, "error": str(e), "stats": {}}

def upsert_jobs(df) -> int:
    ensure_indexes()
    jc = jobs_coll()
    if jc is None or df is None or len(df) == 0:
        return 0

    now = int(time.time())
    reset_views = str(os.getenv("RESET_VIEWS_ON_SYNC", "0")).strip().lower() in {"1", "true", "yes", "y", "on"}

    ops: List[UpdateOne] = []
    for _, r in df.iterrows():
        jid = str(r.get("job_id", "")).strip()
        if not jid:
            continue

        base_doc = {
            "job_id": jid,
            "title": str(r.get("title", "")),
            "company_name": str(r.get("company_name", "")),
            "location": str(r.get("location", "")),
            "skills": str(r.get("skills", "")),
            "description": str(r.get("description", "")),
            "updated_at": now,
        }

        update_doc: Dict[str, Any] = {
            "$set": base_doc,
            "$setOnInsert": {"created_at": now, "views": 0, "last_view_at": None},
        }

        if reset_views:
            update_doc["$set"]["views"] = 0
            update_doc["$set"]["last_view_at"] = None

        ops.append(UpdateOne({"job_id": jid}, update_doc, upsert=True))

    if not ops:
        return 0

    jc.bulk_write(ops, ordered=False)
    return len(ops)


def log_event(event_type: str, payload: Dict[str, Any]) -> None:
    ensure_indexes()
    events_coll().insert_one({"type": str(event_type), "ts": int(time.time()), "payload": payload})


def get_job_popularity(job_id: str | int) -> int:
    ensure_indexes()
    jc = jobs_coll()
    jid = str(job_id)
    doc = jc.find_one({"job_id": jid}, {"views": 1})
    if not doc:
        jc.update_one({"job_id": jid}, {"$setOnInsert": {"job_id": jid, "views": 0, "created_at": int(time.time())}}, upsert=True)
        return 0
    return int(doc.get("views") or 0)

def increment_job_popularity(job_id: str, inc: int = 1, user_id: str | None = None) -> int:
    ensure_indexes()
    jc = jobs_coll()
    ec = events_coll()
    if jc is None:
        return 0

    jid = str(job_id).strip()
    if not jid:
        return 0

    now = int(time.time())
    inc_val = int(inc) if int(inc) > 0 else 1

    jc.update_one(
        {"job_id": jid},
        {
            "$inc": {"views": inc_val},
            "$set": {"updated_at": now, "last_view_at": now},
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )

    if ec is not None:
        payload: Dict[str, Any] = {"job_id": jid}
        if user_id is not None:
            payload["user_id"] = str(user_id)

        ec.insert_one({"type": "view", "ts": now, "payload": payload})

    doc = jc.find_one({"job_id": jid}, {"_id": 0, "views": 1})
    return int((doc or {}).get("views", 0))

def analytics_popularity(limit: int = 15) -> Dict[str, Any]:
    ensure_indexes()
    jc = jobs_coll()
    ec = events_coll()
    if jc is None or ec is None:
        return {"top_jobs": [], "recent_events": []}

    lim = int(limit) if int(limit) > 0 else 15

    top_jobs = list(
        jc.find({}, {"_id": 0})
        .sort([("views", -1), ("last_view_at", -1), ("job_id", 1)])
        .limit(lim)
    )

    recent_events = list(
        ec.find({}, {"_id": 0})
        .sort([("ts", -1)])
        .limit(lim)
    )

    for j in top_jobs:
        if "views" not in j or j["views"] is None:
            j["views"] = 0

    return {"top_jobs": top_jobs, "recent_events": recent_events}

def _l2_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


class JobVectorIndex:
    def __init__(self):
        if REQUIRE_FAISS and not _HAS_FAISS:
            raise RuntimeError("FAISS is required but not installed")
        emb = np.load(JOB_EMB_NPY, allow_pickle=True).astype("float32")
        ids = np.load(JOB_IDS_NPY, allow_pickle=True).astype(str)
        if emb.ndim != 2:
            raise ValueError("job embeddings must be 2D")
        self.ids = ids
        self.emb = _l2_rows(emb)
        d = self.emb.shape[1]
        idx = faiss.IndexFlatIP(d)
        idx.add(self.emb)
        self._faiss = idx

    def job_vector(self, job_id: str) -> Optional[np.ndarray]:
        m = self.ids == str(job_id)
        if not m.any():
            return None
        return self.emb[int(np.argmax(m))]

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[str, float]]:
        q = query_vec.reshape(1, -1).astype("float32")
        q = _l2_rows(q)
        D, I = self._faiss.search(q, int(k))
        out: List[Tuple[str, float]] = []
        for s, i in zip(D[0], I[0]):
            if int(i) < 0:
                continue
            out.append((str(self.ids[int(i)]), float(s)))
        return out

def rag_knn(query_vec: np.ndarray, k: int) -> List[Dict[str, Any]]:
    idx = rag_index()
    if idx is None:
        return []
    hits = idx.search(query_vec, int(k))
    if not hits:
        return []
    df = rag_chunks_df()
    out: List[Dict[str, Any]] = []
    for i, s in hits:
        out.append(
            {
                "chunk_id": str(idx.chunk_ids[i]),
                "job_id": str(idx.job_ids[i]),
                "score": float(s),
                "text": str(df.iloc[i]["text"]),
            }
        )
    return out

@lru_cache(maxsize=1)
def jobs_df():
    df = load_jobs_df()
    df["job_id"] = df["job_id"].astype(str)
    return df
@lru_cache(maxsize=1)
def rag_index() -> Optional[RagChunkIndex]:
    try:
        return RagChunkIndex()
    except Exception:
        return None

@lru_cache(maxsize=1)
def users_df():
    df = load_users_df()
    df["user_id"] = df["user_id"].astype(str)
    return df


def sync_neo4j_graph(batch_size: int = 600) -> Dict[str, Any]:
    if not ENABLE_NEO4J_GRAPH:
        return {"ok": False, "error": "neo4j disabled"}

    g = get_neo4j_graph()
    if g is None:
        return {"ok": False, "error": "neo4j unavailable"}

    jdf = jobs_df()
    udf = users_df()

    users_rows: List[Dict[str, Any]] = []
    for _, r in udf.iterrows():
        uid = str(r.get("user_id", "")).strip()
        if not uid:
            continue
        users_rows.append(
            {
                "user_id": uid,
                "name": str(r.get("name", "")),
                "preferred_location": str(r.get("preferred_location", "")),
                "skills": sorted(list(parse_skills(r.get("skills", "")))),
            }
        )

    jobs_rows: List[Dict[str, Any]] = []
    for _, r in jdf.iterrows():
        jid = str(r.get("job_id", "")).strip()
        if not jid:
            continue
        jobs_rows.append(
            {
                "job_id": jid,
                "title": str(r.get("title", "")),
                "company_name": str(r.get("company_name", "")),
                "location": str(r.get("location", "")),
                "skills": sorted(list(parse_skills(r.get("skills", "")))),
            }
        )

    try:
        g.ensure_schema()
        g.upsert_users(users_rows, batch_size=int(batch_size))
        g.upsert_jobs(jobs_rows, batch_size=int(batch_size))
        co = g.rebuild_skill_cooccurrence(jobs_rows)
        return {"ok": True, "users": int(len(users_rows)), "jobs": int(len(jobs_rows)), "co_occurs": co}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def compute_graph_scores(user_id: str, job_ids: Optional[List[str]] = None) -> Dict[str, float]:
    if job_ids is None:
        job_ids = jobs_df()["job_id"].astype(str).tolist()

    if not ENABLE_NEO4J_GRAPH:
        return {str(j): 0.0 for j in job_ids}

    g = get_neo4j_graph()
    if g is None:
        return {str(j): 0.0 for j in job_ids}

    try:
        return g.graph_scores(str(user_id), [str(x) for x in job_ids])
    except Exception:
        return {str(j): 0.0 for j in job_ids}


def ensure_backends_ready() -> Dict[str, Any]:
    global _backends_ready
    if _backends_ready:
        return {"ok": True}

    # Ensure Mongo has job docs for popularity analytics
    jdf = jobs_df()
    n_mongo = upsert_jobs(jdf)

    # Neo4j graph is optional based on config
    neo = sync_neo4j_graph() if ENABLE_NEO4J_GRAPH else {"ok": True, "skipped": True}

    _backends_ready = True
    return {"ok": True, "mongo_upserts": int(n_mongo), "neo4j": neo}
