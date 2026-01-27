from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL_NAME,
    HYBRID_ALPHA,
    HYBRID_BETA,
    HYBRID_DELTA,
    HYBRID_GAMMA,
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_RETRIES,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OVERSAMPLE,
)
from src.features import build_user_text, ensure_ready, load_jobs_df, load_users_df, parse_skills
from src.stores import (
    rag_knn,
    JobVectorIndex,
    analytics_popularity,  # ✅ used to get top popular jobs efficiently
    compute_graph_scores,
    ensure_backends_ready,
    get_job_popularity,
    increment_job_popularity,
    log_event,
)


class LLMError(RuntimeError):
    pass


def call_llm(messages: List[Dict[str, str]]) -> str:
    if not LLM_API_KEY:
        raise LLMError("LLM_API_KEY/GROQ_API_KEY is not set")
    url = f"{LLM_API_BASE.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": float(LLM_TEMPERATURE),
    }
    last_err: Optional[Exception] = None
    for _ in range(int(LLM_RETRIES)):
        try:
            with httpx.Client(timeout=float(LLM_TIMEOUT)) as client:
                r = client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            time.sleep(0.25)
    raise LLMError(str(last_err))


def _l2(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = float(np.linalg.norm(v) + 1e-12)
    return v / n


def _minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx <= mn + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def _unique_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@dataclass
class Rec:
    job_id: str
    title: str
    company_name: str
    location: str
    embedding_score: float
    graph_score: float
    popularity_views: int
    overlap_count: int
    overlap_skills: str
    hybrid_score: float


class HybridRecommender:
    def __init__(self):
        ensure_ready()
        self.jobs = load_jobs_df()
        self.users = load_users_df()

        # ✅ Build stable indices once (avoid repeated copy/set_index in every call)
        self.jobs = self.jobs.copy()
        self.jobs["job_id"] = self.jobs["job_id"].astype(str)
        self.jobs_idx = self.jobs.set_index("job_id", drop=False)

        self.users = self.users.copy()
        self.users["user_id"] = self.users["user_id"].astype(str)

        # ✅ Pre-parse job skills once (overlap retrieval becomes fast)
        self._job_skills: Dict[str, set[str]] = {}
        for jid, r in self.jobs_idx.iterrows():
            self._job_skills[str(jid)] = set(parse_skills(str(r.get("skills", ""))))

        ensure_backends_ready()
        self.index = JobVectorIndex()

        # default weights (can be overridden per-call from UI sliders)
        # ✅ Keep your mapping: alpha=dense, beta=graph, gamma=popularity, delta=overlap
        self.alpha = float(HYBRID_ALPHA)  # embedding
        self.beta = float(HYBRID_BETA)    # graph
        self.gamma = float(HYBRID_GAMMA)  # popularity
        self.delta = float(HYBRID_DELTA)  # skill overlap

    @staticmethod
    def _normalized_weights(
        base_a: float,
        base_b: float,
        base_g: float,
        base_d: float,
        alpha: float | None,
        beta: float | None,
        gamma: float | None,
        delta: float | None,
    ) -> Tuple[float, float, float, float]:
        a = float(base_a if alpha is None else alpha)
        b = float(base_b if beta is None else beta)
        g = float(base_g if gamma is None else gamma)
        d = float(base_d if delta is None else delta)

        s = a + b + g + d
        if not np.isfinite(s) or s <= 0.0:
            # safe fallback: pure embedding signal
            return 1.0, 0.0, 0.0, 0.0

        return a / s, b / s, g / s, d / s

    def user_embedding(self, user_id: str) -> np.ndarray:
        uid = str(user_id)
        urow = self.users[self.users["user_id"].astype(str) == uid]
        if urow.empty:
            raise ValueError(f"Unknown user_id: {uid}")

        # build_user_text expects a DataFrame and returns a Series
        text = str(build_user_text(urow).iloc[0])

        vec = _model().encode([text], normalize_embeddings=False)[0]
        return _l2(np.asarray(vec, dtype=np.float32))

    def semantic_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        qv = _model().encode([q], normalize_embeddings=False)[0]
        qv = _l2(np.asarray(qv, dtype=np.float32))
        hits = self.index.search(qv, int(top_k))

        out: List[Dict[str, Any]] = []
        for jid, score in hits:
            jid = str(jid)
            if jid not in self.jobs_idx.index:
                continue
            r = self.jobs_idx.loc[jid]
            out.append(
                {
                    "job_id": jid,
                    "score": float(score),
                    "title": r.get("title", ""),
                    "company_name": r.get("company_name", ""),
                    "location": r.get("location", ""),
                    "skills": r.get("skills", ""),
                }
            )
        return out

    def _top_overlap_ids(self, u_sk: set[str], k: int) -> List[str]:
        if k <= 0 or not u_sk:
            return []
        scored: List[Tuple[str, int]] = []
        for jid, j_sk in self._job_skills.items():
            c = len(u_sk.intersection(j_sk))
            if c > 0:
                scored.append((jid, c))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [jid for jid, _ in scored[:k]]

    def _top_popular_ids(self, k: int) -> List[str]:
        if k <= 0:
            return []
        try:
            top = analytics_popularity(limit=int(k)).get("top_jobs", []) or []
            # each element is a doc like {"job_id": "...", "views": ...}
            out = [str(d.get("job_id", "")).strip() for d in top]
            return [j for j in out if j]
        except Exception:
            return []

    def _top_graph_ids(self, user_id: str, k: int) -> List[str]:
        if k <= 0:
            return []
        try:
            # local import to avoid import-time failures
            from src.neo4j_store import get_neo4j_graph

            g = get_neo4j_graph()
            if not g:
                return []
            hits = g.graph_search_jobs(
                user_id=str(user_id),
                top_k=int(k),
                neighbor_limit=200,
                min_w=1,
            ) or []
            out = [str(h.get("job_id", "")).strip() for h in hits]
            return [j for j in out if j]
        except Exception:
            return []

    def recommend(
        self,
        user_id: str,
        top_k: int = 20,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        delta: float | None = None,
    ) -> List[Dict[str, Any]]:
        uid = str(user_id)
        urow = self.users[self.users["user_id"].astype(str) == uid]
        if urow.empty:
            raise ValueError(f"Unknown user_id: {uid}")

        # normalize weights using UI overrides if provided
        # ✅ keep mapping: alpha=dense, beta=graph, gamma=popularity, delta=overlap
        a, b, g, d = self._normalized_weights(
            self.alpha, self.beta, self.gamma, self.delta,
            alpha, beta, gamma, delta
        )

        u_sk = set(parse_skills(str(urow.iloc[0].get("skills", ""))))
        u_vec = self.user_embedding(uid)

        base = max(int(top_k), 1)
        k_dense = max(base * int(OVERSAMPLE), base)
        k_graph = max(base * int(OVERSAMPLE), base)
        k_overlap = max(base * 10, base)
        k_pop = max(base * 10, base)

        # 1) Dense candidates
        dense_hits = self.index.search(u_vec, int(k_dense)) or []
        dense_map = {str(jid): float(s) for jid, s in dense_hits}
        dense_ids = [str(jid) for jid, _ in dense_hits]

        # 2) Graph candidates (independent)
        graph_ids = self._top_graph_ids(uid, int(k_graph))

        # 3) Overlap candidates (independent)
        overlap_ids = self._top_overlap_ids(u_sk, int(k_overlap))

        # 4) Popularity candidates (independent)
        pop_ids = self._top_popular_ids(int(k_pop))

        # Union candidates (preserve order preference: dense → graph → overlap → pop)
        cand_ids = _unique_preserve_order(dense_ids + graph_ids + overlap_ids + pop_ids)
        if not cand_ids:
            return []

        # Compute arrays over union
        dense_arr = np.array([dense_map.get(jid, 0.0) for jid in cand_ids], dtype=np.float32)

        graph_scores = compute_graph_scores(uid, job_ids=cand_ids) or {}
        graph_arr = np.array([float(graph_scores.get(jid, 0.0)) for jid in cand_ids], dtype=np.float32)

        # Popularity per candidate (this is OK now because union pool is limited)
        pop_arr = np.array([float(get_job_popularity(jid)) for jid in cand_ids], dtype=np.float32)

        overlap_counts: List[int] = []
        overlap_skills: List[str] = []
        for jid in cand_ids:
            inter = sorted(list(u_sk.intersection(self._job_skills.get(jid, set()))))
            overlap_counts.append(len(inter))
            overlap_skills.append(", ".join(inter))

        overlap_arr = np.array(overlap_counts, dtype=np.float32)

        # Normalize signals within candidate pool
        dense_n = _minmax(dense_arr)
        graph_n = _minmax(graph_arr)
        pop_n = _minmax(np.log1p(pop_arr))
        overlap_n = _minmax(overlap_arr)

        recs: List[Rec] = []
        for i, jid in enumerate(cand_ids):
            if jid not in self.jobs_idx.index:
                continue
            r = self.jobs_idx.loc[jid]
            hybrid = (
                a * float(dense_n[i])
                + b * float(graph_n[i])
                + g * float(pop_n[i])
                + d * float(overlap_n[i])
            )
            recs.append(
                Rec(
                    job_id=jid,
                    title=str(r.get("title", "")),
                    company_name=str(r.get("company_name", "")),
                    location=str(r.get("location", "")),
                    embedding_score=float(dense_n[i]),
                    graph_score=float(graph_n[i]),
                    popularity_views=int(pop_arr[i]),
                    overlap_count=int(overlap_counts[i]),
                    overlap_skills=str(overlap_skills[i]),
                    hybrid_score=float(hybrid),
                )
            )

        recs.sort(key=lambda r: r.hybrid_score, reverse=True)
        picked = recs[: int(top_k)]

        out = [asdict(r) for r in picked]
        log_event(
            "recommend",
            {
                "user_id": uid,
                "top_k": int(top_k),
                "returned": len(out),
                "candidate_pool": len(cand_ids),
                "weights": {"alpha": a, "beta": b, "gamma": g, "delta": d},
            },
        )
        return out

    def recommend_graph_only(
        self,
        user_id: str,
        top_k: int = 20,
        neighbor_limit: int = 200,
        min_w: int = 1,
    ) -> List[Dict[str, Any]]:
        from src.neo4j_store import get_neo4j_graph  # local import to avoid import-time Neo4j failures

        uid = str(user_id)
        g = get_neo4j_graph()
        if not g:
            return []

        hits = g.graph_search_jobs(
            user_id=uid,
            top_k=int(top_k),
            neighbor_limit=int(neighbor_limit),
            min_w=int(min_w),
        )
        if not hits:
            return []

        out: List[Dict[str, Any]] = []
        for h in hits:
            jid = str(h.get("job_id", ""))
            if not jid or jid not in self.jobs_idx.index:
                continue
            r = self.jobs_idx.loc[jid]
            out.append(
                {
                    "job_id": jid,
                    "title": r.get("title", ""),
                    "company_name": r.get("company_name", ""),
                    "location": r.get("location", ""),
                    "skills": r.get("skills", ""),
                    "description": r.get("description", ""),
                    "graph_score": float(h.get("graph_score", 0.0)),
                    "graph_raw": float(h.get("raw", 0.0)),
                    "shared_skills": int(h.get("shared", 0)),
                    "co_occurrence_sum": float(h.get("co", 0.0)),
                }
            )
        return out


def explain_global(user_id: str, recs: List[Dict[str, Any]]) -> str:
    if not recs:
        return "No recommendations to explain."
    lines = []
    for r in recs:
        lines.append(
            f"- {r.get('title','')} | {r.get('company_name','')} | {r.get('location','')} "
            f"(hybrid={r.get('hybrid_score',0):.3f}, embed={r.get('embedding_score',0):.3f}, "
            f"graph={r.get('graph_score',0):.3f}, views={r.get('popularity_views',0)}, "
            f"overlap={r.get('overlap_skills','')})"
        )
    prompt = (
        "Explain why these jobs were recommended for the user based on the ranking signals. "
        "Be concise, bullet-pointed, and actionable.\n\nRanked jobs (evidence):\n"
        + "\n".join(lines)
    )
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that explains job recommendations based on evidence.",
        },
        {"role": "user", "content": prompt},
    ]
    return call_llm(messages)


def _format_sources(chunks: List[Dict[str, Any]]) -> str:
    out = []
    for i, c in enumerate(chunks, start=1):
        txt = str(c.get("text", "")).strip()
        if not txt:
            continue
        out.append(f"[C{i}] {txt}")
    return "\n\n".join(out)


def _missing_citations(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    return ("[C1]" not in t) and ("[C2]" not in t) and ("[C3]" not in t)


def explain_rag(user_id: str, recs: List[Dict[str, Any]], jobs_df) -> List[str]:
    if not recs:
        return []
    uid = str(user_id)
    u_text = f"User({uid})"

    outs: List[str] = []
    for r in recs:
        jid = str(r.get("job_id", ""))
        title = str(r.get("title", ""))
        skills = str(r.get("skills", ""))
        desc = str(r.get("description", ""))

        evidence = (
            f"User: {u_text}\n"
            f"Job: {jid} | {title}\n"
            f"Skills: {skills}\n"
            f"Signals: hybrid={r.get('hybrid_score',0):.3f}, embed={r.get('embedding_score',0):.3f}, "
            f"graph={r.get('graph_score',0):.3f}, views={r.get('popularity_views',0)}, "
            f"overlap={r.get('overlap_skills','')}\n"
        )

        query = f"{u_text} {title} skills {skills}\n{desc[:800]}"
        qv = _model().encode([query], normalize_embeddings=False)[0]
        qv = _l2(np.asarray(qv, dtype=np.float32))

        # get more, then filter to this job_id
        all_chunks = rag_knn(qv, k=30) or []
        chunks = [c for c in all_chunks if str(c.get("job_id", "")) == jid][:4]

        # fallback: if nothing matches that job_id, just use top chunks
        if not chunks:
            chunks = all_chunks[:4]

        if chunks:
            sources = _format_sources(chunks)
            evidence = evidence + "\nSources:\n" + sources

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are doing grounded generation. "
                        "Every sentence MUST cite one of the provided Sources with [C#]. "
                        "Do not invent facts. Keep it concise."
                    ),
                },
                {"role": "user", "content": f"Evidence:\n{evidence}\n\nExplain the match and ranking."},
            ]
            ans = call_llm(messages)

            if _missing_citations(ans):
                messages2 = [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the answer so that every sentence contains at least one citation [C#]. "
                            "Only use the provided Sources."
                        ),
                    },
                    {"role": "user", "content": f"Sources:\n{sources}\n\nDraft:\n{ans}\n\nRewrite with citations."},
                ]
                ans = call_llm(messages2)

            outs.append(ans)
        else:
            evidence = evidence + f"\nDescription:\n{desc[:1200]}"
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are doing grounded generation. "
                        "Ground the answer only in the Evidence block. "
                        "If you don't see evidence for a claim, say so. "
                        "Return: 2–3 sentences + 2 bullets (skills to highlight, skills to improve)."
                    ),
                },
                {"role": "user", "content": f"Evidence:\n{evidence}\n\nExplain the match and ranking."},
            ]
            outs.append(call_llm(messages))
    return outs


def record_view(user_id: str, job_id: str) -> int:
    return increment_job_popularity(str(job_id), 1, user_id=str(user_id))
