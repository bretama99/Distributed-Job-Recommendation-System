from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

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
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_RETRIES,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MMR_LAMBDA,
    MMR_MAX_CANDIDATES,
    OVERSAMPLE,
)
from src.features import build_user_text, ensure_ready, load_jobs_df, load_users_df, parse_skills
from src.stores import (
    rag_knn,
    JobVectorIndex,
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
        "temperature": float(LLM_TEMPERATURE),
        "max_tokens": int(LLM_MAX_TOKENS),
        "messages": messages,
    }
    last: Optional[Exception] = None
    for a in range(max(1, int(LLM_RETRIES) + 1)):
        try:
            r = httpx.post(url, headers=headers, json=payload, timeout=float(LLM_TIMEOUT))
            r.raise_for_status()
            j = r.json()
            return j["choices"][0]["message"]["content"]
        except Exception as exc:
            last = exc
            time.sleep(0.25 * (a + 1))
    raise LLMError(f"LLM request failed: {last}")


def _l2(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return x / max(n, 1e-12)


def _minmax(vals: np.ndarray) -> np.ndarray:
    if vals.size == 0:
        return vals
    mn = float(vals.min())
    mx = float(vals.max())
    if mx <= mn:
        return np.zeros_like(vals, dtype="float32")
    return ((vals - mn) / (mx - mn)).astype("float32")


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@dataclass(frozen=True)
class Rec:
    user_id: str
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


def _mmr_select(cands: List[Rec], index: JobVectorIndex, k: int, lam: float) -> List[Rec]:
    if not cands:
        return []
    chosen: List[Rec] = []
    pool = cands[:]
    vec_cache: Dict[str, Optional[np.ndarray]] = {}

    def vec(jid: str) -> Optional[np.ndarray]:
        if jid in vec_cache:
            return vec_cache[jid]
        v = index.job_vector(jid)
        vec_cache[jid] = v
        return v

    while pool and len(chosen) < k:
        if not chosen:
            chosen.append(pool.pop(0))
            continue
        best_i = 0
        best = -1e18
        for i, r in enumerate(pool):
            rel = float(r.hybrid_score)
            rv = vec(r.job_id)
            if rv is None:
                score = rel
            else:
                max_sim = 0.0
                for c in chosen:
                    cv = vec(c.job_id)
                    if cv is None:
                        continue
                    sim = float(np.dot(rv, cv))
                    if sim > max_sim:
                        max_sim = sim
                score = lam * rel - (1.0 - lam) * max_sim
            if score > best:
                best = score
                best_i = i
        chosen.append(pool.pop(best_i))
    return chosen


class HybridRecommender:
    def __init__(
        self,
        alpha: float = HYBRID_ALPHA,
        beta: float = HYBRID_BETA,
        gamma: float = HYBRID_GAMMA,
        delta: float = HYBRID_DELTA,
        mmr_lambda: float = MMR_LAMBDA,
    ):
        ensure_ready()
        self.jobs = load_jobs_df()
        self.users = load_users_df()
        ensure_backends_ready()
        self.index = JobVectorIndex()
        self.model = _model()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.mmr_lambda = float(mmr_lambda)

    def user_embedding(self, user_id: str) -> np.ndarray:
        row = self.users[self.users["user_id"].astype(str) == str(user_id)]
        if row.empty:
            raise ValueError(f"Unknown user_id: {user_id}")
        text = build_user_text(row).iloc[0]
        v = self.model.encode([text], convert_to_numpy=True)[0].astype("float32")
        return _l2(v)

    def semantic_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if not str(query).strip():
            return []
        v = self.model.encode([str(query)], convert_to_numpy=True)[0].astype("float32")
        v = _l2(v)
        hits = self.index.search(v, int(top_k))
        out = []
        for jid, score in hits:
            row = self.jobs[self.jobs["job_id"].astype(str) == str(jid)]
            if row.empty:
                continue
            r = row.iloc[0]
            out.append(
                {
                    "job_id": str(jid),
                    "score": float(score),
                    "title": r.get("title", ""),
                    "company_name": r.get("company_name", ""),
                    "location": r.get("location", ""),
                    "skills": r.get("skills", ""),
                }
            )
        return out

    def recommend(self, user_id: str, top_k: int = 20, diversify: bool = True) -> List[Dict[str, Any]]:
        uid = str(user_id)
        urow = self.users[self.users["user_id"].astype(str) == uid]
        if urow.empty:
            raise ValueError(f"Unknown user_id: {uid}")

        u_sk = parse_skills(urow.iloc[0].get("skills", ""))
        u_vec = self.user_embedding(uid)

        cand_k = max(int(top_k) * int(OVERSAMPLE), int(top_k))
        dense_hits = self.index.search(u_vec, cand_k)
        cand_ids = [str(jid) for jid, _ in dense_hits]

        graph_scores = compute_graph_scores(uid, job_ids=cand_ids)

        dense_arr = np.array([s for _, s in dense_hits], dtype="float32")
        graph_arr = np.array([float(graph_scores.get(str(jid), 0.0)) for jid, _ in dense_hits], dtype="float32")
        pop_arr = np.array([float(get_job_popularity(str(jid))) for jid, _ in dense_hits], dtype="float32")

        dense_n = _minmax(dense_arr)
        graph_n = _minmax(graph_arr)
        pop_n = _minmax(np.log1p(pop_arr))

        overlap_counts: List[float] = []
        job_rows: Dict[str, Any] = {}

        for jid in cand_ids:
            row = self.jobs[self.jobs["job_id"].astype(str) == jid]
            if row.empty:
                job_rows[jid] = None
                overlap_counts.append(0.0)
                continue
            jr = row.iloc[0]
            job_rows[jid] = jr
            j_sk = parse_skills(jr.get("skills", ""))
            overlap_counts.append(float(len(u_sk.intersection(j_sk))))

        overlap_arr = np.array(overlap_counts, dtype="float32")
        overlap_n = _minmax(overlap_arr)

        recs: List[Rec] = []
        for i, (jid, dense_raw) in enumerate(dense_hits):
            jid = str(jid)
            jr = job_rows.get(jid)
            if jr is None:
                continue

            j_sk = parse_skills(jr.get("skills", ""))
            ov = u_sk.intersection(j_sk)
            ov_count = int(overlap_arr[i])
            ov_str = ", ".join(sorted(ov)) if ov else "—"

            hybrid = (
                self.alpha * float(dense_n[i])
                + self.beta * float(graph_n[i])
                + self.gamma * float(pop_n[i])
                + self.delta * float(overlap_n[i])
            )

            recs.append(
                Rec(
                    user_id=uid,
                    job_id=jid,
                    title=str(jr.get("title", "")),
                    company_name=str(jr.get("company_name", "")),
                    location=str(jr.get("location", "")),
                    embedding_score=float(dense_raw),
                    graph_score=float(graph_scores.get(jid, 0.0)),
                    popularity_views=int(pop_arr[i]),
                    overlap_count=ov_count,
                    overlap_skills=ov_str,
                    hybrid_score=float(hybrid),
                )
            )

        recs.sort(key=lambda r: r.hybrid_score, reverse=True)
        trimmed = recs[: max(int(top_k) * 6, int(top_k))]

        if diversify:
            picked = _mmr_select(trimmed[: int(MMR_MAX_CANDIDATES)], self.index, int(top_k), float(self.mmr_lambda))
        else:
            picked = trimmed[: int(top_k)]

        out = [asdict(r) for r in picked]
        log_event("recommend", {"user_id": uid, "top_k": int(top_k), "returned": len(out)})
        return out


def explain_global(user_id: str, recs: List[Dict[str, Any]]) -> str:
    if not recs:
        return "No recommendations to explain."
    user_id = str(user_id)
    lines = []
    for i, r in enumerate(recs, start=1):
        lines.append(
            f"{i}. {r.get('title','')} | {r.get('company_name','')} | {r.get('location','')} | "
            f"hybrid={float(r.get('hybrid_score',0.0)):.3f} "
            f"embed={float(r.get('embedding_score',0.0)):.3f} "
            f"graph={float(r.get('graph_score',0.0)):.3f} "
            f"views={int(r.get('popularity_views',0))} "
            f"overlap={r.get('overlap_skills','—')}"
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert AI/Big-Data engineer presenting a hybrid recommender. "
                "Explain rankings using only the numeric signals and retrieved fields. "
                "Be structured, concise, and technical."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User_id: {user_id}\n\n"
                f"Ranked jobs (evidence):\n" + "\n".join(lines) + "\n\n"
                "Output format:\n"
                "A) 3 bullets: why the list matches the user overall\n"
                "B) 1–2 sentences per job: mention which signals drive the rank\n"
                "C) 3 bullets: architecture (ETL → embeddings → vector search → Neo4j graph → Mongo → hybrid + MMR)\n"
            ),
        },
    ]
    return call_llm(messages)


def _format_sources(chunks: List[Dict[str, Any]], max_chars: int = 260) -> str:
    lines: List[str] = []
    for i, c in enumerate(chunks, start=1):
        tag = f"C{i}"
        txt = str(c.get("text", "")).strip().replace("\n", " ")
        if len(txt) > max_chars:
            txt = txt[:max_chars].rstrip() + " …"
        lines.append(f"[{tag}] chunk_id={c.get('chunk_id')} score={float(c.get('score',0.0)):.3f} :: {txt}")
    return "\n".join(lines)


def _missing_citations(s: str) -> bool:
    return ("[C1]" not in s) and ("(C1" not in s) and ("C1" not in s)


def explain_rag(user_id: str, recs: List[Dict[str, Any]], jobs_df) -> List[str]:
    """
    Advanced grounded explanations (10/10 version).

    - If chunk-level RAG assets exist, retrieve top chunks for the target job and force citations [C1]..[Ck].
    - If RAG is unavailable or evidence is weak, fall back to row-only evidence.
    """
    import src.config as cfg
    ENABLE_ADVANCED_RAG = getattr(cfg, 'ENABLE_ADVANCED_RAG', True)
    RAG_OVERFETCH = getattr(cfg, 'RAG_OVERFETCH', 25)
    RAG_TOPK_CHUNKS = getattr(cfg, 'RAG_TOPK_CHUNKS', 5)
    RAG_MIN_SIM = getattr(cfg, 'RAG_MIN_SIM', 0.2)

    outs: List[str] = []

    # Build a compact "user intent" query for retrieval
    user_query = ""
    try:
        from src.features import load_users_df
        udf = load_users_df()
        m = udf["user_id"].astype(str) == str(user_id)
        if m.any():
            u = udf[m].iloc[0]
            user_query = (
                f"Preferred location: {u.get('preferred_location','')}. "
                f"Skills: {u.get('skills','')}. "
                f"Summary: {u.get('summary','')}."
            )
    except Exception:
        user_query = ""

    for r in recs:
        jid = str(r.get("job_id", ""))
        row = jobs_df[jobs_df["job_id"].astype(str) == jid]
        if row.empty:
            outs.append(f"No retrieved job record for job_id={jid}.")
            continue

        jr = row.iloc[0]

        base_evidence = (
            f"Job:\n"
            f"- job_id: {jid}\n"
            f"- title: {jr.get('title','')}\n"
            f"- company: {jr.get('company_name','')}\n"
            f"- location: {jr.get('location','')}\n"
            f"- skills: {jr.get('skills','')}\n\n"
            f"Signals:\n"
            f"- embedding_score: {float(r.get('embedding_score',0.0)):.3f}\n"
            f"- graph_score: {float(r.get('graph_score',0.0)):.3f}\n"
            f"- popularity_views: {int(r.get('popularity_views',0))}\n"
            f"- hybrid_score: {float(r.get('hybrid_score',0.0)):.3f}\n"
            f"- overlap_skills: {r.get('overlap_skills','—')}\n"
        )

        chunks: List[Dict[str, Any]] = []
        if ENABLE_ADVANCED_RAG:
            q = (
                user_query
                + "\nTarget job: "
                + f"{jr.get('title','')} at {jr.get('company_name','')}. "
                + f"Job skills: {jr.get('skills','')}. "
                + f"Job description (preview): {str(jr.get('description',''))[:400]}"
            ).strip()

            qvec = embed_text(q)
            hits = rag_knn(qvec, k=max(int(RAG_OVERFETCH), int(RAG_TOPK_CHUNKS)))
            hits = [h for h in hits if str(h.get("job_id", "")) == jid]
            hits = sorted(hits, key=lambda x: float(x.get("score", 0.0)), reverse=True)
            chunks = hits[: int(RAG_TOPK_CHUNKS)]

            if chunks and float(chunks[0].get("score", 0.0)) < float(RAG_MIN_SIM):
                chunks = []

        if chunks:
            sources = _format_sources(chunks)
            evidence = base_evidence + "\nEvidence Chunks (cite as [C1], [C2], ...):\n" + sources

            system = (
                "You are a job recommender explanation assistant.\n"
                "STRICT RULES:\n"
                "1) Use ONLY the provided Evidence (Job + Signals + Evidence Chunks).\n"
                "2) Every factual claim about the job MUST include a citation tag like [C1] or [C2].\n"
                "3) If evidence is insufficient for a claim, say you cannot infer it.\n"
                "Output format:\n"
                "- 2–3 sentences explaining the match (each sentence must include citations)\n"
                "- Skills to highlight: 2 bullets (cited)\n"
                "- Skills to improve: 2 bullets (cited)\n"
                "- Why ranked here: 1 line mentioning at least 2 Signals (no extra job facts)\n"
            )

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Evidence:\n{evidence}\n\nWrite the explanation now."},
            ]
            ans = call_llm(messages).strip()

            if _missing_citations(ans):
                messages.append(
                    {
                        "role": "user",
                        "content": "Rewrite and ensure every sentence includes [C#] citations to the chunks.",
                    }
                )
                ans = call_llm(messages).strip()

            outs.append(ans + "\n\nSources:\n" + sources)
        else:
            # Fallback to row-only evidence
            desc = str(jr.get("description", "")).strip()
            if len(desc) > 1200:
                desc = desc[:1200] + " ..."
            evidence = base_evidence + f"\nDescription (truncated):\n{desc}\n"

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
