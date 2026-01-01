from __future__ import annotations
import math
from functools import lru_cache
from typing import Any, Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    USER_EMB_NPY,
    USER_IDS_NPY,
    EMBEDDING_MODEL_NAME,
    ALPHA,
    BETA,
    GAMMA,
)
from src.features.preprocessing import (
    load_jobs_df,
    load_users_df,
    build_user_text,
    parse_skills,
)
from src.stores.vector_store import JobVectorIndex
from src.stores.graph_store import compute_graph_scores
from src.stores.nosql_store import get_job_popularity

@lru_cache(maxsize=1)
def _jobs_df():
    return load_jobs_df()

@lru_cache(maxsize=1)
def _users_df():
    return load_users_df()

class HybridRecommender:
    """
    Hybrid recommender that combines:

      - Dense semantic similarity (embeddings + FAISS)
      - Skill-overlap graph score (Jaccard between user & job skills)
      - Popularity from NoSQL (MongoDB views)

    final_score = α * dense_similarity
                  + β * graph_skill_overlap
                  + γ * log(1 + popularity_views)
    """

    def __init__(
        self,
        alpha: float = ALPHA,
        beta: float = BETA,
        gamma: float = GAMMA,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.jobs_df = _jobs_df()
        self.users_df = _users_df()

        # Vector index for job embeddings
        self.vector_index = JobVectorIndex()

        # Embedding model for on-the-fly user embeddings (fallback)
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Optional: cached precomputed user embeddings if files exist
        try:
            self.user_ids = np.load(USER_IDS_NPY, allow_pickle=True).astype(str)
            self.user_emb = np.load(USER_EMB_NPY, allow_pickle=True).astype("float32")
        except Exception:
            self.user_ids = None
            self.user_emb = None

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------
    def _user_embedding(self, user_id: str) -> np.ndarray:
        """
        Return a dense embedding for the user.

        Prefer precomputed embeddings if available; otherwise embed on the fly.
        """
        user_id = str(user_id)

        # 1) Try precomputed embedding
        if self.user_ids is not None and self.user_emb is not None:
            mask = self.user_ids == user_id
            if mask.any():
                return self.user_emb[mask.argmax()]

        # 2) Fallback: embed user text on the fly
        user_rows = self.users_df[self.users_df["user_id"].astype(str) == user_id]
        if user_rows.empty:
            raise ValueError(f"Unknown user_id: {user_id}")

        text = build_user_text(user_rows).iloc[0]
        vec = self.embed_model.encode([text], convert_to_numpy=True)[0]
        return vec.astype("float32")

    def _raw_recommendations(
        self,
        user_id: str,
        top_k: int = 20,
        oversample: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Core logic: compute dense similarity, graph scores and popularity
        for the best candidate jobs, and assemble a list of recommendation dicts.

        This does NOT apply any filtering or final ranking logic; that is
        handled by higher-level methods (recommend_overview, recommend_graph, etc.).
        """
        user_id = str(user_id)
        user_vec = self._user_embedding(user_id)

        # 1) Dense similarity: FAISS search
        dense_candidates = self.vector_index.search_for_user_vector(
            user_vec, top_k * oversample
        )  # list[(job_id, dense_score)]

        # 2) Graph scores (skill overlap)
        graph_scores = compute_graph_scores(user_id)

        # 3) User skills set
        user_row = self.users_df[self.users_df["user_id"].astype(str) == user_id]
        if user_row.empty:
            raise ValueError(f"Unknown user_id: {user_id}")
        user_skills = parse_skills(user_row.iloc[0]["skills"])

        results: List[Dict[str, Any]] = []

        for job_id, dense_score in dense_candidates:
            job_id_str = str(job_id)
            row = self.jobs_df[self.jobs_df["job_id"].astype(str) == job_id_str]
            if row.empty:
                continue
            row = row.iloc[0]

            # Graph / skill overlap score
            g_score = float(graph_scores.get(job_id_str, 0.0))

            # Popularity from NoSQL (MongoDB views)
            try:
                views = int(get_job_popularity(job_id_str) or 0)
            except Exception:
                views = 0

            # Hybrid score
            hybrid = (
                self.alpha * float(dense_score)
                + self.beta * g_score
                + self.gamma * math.log1p(views)
            )

            # Overlap skills (for explanation)
            job_skills = parse_skills(row.get("skills", ""))
            overlap = sorted(user_skills.intersection(job_skills))
            overlap_str = ", ".join(overlap) if overlap else "—"

            results.append(
                {
                    "user_id": user_id,
                    "job_id": job_id_str,
                    "title": row.get("title", ""),
                    "company_name": row.get("company_name", ""),
                    "location": row.get("location", ""),
                    "embedding_score": float(dense_score),
                    "graph_score": g_score,
                    "popularity_views": views,
                    "hybrid_score": float(hybrid),
                    "overlap_skills": overlap_str,
                }
            )

        return results

    @staticmethod
    def _filter_good_matches(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out rows that clearly look like "no match":
        - embedding_score == 0 AND graph_score == 0 AND overlap_skills == "—"
        """
        filtered: List[Dict[str, Any]] = []
        for r in recs:
            emb = float(r.get("embedding_score", 0.0) or 0.0)
            g = float(r.get("graph_score", 0.0) or 0.0)
            ov = str(r.get("overlap_skills", "—")).strip()
            if not (emb <= 0.0 and g <= 0.0 and ov in {"", "—"}):
                filtered.append(r)
        return filtered

    # ---------------------------------------------------------
    # Public APIs – different "views" over the same core logic
    # ---------------------------------------------------------
    def recommend(self, user_id: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Main hybrid recommendation:
          - filter out junk,
          - rank by hybrid_score (final combined score),
          - return top_k items.
        """
        recs = self._raw_recommendations(user_id=user_id, top_k=top_k, oversample=4)
        recs = self._filter_good_matches(recs)
        recs.sort(key=lambda r: float(r.get("hybrid_score", 0.0) or 0.0), reverse=True)
        return recs[:top_k]

    def recommend_overview(self, user_id: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Overview view:
          - uses same underlying recommendations,
          - filters junk,
          - ranks by the average of embedding_score and graph_score.

        The hybrid_score is still included in each record for reference.
        """
        recs = self._raw_recommendations(user_id=user_id, top_k=top_k, oversample=4)
        recs = self._filter_good_matches(recs)

        for r in recs:
            emb = float(r.get("embedding_score", 0.0) or 0.0)
            g = float(r.get("graph_score", 0.0) or 0.0)
            r["_combined_score"] = (emb + g) / 2.0

        recs.sort(key=lambda r: float(r.get("_combined_score", 0.0) or 0.0), reverse=True)
        return recs[:top_k]

    def recommend_semantic(self, user_id: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Semantic-only view:
          - filters junk,
          - ranks by embedding_score only.
        """
        recs = self._raw_recommendations(user_id=user_id, top_k=top_k, oversample=4)
        recs = self._filter_good_matches(recs)
        recs.sort(key=lambda r: float(r.get("embedding_score", 0.0) or 0.0), reverse=True)
        return recs[:top_k]

    def recommend_graph(self, user_id: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Graph-only view:
          - filters junk,
          - ranks by graph_score only (skill overlap).
        """
        recs = self._raw_recommendations(user_id=user_id, top_k=top_k, oversample=4)
        recs = self._filter_good_matches(recs)
        recs.sort(key=lambda r: float(r.get("graph_score", 0.0) or 0.0), reverse=True)
        return recs[:top_k]

    def recommend_with_popularity(
        self,
        user_id: str,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        User-centric popularity view:
          - filters junk,
          - ranks by popularity_views (MongoDB),
          - used by the NoSQL tab.
        """
        recs = self._raw_recommendations(user_id=user_id, top_k=top_k, oversample=4)
        recs = self._filter_good_matches(recs)
        recs.sort(key=lambda r: int(r.get("popularity_views", 0) or 0), reverse=True)
        return recs[:top_k]
