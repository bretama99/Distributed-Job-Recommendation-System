# src/stores/graph_store.py
from __future__ import annotations

from functools import lru_cache
from typing import Dict

from src.features.preprocessing import load_jobs_df, load_users_df, parse_skills


@lru_cache(maxsize=1)
def _jobs_df():
    return load_jobs_df()


@lru_cache(maxsize=1)
def _users_df():
    return load_users_df()


def compute_graph_scores(user_id: str) -> Dict[str, float]:
    """
    Lightweight "graph" scoring: Jaccard similarity between
    user skills and job skills.

    Returns:
        dict[job_id -> jaccard_score]
    """
    jobs_df = _jobs_df()
    users_df = _users_df()

    row = users_df[users_df["user_id"].astype(str) == str(user_id)]
    if row.empty:
        return {str(jid): 0.0 for jid in jobs_df["job_id"]}

    u_skills = parse_skills(row.iloc[0]["skills"])
    if not u_skills:
        return {str(jid): 0.0 for jid in jobs_df["job_id"]}

    scores: Dict[str, float] = {}

    for _, job_row in jobs_df.iterrows():
        job_id = str(job_row["job_id"])
        j_skills = parse_skills(job_row["skills"])
        if not j_skills:
            scores[job_id] = 0.0
            continue

        inter = u_skills.intersection(j_skills)
        union = u_skills.union(j_skills)
        scores[job_id] = len(inter) / float(len(union)) if union else 0.0

    return scores
