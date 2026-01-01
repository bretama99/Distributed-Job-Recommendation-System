# src/llm/explainer.py
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

import pandas as pd

from src.llm.client import call_llm
from src.features.preprocessing import load_users_df, load_jobs_df


# ----------------------------------------------------------------------
# Cached DataFrames (retrieval layer)
# ----------------------------------------------------------------------

@lru_cache(maxsize=1)
def _users_df() -> pd.DataFrame:
    return load_users_df()


@lru_cache(maxsize=1)
def _jobs_df() -> pd.DataFrame:
    return load_jobs_df()


# ----------------------------------------------------------------------
# Global (multi-job) explanation – same as before
# ----------------------------------------------------------------------

def build_explanation_prompt(user_id: str, recs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build an LLM prompt that explains hybrid job recommendations
    for a given user, based on:
      - user profile (name, skills, experience, location)
      - hybrid_score
      - overlapping skills
    """
    users_df = _users_df()
    row = users_df[users_df["user_id"].astype(str) == str(user_id)]
    if row.empty:
        user_profile = "Unknown user (no profile found)."
    else:
        r = row.iloc[0]
        name = str(r.get("name", "")).strip() or "N/A"
        skills = str(r.get("skills", "")).strip() or "N/A"
        years = r.get("years_experience", "N/A")
        loc = str(r.get("preferred_location", "")).strip() or "N/A"
        user_profile = (
            f"Name: {name}\n"
            f"Skills: {skills}\n"
            f"Years of experience: {years}\n"
            f"Preferred location: {loc}"
        )

    lines: List[str] = []
    for idx, rec in enumerate(recs, start=1):
        job_id = rec.get("job_id", "")
        title = rec.get("title", "")
        company = rec.get("company_name", "")
        location = rec.get("location", "")
        hybrid = float(rec.get("hybrid_score", 0.0))
        overlap = rec.get("overlap_skills", "—")
        lines.append(
            f"{idx}. Job {job_id}: {title} at {company} ({location}). "
            f"Hybrid match score: {hybrid:.3f}. Overlapping skills: {overlap}."
        )

    context = "\n".join(lines) if lines else "No jobs provided."

    system_msg = {
        "role": "system",
        "content": (
            "You are an expert career coach and data scientist. "
            "You explain job recommendations in a clear, structured way, "
            "based on an overall hybrid match score and overlapping skills. "
            "First provide a short overall summary of the user's profile and the types of jobs that fit. "
            "Then, for each job, briefly explain why it matches this user."
        ),
    }

    user_msg = {
        "role": "user",
        "content": (
            f"User profile:\n{user_profile}\n\n"
            f"Recommended jobs (with hybrid scores and overlapping skills):\n{context}\n\n"
            "Please:\n"
            "1. Summarize in 2–3 bullet points why these jobs fit the user overall.\n"
            "2. Then, for each job, write 1–2 sentences explaining the match "
            "(mention key skills and, if relevant, location or experience)."
        ),
    }

    return [system_msg, user_msg]


def explain_recommendations_llm(user_id: str, recs: List[Dict[str, Any]]) -> str:
    messages = build_explanation_prompt(user_id, recs)
    return call_llm(messages)


# ----------------------------------------------------------------------
# NEW: Per-job RAG explanations (true RAG)
# ----------------------------------------------------------------------

def _build_user_profile_text(user_id: str) -> str:
    """Small helper to build a user profile snippet."""
    users_df = _users_df()
    row = users_df[users_df["user_id"].astype(str) == str(user_id)]
    if row.empty:
        return "Unknown user (no profile found)."
    r = row.iloc[0]
    name = str(r.get("name", "")).strip() or "N/A"
    skills = str(r.get("skills", "")).strip() or "N/A"
    years = r.get("years_experience", "N/A")
    loc = str(r.get("preferred_location", "")).strip() or "N/A"
    return (
        f"Name: {name}\n"
        f"Skills: {skills}\n"
        f"Years of experience: {years}\n"
        f"Preferred location: {loc}"
    )

def _build_per_job_rag_messages(
    user_profile: str,
    job_row: pd.Series,
    rec: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Build a chat-style RAG prompt for a *single* job:
      - retrieved job information: title, company, location, skills, description
      - numeric signals: embedding_score, graph_score, hybrid_score
      - overlap_skills from the recommender

    The wording is explicitly constructive: even for weaker matches, the model
    should focus on how the job could be relevant (as a stretch option, or with
    upskilling), not simply say "this is not a good fit".
    """
    title = str(job_row.get("title", "N/A")).strip()
    company = str(job_row.get("company_name", "N/A")).strip()
    location = str(job_row.get("location", "N/A")).strip()
    skills = str(job_row.get("skills", "")).strip() or "N/A"
    description = str(job_row.get("description", "")).strip()

    # To avoid huge prompts, truncate very long descriptions
    max_chars = 1200
    if len(description) > max_chars:
        description = description[:max_chars] + " ... [truncated]"

    emb = float(rec.get("embedding_score", 0.0) or 0.0)
    g = float(rec.get("graph_score", 0.0) or 0.0)
    hybrid = float(rec.get("hybrid_score", 0.0) or 0.0)
    overlap = str(rec.get("overlap_skills", "—"))

    # Precompute a simple match label for the model to use
    if hybrid >= 0.65:
        match_label = "high match"
    elif hybrid >= 0.55:
        match_label = "good match"
    elif hybrid >= 0.45:
        match_label = "partial match"
    else:
        match_label = "exploratory / stretch option"

    job_context = (
        f"Job title: {title}\n"
        f"Company: {company}\n"
        f"Location: {location}\n"
        f"Required/mentioned skills: {skills}\n"
        f"Job description:\n{description or 'N/A'}"
    )

    system_msg = {
        "role": "system",
        "content": (
            "You are an encouraging career coach and job-matching assistant. "
            "Given a user's profile and one job's detailed description, you explain "
            "clearly why this specific job could be relevant for the user. "
            "Always be constructive and supportive: highlight strengths and concrete "
            "growth opportunities.\n\n"
            "IMPORTANT STYLE RULES:\n"
            "- Do NOT use negative phrases such as 'not a good fit', 'not a strong match', "
            "'does not match', 'may not be the best fit', or similar.\n"
            "- For weaker matches, describe them as 'lower-priority options', "
            "'exploratory choices', or 'stretch roles that would require upskilling in X'.\n"
            "- Always mention at least one positive aspect (matching skills, domain, "
            "location, or experience level), even if the overall match is weak.\n"
            "- Keep explanations short (2–3 sentences) and focused."
        ),
    }

    user_msg = {
        "role": "user",
        "content": (
            f"User profile:\n{user_profile}\n\n"
            f"Job information (retrieved from the job database):\n{job_context}\n\n"
            f"Matching signals from the recommender system:\n"
            f"- match label (derived from hybrid_score): {match_label}\n"
            f"- embedding_score (semantic similarity between user and job text): {emb:.3f}\n"
            f"- graph_score (skill-overlap score): {g:.3f}\n"
            f"- hybrid_score (overall combined score): {hybrid:.3f}\n"
            f"- overlapping skills extracted: {overlap}\n\n"
            "Write 2–3 sentences explaining how this job relates to the user:\n"
            "- If the match label is 'high match' or 'good match', emphasise why it is a "
            "strong opportunity.\n"
            "- If the match label is 'partial match' or 'exploratory / stretch option', "
            "present it as a lower-priority or stretch role: explain what is positive "
            "and what new skills or experience the user would need to build.\n"
            "Always follow the style rules above and avoid harsh or discouraging language."
        ),
    }

    return [system_msg, user_msg]

def rag_explain_per_job(user_id: str, recs: List[Dict[str, Any]]) -> List[str]:
    """
    True RAG-style per-job explanations.

    For each recommended job:
      1. RETRIEVE: look up the job row in jobs_df (title, skills, description, ...).
      2. AUGMENT: build an LLM prompt including:
         - user profile
         - retrieved job text
         - numeric scores and overlapping skills
      3. GENERATE: call the LLM and collect its explanation text.

    Returns a list of explanation strings in the same order as `recs`.
    """
    jobs_df = _jobs_df()
    user_profile = _build_user_profile_text(user_id)

    explanations: List[str] = []
    for rec in recs:
        job_id = str(rec.get("job_id", ""))
        job_row_df = jobs_df[jobs_df["job_id"].astype(str) == job_id]
        if job_row_df.empty:
            explanations.append(
                f"No detailed job information found for job_id={job_id}. "
                "Cannot generate a RAG explanation for this job."
            )
            continue

        job_row = job_row_df.iloc[0]

        try:
            messages = _build_per_job_rag_messages(user_profile, job_row, rec)
            text = call_llm(messages)
        except Exception as exc:
            # If anything goes wrong for this job, add a fallback message
            text = f"Error while generating LLM explanation for job {job_id}: {exc}"
        explanations.append(text)

    return explanations
