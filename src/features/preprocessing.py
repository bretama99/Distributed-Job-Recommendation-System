# src/features/preprocessing.py
from __future__ import annotations

from typing import Set

import pandas as pd

from src.config import JOBS_CSV, USERS_CSV


# ---------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------
def load_jobs_df() -> pd.DataFrame:
    """Load jobs table created by the ETL step."""
    return pd.read_csv(JOBS_CSV)


def load_users_df() -> pd.DataFrame:
    """Load users table created by the ETL step."""
    return pd.read_csv(USERS_CSV)


# ---------------------------------------------------------------------
# Skills parsing
# ---------------------------------------------------------------------
def parse_skills(raw: str) -> Set[str]:
    """
    Normalise a raw skills string into a set of lowercase tokens.

    Supports separators: ',', ';', '|', '/'.
    """
    if not isinstance(raw, str):
        return set()

    s = raw
    for sep in [";", "|", "/"]:
        s = s.replace(sep, ",")

    parts = [p.strip().lower() for p in s.split(",")]
    return {p for p in parts if p}


# ---------------------------------------------------------------------
# Text builders for embeddings
# ---------------------------------------------------------------------
def build_job_text(df: pd.DataFrame) -> pd.Series:
    """
    Build natural-language texts for jobs, to feed into the
    sentence-transformer when computing JOB embeddings.

    Uses title, company, location, skills, and description.
    """

    def _row_to_text(row: pd.Series) -> str:
        title = row.get("title", "")
        company = row.get("company_name", "")
        location = row.get("location", "")
        skills = row.get("skills", "")
        desc = row.get("description", "")

        return (
            f"Job title: {title} at {company} in {location}. "
            f"Required skills: {skills}. "
            f"Job description: {desc}"
        )

    return df.apply(_row_to_text, axis=1)


def build_user_text(df: pd.DataFrame) -> pd.Series:
    """
    Build natural-language profiles for users, to feed into the
    sentence-transformer when computing USER embeddings.
    """

    def _row_to_text(row: pd.Series) -> str:
        name = row.get("name", "User")
        skills = row.get("skills", "")
        years = row.get("years_experience", 0)
        pref_loc = row.get("preferred_location", "any location")

        return (
            f"{name} with skills {skills} and {years} years of experience, "
            f"preferring jobs in {pref_loc}."
        )

    return df.apply(_row_to_text, axis=1)
