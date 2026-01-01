# src/etl/spark_jobs.py
from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Tuple

import pandas as pd

from src.config import DATA_EXTERNAL, JOBS_CSV, USERS_CSV

log = logging.getLogger(__name__)

# Hard cap to keep it laptop-friendly
MAX_JOBS = 8000

FIRST_NAMES = [
    "Nicole",
    "Amina",
    "Salvatore",
    "Robel Gebrehiwot",
    "Francesco",
    "Alessandro",
    "Ernesto",
    "Aurachiara",
    "Elio Maria",
    "Federico",
    "Alexander",
    "Ilaria",
    "Fortunato Andrea",
    "Emanuele",
    "Brhane Teamrat",
    "Giuseppe",
    "Megan",
    "Marco",
    "Matteo",
    "Davide",
    "Michael",
    "Francesco",
    "Samuele",
    "Pierpaolo",
    "Pasquale",
    "Ilaria Raffaela",
    "Luigi",
    "Domenico",
    "Yaekob Beyene",
    "Jakub",
    "Francesco",
]

LAST_NAMES = [
    "Arnieri",
    "Benkacem",
    "Biamonte",
    "Brhane",
    "Campagna",
    "Casella",
    "Cesario",
    "Chirillo",
    "D'Alessandro",
    "Di Franco",
    "Fichtenberg",
    "Frandina",
    "Gagliardi",
    "Galardo",
    "Gidey",
    "Lentini",
    "Macrì",
    "Martino",
    "Paparo",
    "Pirrò",
    "Posteraro",
    "Serratore",
    "Siciliano",
    "Spadafora",
    "Tudda",
    "Vasile",
    "Villella",
    "Visciglia",
    "Yowhanns",
    "Zeglinski",
    "Magnone",
]



def sample_full_name() -> str:
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return f"{first} {last}"


def _clean_text(s: str) -> str:
    return str(s).replace("\n", " ").strip()


def load_linkedin_dfs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = DATA_EXTERNAL
    posts_path = base / "linkedin_job_postings.csv"
    summary_path = base / "job_summary.csv"
    skills_path = base / "job_skills.csv"

    posts = pd.read_csv(posts_path)
    summary = pd.read_csv(summary_path)
    skills = pd.read_csv(skills_path)

    return posts, summary, skills


def build_jobs_df(max_rows: int = MAX_JOBS) -> pd.DataFrame:
    """
    Merge the three LinkedIn CSVs using pandas and produce a cleaned jobs DataFrame with columns:

        job_id, title, description, company_name, location, skills
    """
    posts, summary, skills = load_linkedin_dfs()

    required_posts = ["job_link", "job_title", "company", "job_location"]
    required_summary = ["job_link", "job_summary"]
    required_skills = ["job_link", "job_skills"]

    for col in required_posts:
        if col not in posts.columns:
            raise ValueError(
                f"Expected column '{col}' in linkedin_job_postings.csv, got {list(posts.columns)}"
            )
    for col in required_summary:
        if col not in summary.columns:
            raise ValueError(
                f"Expected column '{col}' in job_summary.csv, got {list(summary.columns)}"
            )
    for col in required_skills:
        if col not in skills.columns:
            raise ValueError(
                f"Expected column '{col}' in job_skills.csv, got {list(skills.columns)}"
            )

    df = posts.merge(summary, on="job_link", how="left").merge(skills, on="job_link", how="left")
    df = df.head(max_rows).reset_index(drop=True)

    df["job_id"] = df.index + 1
    df = df.rename(
        columns={
            "job_title": "title",
            "company": "company_name",
            "job_location": "location",
            "job_summary": "description",
            "job_skills": "skills",
        }
    )

    df["description"] = df["description"].fillna("").apply(_clean_text)
    df["skills"] = df["skills"].fillna("").apply(_clean_text)

    jobs = df[["job_id", "title", "description", "company_name", "location", "skills"]]
    log.info("Cleaned jobs: %d rows", len(jobs))
    return jobs


def build_users_df(jobs_df: pd.DataFrame, n_users: int = 2000) -> pd.DataFrame:
    """
    Build synthetic users based on job skills and locations.

    Output columns:
      user_id, name, skills, years_experience, preferred_location
    """
    skills_series = jobs_df["skills"].dropna().astype(str)

    skill_list: list[str] = []
    for row in skills_series:
        for s in row.split(","):
            s_norm = s.strip().lower()
            if s_norm:
                skill_list.append(s_norm)

    """if not skill_list:
        skill_list = ["python"]"""

    skill_counts = Counter(skill_list)
    skills_unique = list(skill_counts.keys())
    weights = [skill_counts[s] for s in skills_unique]
    total = float(sum(weights))
    probs = [w / total for w in weights]

    loc_series = jobs_df["location"].dropna().astype(str)
    loc_list = loc_series.value_counts().index.tolist() or ["Remote"]

    def sample_skills() -> str:
        k = random.randint(4, 8)
        chosen = random.choices(skills_unique, weights=probs, k=k)
        seen = set()
        uniq: list[str] = []
        for s in chosen:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return ", ".join(uniq)

    def sample_location() -> str:
        return random.choice(loc_list)

    users_data = []
    for i in range(1, n_users + 1):
        users_data.append(
            {
                "user_id": str(i),
                "name": sample_full_name(),
                "skills": sample_skills(),
                "years_experience": random.randint(1, 10),
                "preferred_location": sample_location(),
            }
        )

    users_df = pd.DataFrame(users_data)
    log.info("Generated synthetic users: %d rows", len(users_df))
    return users_df


def run_etl_jobs_and_users() -> None:
    """
    High-level ETL orchestration:
      1. Build jobs from LinkedIn CSVs.
      2. Build synthetic users.
      3. Save both to data/raw.
    """
    jobs_df = build_jobs_df()
    users_df = build_users_df(jobs_df)

    jobs_df.to_csv(JOBS_CSV, index=False)
    users_df.to_csv(USERS_CSV, index=False)

    log.info("Written jobs to %s", JOBS_CSV)
    log.info("Written users to %s", USERS_CSV)
    log.info("ETL (pandas) finished.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    run_etl_jobs_and_users()
