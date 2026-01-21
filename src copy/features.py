from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (
    DATA_PROCESSED,
    JOBS_CSV,
    USERS_CSV,
    JOB_EMB_NPY,
    JOB_IDS_NPY,
    USER_EMB_NPY,
    USER_IDS_NPY,
    JOB_FAISS_INDEX,
    PIPELINE_META_JSON,
    EMBEDDING_MODEL_NAME,
    LINKEDIN_POSTS,
    LINKEDIN_SUMMARY,
    LINKEDIN_SKILLS,
    MAX_JOBS,
    N_USERS,
)


# ============================================================
# Helpers
# ============================================================

def _sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8", errors="ignore")).hexdigest()

def _norm_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()

def _norm_link(x: Any) -> str:
    """Normalize job_link for stable joining.
    We remove whitespace. (We do NOT try to canonicalize beyond that, because
    your datasets already match on job_link.)
    """
    return _norm_text(x)

def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin-1")

def _skills_to_tokens(raw: Any) -> List[str]:
    """Parse skills from either:
      - comma-separated string in one cell (job_skills / skills)
      - single skill string
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    for sep in [";", "|", "/"]:
        s = s.replace(sep, ",")
    toks = []
    for p in s.split(","):
        t = p.strip().lower()
        if not t:
            continue
        t = " ".join(t.split())
        toks.append(t)
    return toks

def parse_skills(raw: Any) -> Set[str]:
    return set(_skills_to_tokens(raw))


def load_jobs_df() -> pd.DataFrame:
    """Load jobs table created by the ETL step."""
    return pd.read_csv(JOBS_CSV)


def load_users_df() -> pd.DataFrame:
    """Load users table created by the ETL step."""
    return pd.read_csv(USERS_CSV)


def _stable_job_id_from_link(job_link: str, fallback: str) -> str:
    link = _norm_link(job_link)
    if link:
        return _sha1(link)[:12]
    return _sha1(fallback)[:12]


# ============================================================
# ETL: build jobs/users CSVs
# ============================================================

def _etl_jobs_from_linkedin(limit: int = MAX_JOBS) -> pd.DataFrame:
    """
    Build jobs table from the 3 LinkedIn CSVs.

    IMPORTANT:
    - Your job_skills.csv has columns: job_link, job_skills (skills list in ONE cell)
      so the merge MUST be on job_link.

    Output columns:
      job_id, title, company_name, location, skills, description, job_link
    """
    posts = _read_csv_safe(Path(LINKEDIN_POSTS))
    summary = _read_csv_safe(Path(LINKEDIN_SUMMARY))
    skills = _read_csv_safe(Path(LINKEDIN_SKILLS))

    if posts.empty:
        raise RuntimeError(f"LinkedIn posts CSV not found or empty: {LINKEDIN_POSTS}")

    # posts job_link
    p_link = _pick(posts, ["job_link", "link", "url", "job_url"])
    if not p_link:
        raise RuntimeError(f"Posts CSV missing job_link column. Columns={list(posts.columns)}")
    posts["_job_link_norm"] = posts[p_link].map(_norm_link)

    # posts fields
    p_title = _pick(posts, ["title", "job_title", "jobtitle", "position"])
    p_company = _pick(posts, ["company_name", "company", "organization", "employer"])
    p_loc = _pick(posts, ["location", "job_location", "joblocation", "city"])

    base = pd.DataFrame({
        "job_link": posts["_job_link_norm"],
        "title": posts[p_title].astype(str) if p_title else "",
        "company_name": posts[p_company].astype(str) if p_company else "",
        "location": posts[p_loc].astype(str) if p_loc else "",
    })

    # summary merge by job_link
    if not summary.empty:
        s_link = _pick(summary, ["job_link", "link", "url", "job_url"])
        s_desc = _pick(summary, ["description", "job_description", "job_summary", "jobsummary", "summary", "content"])
        if s_link:
            summary["_job_link_norm"] = summary[s_link].map(_norm_link)
            s = pd.DataFrame({
                "job_link": summary["_job_link_norm"],
                "description": summary[s_desc].astype(str) if s_desc else "",
            })
            if not s.empty:
                s["description"] = s["description"].fillna("").astype(str)
                s = s.sort_values("description", key=lambda x: x.str.len(), ascending=False)
                s = s.drop_duplicates("job_link", keep="first")
            base = base.merge(s, on="job_link", how="left")
        else:
            base["description"] = ""
    else:
        base["description"] = ""

    base["description"] = base["description"].fillna("").astype(str)

    # skills merge by job_link
    skills_out = pd.DataFrame({"job_link": base["job_link"], "skills": ""})

    if not skills.empty:
        k_link = _pick(skills, ["job_link", "link", "url", "job_url"])
        k_skill = _pick(skills, ["job_skills", "jobskills", "skills", "skill", "skill_name", "name"])
        if k_link and k_skill:
            skills["_job_link_norm"] = skills[k_link].map(_norm_link)
            tmp = skills[["_job_link_norm", k_skill]].copy()
            tmp = tmp.rename(columns={"_job_link_norm": "job_link", k_skill: "raw_skill"})
            tmp["raw_skill"] = tmp["raw_skill"].fillna("").astype(str)

            # explode into one skill per row
            rows: List[Tuple[str, str]] = []
            for jl, rs in zip(tmp["job_link"].tolist(), tmp["raw_skill"].tolist()):
                for t in _skills_to_tokens(rs):
                    rows.append((jl, t))

            if rows:
                kdf = pd.DataFrame(rows, columns=["job_link", "skill"])
                agg = (
                    kdf.groupby("job_link")["skill"]
                    .apply(lambda x: ", ".join(sorted(set(x.tolist()))))
                    .reset_index()
                    .rename(columns={"skill": "skills"})
                )
                skills_out = agg

    base = base.merge(skills_out, on="job_link", how="left")
    base["skills"] = base["skills"].fillna("").astype(str)

    # cap and create stable job_id
    if limit and int(limit) > 0:
        base = base.head(int(limit)).copy()

    job_ids: List[str] = []
    for i, r in base.iterrows():
        fb = f"{r.get('title','')}|{r.get('company_name','')}|{r.get('location','')}|{i}"
        job_ids.append(_stable_job_id_from_link(str(r.get("job_link", "")), fb))
    base["job_id"] = job_ids

    base = base[["job_id", "title", "company_name", "location", "skills", "description", "job_link"]].copy()
    return base


def _build_users_from_jobs(jobs: pd.DataFrame, n_users: int = N_USERS, seed: int = 42) -> pd.DataFrame:
    """
    Build synthetic users based on jobs' skills and locations.

    Output columns:
      user_id, name, preferred_location, skills, summary
    """
    rng = random.Random(seed)

    skill_list: List[str] = []
    if "skills" in jobs.columns:
        for raw in jobs["skills"].fillna("").astype(str).tolist():
            for t in _skills_to_tokens(raw):
                skill_list.append(t)

    # safe fallback (prevents empty users skills)
    if not skill_list:
        skill_list = ["python", "sql", "machine learning", "data analysis", "communication", "teamwork", "problem solving"]

    from collections import Counter
    counts = Counter(skill_list)
    skills_unique = list(counts.keys())
    weights = [counts[s] for s in skills_unique]

    locs = jobs.get("location", pd.Series([], dtype=str)).fillna("").astype(str)
    loc_pool = locs[locs.str.strip() != ""].value_counts().index.tolist()
    if not loc_pool:
        loc_pool = ["Remote"]

    FIRST_NAMES = [
        "Nicole","Amina","Salvatore","Robel","Francesco","Alessandro","Ernesto","Aurachiara","Elio","Federico",
        "Alexander","Ilaria","Fortunato","Emanuele","Brhane","Giuseppe","Megan","Marco","Matteo","Davide",
        "Michael","Samuele","Pierpaolo","Pasquale","Luigi","Domenico","Yaekob","Jakub"
    ]
    LAST_NAMES = [
        "Arnieri","Benkacem","Biamonte","Campagna","Casella","Cesario","Chirillo","DAlessandro","DiFranco",
        "Gagliardi","Galardo","Gidey","Lentini","Macri","Martino","Paparo","Pirro","Serratore","Siciliano",
        "Spadafora","Tudda","Vasile","Villella","Visciglia","Yowhanns","Zeglinski"
    ]

    def sample_name() -> str:
        return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"

    def sample_skills() -> str:
        k = rng.randint(4, 8)
        chosen = rng.choices(skills_unique, weights=weights, k=k)
        seen, uniq = set(), []
        for s in chosen:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return ", ".join(uniq)

    def sample_location() -> str:
        return rng.choice(loc_pool)

    rows: List[Dict[str, Any]] = []
    for i in range(1, int(n_users) + 1):
        uid = f"u{i:06d}"
        sk = sample_skills()
        loc = sample_location()
        rows.append(
            {
                "user_id": uid,
                "name": sample_name(),
                "preferred_location": loc,
                "skills": sk,
                "summary": "AI/CS candidate interested in roles matching skills and location preferences.",
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# Embeddings artifacts
# ============================================================

def build_job_text(df: pd.DataFrame) -> pd.Series:
    def f(r: pd.Series) -> str:
        return (
            f"Job title: {r.get('title','')} | Company: {r.get('company_name','')} | Location: {r.get('location','')} | "
            f"Skills: {r.get('skills','')} | Description: {r.get('description','')}"
        )
    return df.apply(f, axis=1)

def build_user_text(df: pd.DataFrame) -> pd.Series:
    def f(r: pd.Series) -> str:
        return (
            f"User: {r.get('name','')} | Preferred location: {r.get('preferred_location','')} | "
            f"Skills: {r.get('skills','')} | Summary: {r.get('summary','')}"
        )
    return df.apply(f, axis=1)


def compute_and_save_embeddings(jobs: pd.DataFrame, users: pd.DataFrame) -> Dict[str, Any]:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    job_texts = build_job_text(jobs).tolist()
    job_ids = jobs["job_id"].astype(str).to_numpy()
    job_emb = model.encode(job_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True).astype("float32")

    user_texts = build_user_text(users).tolist()
    user_ids = users["user_id"].astype(str).to_numpy()
    user_emb = model.encode(user_texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True).astype("float32")

    np.save(JOB_EMB_NPY, job_emb)
    np.save(JOB_IDS_NPY, job_ids)
    np.save(USER_EMB_NPY, user_emb)
    np.save(USER_IDS_NPY, user_ids)

    return {"jobs": len(job_ids), "users": len(user_ids)}


def _pipeline_meta() -> Dict[str, Any]:
    return {
        "linkedin_posts": str(LINKEDIN_POSTS),
        "linkedin_summary": str(LINKEDIN_SUMMARY),
        "linkedin_skills": str(LINKEDIN_SKILLS),
        "max_jobs": int(MAX_JOBS),
        "n_users": int(N_USERS),
        "embedding_model": EMBEDDING_MODEL_NAME,
    }


def _read_meta() -> Dict[str, Any]:
    if PIPELINE_META_JSON.exists():
        try:
            return json.loads(PIPELINE_META_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _write_meta(meta: Dict[str, Any]) -> None:
    PIPELINE_META_JSON.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_META_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def ensure_ready(force: bool = False) -> Dict[str, Any]:
    """
    Ensure data/raw and data/processed artifacts exist.
    If force=True, rebuild everything deterministically.
    """
    meta_now = _pipeline_meta()
    meta_old = _read_meta()

    need_etl = force or (meta_old != meta_now) or (not JOBS_CSV.exists()) or (not USERS_CSV.exists())
    need_emb = force or (not JOB_EMB_NPY.exists()) or (not USER_EMB_NPY.exists()) or (not JOB_IDS_NPY.exists()) or (not USER_IDS_NPY.exists())

    out: Dict[str, Any] = {"ok": True, "rebuild_etl": bool(need_etl), "rebuild_emb": bool(need_emb)}

    if need_etl:
        jobs = _etl_jobs_from_linkedin(limit=MAX_JOBS)
        users = _build_users_from_jobs(jobs, n_users=N_USERS)
        JOBS_CSV.parent.mkdir(parents=True, exist_ok=True)
        USERS_CSV.parent.mkdir(parents=True, exist_ok=True)
        jobs.to_csv(JOBS_CSV, index=False)
        users.to_csv(USERS_CSV, index=False)

        empty_ratio = float((jobs["skills"].fillna("").astype(str).str.strip() == "").mean()) if "skills" in jobs.columns else 1.0
        out["jobs_empty_skills_ratio"] = empty_ratio

        _write_meta(meta_now)
        out["jobs_rows"] = int(len(jobs))
        out["users_rows"] = int(len(users))

    if need_emb:
        jobs_df = pd.read_csv(JOBS_CSV)
        users_df = pd.read_csv(USERS_CSV)
        emb_info = compute_and_save_embeddings(jobs_df, users_df)
        out["embeddings"] = emb_info

    return out

def run_etl(limit: int = MAX_JOBS) -> Dict[str, Any]:
    """
    UI-facing wrapper. Builds jobs/users and writes them to data/raw.
    Returns basic stats.
    """
    jobs = _etl_jobs_from_linkedin(limit=limit)
    users = _build_users_from_jobs(jobs, n_users=N_USERS)

    JOBS_CSV.parent.mkdir(parents=True, exist_ok=True)
    USERS_CSV.parent.mkdir(parents=True, exist_ok=True)

    jobs.to_csv(JOBS_CSV, index=False)
    users.to_csv(USERS_CSV, index=False)

    empty_ratio = float((jobs["skills"].fillna("").astype(str).str.strip() == "").mean())
    return {
        "ok": True,
        "jobs_rows": int(len(jobs)),
        "users_rows": int(len(users)),
        "jobs_empty_skills_ratio": empty_ratio,
        "jobs_csv": str(JOBS_CSV),
        "users_csv": str(USERS_CSV),
    }


def compute_embeddings() -> Dict[str, Any]:
    """
    UI-facing wrapper. Loads data/raw CSVs and computes embeddings.
    Returns basic stats.
    """
    jobs_df = pd.read_csv(JOBS_CSV)
    users_df = pd.read_csv(USERS_CSV)
    info = compute_and_save_embeddings(jobs_df, users_df)
    return {"ok": True, **info}
