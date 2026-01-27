from __future__ import annotations

import atexit
import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd

from src.config import APP_TITLE, DATA_PROCESSED, TOP_K_DEFAULT
from src.core import HybridRecommender, explain_global, explain_rag, record_view
from src.features import compute_embeddings, ensure_ready, load_jobs_df, load_users_df, parse_skills, run_etl
from src.stores import (
    analytics_popularity,
    close_mongo,
    ensure_backends_ready,
    get_job_popularity,
    mongo_status,
    neo4j_status,
    sync_neo4j_graph,
    upsert_jobs,
)

CSS = """
html, body, #root { width: 100% !important; margin: 0 !important; padding: 0 !important; }
.gradio-container { max-width: 100% !important; width: 100% !important; margin: 0 !important; padding: 0 !important;
  --layout-max-width: 100% !important; --prose-max-width: 100% !important; --block-max-width: 100% !important; }
.gradio-container .contain, .gradio-container .wrap, .gradio-container .container, .gradio-container .main,
.gradio-container .prose, .gradio-container .content, .gradio-container .app {
  max-width: 100% !important; width: 100% !important; margin: 0 !important; padding: 12px !important; }

h1 { font-size: 1.6rem !important; font-weight: 700 !important; margin: 0 0 8px 0 !important; }
h2 { font-size: 1.05rem !important; font-weight: 650 !important; margin: 0 0 8px 0 !important; }

.tabs { border-bottom: 1px solid rgba(0,0,0,0.08) !important; margin-bottom: 12px !important; }
.tab-nav button { padding: 10px 14px !important; border: none !important; border-bottom: 2px solid transparent !important; }
.tab-nav button.selected { border-bottom-color: #4f46e5 !important; }

.gr-button { background: #4f46e5 !important; border: none !important; color: white !important; padding: 9px 14px !important;
  border-radius: 8px !important; font-size: 0.9rem !important; font-weight: 600 !important; }
.gr-button.secondary { background: #0f172a !important; }
.gr-button:hover { filter: brightness(0.95) !important; }

.gr-box, .gr-input, .gr-dropdown { border: 1px solid rgba(0,0,0,0.12) !important; border-radius: 10px !important; padding: 8px 12px !important; }
label { font-size: 0.86rem !important; font-weight: 600 !important; margin-bottom: 4px !important; text-transform: none !important; }

.gradio-dataframe { width: 100% !important; overflow: auto !important; border: 1px solid rgba(0,0,0,0.10) !important; border-radius: 10px !important; }
.gradio-dataframe table { table-layout: auto !important; border-collapse: collapse !important; font-size: 0.88rem !important; }
.gradio-dataframe td, .gradio-dataframe th { white-space: nowrap !important; padding: 10px 12px !important; border-bottom: 1px solid rgba(0,0,0,0.06) !important; }
.gradio-dataframe thead th { position: sticky !important; top: 0 !important; z-index: 2 !important; background: #f8fafc !important;
  font-weight: 700 !important; font-size: 0.75rem !important; border-bottom: 2px solid rgba(0,0,0,0.10) !important; }
.gradio-dataframe tbody tr:hover { background: #f8fafc !important; }

.df-big .gradio-dataframe { height: 75vh !important; max-height: 75vh !important; }
.df-mid .gradio-dataframe { height: 55vh !important; max-height: 55vh !important; }
.df-small .gradio-dataframe { height: 40vh !important; max-height: 40vh !important; }

.gr-row { gap: 12px !important; }

::-webkit-scrollbar { width: 8px !important; height: 8px !important; }
::-webkit-scrollbar-track { background: #f1f5f9 !important; }
::-webkit-scrollbar-thumb { background: #94a3b8 !important; border-radius: 10px !important; }
::-webkit-scrollbar-thumb:hover { background: #64748b !important; }

.kpi { border: 1px solid rgba(148,163,184,0.22); border-radius: 12px; padding: 10px 12px;
  background: rgba(2,6,23,0.55); color: rgba(226,232,240,0.95); }
.kpi b { font-size: 1.05rem; }
.kpi small { color: rgba(226,232,240,0.72); }

"""

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet", neutral_hue="slate")

MAX_USER_CHOICES = 30
REC: HybridRecommender | None = None


def _reset_runtime() -> None:
    global REC
    REC = None
    try:
        import src.core as _core_mod

        try:
            _core_mod._model.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass
    try:
        close_mongo()
    except Exception:
        pass


def _clip(x: Any, n: int = 240) -> str:
    s = "" if x is None else str(x)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n].rstrip() + " ..."


def get_rec() -> HybridRecommender:
    global REC
    if REC is None:
        ensure_ready()
        ensure_backends_ready()
        REC = HybridRecommender()
    return REC


def df_users() -> pd.DataFrame:
    df = load_users_df()
    df["user_id"] = df["user_id"].astype(str)
    for c in ["name", "preferred_location", "skills", "summary"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    return df


def df_jobs() -> pd.DataFrame:
    df = load_jobs_df()
    df["job_id"] = df["job_id"].astype(str)
    for c in ["title", "company_name", "location", "skills", "description"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    return df


def build_user_choices(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    labels: List[str] = []
    mp: Dict[str, str] = {}
    if df.empty or "user_id" not in df.columns:
        return labels, mp
    for _, r in df.iterrows():
        uid = str(r["user_id"])
        name = str(r.get("name", "")).strip()
        label = f"{name} ({uid})" if name else f"User {uid}"
        labels.append(label)
        mp[label] = uid
    return labels, mp


def refresh_users_state() -> Tuple[List[str], Dict[str, str], str | None]:
    u = df_users()
    labels, mp = build_user_choices(u)
    default = labels[0] if labels else None
    return labels, mp, default


def resolve_user(user_label: str, mapping: Dict[str, str]) -> Tuple[str | None, str | None]:
    if not user_label:
        return None, "Select a user."
    uid = mapping.get(user_label)
    if not uid:
        return None, "Invalid user selection. Click Refresh Users."
    return uid, None


def _empty_recs_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Rank",
            "Job ID",
            "Title",
            "Company",
            "Location",
            "Job Skills",
            "Description",
            "Embedding Score",
            "Graph Score",
            "Popularity",
            "Overlap Count",
            "Overlap Skills",
            "Hybrid Score",
        ]
    )


def _empty_search_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Type", "ID", "Score", "Title/Name", "Company", "Location/Preferred", "Skills", "Details"])


def _empty_graph_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "Rank",
            "Job ID",
            "Title",
            "Company",
            "Location",
            "User Skills",
            "Job Skills",
            "Overlap Skills",
            "Shared Skills",
            "Co-Occur Sum",
            "Graph Raw",
            "Graph Score",
        ]
    )


def _empty_pop_jobs_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Job ID", "Title", "Company", "Location", "Views", "Skills", "Description", "Updated", "Created", "Last View"])


def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Event Type", "Timestamp", "Readable Time", "Details"])


def _empty_job_browser_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["👁", "Job ID", "Title", "Company", "Location", "Skills"])


def _jobs_for_browser(query: str = "", limit: int = 30) -> pd.DataFrame:
    jobs = df_jobs().copy()
    q = (query or "").strip().lower()
    if q:
        blob = (
            jobs["title"].astype(str)
            + " "
            + jobs["company_name"].astype(str)
            + " "
            + jobs["location"].astype(str)
            + " "
            + jobs["skills"].astype(str)
            + " "
            + jobs["description"].astype(str)
        ).str.lower()
        jobs = jobs[blob.str.contains(q, na=False)]
    jobs = jobs.head(int(limit)).copy()
    out = pd.DataFrame(
        {
            "👁": ["👁"] * len(jobs),
            "Job ID": jobs["job_id"].astype(str),
            "Title": jobs["title"].astype(str),
            "Company": jobs["company_name"].astype(str),
            "Location": jobs["location"].astype(str),
            "Skills": jobs["skills"].astype(str).apply(lambda x: _clip(x, 240)),
        }
    )
    return out if not out.empty else _empty_job_browser_df()


def _format_job_details(job_id: str, uid: str | None = None, just_viewed: bool = False) -> str:
    jobs = df_jobs().set_index("job_id", drop=False)
    jid = str(job_id).strip()
    if not jid or jid not in jobs.index:
        return "### ⚠️ Select a job from the table."
    jr = jobs.loc[jid]
    title = str(jr.get("title", "")).strip()
    comp = str(jr.get("company_name", "")).strip()
    loc = str(jr.get("location", "")).strip()
    skills = str(jr.get("skills", "")).strip()
    desc = str(jr.get("description", "")).strip()
    views = int(get_job_popularity(jid))
    badge = "✅ **View Recorded**" if just_viewed else "ℹ️ **Details**"
    who = f"\n\n**Viewer (User ID):** `{uid}`" if uid else ""
    if not desc:
        desc = "_(no description provided in dataset)_"
    return (
        f"### {badge}\n"
        f"**Job ID:** `{jid}`\n\n"
        f"**Title:** {title}\n\n"
        f"**Company:** {comp}\n\n"
        f"**Location:** {loc}\n\n"
        f"**Views:** **{views}**"
        f"{who}\n\n"
        f"**Skills:**\n\n{skills}\n\n"
        f"**Description:**\n\n{desc}"
    )


def _on_job_row_select(browser_df: pd.DataFrame, evt: gr.SelectData):
    try:
        row_idx = evt.index[0] if isinstance(evt.index, (tuple, list)) else int(evt.index)
    except Exception:
        row_idx = 0
    if browser_df is None or len(browser_df) == 0:
        return "", "### ⚠️ No jobs to select."
    row_idx = max(0, min(int(row_idx), len(browser_df) - 1))
    jid = str(browser_df.iloc[row_idx].get("Job ID", "")).strip()
    return jid, _format_job_details(jid)


def _record_view_and_show(user_label: str, mapping: Dict[str, str], job_id: str):
    uid, err = resolve_user(user_label, mapping)
    if err:
        return f"### ❌ {err}"
    jid = str(job_id).strip()
    if not jid:
        return "### ⚠️ Select a job first."
    record_view(uid, jid)
    return _format_job_details(jid, uid=uid, just_viewed=True)


def overview(user_label: str, mapping: Dict[str, str], top_k: int, a: float, b: float, g: float, d: float):
    uid, err = resolve_user(user_label, mapping)
    if err:
        return f"### ❌ {err}", _empty_recs_df()

    rec = get_rec()
    s = float(a) + float(b) + float(g) + float(d)
    if s <= 0.0:
        a, b, g, d = 1.0, 0.0, 0.0, 0.0
        s = 1.0
    rec.alpha, rec.beta, rec.gamma, rec.delta = float(a) / s, float(b) / s, float(g) / s, float(d) / s

    recs = rec.recommend(uid, int(top_k))
    df = pd.DataFrame(recs)
    if df.empty:
        return "### ℹ️ No results.", _empty_recs_df()

    jobs = df_jobs().set_index("job_id", drop=False)
    df["job_skills"] = df["job_id"].astype(str).apply(lambda jid: jobs.loc[jid]["skills"] if jid in jobs.index else "")
    df["desc_snip"] = df["job_id"].astype(str).apply(lambda jid: _clip(jobs.loc[jid]["description"], 240) if jid in jobs.index else "")
    df.insert(0, "rank", range(1, len(df) + 1))

    loc_counts = df["location"].astype(str).value_counts().head(6)
    loc_str = ", ".join([f"{k} ({v})" for k, v in loc_counts.items()]) if not loc_counts.empty else "N/A"

    skills: List[str] = []
    for s2 in df["overlap_skills"].astype(str).tolist():
        for t in s2.split(","):
            t = t.strip()
            if t and t != "—":
                skills.append(t)
    top_sk = Counter(skills).most_common(10)
    sk_str = ", ".join([f"{k} ({v})" for k, v in top_sk]) if top_sk else "N/A"

    msg = (
        f"<div class='kpi'><b>Hybrid Summary</b><br/>"
        f"<small>Avg Embedding</small> <b>{df['embedding_score'].astype(float).mean():.3f}</b> &nbsp; "
        f"<small>Avg Graph</small> <b>{df['graph_score'].astype(float).mean():.3f}</b> &nbsp; "
        f"<small>Avg Hybrid</small> <b>{df['hybrid_score'].astype(float).mean():.3f}</b><br/>"
        f"<small>Top Locations:</small> {loc_str}<br/>"
        f"<small>Top Overlap Skills:</small> {sk_str}"
        f"</div>"
    )

    out = pd.DataFrame(
        {
            "Rank": df["rank"],
            "Job ID": df["job_id"],
            "Title": df["title"],
            "Company": df["company_name"],
            "Location": df["location"],
            "Job Skills": df["job_skills"].astype(str).apply(lambda x: _clip(x, 260)),
            "Description": df["desc_snip"].astype(str),
            "Embedding Score": df["embedding_score"].astype(float).round(4),
            "Graph Score": df["graph_score"].astype(float).round(4),
            "Popularity": df["popularity_views"].astype(int),
            "Overlap Count": df["overlap_count"].astype(int),
            "Overlap Skills": df["overlap_skills"],
            "Hybrid Score": df["hybrid_score"].astype(float).round(4),
        }
    )
    return msg, out

def semantic_search(query: str, top_k: int):
    q = (query or "").strip()
    if not q:
        return _empty_search_df()

    rec = get_rec()
    job_hits = rec.semantic_search(q, int(top_k))

    jobs = df_jobs()
    jobs_idx = jobs.set_index("job_id", drop=False)

    rows: List[Dict[str, Any]] = []
    for r in job_hits:
        jid = str(r.get("job_id", ""))
        score = float(r.get("score", 0.0))

        if jid in jobs_idx.index:
            jr = jobs_idx.loc[jid]
            title = str(jr.get("title", ""))
            comp = str(jr.get("company_name", ""))
            loc = str(jr.get("location", ""))
            skills = str(jr.get("skills", ""))
            desc = str(jr.get("description", ""))
        else:
            title = str(r.get("title", ""))
            comp = str(r.get("company_name", ""))
            loc = str(r.get("location", ""))
            skills = str(r.get("skills", ""))
            desc = ""

        rows.append(
            {
                "Type": "Job",
                "ID": jid,
                "Score": round(score, 6),
                "Title/Name": title,
                "Company": comp,
                "Location/Preferred": loc,
                "Skills": _clip(skills, 300),
                "Details": _clip(desc, 280),
            }
        )

    return pd.DataFrame(rows) if rows else _empty_search_df()

def skill_graph_insights(user_label: str, mapping: Dict[str, str], top_k: int):
    uid, err = resolve_user(user_label, mapping)
    if err:
        return f"### ❌ {err}", _empty_graph_df()

    rec = get_rec()
    recs = rec.recommend_graph_only(uid, int(top_k), neighbor_limit=100000, min_w=1)
    df = pd.DataFrame(recs)
    if df.empty:
        return "### ℹ️ No results.", _empty_graph_df()

    users = df_users().set_index("user_id", drop=False)
    jobs = df_jobs().set_index("job_id", drop=False)

    user_sk = users.loc[uid]["skills"] if uid in users.index else ""
    u_set = set(parse_skills(str(user_sk)))

    df["job_skills"] = df["job_id"].astype(str).apply(lambda jid: jobs.loc[jid]["skills"] if jid in jobs.index else "")
    df["user_skills"] = str(user_sk)

    def overlap(sk: str) -> str:
        j_set = set(parse_skills(str(sk)))
        inter = sorted(u_set.intersection(j_set))
        return ", ".join(inter)

    df["overlap_skills"] = df["job_skills"].apply(overlap)
    df.insert(0, "rank", range(1, len(df) + 1))

    out = pd.DataFrame(
        {
            "Rank": df["rank"],
            "Job ID": df["job_id"],
            "Title": df.get("title", ""),
            "Company": df.get("company_name", ""),
            "Location": df.get("location", ""),
            "User Skills": df["user_skills"].astype(str).apply(lambda x: _clip(x, 260)),
            "Job Skills": df["job_skills"].astype(str).apply(lambda x: _clip(x, 260)),
            "Overlap Skills": df["overlap_skills"].astype(str).apply(lambda x: _clip(x, 260)),
            "Shared Skills": df.get("shared_skills", 0).astype(int),
            "Co-Occur Sum": df.get("co_occurrence_sum", 0.0).astype(float).round(2),
            "Graph Raw": df.get("graph_raw", 0.0).astype(float).round(3),
            "Graph Score": df.get("graph_score", 0.0).astype(float).round(4),
        }
    )

    return "### 🕸️ Skill Graph Analysis", out


def popularity_analytics():
    ensure_backends_ready()
    a = analytics_popularity(25)
    top_jobs = pd.DataFrame(a.get("top_jobs", []))
    events = pd.DataFrame(a.get("recent_events", []))

    if top_jobs.empty:
        j = df_jobs()
        upsert_jobs(j)
        a = analytics_popularity(25)
        top_jobs = pd.DataFrame(a.get("top_jobs", []))

    if top_jobs.empty:
        top_jobs = _empty_pop_jobs_df()
    else:
        for c in ["job_id", "title", "company_name", "location", "skills", "description"]:
            if c not in top_jobs.columns:
                top_jobs[c] = ""
            top_jobs[c] = top_jobs[c].fillna("").astype(str)

        top_jobs = top_jobs.rename(
            columns={
                "job_id": "Job ID",
                "title": "Title",
                "company_name": "Company",
                "location": "Location",
                "views": "Views",
                "skills": "Skills",
                "description": "Description",
                "updated_at": "Updated",
                "created_at": "Created",
                "last_view_at": "Last View",
            }
        )
        top_jobs["Skills"] = top_jobs["Skills"].astype(str).apply(lambda x: _clip(x, 280))
        top_jobs["Description"] = top_jobs["Description"].astype(str).apply(lambda x: _clip(x, 260))
        keep = ["Job ID", "Title", "Company", "Location", "Views", "Skills", "Description", "Updated", "Created", "Last View"]
        top_jobs = top_jobs[[c for c in keep if c in top_jobs.columns]]

    if events.empty:
        events = _empty_events_df()
    else:
        if "payload" in events.columns:
            events["payload"] = events["payload"].apply(lambda x: x if isinstance(x, str) else json.dumps(x, ensure_ascii=False))
        if "ts" in events.columns:
            events["ts_readable"] = pd.to_datetime(events["ts"], unit="s", errors="coerce")
        events = events.rename(columns={"type": "Event Type", "ts": "Timestamp", "ts_readable": "Readable Time", "payload": "Details"})
        keep = ["Event Type", "Timestamp", "Readable Time", "Details"]
        events = events[[c for c in keep if c in events.columns]]

    return top_jobs, events


def explain_tab(user_label: str, mapping: Dict[str, str], top_k: int, mode: str):
    uid, err = resolve_user(user_label, mapping)
    if err:
        return f"### ❌ {err}", _empty_recs_df()

    rec = get_rec()
    recs = rec.recommend(uid, int(top_k))
    df = pd.DataFrame(recs)
    if df.empty:
        return "### ℹ️ No results.", _empty_recs_df()

    jobs = df_jobs().set_index("job_id", drop=False)
    df["job_skills"] = df["job_id"].astype(str).apply(lambda jid: jobs.loc[jid]["skills"] if jid in jobs.index else "")
    df["desc_snip"] = df["job_id"].astype(str).apply(lambda jid: _clip(jobs.loc[jid]["description"], 240) if jid in jobs.index else "")

    if mode == "Global":
        txt = explain_global(uid, recs)
        out = pd.DataFrame(
            {
                "Job ID": df["job_id"],
                "Title": df["title"],
                "Company": df["company_name"],
                "Location": df["location"],
                "Job Skills": df["job_skills"].astype(str).apply(lambda x: _clip(x, 260)),
                "Description": df["desc_snip"].astype(str),
                "Embedding Score": df["embedding_score"].astype(float).round(4),
                "Graph Score": df["graph_score"].astype(float).round(4),
                "Popularity": df["popularity_views"].astype(int),
                "Overlap Skills": df["overlap_skills"],
                "Hybrid Score": df["hybrid_score"].astype(float).round(4),
            }
        )
        return f"### 🤖 Global Explanation\n\n{txt}", out

    exps = explain_rag(uid, recs, df_jobs())
    df = df.copy()
    df["rag_explanation"] = exps

    out = pd.DataFrame(
        {
            "Job ID": df["job_id"],
            "Title": df["title"],
            "Company": df["company_name"],
            "Location": df["location"],
            "Job Skills": df["job_skills"].astype(str).apply(lambda x: _clip(x, 260)),
            "Description": df["desc_snip"].astype(str),
            "Embedding Score": df["embedding_score"].astype(float).round(4),
            "Graph Score": df["graph_score"].astype(float).round(4),
            "Popularity": df["popularity_views"].astype(int),
            "Overlap Skills": df["overlap_skills"],
            "Hybrid Score": df["hybrid_score"].astype(float).round(4),
            "rag_explanation": df["rag_explanation"].astype(str),
        }
    )
    return "### 🤖 AI Explanations", out


def pipeline_health():
    try:
        etl = ensure_ready()
    except Exception as e:
        etl = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    stores = {"mongo": mongo_status(), "neo4j": neo4j_status()}
    j = df_jobs()
    u = df_users()
    j_empty = int((j["skills"].astype(str).str.strip() == "").sum()) if "skills" in j.columns else -1
    u_empty = int((u["skills"].astype(str).str.strip() == "").sum()) if "skills" in u.columns else -1
    diag = {"jobs": int(len(j)), "users": int(len(u)), "jobs_empty_skills": j_empty, "users_empty_skills": u_empty}
    return pd.DataFrame([{"etl": etl, "stores": stores, "data_quality": diag}])


def run_etl_btn():
    _reset_runtime()
    started = time.time()
    try:
        r = run_etl()
        secs = round(time.time() - started, 3)
        jobs_n = r.get("jobs_rows", r.get("jobs", 0))
        users_n = r.get("users_rows", r.get("users", 0))
        ratio = r.get("jobs_empty_skills_ratio", None)
        ratio_s = f"{float(ratio):.3f}" if ratio is not None else "N/A"
        return (
            "### ✅ ETL Completed\n"
            f"- Jobs: **{jobs_n}**\n"
            f"- Users: **{users_n}**\n"
            f"- Empty skills ratio (jobs): **{ratio_s}**\n"
            f"- Saved: `{r.get('jobs_csv', '')}` and `{r.get('users_csv', '')}`\n"
            f"- Seconds: **{secs}**"
        )
    except Exception as e:
        return f"### ❌ ETL Failed\n`{type(e).__name__}: {e}`"


def run_emb_btn():
    _reset_runtime()
    started = time.time()
    try:
        r = compute_embeddings()
        secs = round(time.time() - started, 3)
        jobs_n = r.get("jobs", r.get("jobs_emb", 0))
        users_n = r.get("users", r.get("users_emb", 0))
        return (
            "### ✅ Embeddings Computed\n"
            f"- Jobs embeddings: **{jobs_n}**\n"
            f"- Users embeddings: **{users_n}**\n"
            f"- Output folder: `{DATA_PROCESSED}`\n"
            f"- Seconds: **{secs}**"
        )
    except Exception as e:
        return f"### ❌ Embeddings Failed\n`{type(e).__name__}: {e}`"


def mongo_sync_btn():
    _reset_runtime()
    j = df_jobs()
    n = upsert_jobs(j)
    return f"### ✅ Mongo Sync\nJobs upserted: **{n}**"


def neo4j_sync_btn():
    _reset_runtime()
    r = sync_neo4j_graph()
    if not (r or {}).get("ok", False):
        return f"### ❌ Neo4j Sync Failed\n`{(r or {}).get('error', 'unknown error')}`"
    pairs = ((r.get("co_occurs") or {}) if isinstance(r, dict) else {}).get("pairs", "")
    return f"### ✅ Neo4j Sync\nUsers: **{r.get('users')}** | Jobs: **{r.get('jobs')}** | CO_OCCURS pairs: **{pairs}**"


def main():
    ensure_ready()
    atexit.register(close_mongo)

    users = df_users()
    labels, mp = build_user_choices(users)
    initial_labels = labels[:MAX_USER_CHOICES]
    default_user = initial_labels[0] if initial_labels else None

    with gr.Blocks(title=APP_TITLE, css=CSS, theme=THEME) as demo:
        gr.Markdown(
            f"# {APP_TITLE}\n"
            "A compact analytics UI for hybrid recommendation, semantic search, graph-based insights (Neo4j), and popularity signals (MongoDB)."
        )

        mapping_state = gr.State(mp)

        with gr.Row():
            user_dd = gr.Dropdown(
                choices=initial_labels,
                value=default_user,
                label="Select User",
                scale=3,
                filterable=True,
            )
            refresh_btn = gr.Button("Refresh Users", scale=1)

        def _refresh(current_label: str | None):
            labels2, mp2, default2 = refresh_users_state()
            all_labels2 = list(labels2)
            val2 = current_label if (current_label in all_labels2) else (default2 if default2 in all_labels2 else (all_labels2[0] if all_labels2 else None))
            return gr.update(choices=all_labels2, value=val2), mp2

        refresh_btn.click(_refresh, inputs=[user_dd], outputs=[user_dd, mapping_state])

        def _on_user_focus(mapping: Dict[str, str], current_label: str | None):
            labels_all = list((mapping or {}).keys())
            if not labels_all:
                return gr.update()
            val = current_label if (current_label in labels_all) else labels_all[0]
            return gr.update(choices=labels_all, value=val)

        user_dd.focus(_on_user_focus, inputs=[mapping_state, user_dd], outputs=[user_dd])

        with gr.Tabs():
            with gr.Tab("🎯 Hybrid Recommendations"):
                with gr.Row():
                    topk = gr.Slider(5, 50, value=TOP_K_DEFAULT, step=1, label="Top K")
                with gr.Row():
                    a = gr.Slider(0, 1, value=0.62, step=0.01, label="α Semantic")
                    b = gr.Slider(0, 1, value=0.23, step=0.01, label="β Graph")
                    g = gr.Slider(0, 1, value=0.06, step=0.01, label="γ Popularity")
                    d = gr.Slider(0, 1, value=0.09, step=0.01, label="δ Overlap")

                msg = gr.HTML()
                table = gr.Dataframe(value=_empty_recs_df(), wrap=False, elem_classes=["df-big"])
                recommend_btn = gr.Button("Generate Recommendations")
                recommend_btn.click(overview, inputs=[user_dd, mapping_state, topk, a, b, g, d], outputs=[msg, table])

            with gr.Tab("🔍 Semantic Search"):
                q = gr.Textbox(label="Search jobs/users by text", placeholder="e.g., software, pytorch, data analyst, milan", lines=2)
                out = gr.Dataframe(value=_empty_search_df(), wrap=False, elem_classes=["df-big"])
                search_btn = gr.Button("Search")
                search_btn.click(semantic_search, inputs=[q, topk], outputs=[out])

            with gr.Tab("🕸️ Skill Graph Analysis (Neo4j)"):
                msg2 = gr.Markdown()
                t2 = gr.Dataframe(value=_empty_graph_df(), wrap=False, elem_classes=["df-big"])
                graph_btn = gr.Button("Show Graph Insights")
                graph_btn.click(skill_graph_insights, inputs=[user_dd, mapping_state, topk], outputs=[msg2, t2])

            with gr.Tab("📈 Popularity Analytics"):
                with gr.Row():
                    top_jobs = gr.Dataframe(value=_empty_pop_jobs_df(), wrap=False, elem_classes=["df-mid"])
                    events = gr.Dataframe(value=_empty_events_df(), wrap=False, elem_classes=["df-mid"])
                analytics_btn = gr.Button("Refresh Analytics")
                analytics_btn.click(popularity_analytics, outputs=[top_jobs, events])

            with gr.Tab("🤖 AI Explanations"):
                mode = gr.Radio(["Per-job RAG", "Global"], value="Per-job RAG", label="Explanation Mode")
                msg3 = gr.Markdown()
                t3 = gr.Dataframe(value=_empty_recs_df(), wrap=False, elem_classes=["df-big"])
                explain_btn = gr.Button("Generate Explanations")
                explain_btn.click(explain_tab, inputs=[user_dd, mapping_state, topk, mode], outputs=[msg3, t3])

            with gr.Tab("👁️ Job Viewer"):
                gr.Markdown("Browse jobs, open details, and record a view (updates Mongo popularity).")

                with gr.Row():
                    q_view = gr.Textbox(label="Search jobs to view", placeholder="Search by title, company, location, skills, description...", scale=5)
                    btn_search_view = gr.Button("Search", scale=1)
                    btn_reset_view = gr.Button("Reset", scale=1)

                browser_state = gr.State(_jobs_for_browser("", 30))

                jobs_table = gr.Dataframe(value=_jobs_for_browser("", 30), wrap=False, elem_classes=["df-big"])
                selected_job_id = gr.Textbox(label="Selected Job ID", interactive=False)
                details_md = gr.Markdown(value="### Select a job to see details.")

                btn_view = gr.Button("View / Record View", variant="primary")

                def _search_jobs(qs: str):
                    dfb = _jobs_for_browser(qs, 30)
                    return dfb, dfb

                btn_search_view.click(fn=_search_jobs, inputs=[q_view], outputs=[jobs_table, browser_state])
                btn_reset_view.click(
                    fn=lambda: (_jobs_for_browser("", 30), _jobs_for_browser("", 30), "", "### Select a job to see details."),
                    outputs=[jobs_table, browser_state, selected_job_id, details_md],
                )

                jobs_table.select(fn=_on_job_row_select, inputs=[browser_state], outputs=[selected_job_id, details_md])
                btn_view.click(fn=_record_view_and_show, inputs=[user_dd, mapping_state, selected_job_id], outputs=[details_md])

            with gr.Tab("🏥 Pipeline & Health"):
                h = gr.Dataframe(value=pipeline_health(), wrap=False, elem_classes=["df-small"])
                health_btn = gr.Button("Refresh Health")
                health_btn.click(lambda: pipeline_health(), outputs=[h])

                outp = gr.Markdown()
                with gr.Row():
                    etl_btn = gr.Button("Run ETL")
                    emb_btn = gr.Button("Compute Embeddings")
                    mongo_btn = gr.Button("Sync Mongo")
                    neo_btn = gr.Button("Sync Neo4j")

                etl_btn.click(lambda: run_etl_btn(), outputs=[outp])
                emb_btn.click(lambda: run_emb_btn(), outputs=[outp])
                mongo_btn.click(lambda: mongo_sync_btn(), outputs=[outp])
                neo_btn.click(lambda: neo4j_sync_btn(), outputs=[outp])

            with gr.Tab("📚 Data Browser"):
                u_tab = gr.Dataframe(value=df_users(), wrap=False, elem_classes=["df-big"])
                j_tab = gr.Dataframe(value=df_jobs(), wrap=False, elem_classes=["df-big"])
                reload_btn = gr.Button("Reload Data")
                reload_btn.click(lambda: (df_users(), df_jobs()), outputs=[u_tab, j_tab])

    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
