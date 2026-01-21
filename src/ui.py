from __future__ import annotations

import atexit
import json
from collections import Counter
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd

from src.config import APP_TITLE, TOP_K_DEFAULT, PROMETHEUS_ENABLED, PROMETHEUS_PORT
from src.core import HybridRecommender, explain_global, explain_rag, record_view
from src.features import ensure_ready, load_jobs_df, load_users_df, compute_embeddings, run_etl
from src.stores import (
    analytics_popularity,
    close_mongo,
    ensure_backends_ready,
    mongo_status,
    neo4j_status,
    sync_neo4j_graph,
    upsert_jobs,
    get_job_popularity,

)

try:
    if PROMETHEUS_ENABLED:
        from prometheus_client import start_http_server

        start_http_server(PROMETHEUS_PORT)
except Exception:
    pass


CSS = """/* =======================================================================
   FIX: Remove Gradio max-width constraints that cause big left/right margins
   ======================================================================= */
html, body, #root {
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}

.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  box-sizing: border-box !important;

  /* Gradio layout variables that often enforce centered/narrow content */
  --layout-max-width: 100% !important;
  --prose-max-width: 100% !important;
  --block-max-width: 100% !important;
}

/* Common wrappers used by Gradio to clamp width */
.gradio-container .contain,
.gradio-container .wrap,
.gradio-container .container,
.gradio-container .main,
.gradio-container .prose,
.gradio-container .content,
.gradio-container .app {
  max-width: 100% !important;
  width: 100% !important;
  margin-left: 0 !important;
  margin-right: 0 !important;
  box-sizing: border-box !important;
}

/* Small comfortable side gutter (tune: 6px / 8px / 10px) */
.gradio-container .wrap,
.gradio-container .container,
.gradio-container .main,
.gradio-container .prose,
.gradio-container .content {
  padding-left: 8px !important;
  padding-right: 8px !important;
}

/* Main content area (reduce if you want even tighter) */
.gradio-container > .main {
  padding: 12px 8px !important;
}

/* =======================================================================
   Your existing styling (kept)
   ======================================================================= */

/* Header styling */
h1 {
  font-size: 2rem !important;
  font-weight: 700 !important;
  margin-bottom: 8px !important;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
  background-clip: text !important;
}

/* Subtitle styling */
.gradio-container p {
  color: #94a3b8 !important;
  margin-bottom: 24px !important;
}

/* Tab styling */
.tabs {
  border-bottom: 2px solid #334155 !important;
  margin-bottom: 24px !important;
}

.tab-nav button {
  font-weight: 600 !important;
  padding: 12px 20px !important;
  color: #94a3b8 !important;
  border: none !important;
  border-bottom: 3px solid transparent !important;
  transition: all 0.3s ease !important;
}

.tab-nav button:hover {
  color: #e2e8f0 !important;
  background: rgba(100, 116, 139, 0.1) !important;
}

.tab-nav button.selected {
  color: #667eea !important;
  border-bottom-color: #667eea !important;
  background: rgba(102, 126, 234, 0.1) !important;
}

/* Button styling */
.gr-button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
  padding: 12px 24px !important;
  border-radius: 8px !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3) !important;
}

.gr-button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4) !important;
}

.gr-button:active {
  transform: translateY(0) !important;
}

/* Input styling */
.gr-box, .gr-input, .gr-dropdown {
  background: #1e293b !important;
  border: 1px solid #334155 !important;
  border-radius: 8px !important;
  color: #e2e8f0 !important;
  transition: all 0.3s ease !important;
}

.gr-box:focus, .gr-input:focus, .gr-dropdown:focus {
  border-color: #667eea !important;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
}

/* Slider styling */
.gr-slider input[type="range"] {
  background: linear-gradient(to right, #667eea 0%, #764ba2 100%) !important;
  height: 6px !important;
  border-radius: 3px !important;
}

.gr-slider input[type="range"]::-webkit-slider-thumb {
  background: #667eea !important;
  border: 3px solid white !important;
  box-shadow: 0 2px 8px rgba(102, 126, 234, 0.5) !important;
  width: 20px !important;
  height: 20px !important;
}

/* Checkbox styling */
.gr-checkbox {
  accent-color: #667eea !important;
}

.gr-checkbox-label {
  color: #e2e8f0 !important;
  font-weight: 500 !important;
}

/* Label styling */
label {
  color: #cbd5e1 !important;
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  margin-bottom: 8px !important;
}

/* Dataframe styling */
.gradio-dataframe {
  width: 100% !important;
  overflow: auto !important;
  border-radius: 8px !important;
  border: 1px solid #334155 !important;
  background: #1e293b !important;
}

.gradio-dataframe table {
  table-layout: auto !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
}

.gradio-dataframe td,
.gradio-dataframe th {
  white-space: nowrap !important;
  padding: 12px 16px !important;
  border-bottom: 1px solid #334155 !important;
  color: #e2e8f0 !important;
}

.gradio-dataframe thead th {
  position: sticky !important;
  top: 0 !important;
  z-index: 2 !important;
  background: #0f172a !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.05em !important;
  color: #94a3b8 !important;
  border-bottom: 2px solid #667eea !important;
}

.gradio-dataframe tbody tr:hover {
  background: rgba(102, 126, 234, 0.1) !important;
}

/* Heights for different dataframe sizes */
.df-big .gradio-dataframe {
  height: 72vh !important;
  max-height: 72vh !important;
}

.df-mid .gradio-dataframe {
  height: 52vh !important;
  max-height: 52vh !important;
}

.df-small .gradio-dataframe {
  height: 38vh !important;
  max-height: 38vh !important;
}

/* Markdown styling */
.markdown-text {
  color: #e2e8f0 !important;
  line-height: 1.7 !important;
}

.markdown-text h3 {
  color: #667eea !important;
  font-weight: 700 !important;
  margin-top: 16px !important;
  margin-bottom: 12px !important;
}

.markdown-text strong {
  color: #e2e8f0 !important;
  font-weight: 700 !important;
}

/* Row styling */
.gr-row {
  gap: 16px !important;
}

/* Card-like appearance for sections */
.gr-panel {
  background: #1e293b !important;
  border-radius: 12px !important;
  padding: 20px !important;
  border: 1px solid #334155 !important;
  margin-bottom: 16px !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 10px !important;
  height: 10px !important;
}

::-webkit-scrollbar-track {
  background: #0f172a !important;
  border-radius: 5px !important;
}

::-webkit-scrollbar-thumb {
  background: #334155 !important;
  border-radius: 5px !important;
}

::-webkit-scrollbar-thumb:hover {
  background: #475569 !important;
}

/* Radio button styling */
.gr-radio label {
  color: #e2e8f0 !important;
  padding: 8px 16px !important;
  border-radius: 6px !important;
  transition: all 0.2s ease !important;
}

.gr-radio label:hover {
  background: rgba(102, 126, 234, 0.1) !important;
}

.gr-radio input:checked + label {
  background: rgba(102, 126, 234, 0.2) !important;
  color: #667eea !important;
  font-weight: 600 !important;
}

/* Responsive padding adjustments */
@media (max-width: 768px) {
  .gradio-container > .main {
    padding: 10px 8px !important;
  }

  .gr-button {
    padding: 10px 16px !important;
    font-size: 0.875rem !important;
  }
}

"""

REC: HybridRecommender | None = None


def _empty_job_browser_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["👁", "Job ID", "Title", "Company", "Location", "Skills"])


def _jobs_for_browser(query: str = "", limit: int = 30) -> pd.DataFrame:
    jobs = df_jobs().copy()
    for c in ["job_id", "title", "company_name", "location", "skills", "description"]:
        if c not in jobs.columns:
            jobs[c] = ""
        jobs[c] = jobs[c].fillna("").astype(str)

    q = (query or "").strip().lower()

    if q:
        # simple search for browsing (NOT affecting your recommender)
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

    # current views from Mongo (even before recording a new view)
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
    # evt.index is typically (row, col). We only care about row.
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

    # this increments Mongo views + logs event(type="view")
    _ = record_view(uid, jid)  # returns updated views count :contentReference[oaicite:5]{index=5}
    return _format_job_details(jid, uid=uid, just_viewed=True)


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
            "Job ID",
            "Title",
            "Company",
            "Location",
            "User Skills",
            "Job Skills",
            "Overlap Skills",
            "Graph Score",
            "Embedding Score",
            "Popularity",
            "Hybrid Score",
        ]
    )


def _empty_pop_jobs_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Job ID", "Title", "Company", "Location", "Views", "Skills", "Description", "Updated", "Created", "Last View"])


def _empty_events_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Event Type", "Timestamp", "Readable Time", "Details"])


def overview(
    user_label: str,
    mapping: Dict[str, str],
    top_k: int,
    a: float,
    b: float,
    g: float,
    d: float,
):
    uid, err = resolve_user(user_label, mapping)
    if err:
        return f"### ❌ {err}", _empty_recs_df()

    rec = get_rec()

    # Normalize weights to sum to 1 (expert-level stable mixing)
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
        f"### ✅ Recommendations\n"
        f"- Avg Embedding: **{df['embedding_score'].astype(float).mean():.3f}**\n"
        f"- Avg Graph: **{df['graph_score'].astype(float).mean():.3f}**\n"
        f"- Avg Hybrid: **{df['hybrid_score'].astype(float).mean():.3f}**\n"
        f"- Top Locations: **{loc_str}**\n"
        f"- Top Overlap Skills: **{sk_str}**"
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

    users = df_users()
    jobs = df_jobs()

    m_user = (
        users["name"].astype(str).str.lower().str.contains(q.lower(), na=False)
        | users["skills"].astype(str).str.lower().str.contains(q.lower(), na=False)
    )
    users_hits = users[m_user].head(int(top_k))

    rec = get_rec()
    job_hits = rec.semantic_search(q, int(top_k))

    rows: List[Dict[str, Any]] = []

    for _, r in users_hits.iterrows():
        rows.append(
            {
                "Type": "User",
                "ID": str(r.get("user_id", "")),
                "Score": "",
                "Title/Name": str(r.get("name", "")),
                "Company": "",
                "Location/Preferred": str(r.get("preferred_location", "")),
                "Skills": _clip(r.get("skills", ""), 300),
                "Details": _clip(r.get("summary", ""), 280),
            }
        )

    jobs_idx = jobs.set_index("job_id", drop=False)
    for r in job_hits:
        jid = str(r.get("job_id", ""))
        if jid in jobs_idx.index:
            jr = jobs_idx.loc[jid]
            skills = jr.get("skills", "")
            desc = jr.get("description", "")
            title = jr.get("title", r.get("title", ""))
            comp = jr.get("company_name", r.get("company_name", ""))
            loc = jr.get("location", r.get("location", ""))
        else:
            skills = r.get("skills", "")
            desc = ""
            title = r.get("title", "")
            comp = r.get("company_name", "")
            loc = r.get("location", "")
        rows.append(
            {
                "Type": "Job",
                "ID": jid,
                "Score": float(r.get("score", 0.0)),
                "Title/Name": str(title),
                "Company": str(comp),
                "Location/Preferred": str(loc),
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
    recs = rec.recommend(uid, int(top_k))
    df = pd.DataFrame(recs)
    if df.empty:
        return "### ℹ️ No results.", _empty_graph_df()

    users = df_users().set_index("user_id", drop=False)
    jobs = df_jobs().set_index("job_id", drop=False)
    user_sk = users.loc[uid]["skills"] if uid in users.index else ""

    df["user_skills"] = user_sk
    df["job_skills"] = df["job_id"].astype(str).apply(lambda jid: jobs.loc[jid]["skills"] if jid in jobs.index else "")

    out = pd.DataFrame(
        {
            "Job ID": df["job_id"],
            "Title": df["title"],
            "Company": df["company_name"],
            "Location": df["location"],
            "User Skills": df["user_skills"].astype(str).apply(lambda x: _clip(x, 260)),
            "Job Skills": df["job_skills"].astype(str).apply(lambda x: _clip(x, 260)),
            "Overlap Skills": df["overlap_skills"],
            "Graph Score": df["graph_score"].astype(float).round(4),
            "Embedding Score": df["embedding_score"].astype(float).round(4),
            "Popularity": df["popularity_views"].astype(int),
            "Hybrid Score": df["hybrid_score"].astype(float).round(4),
        }
    )
    return "### 🔗 Skill Graph Analysis (Neo4j)", out


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

        events = events.rename(
            columns={
                "type": "Event Type",
                "ts": "Timestamp",
                "ts_readable": "Readable Time",
                "payload": "Details",
            }
        )
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


def record_view_ui(user_label: str, mapping: Dict[str, str], job_id: str):
    uid, err = resolve_user(user_label, mapping)
    if err:
        return f"### ❌ {err}"
    if not str(job_id).strip():
        return "### ⚠️ Enter Job ID."
    v = record_view(uid, str(job_id).strip())
    return f"### ✅ View Recorded\n\nJob ID: **{job_id}**\nViews: **{v}**\nUser: **{uid}**"


def pipeline_health():
    etl = ensure_ready()
    stores = {"mongo": mongo_status(), "neo4j": neo4j_status()}
    j = df_jobs()
    u = df_users()
    j_empty = int((j["skills"].astype(str).str.strip() == "").sum()) if "skills" in j.columns else -1
    u_empty = int((u["skills"].astype(str).str.strip() == "").sum()) if "skills" in u.columns else -1
    diag = {"jobs": int(len(j)), "users": int(len(u)), "jobs_empty_skills": j_empty, "users_empty_skills": u_empty}
    return pd.DataFrame([{"etl": etl, "stores": stores, "data_quality": diag}])


def run_etl_btn():
    r = run_etl()
    return f"### ✅ ETL Completed\nJobs: **{r.get('jobs')}** | Users: **{r.get('users')}** | Seconds: **{r.get('seconds')}**"


def run_emb_btn():
    r = compute_embeddings()
    return f"### ✅ Embeddings Computed\nJobsEmb: **{r.get('jobs_emb')}** | UsersEmb: **{r.get('users_emb')}** | Seconds: **{r.get('seconds')}**"


def mongo_sync_btn():
    j = df_jobs()
    n = upsert_jobs(j)
    return f"### ✅ Mongo Sync\nJobs upserted: **{n}**"


def neo4j_sync_btn():
    r = sync_neo4j_graph()
    return f"### ✅ Neo4j Sync\nUsers: **{r.get('users')}** | Jobs: **{r.get('jobs')}** | CO_OCCURS: **{(r.get('co_occurs') or {}).get('pairs')}**"


def main():
    ensure_ready()
    atexit.register(close_mongo)

    users = df_users()
    labels, mp = build_user_choices(users)
    default_user = labels[0] if labels else None

    with gr.Blocks(
        title=APP_TITLE,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
    ) as demo:
        gr.Markdown(f"# {APP_TITLE}\nEmbeddings + Neo4j Graph + Mongo Popularity")

        mapping_state = gr.State(mp)

        with gr.Row():
            user_dd = gr.Dropdown(choices=labels, value=default_user, label="Select User")
            refresh_btn = gr.Button("🔄 Refresh Users", scale=0)

        def _refresh():
            labels2, mp2, default2 = refresh_users_state()
            return gr.update(choices=labels2, value=default2), mp2

        refresh_btn.click(_refresh, outputs=[user_dd, mapping_state])

        with gr.Tabs():
            with gr.Tab("🎯 Hybrid Recommendations"):
                with gr.Row():
                    topk = gr.Slider(5, 50, value=TOP_K_DEFAULT, step=1, label="Top K")

                    a = gr.Slider(0, 1, value=0.62, step=0.01, label="α Semantic")
                    b = gr.Slider(0, 1, value=0.23, step=0.01, label="β Graph")
                    g = gr.Slider(0, 1, value=0.06, step=0.01, label="γ Popularity")
                    d = gr.Slider(0, 1, value=0.09, step=0.01, label="δ Overlap")

                msg = gr.Markdown()
                table = gr.Dataframe(value=_empty_recs_df(), wrap=False, elem_classes=["df-big"])
                recommend_btn = gr.Button("🚀 Generate Recommendations")
                recommend_btn.click(
                    overview,
                    inputs=[user_dd, mapping_state, topk, a, b, g, d],
                    outputs=[msg, table],
                )

            with gr.Tab("🔍 Semantic Search"):
                q = gr.Textbox(label="Search jobs/users by text", placeholder="e.g., software, pytorch, data analyst, milan", lines=2)
                out = gr.Dataframe(value=_empty_search_df(), wrap=False, elem_classes=["df-big"])
                search_btn = gr.Button("🔎 Search")
                search_btn.click(semantic_search, inputs=[q, topk], outputs=[out])

            with gr.Tab("🕸️ Skill Graph Analysis (Neo4j)"):
                msg2 = gr.Markdown()
                t2 = gr.Dataframe(value=_empty_graph_df(), wrap=False, elem_classes=["df-big"])
                graph_btn = gr.Button("📊 Show Graph Insights")
                graph_btn.click(skill_graph_insights, inputs=[user_dd, mapping_state, topk], outputs=[msg2, t2])

            with gr.Tab("📈 Popularity Analytics"):
                with gr.Row():
                    top_jobs = gr.Dataframe(value=_empty_pop_jobs_df(), wrap=False, elem_classes=["df-mid"])
                    events = gr.Dataframe(value=_empty_events_df(), wrap=False, elem_classes=["df-mid"])
                analytics_btn = gr.Button("🔄 Refresh Analytics")
                analytics_btn.click(popularity_analytics, outputs=[top_jobs, events])

            with gr.Tab("🤖 AI Explanations"):
                mode = gr.Radio(["Per-job RAG", "Global"], value="Per-job RAG", label="Explanation Mode")
                msg3 = gr.Markdown()
                t3 = gr.Dataframe(value=_empty_recs_df(), wrap=False, elem_classes=["df-big"])
                explain_btn = gr.Button("💡 Generate Explanations")
                explain_btn.click(explain_tab, inputs=[user_dd, mapping_state, topk, mode], outputs=[msg3, t3])

            with gr.Tab("👁️ Job Viewer"):
                gr.Markdown("Browse jobs, open details, and record a view (updates Mongo popularity).")

                with gr.Row():
                    q_view = gr.Textbox(
                        label="Search jobs to view",
                        placeholder="Search by title, company, location, skills, description...",
                        scale=5,
                    )
                    btn_search_view = gr.Button("🔎 Search", scale=1)
                    btn_reset_view = gr.Button("↩ Reset", scale=1)

                browser_state = gr.State(_jobs_for_browser("", 30))

                jobs_table = gr.Dataframe(value=_jobs_for_browser("", 30), wrap=False, elem_classes=["df-big"])
                selected_job_id = gr.Textbox(label="Selected Job ID", interactive=False)
                details_md = gr.Markdown(value="### Select a job to see details.")

                with gr.Row():
                    btn_view = gr.Button("👁 View / Record View", variant="primary")

                # search / reset
                btn_search_view.click(
                    fn=lambda q: (_jobs_for_browser(q, 30), _jobs_for_browser(q, 30)),
                    inputs=[q_view],
                    outputs=[jobs_table, browser_state],
                )
                btn_reset_view.click(
                    fn=lambda: (_jobs_for_browser("", 30), _jobs_for_browser("", 30), "", "### Select a job to see details."),
                    outputs=[jobs_table, browser_state, selected_job_id, details_md],
                )

                # select row -> show details (no increment yet)
                jobs_table.select(
                    fn=_on_job_row_select,
                    inputs=[browser_state],
                    outputs=[selected_job_id, details_md],
                )

                # click view -> increment views + show details
                btn_view.click(
                    fn=_record_view_and_show,
                    inputs=[user_dd, mapping_state, selected_job_id],
                    outputs=[details_md],
                )


            with gr.Tab("🏥 Pipeline & Health"):
                h = gr.Dataframe(value=pipeline_health(), wrap=False, elem_classes=["df-small"])
                health_btn = gr.Button("🔄 Refresh Health")
                health_btn.click(lambda: pipeline_health(), outputs=[h])

                outp = gr.Markdown()
                with gr.Row():
                    etl_btn = gr.Button("⚙️ Run ETL")
                    emb_btn = gr.Button("🧮 Compute Embeddings")
                    mongo_btn = gr.Button("🍃 Sync Mongo")
                    neo_btn = gr.Button("🔷 Sync Neo4j")

                etl_btn.click(lambda: run_etl_btn(), outputs=[outp])
                emb_btn.click(lambda: run_emb_btn(), outputs=[outp])
                mongo_btn.click(lambda: mongo_sync_btn(), outputs=[outp])
                neo_btn.click(lambda: neo4j_sync_btn(), outputs=[outp])

            with gr.Tab("📚 Data Browser"):
                u_tab = gr.Dataframe(value=df_users(), wrap=False, elem_classes=["df-big"])
                j_tab = gr.Dataframe(value=df_jobs(), wrap=False, elem_classes=["df-big"])
                reload_btn = gr.Button("🔄 Reload Data")
                reload_btn.click(lambda: (df_users(), df_jobs()), outputs=[u_tab, j_tab])

    demo.launch(server_name="0.0.0.0", share=False, show_error=True)


if __name__ == "__main__":
    main()
