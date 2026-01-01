from __future__ import annotations
from collections import Counter
from typing import Any, Dict, List, Tuple
import gradio as gr
import numpy as np
import pandas as pd

from src.config import (
    USERS_CSV,
    JOBS_CSV,
    JOB_EMB_NPY,
    USER_EMB_NPY,
    TOP_K_DEFAULT,
)
from src.features.preprocessing import load_users_df, load_jobs_df

def _safe_load_users() -> pd.DataFrame:
    try:
        df = load_users_df()
    except Exception:
        try:
            df = pd.read_csv(USERS_CSV)
        except Exception:
            df = pd.DataFrame()

    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype(str)
    return df

def _safe_load_jobs() -> pd.DataFrame:
    try:
        df = load_jobs_df()
    except Exception:
        try:
            df = pd.read_csv(JOBS_CSV)
        except Exception:
            df = pd.DataFrame()

    if "job_id" in df.columns:
        df["job_id"] = df["job_id"].astype(str)
    return df

USERS_DF = _safe_load_users()
JOBS_DF = _safe_load_jobs()

def _build_user_choices(df: pd.DataFrame) -> Tuple[List[str], Dict[str, str]]:
    labels: List[str] = []
    mapping: Dict[str, str] = {}
    if df.empty or "user_id" not in df.columns:
        return [], {}
    has_name = "name" in df.columns
    for _, row in df.iterrows():
        uid = str(row["user_id"])
        if has_name and str(row["name"]).strip():
            label = f"{row['name']} ({uid})"
        else:
            label = uid
        labels.append(label)
        mapping[label] = uid

    return labels, mapping

USER_LABELS, USER_LABEL_TO_ID = _build_user_choices(USERS_DF)
DEFAULT_USER_LABEL = USER_LABELS[0] if USER_LABELS else None

# ======================================================================
#  Column schemas – embedding + graph + overlap
# ======================================================================

OVERVIEW_COLS = [
    "rank",
    "job_id",
    "title",
    "company_name",
    "location",
    "embedding_score",
    "graph_score",
    "hybrid_score",
    "overlap_skills",
]

NOSQL_USER_COLS = [
    "rank_popularity",
    "job_id",
    "title",
    "company_name",
    "location",
    "views",
    "overlap_skills",
]

EXPLAIN_COLS = [
    "rank",
    "job_id",
    "title",
    "company_name",
    "location",
    "hybrid_score",
    "overlap_skills",
    "rag_explanation",
]

GRAPH_COLS = [
    "rank_graph",
    "job_id",
    "title",
    "company_name",
    "location",
    "graph_score",
    "overlap_skills",
]

SEMANTIC_COLS = [
    "rank_semantic",
    "job_id",
    "title",
    "company_name",
    "location",
    "embedding_score",
    "overlap_skills",
]

# ======================================================================
#  Recommender access (lazy) & helpers
# ======================================================================

_REC = None           # HybridRecommender instance
_REC_ERROR: str | None = None

def _get_recommender():
    """Lazy initialise HybridRecommender; never crash the UI."""
    global _REC, _REC_ERROR
    if _REC is not None or _REC_ERROR is not None:
        return _REC
    try:
        from src.recommender.core import HybridRecommender
        _REC = HybridRecommender()
        return _REC
    except Exception as exc:
        _REC_ERROR = f"Failed to initialise HybridRecommender: {exc}"
        print(_REC_ERROR)
        return None

def _ensure_user_id(user_label: str) -> Tuple[str | None, str | None]:
    if not user_label:
        return None, "Please select a user."
    uid = USER_LABEL_TO_ID.get(user_label)
    if not uid:
        return None, f"Unknown user selection: {user_label}."
    return uid, None

def _resolve_rec_and_user(user_label: str):
    """
    Helper: resolve user_id and HybridRecommender instance,
    return (rec, user_id, error_message_or_None).
    """
    uid, err = _ensure_user_id(user_label)
    if err:
        return None, None, f"❗ {err}"
    rec = _get_recommender()
    if rec is None:
        return None, None, _REC_ERROR or "❗ Recommender not available. Check backend configuration."
    return rec, uid, None

def _to_dataframe(recs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list[dict] to DataFrame and ensure core columns exist
    with reasonable types. This is purely a view/formatting helper.
    """
    df = pd.DataFrame(recs)
    # Ensure expected columns exist
    for col in [
        "user_id",
        "job_id",
        "title",
        "company_name",
        "location",
        "embedding_score",
        "graph_score",
        "hybrid_score",
        "popularity_views",
        "overlap_skills",
    ]:
        if col not in df.columns:
            df[col] = None
    if "job_id" in df.columns:
        df["job_id"] = df["job_id"].astype(str)
    for col in ["embedding_score", "graph_score", "hybrid_score"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "popularity_views" in df.columns:
        df["popularity_views"] = df["popularity_views"].fillna(0).astype(int)

    # overlap_skills is already a string from core, but we normalise as fallback
    def _ov(x):
        if isinstance(x, (list, set, tuple)):
            s = ", ".join(sorted(str(s) for s in x))
            return s if s else "—"
        if x is None or str(x).strip() == "":
            return "—"
        return str(x)

    df["overlap_skills"] = df["overlap_skills"].apply(_ov)

    return df

# ======================================================================
#  Tab: Overview – combined embedding + graph summary
# ======================================================================

def tab_overview(user_label: str, top_k: int):
    """
    Overview tab:
      - calls core.recommend_overview (logic),
      - builds Markdown summary and table (view).
    """
    rec, uid, err = _resolve_rec_and_user(user_label)
    if err:
        return err, pd.DataFrame(columns=OVERVIEW_COLS)
    try:
        recs = rec.recommend_overview(uid, top_k)
    except Exception as exc:
        msg = f"❗ Error while generating overview: {exc}"
        return msg, pd.DataFrame(columns=OVERVIEW_COLS)
    if not recs:
        msg = (
            "⚠️ No reasonable matches (non-zero embedding/graph or overlapping skills) "
            "were found for this user."
        )
        return msg, pd.DataFrame(columns=OVERVIEW_COLS)

    df = _to_dataframe(recs)
    df = df.reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    n_jobs = len(df)
    mean_e = df["embedding_score"].mean()
    mean_g = df["graph_score"].mean()
    mean_h = df["hybrid_score"].mean()

    loc_counts = df["location"].value_counts().head(5)
    loc_str = ", ".join([f"{loc} ({cnt})" for loc, cnt in loc_counts.items()]) or "N/A"

    all_skills: List[str] = []
    for val in df["overlap_skills"]:
        for s in str(val).split(","):
            s = s.strip()
            if s and s != "—":
                all_skills.append(s)
    skill_counts = Counter(all_skills)
    skills_str = ", ".join([f"{sk} ({cnt})" for sk, cnt in skill_counts.most_common(8)]) or "N/A"

    summary = (
        f"### 📊 Comprehensive Job Matching Overview\n\n"
        f"**Match Statistics:**\n"
        f"- 🎯 Displaying top **{n_jobs}** jobs ranked by intelligent hybrid algorithm\n"
        f"- 🧠 Average embedding score: **{mean_e:.3f}** (semantic similarity)\n"
        f"- 🕸️ Average graph score: **{mean_g:.3f}** (skill overlap)\n"
        f"- ⚡ Average hybrid score: **{mean_h:.3f}** (combined intelligence)\n\n"
        f"**Geographic Distribution:**\n"
        f"- 📍 {loc_str}\n\n"
        f"**Skills Analysis:**\n"
        f"- 💼 {skills_str}\n"
    )
    return summary, df[OVERVIEW_COLS]

# ======================================================================
#  Tab: NoSQL – popularity (user-centric only)
# ======================================================================

def tab_popularity_user(user_label: str, top_k: int):
    """
    For a given user, call core.recommend_with_popularity and display
    how many times each recommended job was viewed in MongoDB.
    """
    rec, uid, err = _resolve_rec_and_user(user_label)
    if err:
        return pd.DataFrame(columns=NOSQL_USER_COLS), err
    try:
        recs = rec.recommend_with_popularity(uid, top_k)
    except Exception as exc:
        msg = f"❗ Error while generating popularity ranking: {exc}"
        return pd.DataFrame(columns=NOSQL_USER_COLS), msg
    if not recs:
        msg = "⚠️ No recommendations available to compute popularity for this user."
        return pd.DataFrame(columns=NOSQL_USER_COLS), msg
    df = _to_dataframe(recs)

    # Map popularity_views -> views (UI column)
    df["views"] = df["popularity_views"].fillna(0).astype(int)
    df = df.sort_values("views", ascending=False).reset_index(drop=True)
    df.insert(0, "rank_popularity", df.index + 1)
    return df[NOSQL_USER_COLS], ""

# ======================================================================
#  Tab: RAG & LLM – explanations based on hybrid score + skills
# ======================================================================

def _rag_reason(row: pd.Series) -> str:
    """
    Multi-sentence explanation based on:
      - hybrid_score (overall match strength)
      - embedding_score
      - graph_score
      - overlap_skills
    The value in parenthesis is explicitly the hybrid score.
    """
    score = float(row.get("hybrid_score", 0.0))
    emb = float(row.get("embedding_score", 0.0))
    g = float(row.get("graph_score", 0.0))
    title = str(row.get("title", "this role"))
    location = str(row.get("location", "")).strip()
    ov_raw = str(row.get("overlap_skills", "—"))
    # 1) Match strength label
    if score >= 0.65:
        strength = "🌟 Excellent match"
    elif score >= 0.55:
        strength = "✨ Strong match"
    elif score >= 0.45:
        strength = "👍 Good match"
    else:
        strength = "💡 Reasonable match"
    # 2) Skill phrase
    skills_list = [
        s.strip()
        for s in ov_raw.split(",")
        if s.strip() and s.strip() != "—"
    ]
    if skills_list:
        if len(skills_list) == 1:
            skills_text = f"it closely matches your skill in **{skills_list[0]}**"
        elif len(skills_list) == 2:
            skills_text = (
                f"it aligns well with your skills in **{skills_list[0]}** and **{skills_list[1]}**"
            )
        else:
            examples = ", ".join(skills_list[:3])
            skills_text = f"it covers several of your key skills (for example **{examples}**)"
    else:
        skills_text = (
            "it fits your overall experience even though no explicit overlapping skills "
            "were extracted"
        )
    loc_part = f" in **{location}**" if location else ""
    # 3) Final explanation (2–3 sentences)
    return (
        f"{strength} (hybrid score = {score:.3f}) for **{title}**{loc_part}. "
        f"This job is recommended because {skills_text}. "
        f"The system observes good semantic similarity between your profile and the job "
        f"description (embedding_score = {emb:.3f}) together with a skill-overlap "
        f"score of {g:.3f}, which makes this one of the more suitable options among "
        f"the available jobs."
    )

def tab_explain(user_label: str, top_k: int):
    """
    RAG & LLM tab:
      - calls core.recommend (hybrid ranking),
      - builds per-job explanations (RAG),
      - optionally calls LLM for a global explanation.
    """
    rec, uid, err = _resolve_rec_and_user(user_label)
    if err:
        return pd.DataFrame(columns=EXPLAIN_COLS), err
    try:
        recs = rec.recommend(uid, top_k)
    except Exception as exc:
        msg = f"❗ Error while generating explanations: {exc}"
        return pd.DataFrame(columns=EXPLAIN_COLS), msg
    if not recs:
        msg = (
            "⚠️ No reasonable matches were found (all had zero embedding/graph "
            "and no overlapping skills). Cannot generate explanations."
        )
        return pd.DataFrame(columns=EXPLAIN_COLS), msg
    df = _to_dataframe(recs)
    df = df.reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    # Per-job explanations using true RAG (LLM + retrieved job text),
    # with a fallback to the old rule-based explanation if something goes wrong.
    try:
        from src.llm.explainer import rag_explain_per_job
        recs_for_rag = df.to_dict(orient="records")
        rag_texts = rag_explain_per_job(uid, recs_for_rag)

        # Safety: ensure lengths match, otherwise fallback
        if len(rag_texts) == len(df):
            df["rag_explanation"] = rag_texts
        else:
            df["rag_explanation"] = df.apply(_rag_reason, axis=1)
    except Exception:
        # Fallback: rule-based explanation from scores only
        df["rag_explanation"] = df.apply(_rag_reason, axis=1)

    # Global LLM explanation using the same matches as context (if available)
    llm_text = ""
    llm_err = ""
    if uid is not None:
        try:
            from src.llm.explainer import explain_recommendations_llm
            recs_for_llm = df.to_dict(orient="records")
            llm_text = explain_recommendations_llm(uid, recs_for_llm)
        except Exception as exc:
            llm_err = f"Error calling LLM: {exc}"
    parts: List[str] = []
    if llm_err:
        parts.append(f"❗ {llm_err}")
    if llm_text:
        parts.append(f"### 🤖 AI-Powered Career Analysis\n\n{llm_text}")
    if not parts:
        parts.append("_No global LLM explanation available._")
    markdown = "\n\n".join(parts)
    return df[EXPLAIN_COLS], markdown

# ======================================================================
#  Tab: Skill Graph – ranking by graph_score (user-centric only)
# ======================================================================

def tab_skill_graph(user_label: str, top_k: int):
    rec, uid, err = _resolve_rec_and_user(user_label)
    if err:
        return pd.DataFrame(columns=GRAPH_COLS), err
    try:
        recs = rec.recommend_graph(uid, top_k)
    except Exception as exc:
        msg = f"❗ Error while running graph-based matching: {exc}"
        return pd.DataFrame(columns=GRAPH_COLS), msg
    if not recs:
        msg = (
            "⚠️ No jobs with meaningful graph_score or overlapping skills were found "
            "for this user."
        )
        return pd.DataFrame(columns=GRAPH_COLS), msg

    df = _to_dataframe(recs)
    df = df.reset_index(drop=True)
    df.insert(0, "rank_graph", df.index + 1)
    return df[GRAPH_COLS], ""

# ======================================================================
#  Tab: Embeddings – ranking by embedding_score
# ======================================================================

def embeddings_status() -> str:
    parts: List[str] = []
    try:
        if JOB_EMB_NPY.exists():
            job_emb = np.load(JOB_EMB_NPY, allow_pickle=True)
            parts.append(f"✅ **Job embeddings loaded:** shape {job_emb.shape}")
        else:
            parts.append(f"❌ **Job embeddings:** file not found at `{JOB_EMB_NPY}`")
    except Exception as exc:
        parts.append(f"⚠️ **Error loading job embeddings:** {exc}")
    try:
        if USER_EMB_NPY.exists():
            user_emb = np.load(USER_EMB_NPY, allow_pickle=True)
            parts.append(f"✅ **User embeddings loaded:** shape {user_emb.shape}")
        else:
            parts.append(f"❌ **User embeddings:** file not found at `{USER_EMB_NPY}`")
    except Exception as exc:
        parts.append(f"⚠️ **Error loading user embeddings:** {exc}")
    return "\n\n".join(parts) if parts else "No embeddings information available."

def tab_semantic(user_label: str, top_k: int):
    rec, uid, err = _resolve_rec_and_user(user_label)
    if err:
        return pd.DataFrame(columns=SEMANTIC_COLS), err
    try:
        recs = rec.recommend_semantic(uid, top_k)
    except Exception as exc:
        msg = f"❗ Error while running semantic matching: {exc}"
        return pd.DataFrame(columns=SEMANTIC_COLS), msg
    if not recs:
        msg = (
            "⚠️ No meaningful semantic matches were found for this user "
            "(all had zero embedding_score and no overlapping skills)."
        )
        return pd.DataFrame(columns=SEMANTIC_COLS), msg

    df = _to_dataframe(recs)
    df = df.reset_index(drop=True)
    df.insert(0, "rank_semantic", df.index + 1)
    return df[SEMANTIC_COLS], ""

# ======================================================================
#  UI construction
# ======================================================================

def build_ui():
    css = """
    .gradio-container {
        max-width: 100% !important; 
        padding: 0 2rem !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    .tab-nav button {
        font-weight: 600;
        font-size: 0.95rem;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .markdown-text h3 {
        color: #4a5568;
        font-weight: 700;
        margin-top: 1.5rem;
    }
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue")) as demo:
        gr.Markdown(
            """
            # 🎯 Intelligent Job Recommendation System
            ### Powered by Distributed AI Architecture
            
            *Leveraging graph-based skill matching, semantic embeddings, and LLM-powered explanations to deliver personalized career opportunities*
            """
        )
        
        # 1) Overview -------------------------------------------------
        with gr.Tab("🏠 Overview"):
            gr.Markdown(
                """
                ### 📊 Comprehensive Recommendation Dashboard
                This hybrid approach combines multiple AI techniques to provide the most relevant job matches:
                - **Semantic Analysis:** Deep learning embeddings capture contextual similarity
                - **Graph Intelligence:** Skill networks reveal hidden connections
                - **Hybrid Scoring:** Weighted combination for optimal ranking
                """
            )
            with gr.Row():
                with gr.Column(scale=3):
                    user_ov = gr.Dropdown(
                        label="👤 Select User Profile",
                        choices=USER_LABELS,
                        value=DEFAULT_USER_LABEL,
                        interactive=bool(USER_LABELS),
                    )
                with gr.Column(scale=1):
                    topk_ov = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=TOP_K_DEFAULT,
                        step=1,
                        label="📈 Number of Recommendations",
                    )

            btn_ov = gr.Button("🔍 Generate Comprehensive Overview", variant="primary", size="lg")

            summary_box = gr.Markdown()
            table_ov = gr.Dataframe(
                value=pd.DataFrame(columns=OVERVIEW_COLS),
                label="📋 Ranked Job Recommendations",
                interactive=False,
                wrap=True,
            )

            btn_ov.click(
                tab_overview,
                inputs=[user_ov, topk_ov],
                outputs=[summary_box, table_ov],
            )

        # 2) NoSQL -----------------------------------------------------
        with gr.Tab("📊 Popularity Analytics"):
            gr.Markdown(
                """
                ### 🔥 Trending Jobs & User Engagement
                
                Explore job popularity metrics from our NoSQL database (MongoDB). This view shows:
                - **View counts** for each recommended position
                - **Engagement patterns** across the platform
                - **Real-time popularity** trends
                
                *Note: Popularity is integrated into the hybrid score but displayed separately here for transparency.*
                """
            )
            with gr.Row():
                with gr.Column(scale=3):
                    user_pop = gr.Dropdown(
                        label="👤 Select User Profile",
                        choices=USER_LABELS,
                        value=DEFAULT_USER_LABEL,
                        interactive=bool(USER_LABELS),
                    )
                with gr.Column(scale=1):
                    topk_pop = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=TOP_K_DEFAULT,
                        step=1,
                        label="📈 Top Recommendations",
                    )

            btn_pop_user = gr.Button("📊 Analyze User-Specific Popularity", variant="primary", size="lg")

            table_pop_user = gr.Dataframe(
                value=pd.DataFrame(columns=NOSQL_USER_COLS),
                label="🔥 Popularity-Ranked Recommendations",
                interactive=False,
                wrap=True,
            )
            msg_pop_user = gr.Markdown()

            btn_pop_user.click(
                tab_popularity_user,
                inputs=[user_pop, topk_pop],
                outputs=[table_pop_user, msg_pop_user],
            )

        # 3) RAG & LLM -------------------------------------------------
        with gr.Tab("🤖 AI Explanations"):
            gr.Markdown(
                """
                ### 🧠 Explainable AI & Career Insights
                
                Understand *why* each job is recommended through our advanced AI system:
                - **Per-Job RAG Analysis:** Detailed explanations for each recommendation
                - **Hybrid Score Breakdown:** Transparent scoring methodology
                - **LLM Global Insights:** High-level career trajectory analysis
                
                *Powered by Retrieval-Augmented Generation (RAG) for contextually grounded explanations.*
                """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    user_expl = gr.Dropdown(
                        label="👤 Select User Profile",
                        choices=USER_LABELS,
                        value=DEFAULT_USER_LABEL,
                        interactive=bool(USER_LABELS),
                    )
                with gr.Column(scale=1):
                    topk_expl = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=min(TOP_K_DEFAULT, 10),
                        step=1,
                        label="🎯 Jobs to Explain",
                    )

            btn_expl = gr.Button("🤖 Generate AI-Powered Explanations", variant="primary", size="lg")
            
            expl_llm = gr.Markdown(label="💡 LLM Career Analysis")
            
            expl_table = gr.Dataframe(
                value=pd.DataFrame(columns=EXPLAIN_COLS),
                label="📝 Detailed Job Explanations",
                interactive=False,
                wrap=True,
            )

            btn_expl.click(
                tab_explain,
                inputs=[user_expl, topk_expl],
                outputs=[expl_table, expl_llm],
            )

        # 4) Skill Graph ----------------------------------------------
        with gr.Tab("🕸️ Skill Graph"):
            gr.Markdown(
                """
                ### 🔗 Network-Based Skill Matching
                
                Leverage graph database technology to discover jobs through skill relationships:
                - **Direct Skill Matches:** Explicit overlaps between user and job requirements
                - **Network Analysis:** Graph algorithms reveal hidden connections
                - **Skill Proximity:** Jobs ranked by skill neighborhood similarity
                
                *This view prioritizes explicit technical competencies and professional capabilities.*
                """
            )

            with gr.Row():
                with gr.Column(scale=3):
                    user_graph = gr.Dropdown(
                        label="👤 Select User Profile",
                        choices=USER_LABELS,
                        value=DEFAULT_USER_LABEL,
                        interactive=bool(USER_LABELS),
                    )
                with gr.Column(scale=1):
                    topk_graph = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=TOP_K_DEFAULT,
                        step=1,
                        label="🎯 Top Matches",
                    )

            btn_graph = gr.Button("🕸️ Execute Graph-Based Analysis", variant="primary", size="lg")
            table_graph = gr.Dataframe(
                value=pd.DataFrame(columns=GRAPH_COLS),
                label="🔗 Skill Graph Recommendations",
                interactive=False,
                wrap=True,
            )
            msg_graph = gr.Markdown()
            btn_graph.click(
                tab_skill_graph,
                inputs=[user_graph, topk_graph],
                outputs=[table_graph, msg_graph],
            )

        # 5) Embeddings -----------------------------------------------
        with gr.Tab("🧠 Semantic Search"):
            gr.Markdown(
                """
                ### 🎯 Deep Learning-Based Matching
                
                Discover opportunities through advanced neural network embeddings:
                - **Contextual Understanding:** Captures semantic meaning beyond keywords
                - **Latent Similarity:** Identifies conceptually related positions
                - **Vector Space Analysis:** High-dimensional representation of skills and roles
                
                *Powered by state-of-the-art transformer models for nuanced profile-job matching.*
                """
            )
            with gr.Row():
                with gr.Column(scale=3):
                    user_sem = gr.Dropdown(
                        label="👤 Select User Profile",
                        choices=USER_LABELS,
                        value=DEFAULT_USER_LABEL,
                        interactive=bool(USER_LABELS),
                    )
                with gr.Column(scale=1):
                    topk_sem = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=TOP_K_DEFAULT,
                        step=1,
                        label="🎯 Top Matches",
                    )

            btn_sem = gr.Button("🧠 Run Semantic Analysis", variant="primary", size="lg")
            table_sem = gr.Dataframe(
                value=pd.DataFrame(columns=SEMANTIC_COLS),
                label="📊 Semantically Matched Opportunities",
                interactive=False,
                wrap=True,
            )
            msg_sem = gr.Markdown()
            btn_sem.click(
                tab_semantic,
                inputs=[user_sem, topk_sem],
                outputs=[table_sem, msg_sem],
            )

            gr.Markdown("---")
            gr.Markdown("### 📦 Embedding Model Status")
            emb_box = gr.Markdown(value=embeddings_status())
            btn_emb = gr.Button("🔄 Refresh Embeddings Status")
            btn_emb.click(
                embeddings_status,
                inputs=None,
                outputs=emb_box,
            )

        # 6) User List -------------------------------------------------
        with gr.Tab("👥 User Database"):
            gr.Markdown(
                """
                ### 📋 Complete User Registry
                
                Browse all registered users with their professional profiles:
                - **Skills & Competencies:** Technical and soft skills
                - **Experience Level:** Years in industry
                - **Location Preferences:** Geographic constraints
                """
            )
            if not USERS_DF.empty:
                cols = [
                    c
                    for c in [
                        "user_id",
                        "name",
                        "skills",
                        "years_experience",
                        "preferred_location",
                    ]
                    if c in USERS_DF.columns
                ]
                gr.Dataframe(value=USERS_DF[cols], interactive=False, wrap=True)
            else:
                gr.Markdown("⚠️ No users loaded. Please check the data source configuration.")

        # 7) Jobs List -------------------------------------------------
        with gr.Tab("💼 Job Database"):
            gr.Markdown(
                """
                ### 📋 Complete Job Catalog
                
                Explore all available positions in the system:
                - **Job Details:** Title, company, location
                - **Required Skills:** Technical requirements
                - **Full Descriptions:** Comprehensive role information
                """
            )
            if not JOBS_DF.empty:
                cols = [
                    c
                    for c in [
                        "job_id",
                        "title",
                        "company_name",
                      
                        "location",
                        "skills",
                        "description",
                    ]
                    if c in JOBS_DF.columns
                ]
                gr.Dataframe(value=JOBS_DF[cols], interactive=False)
            else:
                gr.Markdown("No jobs loaded.")
    return demo

if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
