# src/features/embeddings.py
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    JOB_EMB_NPY,
    JOB_IDS_NPY,
    USER_EMB_NPY,
    USER_IDS_NPY,
    EMBEDDING_MODEL_NAME,
)
from src.features.preprocessing import (
    load_jobs_df,
    load_users_df,
    build_job_text,
    build_user_text,
)


def compute_and_save_embeddings() -> None:
    jobs_df = load_jobs_df()
    users_df = load_users_df()

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Jobs
    job_texts = build_job_text(jobs_df).tolist()
    job_ids = jobs_df["job_id"].astype(str).to_numpy()

    print(f"[embeddings] encoding {len(job_texts)} job descriptions...")
    job_emb = model.encode(
        job_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    np.save(JOB_EMB_NPY, job_emb)
    np.save(JOB_IDS_NPY, job_ids)

    # Users
    user_texts = build_user_text(users_df).tolist()
    user_ids = users_df["user_id"].astype(str).to_numpy()

    print(f"[embeddings] encoding {len(user_texts)} user profiles...")
    user_emb = model.encode(
        user_texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    np.save(USER_EMB_NPY, user_emb)
    np.save(USER_IDS_NPY, user_ids)

    print("[embeddings] done – saved job & user embeddings to data/processed/")


if __name__ == "__main__":
    compute_and_save_embeddings()
