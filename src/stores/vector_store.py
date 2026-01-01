# src/stores/vector_store.py
from __future__ import annotations

import numpy as np
import faiss

from src.config import JOB_EMB_NPY, JOB_IDS_NPY


class JobVectorIndex:
    """FAISS-based dense vector index over job embeddings."""

    def __init__(self):
        # allow_pickle=True to survive any old files, then cast to float32
        job_emb = np.load(JOB_EMB_NPY, allow_pickle=True)
        job_ids = np.load(JOB_IDS_NPY, allow_pickle=True)

        self.job_embeddings = job_emb.astype("float32")
        self.job_ids = job_ids.astype(str)

        if self.job_embeddings.ndim != 2:
            raise ValueError("job_embeddings must be 2D")

        n, d = self.job_embeddings.shape
        print(f"[vector_store] loaded {n} job vectors, dim={d}")

        # build FAISS index
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(self.job_embeddings)
        self.index.add(self.job_embeddings)

    def search_for_user_vector(self, user_vec: np.ndarray, top_k: int = 20):
        if user_vec.ndim != 1:
            raise ValueError("user_vec must be 1D")

        v = user_vec.reshape(1, -1).astype("float32")
        faiss.normalize_L2(v)
        distances, indices = self.index.search(v, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            job_id = self.job_ids[idx]
            results.append((str(job_id), float(score)))
        return results
