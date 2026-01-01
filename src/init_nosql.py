# src/init_nosql.py
from __future__ import annotations

import logging

from src.features.preprocessing import load_jobs_df
from src.stores.nosql_store import upsert_jobs

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    log.info("[NoSQL Init] Loading jobs from CSV via load_jobs_df() ...")
    jobs_df = load_jobs_df()
    log.info("[NoSQL Init] Loaded %d jobs – upserting into MongoDB", len(jobs_df))

    upsert_jobs(jobs_df)

    log.info("[NoSQL Init] NoSQL initialisation completed.")


if __name__ == "__main__":
    main()
