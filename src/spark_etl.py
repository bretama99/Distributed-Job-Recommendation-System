# src/spark_etl.py
from __future__ import annotations
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import hashlib
from typing import Optional, List

import pandas as pd

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_l = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_l:
            return cols_l[cand.lower()]
    return None

def etl_jobs_with_spark(
    linkedin_posts: str,
    linkedin_summary: str,
    linkedin_skills: str,
    limit: int = 5000,
    master: str = "local[*]",
    app_name: str = "job-rec-etl",
) -> pd.DataFrame:
    from pyspark.sql import SparkSession, functions as F, types as T

    spark = (
        SparkSession.builder
        .master(master)
        .appName(app_name)
        .getOrCreate()
    )

    def read_csv(path: str):
        return (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(path)
        )

    posts = read_csv(linkedin_posts)
    summary = read_csv(linkedin_summary)
    skills = read_csv(linkedin_skills)

    # normalize link: lowercase, trim, remove query/fragment, remove trailing slash
    def norm_link(col):
        c = F.lower(F.trim(col))
        c = F.regexp_replace(c, r"[#?].*$", "")
        c = F.regexp_replace(c, r"/+$", "")
        return c

    p_link = _pick_col(posts.columns, ["job_link", "link", "url", "job_url"])
    if not p_link:
        raise RuntimeError(f"Posts CSV missing job_link column. Columns={posts.columns}")

    s_link = _pick_col(summary.columns, ["job_link", "link", "url", "job_url"])
    k_link = _pick_col(skills.columns, ["job_link", "link", "url", "job_url"])

    p_title = _pick_col(posts.columns, ["title", "job_title", "jobtitle", "position"])
    p_company = _pick_col(posts.columns, ["company_name", "company", "organization", "employer"])
    p_loc = _pick_col(posts.columns, ["location", "job_location", "joblocation", "city"])

    # skills column
    k_skill = _pick_col(skills.columns, ["skill", "skills", "name", "skill_name"])
    if not k_skill:
        # fallback: pick first non-link string column
        k_skill = next((c for c in skills.columns if c != k_link), None)
    if not k_skill:
        raise RuntimeError(f"Skills CSV missing skill column. Columns={skills.columns}")

    # description column
    s_desc = _pick_col(summary.columns, ["description", "job_summary", "summary", "text"])

    posts2 = posts.withColumn("_job_link_norm", norm_link(F.col(p_link)))

    summary2 = summary
    if s_link:
        summary2 = summary2.withColumn("_job_link_norm", norm_link(F.col(s_link)))
    else:
        summary2 = spark.createDataFrame([], schema=T.StructType([
            T.StructField("_job_link_norm", T.StringType(), True),
        ]))

    skills2 = skills
    if k_link:
        skills2 = skills2.withColumn("_job_link_norm", norm_link(F.col(k_link)))
    else:
        skills2 = spark.createDataFrame([], schema=T.StructType([
            T.StructField("_job_link_norm", T.StringType(), True),
            T.StructField(k_skill, T.StringType(), True),
        ]))

    # aggregate skills per job
    skills_agg = (
        skills2.groupBy("_job_link_norm")
        .agg(F.collect_set(F.col(k_skill).cast("string")).alias("_skills_set"))
        .withColumn("skills", F.concat_ws(", ", F.col("_skills_set")))
        .select("_job_link_norm", "skills")
    )

    # attach summary/description
    summary_sel = summary2.select(
        "_job_link_norm",
        (F.col(s_desc).cast("string").alias("description") if s_desc else F.lit("").alias("description")),
    )

    joined = (
        posts2.join(summary_sel, on="_job_link_norm", how="left")
              .join(skills_agg, on="_job_link_norm", how="left")
    )

    # build final columns
    out = joined.select(
        F.col(p_link).cast("string").alias("job_link"),
        (F.col(p_title).cast("string").alias("title") if p_title else F.lit("").alias("title")),
        (F.col(p_company).cast("string").alias("company_name") if p_company else F.lit("").alias("company_name")),
        (F.col(p_loc).cast("string").alias("location") if p_loc else F.lit("").alias("location")),
        F.coalesce(F.col("description"), F.lit("")).alias("description"),
        F.coalesce(F.col("skills"), F.lit("")).alias("skills"),
        F.col("_job_link_norm").alias("_job_link_norm"),
    )

    # deterministic job_id from normalized link
    out = out.withColumn(
        "job_id",
        F.substring(F.sha1(F.col("_job_link_norm")), 1, 12)
    )
    out = out.withColumn("job_id", sha1_udf(F.col("_job_link_norm")))

    # limit + convert to pandas
    out_pd = out.drop("_job_link_norm").limit(int(limit)).toPandas()

    spark.stop()
    return out_pd
