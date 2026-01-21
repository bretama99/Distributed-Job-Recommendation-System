from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from neo4j import GraphDatabase

from src.config import (
    NEO4J_CONNECTION_TIMEOUT,
    NEO4J_DATABASE,
    NEO4J_MAX_CONNECTION_POOL_SIZE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
)


def _chunks(rows: List[Dict[str, Any]], n: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(rows), n):
        yield rows[i : i + n]


def _pairs(skills: List[str]) -> Iterable[Tuple[str, str]]:
    s = sorted(set([x for x in skills if x]))
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            yield s[i], s[j]


@dataclass
class Neo4jGraphStore:
    driver: Any

    @classmethod
    def connect(cls) -> "Neo4jGraphStore":
        d = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_pool_size=int(NEO4J_MAX_CONNECTION_POOL_SIZE),
            connection_timeout=float(NEO4J_CONNECTION_TIMEOUT),
        )
        with d.session(database=NEO4J_DATABASE) as s:
            s.run("RETURN 1").consume()
        return cls(driver=d)

    def close(self) -> None:
        try:
            self.driver.close()
        except Exception:
            pass

    def ping(self) -> None:
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("RETURN 1").consume()

    def ensure_schema(self) -> None:
        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE").consume()
            s.run("CREATE CONSTRAINT job_id_unique IF NOT EXISTS FOR (j:Job) REQUIRE j.job_id IS UNIQUE").consume()
            s.run("CREATE CONSTRAINT skill_name_unique IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE").consume()
            s.run("CREATE INDEX user_user_id IF NOT EXISTS FOR (u:User) ON (u.user_id)").consume()
            s.run("CREATE INDEX job_job_id IF NOT EXISTS FOR (j:Job) ON (j.job_id)").consume()
            s.run("CREATE INDEX skill_name IF NOT EXISTS FOR (s:Skill) ON (s.name)").consume()

    def upsert_users(self, rows: List[Dict[str, Any]], batch_size: int = 600) -> int:
        if not rows:
            return 0
        cy = (
            "UNWIND $rows AS row "
            "MERGE (u:User {user_id: row.user_id}) "
            "SET u.name = row.name, u.preferred_location = row.preferred_location "
            "WITH u, row "
            "UNWIND row.skills AS sk "
            "MERGE (s:Skill {name: sk}) "
            "MERGE (u)-[:HAS_SKILL]->(s)"
        )
        with self.driver.session(database=NEO4J_DATABASE) as s:
            for part in _chunks(rows, int(batch_size)):
                s.run(cy, rows=part).consume()
        return len(rows)

    def upsert_jobs(self, rows: List[Dict[str, Any]], batch_size: int = 600) -> int:
        if not rows:
            return 0
        cy = (
            "UNWIND $rows AS row "
            "MERGE (j:Job {job_id: row.job_id}) "
            "SET j.title = row.title, j.company_name = row.company_name, j.location = row.location "
            "WITH j, row "
            "UNWIND row.skills AS sk "
            "MERGE (s:Skill {name: sk}) "
            "MERGE (j)-[:REQUIRES]->(s)"
        )
        with self.driver.session(database=NEO4J_DATABASE) as s:
            for part in _chunks(rows, int(batch_size)):
                s.run(cy, rows=part).consume()
        return len(rows)

    def rebuild_skill_cooccurrence(self, job_rows: List[Dict[str, Any]], batch_size: int = 800) -> Dict[str, Any]:
        started = time.time()
        pairs: Dict[Tuple[str, str], int] = {}
        for r in job_rows:
            sk = [x for x in (r.get("skills") or []) if x]
            for a, b in _pairs(sk):
                k = (a, b)
                pairs[k] = pairs.get(k, 0) + 1

        rows = [{"a": a, "b": b, "w": int(w)} for (a, b), w in pairs.items() if w > 0]

        with self.driver.session(database=NEO4J_DATABASE) as s:
            s.run("MATCH ()-[r:CO_OCCURS]->() DELETE r").consume()
            cy = (
                "UNWIND $rows AS row "
                "MERGE (a:Skill {name: row.a}) "
                "MERGE (b:Skill {name: row.b}) "
                "MERGE (a)-[r:CO_OCCURS]->(b) "
                "SET r.w = coalesce(r.w, 0) + row.w"
            )
            for part in _chunks(rows, int(batch_size)):
                s.run(cy, rows=part).consume()

        return {"ok": True, "pairs": len(rows), "seconds": round(time.time() - started, 3)}

    def graph_scores(self, user_id: str, job_ids: List[str]) -> Dict[str, float]:
        uid = str(user_id)
        jids = [str(x) for x in job_ids]
        cy = (
            "MATCH (u:User {user_id:$uid})-[:HAS_SKILL]->(us:Skill) "
            "MATCH (j:Job)-[:REQUIRES]->(js:Skill) "
            "WHERE j.job_id IN $jids "
            "OPTIONAL MATCH (us)-[c:CO_OCCURS]->(js) "
            "OPTIONAL MATCH (js)-[c2:CO_OCCURS]->(us) "
            "WITH j, "
            "count(DISTINCT CASE WHEN us = js THEN us END) AS shared, "
            "sum(coalesce(c.w,0)) + sum(coalesce(c2.w,0)) AS co "
            "RETURN j.job_id AS job_id, shared AS shared, co AS co"
        )
        raw: Dict[str, float] = {jid: 0.0 for jid in jids}
        with self.driver.session(database=NEO4J_DATABASE) as s:
            for r in s.run(cy, uid=uid, jids=jids):
                jid = str(r["job_id"])
                shared = float(r["shared"] or 0.0)
                co = float(r["co"] or 0.0)
                raw[jid] = float(10.0 * shared + (co ** 0.5))

        mx = max(raw.values()) if raw else 0.0
        if mx <= 0.0:
            return {k: 0.0 for k in raw.keys()}
        return {k: float(v) / float(mx) for k, v in raw.items()}

    def stats(self) -> Dict[str, int]:
        with self.driver.session(database=NEO4J_DATABASE) as s:
            users = int(s.run("MATCH (u:User) RETURN count(u) AS c").single()["c"])
            jobs = int(s.run("MATCH (j:Job) RETURN count(j) AS c").single()["c"])
            skills = int(s.run("MATCH (s:Skill) RETURN count(s) AS c").single()["c"])
            rels = int(s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"])
            co = int(s.run("MATCH (:Skill)-[r:CO_OCCURS]->(:Skill) RETURN count(r) AS c").single()["c"])
        return {"users": users, "jobs": jobs, "skills": skills, "rels": rels, "co_occurs": co}


# NOTE:
# Do not connect at import time. This keeps the app usable even when Neo4j is down/misconfigured.
_neo4j_graph: Optional["Neo4jGraphStore"] = None


def get_neo4j_graph() -> Optional["Neo4jGraphStore"]:
    global _neo4j_graph
    if _neo4j_graph is not None:
        return _neo4j_graph
    try:
        _neo4j_graph = Neo4jGraphStore.connect()
        return _neo4j_graph
    except Exception:
        _neo4j_graph = None
        return None
