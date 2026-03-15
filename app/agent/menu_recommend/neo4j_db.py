import logging
from typing import List, Optional
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel
import os


class Neo4jService:
    def __init__(self, uri, user, password):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        await self.driver.close()

    async def get_user_context(self, user_id: str) -> List[dict]:
        """계층적 취향 정보를 로드합니다."""
        query = """
        MATCH (u:User {id: $user_id})-[r:PREFERS]->(c:Category)
        // 상위 카테고리 경로를 함께 가져와 문맥을 풍부하게 합니다.
        OPTIONAL MATCH path = (parent:Category {domain: c.domain})-[:PARENT_OF*]->(c)
        RETURN 
            c.name as item, 
            r.type as preference_type, 
            c.domain as domain,
            [n in nodes(path) | n.name] as category_path
        """
        async with self.driver.session() as session:
            result = await session.run(query, user_id=user_id)
            return [record.data() async for record in result]

    async def upsert_hierarchical_preference(self, user_id: str, category_path: str, pref_type: str, domain: str):
        """'음식 > 일식 > 초밥' 형태의 경로를 계층적으로 저장합니다."""
        parts = [p.strip() for p in category_path.split(">")]

        query = """
        MERGE (u:User {id: $user_id})
        WITH u
        // 경로상의 모든 노드를 생성하고 연결
        UNWIND range(0, size($parts)-2) AS i
        MERGE (p:Category {name: $parts[i], domain: $domain})
        MERGE (c:Category {name: $parts[i+1], domain: $domain})
        MERGE (p)-[:PARENT_OF]->(c)
        WITH u, c WHERE c.name = last($parts)
        // 기존 관계 삭제 후 새 관계 생성 (상충 방지)
        OPTIONAL MATCH (u)-[old_r:PREFERS]->(c)
        DELETE old_r
        CREATE (u)-[r:PREFERS {type: $pref_type, updated_at: datetime()}]->(c)
        """
        async with self.driver.session() as session:
            await session.run(query, user_id=user_id, parts=parts, pref_type=pref_type, domain=domain)

    async def reset_db(self):
        """
        데이터베이스 전체 초기화 (노드 및 관계 삭제)
        테스트 환경이나 초기화가 필요할 때 사용합니다.
        """
        query = "MATCH (n) DETACH DELETE n"
        async with self.driver.session() as session:
            try:
                await session.run(query)
            except Exception as e:
                raise

    async def run_query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """
        임의의 Cypher 쿼리를 실행하고 결과를 반환합니다.
        관리자 디버깅 및 그래프 조회용.

        반환: 각 레코드를 dict로 변환한 리스트
        """
        async with self.driver.session() as session:
            result = await session.run(cypher, params or {})
            records = [record.data() async for record in result]
            return records

    async def get_stats(self) -> dict:
        """
        그래프 DB 전체 통계를 반환합니다.
        노드 라벨별 카운트, 관계 타입별 카운트, 총합을 포함합니다.
        """
        async with self.driver.session() as session:
            # 노드 라벨별 카운트
            node_result = await session.run(
                "CALL db.labels() YIELD label "
                "CALL { WITH label MATCH (n) WHERE label IN labels(n) RETURN count(n) AS cnt } "
                "RETURN label, cnt ORDER BY cnt DESC"
            )
            node_counts = {r["label"]: r["cnt"] async for r in node_result}

            # 관계 타입별 카운트
            rel_result = await session.run(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "CALL { WITH relationshipType MATCH ()-[r]->() WHERE type(r) = relationshipType RETURN count(r) AS cnt } "
                "RETURN relationshipType, cnt ORDER BY cnt DESC"
            )
            rel_counts = {r["relationshipType"]: r["cnt"] async for r in rel_result}

            # 총합
            total_result = await session.run(
                "MATCH (n) RETURN count(n) AS total_nodes"
            )
            total_nodes = (await total_result.single())["total_nodes"]

            total_rel_result = await session.run(
                "MATCH ()-[r]->() RETURN count(r) AS total_rels"
            )
            total_rels = (await total_rel_result.single())["total_rels"]

        return {
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
            "nodes_by_label": node_counts,
            "relationships_by_type": rel_counts,
        }



class Neo4jManager:
    _service: Neo4jService = None

    @classmethod
    async def init(cls):
        """앱 시작 시 단 한 번 호출하여 서비스 인스턴스 생성"""
        if cls._service is None:
            cls._service = Neo4jService(
                uri=os.getenv("NEO4J_URI"),
                user=os.getenv("NEO4J_USER"),
                password=os.getenv("NEO4J_PASSWORD")
            )

    @classmethod
    def get_service(cls) -> Neo4jService:
        """노드나 API에서 서비스 인스턴스에 접근할 때 사용"""
        if cls._service is None:
            raise RuntimeError("Neo4jService가 초기화되지 않았습니다. Lifespan을 확인하세요.")
        return cls._service

    @classmethod
    async def close(cls):
        """앱 종료 시 드라이버 연결을 안전하게 닫음"""
        if cls._service:
            await cls._service.close()
            cls._service = None