import logging
from typing import List, Optional
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel



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