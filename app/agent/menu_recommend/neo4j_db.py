import logging
from typing import List, Optional
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel

# 데이터 모델 정의
class UserPreference(BaseModel):
    item: str
    preference_type: str  # "LIKES", "DISLIKES", "ALLERGIC_TO"

class Neo4jService:
    def __init__(self, uri, user, password):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        await self.driver.close()

    async def initialize_constraints(self):
        """초기 필수 제약 조건 및 인덱스 설정"""
        queries = [
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE INDEX menu_name_index IF NOT EXISTS FOR (m:Menu) ON (m.name)",
            "CREATE INDEX feature_name_index IF NOT EXISTS FOR (f:Feature) ON (f.name)"
        ]
        async with self.driver.session() as session:
            for q in queries:
                await session.run(q)

    async def get_user_context(self, user_id: str) -> List[dict]:
        """사용자의 취향 및 팀 정보 로드"""
        query = """
        MATCH (u:User {id: $user_id})
        OPTIONAL MATCH (u)-[r]->(target)
        WHERE type(r) IN ['LIKES', 'DISLIKES', 'ALLERGIC_TO', 'MEMBER_OF']
        RETURN type(r) as relationship, target.name as name, labels(target)[0] as label
        """
        async with self.driver.session() as session:
            result = await session.run(query, user_id=user_id)
            return [record.data() async for record in result]

    async def upsert_preference(self, user_id: str, item_name: str, pref_type: str, label: str = "Menu"):
        """취향 업데이트 (상충 관계 삭제 후 갱신)"""
        query = f"""
        MERGE (u:User {{id: $user_id}})
        MERGE (target:{label} {{name: $item_name}})
        WITH u, target
        // 기존에 존재하던 상충되는 관계들(LIKES, DISLIKES 등) 삭제
        OPTIONAL MATCH (u)-[old_r]-(target)
        WHERE type(old_r) IN ['LIKES', 'DISLIKES', 'ALLERGIC_TO']
        DELETE old_r
        WITH u, target
        // 새로운 관계 생성 (APOC 없이 표준 Cypher로 구현)
        CREATE (u)-[r:{pref_type} {{updated_at: timestamp()}}]->(target)
        RETURN type(r)
        """
        async with self.driver.session() as session:
            await session.run(query, user_id=user_id, item_name=item_name)