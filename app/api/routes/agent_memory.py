import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import func, select

from app.api.deps import SessionDep
from app.agent.menu_recommend.neo4j_db import Neo4jManager
from app.models import (
    EntityDefinition, EntityDefinitionCreate, EntityDefinitionPublic, EntityDefinitionUpdate,
    Message,
    NodeDefinition, NodeDefinitionCreate, NodeDefinitionPublic, NodeDefinitionUpdate,
    NodeEntityDefinition, NodeEntityDefinitionCreate, NodeEntityDefinitionPublic, NodeEntityDefinitionUpdate,
    NodesPublic,
    RelationDefinition, RelationDefinitionCreate, RelationDefinitionPublic, RelationDefinitionUpdate,
    RelationsPublic,
)
from datetime import datetime

router = APIRouter(prefix="/agent-memory", tags=["agent-memory"])


# ── Neo4j 관리 ─────────────────────────────────────────────────

class Neo4jQueryRequest(BaseModel):
    cypher: str
    params: dict | None = None


class Neo4jQueryResponse(BaseModel):
    cypher: str
    row_count: int
    columns: list[str]
    rows: list[dict]


class Neo4jStatsResponse(BaseModel):
    total_nodes: int
    total_relationships: int
    nodes_by_label: dict[str, int]
    relationships_by_type: dict[str, int]


@router.post("/neo4j/reset", response_model=Message)
async def reset_neo4j() -> Any:
    """Neo4j 그래프 DB 전체 초기화 (모든 노드 및 관계 삭제)."""
    try:
        service = Neo4jManager.get_service()
        await service.reset_db()
        return Message(message="Neo4j DB가 초기화되었습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j 초기화 실패: {str(e)}")


@router.post("/neo4j/query", response_model=Neo4jQueryResponse)
async def run_neo4j_query(*, body: Neo4jQueryRequest) -> Any:
    """
    임의의 Cypher 쿼리를 실행하고 결과를 반환합니다.

    [요청 예시]
    {
        "cypher": "MATCH (u:User)-[r:PREFERS]->(p:Preference) RETURN u.id, p.category, p.value LIMIT 10",
        "params": {}
    }
    """
    try:
        service = Neo4jManager.get_service()
        rows = await service.run_query(body.cypher, body.params)
        columns = list(rows[0].keys()) if rows else []
        return Neo4jQueryResponse(
            cypher=body.cypher,
            row_count=len(rows),
            columns=columns,
            rows=rows,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"쿼리 실행 실패: {str(e)}")


@router.get("/neo4j/stats", response_model=Neo4jStatsResponse)
async def get_neo4j_stats() -> Any:
    """Neo4j 그래프 DB 통계 (노드 라벨별 수, 관계 타입별 수, 전체 총합)."""
    try:
        service = Neo4jManager.get_service()
        stats = await service.get_stats()
        return Neo4jStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")


# ── Relations ──────────────────────────────────────────────────

@router.get("/relations", response_model=RelationsPublic)
def list_relations(
    session: SessionDep,
    domain_name: str | None = None,
    action_name: str | None = None,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    statement = select(RelationDefinition)
    count_statement = select(func.count()).select_from(RelationDefinition)

    if domain_name:
        statement = statement.where(RelationDefinition.domain_name == domain_name)
        count_statement = count_statement.where(RelationDefinition.domain_name == domain_name)
    if action_name is not None:
        statement = statement.where(RelationDefinition.action_name == action_name)
        count_statement = count_statement.where(RelationDefinition.action_name == action_name)

    count = session.exec(count_statement).one()
    relations = session.exec(statement.offset(skip).limit(limit).order_by(RelationDefinition.created_at.desc())).all()
    return RelationsPublic(data=list(relations), count=count)


@router.post("/relations", response_model=RelationDefinitionPublic)
def create_relation(
    *,
    session: SessionDep,
    relation_in: RelationDefinitionCreate,
) -> Any:
    relation = RelationDefinition.model_validate(relation_in)
    session.add(relation)
    session.commit()
    session.refresh(relation)
    return relation


@router.get("/relations/{relation_id}", response_model=RelationDefinitionPublic)
def get_relation(
    *,
    session: SessionDep,
    relation_id: uuid.UUID,
) -> Any:
    relation = session.get(RelationDefinition, relation_id)
    if not relation:
        raise HTTPException(status_code=404, detail="RelationDefinition not found")
    return relation


@router.put("/relations/{relation_id}", response_model=RelationDefinitionPublic)
def update_relation(
    *,
    session: SessionDep,
    relation_id: uuid.UUID,
    relation_in: RelationDefinitionUpdate,
) -> Any:
    relation = session.get(RelationDefinition, relation_id)
    if not relation:
        raise HTTPException(status_code=404, detail="RelationDefinition not found")
    update_dict = relation_in.model_dump(exclude_unset=True)
    relation.sqlmodel_update(update_dict)
    relation.updated_at = datetime.now()
    session.add(relation)
    session.commit()
    session.refresh(relation)
    return relation


@router.delete("/relations/{relation_id}")
def delete_relation(
    *,
    session: SessionDep,
    relation_id: uuid.UUID,
) -> Message:
    relation = session.get(RelationDefinition, relation_id)
    if not relation:
        raise HTTPException(status_code=404, detail="RelationDefinition not found")
    session.delete(relation)
    session.commit()
    return Message(message="RelationDefinition deleted successfully")


# ── Entities ───────────────────────────────────────────────────

@router.post("/relations/{relation_id}/entities", response_model=EntityDefinitionPublic)
def add_entity(
    *,
    session: SessionDep,
    relation_id: uuid.UUID,
    entity_in: EntityDefinitionCreate,
) -> Any:
    relation = session.get(RelationDefinition, relation_id)
    if not relation:
        raise HTTPException(status_code=404, detail="RelationDefinition not found")
    entity = EntityDefinition.model_validate(entity_in, update={"relation_definition_id": relation_id})
    session.add(entity)
    session.commit()
    session.refresh(entity)
    return entity


@router.put("/entities/{entity_id}", response_model=EntityDefinitionPublic)
def update_entity(
    *,
    session: SessionDep,
    entity_id: uuid.UUID,
    entity_in: EntityDefinitionUpdate,
) -> Any:
    entity = session.get(EntityDefinition, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="EntityDefinition not found")
    update_dict = entity_in.model_dump(exclude_unset=True)
    entity.sqlmodel_update(update_dict)
    entity.updated_at = datetime.now()
    session.add(entity)
    session.commit()
    session.refresh(entity)
    return entity


@router.delete("/entities/{entity_id}")
def delete_entity(
    *,
    session: SessionDep,
    entity_id: uuid.UUID,
) -> Message:
    entity = session.get(EntityDefinition, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="EntityDefinition not found")
    session.delete(entity)
    session.commit()
    return Message(message="EntityDefinition deleted successfully")


# ── Nodes ───────────────────────────────────────────────────────

@router.get("/nodes", response_model=NodesPublic)
def list_nodes(
    session: SessionDep,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    count = session.exec(select(func.count()).select_from(NodeDefinition)).one()
    nodes = session.exec(select(NodeDefinition).offset(skip).limit(limit).order_by(NodeDefinition.created_at.desc())).all()
    return NodesPublic(data=list(nodes), count=count)


@router.post("/nodes", response_model=NodeDefinitionPublic)
def create_node(
    *,
    session: SessionDep,
    node_in: NodeDefinitionCreate,
) -> Any:
    node = NodeDefinition.model_validate(node_in)
    session.add(node)
    session.commit()
    session.refresh(node)
    return node


@router.get("/nodes/{node_id}", response_model=NodeDefinitionPublic)
def get_node(
    *,
    session: SessionDep,
    node_id: uuid.UUID,
) -> Any:
    node = session.get(NodeDefinition, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="NodeDefinition not found")
    return node


@router.put("/nodes/{node_id}", response_model=NodeDefinitionPublic)
def update_node(
    *,
    session: SessionDep,
    node_id: uuid.UUID,
    node_in: NodeDefinitionUpdate,
) -> Any:
    node = session.get(NodeDefinition, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="NodeDefinition not found")
    update_dict = node_in.model_dump(exclude_unset=True)
    node.sqlmodel_update(update_dict)
    node.updated_at = datetime.now()
    session.add(node)
    session.commit()
    session.refresh(node)
    return node


@router.delete("/nodes/{node_id}")
def delete_node(
    *,
    session: SessionDep,
    node_id: uuid.UUID,
) -> Message:
    node = session.get(NodeDefinition, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="NodeDefinition not found")
    session.delete(node)
    session.commit()
    return Message(message="NodeDefinition deleted successfully")


@router.post("/nodes/{node_id}/entities", response_model=NodeEntityDefinitionPublic)
def add_node_entity(
    *,
    session: SessionDep,
    node_id: uuid.UUID,
    entity_in: NodeEntityDefinitionCreate,
) -> Any:
    node = session.get(NodeDefinition, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="NodeDefinition not found")
    entity = NodeEntityDefinition.model_validate(entity_in, update={"node_definition_id": node_id})
    session.add(entity)
    session.commit()
    session.refresh(entity)
    return entity


@router.put("/node-entities/{entity_id}", response_model=NodeEntityDefinitionPublic)
def update_node_entity(
    *,
    session: SessionDep,
    entity_id: uuid.UUID,
    entity_in: NodeEntityDefinitionUpdate,
) -> Any:
    entity = session.get(NodeEntityDefinition, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="NodeEntityDefinition not found")
    update_dict = entity_in.model_dump(exclude_unset=True)
    entity.sqlmodel_update(update_dict)
    entity.updated_at = datetime.now()
    session.add(entity)
    session.commit()
    session.refresh(entity)
    return entity


@router.delete("/node-entities/{entity_id}")
def delete_node_entity(
    *,
    session: SessionDep,
    entity_id: uuid.UUID,
) -> Message:
    entity = session.get(NodeEntityDefinition, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="NodeEntityDefinition not found")
    session.delete(entity)
    session.commit()
    return Message(message="NodeEntityDefinition deleted successfully")


# ── Config (공개 엔드포인트, 챗봇용) ────────────────────────────

@router.get("/config", response_model=RelationsPublic)
def get_config(
    session: SessionDep,
    domain_name: str | None = None,
    action_name: str | None = None,
) -> Any:
    """
    챗봇이 사용할 RelationDefinition 설정 조회.
    is_active=True인 항목만 반환.
    node_definition_id가 있는 경우 node의 entity_definitions를 병합하여 반환.
    """
    statement = select(RelationDefinition).where(RelationDefinition.is_active == True)
    count_statement = select(func.count()).select_from(RelationDefinition).where(RelationDefinition.is_active == True)

    if domain_name:
        statement = statement.where(RelationDefinition.domain_name == domain_name)
        count_statement = count_statement.where(RelationDefinition.domain_name == domain_name)
    if action_name is not None:
        statement = statement.where(RelationDefinition.action_name == action_name)
        count_statement = count_statement.where(RelationDefinition.action_name == action_name)

    count = session.exec(count_statement).one()
    relations = session.exec(statement.order_by(RelationDefinition.domain_name)).all()

    result = []
    for r in relations:
        r_public = RelationDefinitionPublic.model_validate(r)
        if r.node_definition_id and r.node_definition:
            r_public.entity_definitions = [
                EntityDefinitionPublic(
                    id=ne.id,
                    key=ne.key,
                    value_type=ne.value_type,
                    description=ne.description,
                    example_value=ne.example_value,
                    is_required=ne.is_required,
                    relation_definition_id=r.id,
                    created_at=ne.created_at,
                    updated_at=ne.updated_at,
                )
                for ne in r.node_definition.entity_definitions
            ]
        result.append(r_public)

    return RelationsPublic(data=result, count=count)
