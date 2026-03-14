import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlmodel import func, select

from app.api.deps import CurrentUser, SessionDep
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


# ── Relations ──────────────────────────────────────────────────

@router.get("/relations", response_model=RelationsPublic)
def list_relations(
    session: SessionDep,
    current_user: CurrentUser,
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
    current_user: CurrentUser,
    relation_in: RelationDefinitionCreate,
) -> Any:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="슈퍼유저만 접근 가능합니다.")
    relation = RelationDefinition.model_validate(relation_in)
    session.add(relation)
    session.commit()
    session.refresh(relation)
    return relation


@router.get("/relations/{relation_id}", response_model=RelationDefinitionPublic)
def get_relation(
    *,
    session: SessionDep,
    current_user: CurrentUser,
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
    current_user: CurrentUser,
    relation_id: uuid.UUID,
    relation_in: RelationDefinitionUpdate,
) -> Any:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="슈퍼유저만 접근 가능합니다.")
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
    current_user: CurrentUser,
    relation_id: uuid.UUID,
) -> Message:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="슈퍼유저만 접근 가능합니다.")
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
    current_user: CurrentUser,
    relation_id: uuid.UUID,
    entity_in: EntityDefinitionCreate,
) -> Any:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="슈퍼유저만 접근 가능합니다.")
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
    current_user: CurrentUser,
    entity_id: uuid.UUID,
    entity_in: EntityDefinitionUpdate,
) -> Any:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="슈퍼유저만 접근 가능합니다.")
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
    current_user: CurrentUser,
    entity_id: uuid.UUID,
) -> Message:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="슈퍼유저만 접근 가능합니다.")
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
    current_user: CurrentUser,
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
    current_user: CurrentUser,
    node_in: NodeDefinitionCreate,
) -> Any:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="슈퍼유저만 접근 가능합니다.")
    node = NodeDefinition.model_validate(node_in)
    session.add(node)
    session.commit()
    session.refresh(node)
    return node


@router.get("/nodes/{node_id}", response_model=NodeDefinitionPublic)
def get_node(
    *,
    session: SessionDep,
    current_user: CurrentUser,
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
    current_user: CurrentUser,
    node_id: uuid.UUID,
    node_in: NodeDefinitionUpdate,
) -> Any:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="슈퍼유저만 접근 가능합니다.")
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
    current_user: CurrentUser,
    node_id: uuid.UUID,
) -> Message:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="슈퍼유저만 접근 가능합니다.")
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
    current_user: CurrentUser,
    node_id: uuid.UUID,
    entity_in: NodeEntityDefinitionCreate,
) -> Any:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="슈퍼유저만 접근 가능합니다.")
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
    current_user: CurrentUser,
    entity_id: uuid.UUID,
    entity_in: NodeEntityDefinitionUpdate,
) -> Any:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="슈퍼유저만 접근 가능합니다.")
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
    current_user: CurrentUser,
    entity_id: uuid.UUID,
) -> Message:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="슈퍼유저만 접근 가능합니다.")
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
    인증 불필요 - 챗봇 서비스에서 직접 호출.
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
        # node_definition_id가 있으면 node의 entity를 우선 사용
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
