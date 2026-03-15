import uuid
from enum import Enum
from datetime import datetime
from typing import Optional

from pydantic import EmailStr
from sqlmodel import Field, Relationship, SQLModel
from sqlalchemy import text, Column, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from uuid import UUID


# --- [Enums] ---
class MealType(str, Enum):
    BREAKFAST = "BREAKFAST"
    LUNCH = "LUNCH"
    DINNER = "DINNER"
    SNACK = "SNACK"


# --- [User Models] ---

class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = Field(default=None, max_length=255)


class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)
    created_at: datetime = Field(default_factory=datetime.now)


class UserRegister(SQLModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


class UserUpdate(UserBase):
    email: EmailStr | None = Field(default=None, max_length=255)
    password: str | None = Field(default=None, min_length=8, max_length=128)


class UserUpdateMe(SQLModel):
    full_name: str | None = Field(default=None, max_length=255)
    email: EmailStr | None = Field(default=None, max_length=255)


class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class User(UserBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    hashed_password: str
    items: list["Item"] = Relationship(back_populates="owner", cascade_delete=True)

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        nullable=False,
        sa_column_kwargs={
            "server_default": text("current_timestamp(0)")
        }
    )

    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        nullable=False,
        sa_column_kwargs={
            "server_default": text("current_timestamp(0)"),
            "onupdate": text("current_timestamp(0)")
        }
    )


class UserPublic(UserBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class UsersPublic(SQLModel):
    data: list[UserPublic]
    count: int


# --- [Item Models] ---

class ItemBase(SQLModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)
    is_solved: bool = Field(default=False, nullable=False)


class ItemCreate(ItemBase):
    pass


class ItemUpdate(ItemBase):
    title: str | None = Field(default=None, min_length=1, max_length=255)


class Item(ItemBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship(back_populates="items")
    created_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
        sa_column_kwargs={
            "server_default": text("current_timestamp(0)")
        }
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
        sa_column_kwargs={
            "server_default": text("current_timestamp(0)"),
            "onupdate": text("current_timestamp(0)")
        }
    )


class ItemPublic(ItemBase):
    id: uuid.UUID
    owner_id: uuid.UUID
    owner: UserPublic | None = None
    created_at: datetime
    updated_at: datetime


class ItemsPublic(SQLModel):
    data: list[ItemPublic]
    count: int


# --- [Auth & Tokens] ---

class Message(SQLModel):
    message: str


class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


class TokenPayload(SQLModel):
    sub: str | None = None


class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)


# --- [Menu Agent Models] ---

# 1. 사용자 취향
class UserPreference(SQLModel, table=True):
    user_id: UUID = Field(primary_key=True, foreign_key="user.id")
    tastes: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# 2. 식사 기록
class MealLogBase(SQLModel):
    menu_name: str
    meal_type: MealType = Field(default=MealType.LUNCH, nullable=True)


class MealLog(MealLogBase, table=True):
    id: int | None = Field(default=None, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id")
    # [중요] created_at은 DB 기본값이 있지만, API로 덮어씌울 수 있도록 필드 정의
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# [수정] 식사 기록 생성 시 날짜(created_at)도 받을 수 있도록 추가
class MealLogCreate(MealLogBase):
    created_at: datetime | None = None  # 선택 입력 (없으면 현재 시간)


# 식사 기록 수정 시 사용
class MealLogUpdate(SQLModel):
    menu_name: str | None = None
    meal_type: MealType | None = None
    created_at: datetime | None = None  # 수정 시 날짜 변경 가능하도록


class MealLogPublic(MealLogBase):
    id: int
    created_at: datetime
    updated_at: datetime


# --- [Node Definition Models] ---

class NodeDefinitionBase(SQLModel):
    label: str = Field(max_length=100)
    description: str | None = Field(default=None, max_length=500)


class NodeDefinitionCreate(NodeDefinitionBase):
    pass


class NodeDefinitionUpdate(SQLModel):
    label: str | None = Field(default=None, max_length=100)
    description: str | None = None


class NodeDefinition(NodeDefinitionBase, table=True):
    __tablename__ = "node_definition"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
        sa_column_kwargs={"server_default": text("current_timestamp(0)")}
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
        sa_column_kwargs={
            "server_default": text("current_timestamp(0)"),
            "onupdate": text("current_timestamp(0)")
        }
    )
    entity_definitions: list["NodeEntityDefinition"] = Relationship(
        back_populates="node",
        cascade_delete=True,
        sa_relationship_kwargs={"lazy": "selectin"}
    )
    relations: list["RelationDefinition"] = Relationship(back_populates="node_definition")


class NodeEntityDefinitionBase(SQLModel):
    key: str = Field(max_length=100)
    value_type: str = Field(default="str", max_length=50)
    description: str = Field(max_length=500)
    example_value: str | None = Field(default=None, max_length=255)
    is_required: bool = Field(default=False)


class NodeEntityDefinitionCreate(NodeEntityDefinitionBase):
    pass


class NodeEntityDefinitionUpdate(SQLModel):
    key: str | None = Field(default=None, max_length=100)
    value_type: str | None = Field(default=None, max_length=50)
    description: str | None = Field(default=None, max_length=500)
    example_value: str | None = None
    is_required: bool | None = None


class NodeEntityDefinition(NodeEntityDefinitionBase, table=True):
    __tablename__ = "node_entity_definition"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    node_definition_id: uuid.UUID = Field(
        foreign_key="node_definition.id",
        nullable=False,
        ondelete="CASCADE"
    )
    node: NodeDefinition | None = Relationship(back_populates="entity_definitions")
    created_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
        sa_column_kwargs={"server_default": text("current_timestamp(0)")}
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
        sa_column_kwargs={
            "server_default": text("current_timestamp(0)"),
            "onupdate": text("current_timestamp(0)")
        }
    )


class NodeEntityDefinitionPublic(NodeEntityDefinitionBase):
    model_config = {"from_attributes": True}
    id: uuid.UUID
    node_definition_id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class NodeDefinitionPublic(NodeDefinitionBase):
    model_config = {"from_attributes": True}
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    entity_definitions: list[NodeEntityDefinitionPublic] = []


class NodesPublic(SQLModel):
    data: list[NodeDefinitionPublic]
    count: int


# --- [Agent Memory Config Models] ---

# RelationDefinition: Domain-Action 당 Neo4j Relation 타입 정의
class RelationDefinitionBase(SQLModel):
    domain_name: str = Field(max_length=100)
    action_name: str | None = Field(default=None, max_length=100)  # None = 도메인 레벨
    relation_type: str = Field(max_length=100)  # Neo4j relation type e.g. "LIKES_FOOD"
    description: str = Field(max_length=500)
    target_node_label: str = Field(max_length=100)  # Neo4j target node label e.g. "Food"
    target_node_description: str | None = Field(default=None, max_length=500)  # 노드 설명
    node_definition_id: uuid.UUID | None = Field(default=None)  # FK to NodeDefinition
    is_active: bool = Field(default=True)
    is_global: bool = Field(default=False)  # True: 공유 노드 (장소/음식점 등, 크로스 유저 탐색 허브)
                                            # False: 유저 전용 노드 (개인 선호/메모 등, private)
    ttl_seconds: int | None = Field(default=None)  # Cache TTL in seconds (for future use)


class RelationDefinitionCreate(RelationDefinitionBase):
    pass


class RelationDefinitionUpdate(SQLModel):
    relation_type: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    target_node_label: str | None = Field(default=None, max_length=100)
    target_node_description: str | None = None
    node_definition_id: uuid.UUID | None = None
    is_active: bool | None = None
    is_global: bool | None = None
    ttl_seconds: int | None = None


class RelationDefinition(RelationDefinitionBase, table=True):
    __tablename__ = "relation_definition"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    node_definition_id: uuid.UUID | None = Field(
        default=None,
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("node_definition.id", ondelete="SET NULL"),
            nullable=True,
        )
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
        sa_column_kwargs={"server_default": text("current_timestamp(0)")}
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
        sa_column_kwargs={
            "server_default": text("current_timestamp(0)"),
            "onupdate": text("current_timestamp(0)")
        }
    )
    entity_definitions: list["EntityDefinition"] = Relationship(
        back_populates="relation",
        cascade_delete=True,
        sa_relationship_kwargs={"lazy": "selectin"}
    )
    node_definition: NodeDefinition | None = Relationship(
        back_populates="relations",
        sa_relationship_kwargs={"lazy": "selectin"}
    )


class RelationDefinitionPublic(RelationDefinitionBase):
    model_config = {"from_attributes": True}
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    entity_definitions: list["EntityDefinitionPublic"] = []
    node_definition_id: uuid.UUID | None = None
    node_definition: NodeDefinitionPublic | None = None


class RelationsPublic(SQLModel):
    data: list[RelationDefinitionPublic]
    count: int


# EntityDefinition: Relation에서 추출할 key-value 엔티티 정의
class EntityDefinitionBase(SQLModel):
    key: str = Field(max_length=100)
    value_type: str = Field(default="str", max_length=50)  # "str", "int", "float", "bool", "list"
    description: str = Field(max_length=500)
    example_value: str | None = Field(default=None, max_length=255)
    is_required: bool = Field(default=False)


class EntityDefinitionCreate(EntityDefinitionBase):
    pass


class EntityDefinitionUpdate(SQLModel):
    key: str | None = Field(default=None, max_length=100)
    value_type: str | None = Field(default=None, max_length=50)
    description: str | None = Field(default=None, max_length=500)
    example_value: str | None = None
    is_required: bool | None = None


class EntityDefinition(EntityDefinitionBase, table=True):
    __tablename__ = "entity_definition"
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    relation_definition_id: uuid.UUID = Field(
        foreign_key="relation_definition.id",
        nullable=False,
        ondelete="CASCADE"
    )
    relation: RelationDefinition | None = Relationship(back_populates="entity_definitions")
    created_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
        sa_column_kwargs={"server_default": text("current_timestamp(0)")}
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        nullable=False,
        sa_column_kwargs={
            "server_default": text("current_timestamp(0)"),
            "onupdate": text("current_timestamp(0)")
        }
    )


class EntityDefinitionPublic(EntityDefinitionBase):
    model_config = {"from_attributes": True}
    id: uuid.UUID
    relation_definition_id: uuid.UUID
    created_at: datetime
    updated_at: datetime