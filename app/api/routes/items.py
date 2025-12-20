import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlmodel import func, select
from sqlalchemy.orm import joinedload
from app.api.deps import CurrentUser, SessionDep
from app.models import Item, ItemCreate, ItemPublic, ItemsPublic, ItemUpdate, Message
from datetime import datetime
router = APIRouter(prefix="/items", tags=["items"])


@router.get("/", response_model=ItemsPublic)
def read_items(
    session: SessionDep, current_user: CurrentUser, skip: int = 0, limit: int = 100
) -> Any:
    """
    Retrieve all items (Visible to all logged-in users).
    """
    # 1. 전체 게시글 수 카운트 (주인 여부 상관없이 전체)
    count_statement = select(func.count()).select_from(Item)
    count = session.exec(count_statement).one()

    # 2. 전체 게시글 조회
    # - where 조건을 제거하여 모든 글을 가져옵니다.
    # - joinedload(Item.owner)는 유지하여 작성자 이름을 표시합니다.
    statement = (
        select(Item)
        .options(joinedload(Item.owner))
        .offset(skip)
        .limit(limit)
        .order_by(Item.created_at.desc()) # (선택사항) 최신글 순으로 정렬하려면 추가
    )
    items = session.exec(statement).all()

    return ItemsPublic(data=items, count=count)

@router.post("/", response_model=ItemPublic)
def create_item(
    *, session: SessionDep, current_user: CurrentUser, item_in: ItemCreate
) -> Any:
    """
    Create new item.
    """
    item = Item.model_validate(item_in, update={"owner_id": current_user.id})
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


@router.put("/{id}", response_model=ItemPublic)
def update_item(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    item_in: ItemUpdate,
) -> Any:
    """
    Update an item.
    """
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    if not current_user.is_superuser and (item.owner_id != current_user.id):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    update_dict = item_in.model_dump(exclude_unset=True)
    item.sqlmodel_update(update_dict)
    item.updated_at = datetime.now()
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


@router.delete("/{id}")
def delete_item(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> Message:
    """
    Delete an item.
    """
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    if not current_user.is_superuser and (item.owner_id != current_user.id):
        raise HTTPException(status_code=400, detail="Not enough permissions")
    session.delete(item)
    session.commit()
    return Message(message="Item deleted successfully")


# ... (기존 read_items 함수 아래에 추가)

@router.get("/mine", response_model=ItemsPublic)
def read_my_items(
    session: SessionDep, current_user: CurrentUser, skip: int = 0, limit: int = 100
) -> Any:
    """
    Retrieve only items created by the current user.
    """
    # 1. 내 글 개수 카운트
    count_statement = (
        select(func.count())
        .select_from(Item)
        .where(Item.owner_id == current_user.id)
    )
    count = session.exec(count_statement).one()

    # 2. 내 글 조회 (작성자 정보 포함)
    statement = (
        select(Item)
        .where(Item.owner_id == current_user.id)
        .options(joinedload(Item.owner))
        .offset(skip)
        .limit(limit)
        .order_by(Item.created_at.desc())
    )
    items = session.exec(statement).all()

    return ItemsPublic(data=items, count=count)