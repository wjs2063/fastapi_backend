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
    count_statement = select(func.count()).select_from(Item)
    count = session.exec(count_statement).one()

    statement = (
        select(Item)
        .options(joinedload(Item.owner))
        .offset(skip)
        .limit(limit)
        .order_by(Item.created_at.desc())
    )
    items = session.exec(statement).all()
    return ItemsPublic(data=items, count=count)


@router.post("/", response_model=ItemPublic)
def create_item(
        *, session: SessionDep, current_user: CurrentUser, item_in: ItemCreate
) -> Any:
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
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    # [수정] 권한 체크: 관리자가 아니고 소유자도 아니면 403 에러
    if not current_user.is_superuser and (item.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="본인의 글만 수정할 수 있습니다.")

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
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    # [수정] 권한 체크: 관리자가 아니고 소유자도 아니면 403 에러
    if not current_user.is_superuser and (item.owner_id != current_user.id):
        raise HTTPException(status_code=403, detail="본인의 글만 삭제할 수 있습니다.")

    session.delete(item)
    session.commit()
    return Message(message="Item deleted successfully")


@router.get("/mine", response_model=ItemsPublic)
def read_my_items(
        session: SessionDep, current_user: CurrentUser, skip: int = 0, limit: int = 100
) -> Any:
    count_statement = (
        select(func.count())
        .select_from(Item)
        .where(Item.owner_id == current_user.id)
    )
    count = session.exec(count_statement).one()

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