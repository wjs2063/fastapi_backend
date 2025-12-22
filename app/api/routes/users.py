import uuid
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import col, delete, func, select

from app import crud
from app.api.deps import (
    CurrentUser,
    SessionDep,
    get_current_active_superuser,
)
from datetime import datetime
from app.core.config import settings
from app.core.security import get_password_hash, verify_password
from app.models import (
    Item,
    Message,
    UpdatePassword,
    User,
    UserCreate,
    UserPublic,
    UserRegister,
    UsersPublic,
    UserUpdate,
    UserUpdateMe,
)
from app.utils import generate_new_account_email, send_email
from fastapi import APIRouter, Depends
from sqlmodel import Session, select
from app.api.deps import get_db, get_current_user
from app.models import User, UserPreference, MealLog
from pydantic import BaseModel

router = APIRouter(prefix="/users", tags=["users"])


@router.get(
    "/",
    dependencies=[Depends(get_current_active_superuser)],
    response_model=UsersPublic,
)
def read_users(session: SessionDep, skip: int = 0, limit: int = 100) -> Any:
    """
    Retrieve users.
    """

    count_statement = select(func.count()).select_from(User)
    count = session.exec(count_statement).one()

    statement = select(User).offset(skip).limit(limit)
    users = session.exec(statement).all()

    return UsersPublic(data=users, count=count)


@router.post(
    "/", dependencies=[Depends(get_current_active_superuser)], response_model=UserPublic
)
def create_user(*, session: SessionDep, user_in: UserCreate) -> Any:
    """
    Create new user.
    """
    user = crud.get_user_by_email(session=session, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )

    user = crud.create_user(session=session, user_create=user_in)
    if settings.emails_enabled and user_in.email:
        email_data = generate_new_account_email(
            email_to=user_in.email, username=user_in.email, password=user_in.password
        )
        send_email(
            email_to=user_in.email,
            subject=email_data.subject,
            html_content=email_data.html_content,
        )
    return user


@router.patch("/me", response_model=UserPublic)
def update_user_me(
        *, session: SessionDep, user_in: UserUpdateMe, current_user: CurrentUser
) -> Any:
    """
    Update own user.
    """

    if user_in.email:
        existing_user = crud.get_user_by_email(session=session, email=user_in.email)
        if existing_user and existing_user.id != current_user.id:
            raise HTTPException(
                status_code=409, detail="User with this email already exists"
            )
    user_data = user_in.model_dump(exclude_unset=True)
    current_user.sqlmodel_update(user_data)
    session.add(current_user)
    session.commit()
    session.refresh(current_user)
    return current_user


@router.patch("/me/password", response_model=Message)
def update_password_me(
        *, session: SessionDep, body: UpdatePassword, current_user: CurrentUser
) -> Any:
    """
    Update own password.
    """
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect password")
    if body.current_password == body.new_password:
        raise HTTPException(
            status_code=400, detail="New password cannot be the same as the current one"
        )
    hashed_password = get_password_hash(body.new_password)
    current_user.hashed_password = hashed_password
    session.add(current_user)
    session.commit()
    return Message(message="Password updated successfully")


@router.get("/me", response_model=UserPublic)
def read_user_me(current_user: CurrentUser) -> Any:
    """
    Get current user.
    """
    return current_user


@router.delete("/me", response_model=Message)
def delete_user_me(session: SessionDep, current_user: CurrentUser) -> Any:
    """
    Delete own user.
    """
    if current_user.is_superuser:
        raise HTTPException(
            status_code=403, detail="Super users are not allowed to delete themselves"
        )
    session.delete(current_user)
    session.commit()
    return Message(message="User deleted successfully")


@router.post("/signup", response_model=UserPublic)
def register_user(session: SessionDep, user_in: UserRegister) -> Any:
    """
    Create new user without the need to be logged in.
    """
    user = crud.get_user_by_email(session=session, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system",
        )
    user_create = UserCreate.model_validate(user_in)
    user = crud.create_user(session=session, user_create=user_create)
    return user


@router.get("/{user_id}", response_model=UserPublic)
def read_user_by_id(
        user_id: uuid.UUID, session: SessionDep, current_user: CurrentUser
) -> Any:
    """
    Get a specific user by id.
    """
    user = session.get(User, user_id)
    if user == current_user:
        return user
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403,
            detail="The user doesn't have enough privileges",
        )
    return user


@router.patch(
    "/{user_id}",
    dependencies=[Depends(get_current_active_superuser)],
    response_model=UserPublic,
)
def update_user(
        *,
        session: SessionDep,
        user_id: uuid.UUID,
        user_in: UserUpdate,
) -> Any:
    """
    Update a user.
    """

    db_user = session.get(User, user_id)
    if not db_user:
        raise HTTPException(
            status_code=404,
            detail="The user with this id does not exist in the system",
        )
    if user_in.email:
        existing_user = crud.get_user_by_email(session=session, email=user_in.email)
        if existing_user and existing_user.id != user_id:
            raise HTTPException(
                status_code=409, detail="User with this email already exists"
            )

    db_user = crud.update_user(session=session, db_user=db_user, user_in=user_in)
    return db_user


@router.delete("/{user_id}", dependencies=[Depends(get_current_active_superuser)])
def delete_user(
        session: SessionDep, current_user: CurrentUser, user_id: uuid.UUID
) -> Message:
    """
    Delete a user.
    """
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user == current_user:
        raise HTTPException(
            status_code=403, detail="Super users are not allowed to delete themselves"
        )
    statement = delete(Item).where(col(Item.owner_id) == user_id)
    session.exec(statement)  # type: ignore
    session.delete(user)
    session.commit()
    return Message(message="User deleted successfully")


# 요청 스키마
class PreferenceUpdate(BaseModel):
    tastes: str


class MealCreate(BaseModel):
    menu_name: str


@router.get("/me/preferences", response_model=UserPreference)
def read_user_preferences(
        session: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
) -> Any:
    """
    내 취향 정보를 조회합니다.
    정보가 없으면 기본값(빈 문자열)을 반환합니다.
    """
    pref = session.get(UserPreference, current_user.id)

    if not pref:
        # 데이터가 없으면 404 대신, 빈 객체를 만들어서 반환 (프론트엔드 오류 방지)
        return UserPreference(user_id=current_user.id, tastes="")

    return pref


# 1. 취향 업데이트 (UPSERT)
@router.put("/me/preferences")
def update_preferences(
        data: PreferenceUpdate,
        session: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    # 기존 데이터 확인
    pref = session.get(UserPreference, current_user.id)
    if not pref:
        # 없으면 생성
        pref = UserPreference(user_id=current_user.id, tastes=data.tastes)
        session.add(pref)
    else:
        # 있으면 수정
        pref.tastes = data.tastes
        pref.updated_at = datetime.now()  # 수정 시간 갱신
        session.add(pref)

    session.commit()
    return pref


# 2. 식사 기록 추가 (INSERT)
@router.post("/me/meals")
def add_meal_log(
        data: MealCreate,
        session: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    meal = MealLog(user_id=current_user.id, menu_name=data.menu_name)
    session.add(meal)
    session.commit()
    return {"message": "Meal logged"}


@router.get("/me/meals", response_model=List[MealLog])
def read_my_meals(
        session: SessionDep,
        current_user: CurrentUser,
        skip: int = 0,
        limit: int = 50,  # 기본 50개 조회
) -> Any:
    """
    Retrieve current user's meal history.
    """
    statement = (
        select(MealLog)
        .where(MealLog.user_id == current_user.id)
        .order_by(MealLog.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    meals = session.exec(statement).all()
    return meals


# [추가] 4. 식사 기록 삭제 (DELETE) - 실수로 입력한 경우
@router.delete("/me/meals/{meal_id}")
def delete_my_meal(
        meal_id: int,
        session: SessionDep,
        current_user: CurrentUser,
) -> Any:
    """
    Delete a meal log.
    """
    meal = session.get(MealLog, meal_id)
    if not meal:
        raise HTTPException(status_code=404, detail="Meal log not found")
    if meal.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    session.delete(meal)
    session.commit()
    return {"message": "Meal deleted successfully"}
