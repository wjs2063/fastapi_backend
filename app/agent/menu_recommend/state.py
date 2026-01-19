import uuid
from datetime import datetime
from enum import Enum
from app.api.deps import CurrentUser, SessionDep
from langgraph.types import Command
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, EmailStr
from typing import List


class Gender(str, Enum):
    MALE = 'male'
    FEMALE = 'female'


class UserInfo(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    user_id: UUID
    name: str
    alias_name: str | None = None
    email: EmailStr

    # --- 건강 및 신체 정보 ---
    age: int = Field(default=19, description='age of user', ge=19, le=130)
    gender: Gender = Field(default=Gender.MALE, description='gender of user')
    weight_kg: float | None = Field(default=None, description='weight of user')
    height_cm: float | None = Field(default=None, description='height of user')
    activity_level: str | None = None  # Sedentary, Active, Very Active

    # --- 제한 사항 및 목표 ---
    health_goals: List[str] = Field(default_factory=list)  # ["다이어트", "근성장", "저염식"]
    allergies: List[str] = Field(default_factory=list)  # ["견과류", "갑각류"]
    medical_conditions: List[str] = Field(default_factory=list)  # ["당뇨", "고혈압"]

    # --- 취향 (Preference) ---
    preference: list[str] = Field(default_factory=list)
    disliked_foods: List[str] = Field(default_factory=list)  # ["오이", "민트초코"]
    favorite_cuisines: List[str] = Field(default_factory=list)  # ["한식", "일식"]
    spiciness_preference: int = Field(default=1, ge=1, le=5)  # 1(전혀 안매움) ~ 5(매우 매움)

    # -- 최근 식사 메뉴--
    recent_meal: list = Field(default_factory=list)


class State(BaseModel):
    user: UserInfo = UserInfo(user_id=uuid.uuid4(), name="test_name", email="test_email@test.com", age=19,
                              gender=Gender.MALE)

    messages: str = Field(description='messages of user', max_length=50000)
    current_location: tuple[float, float] = Field(default_factory=tuple, description='current location of user')
    current_time: datetime

    # network (like session)
    # rdb_session : SessionDep


from app.models import User


def init_agent_state(user: User, message: str, *args, **kwargs) -> dict:
    return {
        "user": UserInfo(user_id=user.id, name=user.full_name, email=user.email),
        "messages": message,
        "current_location": (127, 33),
        "current_time": datetime.now(),
    }
