from typing import List, Optional
from uuid import UUID
from sqlalchemy import select, desc
from sqlalchemy.orm import Session
from app.models import MealLog, UserPreference  # 모델 경로에 맞춰 수정하세요


class RDBRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_meal_logs_by_user_id(
            self,
            *,
            user_id: UUID,
            skip: int = 0,
            limit: int = 100
    ) -> List[MealLog]:
        """
        특정 사용자의 과거 식단 이력을 최신순으로 가져옵니다.
        """
        # 1. 쿼리 생성 (created_at 기준 내림차순 정렬)
        query = (
            select(MealLog)
            .where(MealLog.user_id == user_id)
            .order_by(desc(MealLog.created_at))
            .offset(skip)
            .limit(limit)
        )

        result = self.session.execute(query)

        return list(result.scalars().all())

    def get_recent_meal_logs_by_user_id(
            self,
            *,
            user_id: UUID,
            count: int = 5
    ) -> List[MealLog]:
        """
        추천 에이전트가 참고할 수 있도록 가장 최근에 먹은 N개의 식단만 가져옵니다.
        """
        return self.get_meal_logs_by_user_id(user_id=user_id, limit=count)

    def get_user_preferences_by_user_id(
            self,
            *,
            user_id: UUID,
    ) -> List[UserPreference]:
        query = (
            select(UserPreference).where(UserPreference.user_id == user_id)
        )

        result = self.session.execute(query)

        return list(result.scalars().all())
