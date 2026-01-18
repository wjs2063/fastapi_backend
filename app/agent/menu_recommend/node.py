from app.agent.menu_recommend.state import State
from app.infra.repository.rdb import RDBRepository


def fetch_user_info(state: State):
    """
    사용자의 정보를 가져옵니다.
    과거에 먹었던 식단 / 취향 등을 가져옵니다.
    """
    user_id = state.user_id

    recent_meal_logs : list = RDBRepository.get_recent_meal_logs_by_user_id(user_id)
    user_preference : str =
    pass


def fetch_weather_info(state: State):
    """
    현재 좌표의 날씨정보를 가져옵니다.
    """
    pass


