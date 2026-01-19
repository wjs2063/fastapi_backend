from langgraph.config import RunnableConfig
from app.agent.menu_recommend.state import State
from app.infra.repository.rdb import RDBRepository


def fetch_user_info(state: State, config: RunnableConfig):
    """
    사용자의 정보를 가져옵니다.
    과거에 먹었던 식단 / 취향 등을 가져옵니다.
    Args:
        state:
        config:

    Returns:

    """

    rdb_session = config.get("configurable", {}).get("rdb_session")

    user_id = state.user.user_id
    rdb_repository = RDBRepository(rdb_session)
    recent_meal_logs: list = rdb_repository.get_recent_meal_logs_by_user_id(user_id=user_id)
    user_preference: list = rdb_repository.get_user_preferences_by_user_id(user_id=user_id)

    update_user = state.user.model_copy(
        update={"recent_meal": recent_meal_logs, "preference": user_preference})

    return {"user": update_user}


def fetch_weather_info(state: State):
    """
    현재 좌표의 날씨를 가져옵니다
    Args:
        state:

    Returns:

    """
    """
    현재 좌표의 날씨정보를 가져옵니다.
    """
    pass


def retrieve_restaurant(state: State):
    """
    KAKAO
    NAVER
    GOOGLE
    지도API를 이용하여 사용자의 취향에 맞는 응답을 가져옵니다
    Args:
        state:

    Returns:

    """
    # kakako_map

    # naver_map

    # google_map
    pass


def generate_response(state: State):
    """
    Args:
        state:

    Returns:

    """

    pass
