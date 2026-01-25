import urllib.parse
from typing import Any, Dict, Optional, List, Tuple

from app.clients.base import BaseClient
from app.core.config import settings
from app.core.exceptions import ExternalAPIError


class TMapClient(BaseClient):
    """
    T-Map API 연동을 위한 클라이언트 클래스.
    보행자 경로 안내 서비스를 중심으로 SK Open API와의 통신을 담당합니다.

    Attributes:
        base_url (str): T-Map API 기본 URL (https://apis.openapi.sk.com)
        headers (dict): appKey 및 Content-Type을 포함한 공통 헤더
    """

    # 경로 탐색 옵션 상수화
    OPTION_RECOMMENDED = 0  # 추천
    OPTION_RECOMMENDED_MAIN = 4  # 추천 + 대로 우선
    OPTION_SHORTEST = 10  # 최단거리
    OPTION_SHORTEST_NO_STAIR = 30  # 최단거리 + 계단 제외

    def __init__(self):
        super().__init__(base_url="https://apis.openapi.sk.com")
        self.headers = {
            "appKey": settings.SK_MAP_API_KEY,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

    def _url_encode(self, text: str) -> str:
        """
        T-Map 명세에 따라 텍스트를 UTF-8 기반으로 URL 인코딩합니다.

        Args:
            text (str): 인코딩할 문자열 (예: "광치기해변")

        Returns:
            str: 인코딩된 문자열 (예: "%EA%B4%91%EC%B9%98%EA%B8%B0%ED%95%B4%EB%B3%80")
        """
        return urllib.parse.quote(text)

    async def get_pedestrian_route(
            self,
            start_x: float,
            start_y: float,
            end_x: float,
            end_y: float,
            start_name: str = "출발지",
            end_name: str = "목적지",
            search_option: int = 0,
            sort: str = "index",
            pass_list: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        보행자 기준으로 출발지와 목적지까지의 경로 정보를 요청합니다. (GeoJSON 응답)

        Args:
            start_x (float): 출발지 X 좌표 (경도, 예: 126.9246033)
            start_y (float): 출발지 Y 좌표 (위도, 예: 33.45241976)
            end_x (float): 목적지 X 좌표 (경도, 예: 126.9041895)
            end_y (float): 목적지 Y 좌표 (위도, 예: 33.4048969)
            start_name (str): 출발지 명칭 (UTF-8 인코딩 처리됨)
            end_name (str): 목적지 명칭 (UTF-8 인코딩 처리됨)
            search_option (int): 경로 탐색 옵션.
                - 0: 추천 (기본값)
                - 4: 추천 + 대로 우선
                - 10: 최단거리
                - 30: 최단거리 + 계단 제외
            sort (str): 지리 정보 개체 정렬 순서. (index: 인덱스 순, custom: 라인/포인트 순)
            pass_list (str, optional): 경유지 정보. "x1,y1_x2,y2" 형식 (최대 5개)

        Returns:
            Dict[str, Any]: T-Map에서 반환한 GeoJSON 형식의 FeatureCollection 데이터

        Raises:
            ExternalAPIError: API 응답이 4xx, 5xx 에러인 경우 발생
        """
        endpoint = "/tmap/routes/pedestrian?version=1"

        payload = {
            "startX": start_x,
            "startY": start_y,
            "endX": end_x,
            "endY": end_y,
            "startName": self._url_encode(start_name),
            "endName": self._url_encode(end_name),
            "searchOption": str(search_option),
            "sort": sort,
            "resCoordType": "WGS84GEO",
            "reqCoordType": "WGS84GEO"
        }

        if pass_list:
            payload["passList"] = pass_list

        return await self.request(
            "POST",
            endpoint,
            json=payload,
            headers=self.headers
        )

    async def get_route_summary(
            self,
            start_coords: Tuple[float, float],
            end_coords: Tuple[float, float],
            start_name: str = "출발지",
            end_name: str = "목적지",
            option: int = 10
    ) -> Dict[str, Any]:
        """
        보행자 경로 안내 응답에서 실질적으로 필요한 요약 정보(총 거리, 소요 시간)만 추출합니다.

        Args:
            start_coords (Tuple[float, float]): 출발지 (위도, 경도)
            end_coords (Tuple[float, float]): 목적지 (위도, 경도)
            start_name (str): 출발지 명칭
            end_name (str): 목적지 명칭
            option (int): 경로 탐색 옵션 (기본값: 최단거리 10)

        Returns:
            Dict[str, Any]: {
                "total_distance_m": 총 거리 (미터),
                "total_time_sec": 총 소요 시간 (초),
                "description": 첫 번째 안내 메시지
            }

        Example:
            summary = await tmap_client.get_route_summary((37.1, 127.1), (37.2, 127.2))
            print(summary["total_distance_m"]) # 6337
        """
        full_data = await self.get_pedestrian_route(
            start_x=start_coords[1],  # 경도
            start_y=start_coords[0],  # 위도
            end_x=end_coords[1],
            end_y=end_coords[0],
            start_name=start_name,
            end_name=end_name,
            search_option=option
        )

        try:
            # GeoJSON의 첫 번째 Feature(Point/SP)에 전체 요약 정보가 포함됨
            properties = full_data["features"][0]["properties"]
            return {
                "total_distance_m": properties.get("totalDistance"),
                "total_time_sec": properties.get("totalTime"),
                "description": properties.get("description")
            }
        except (KeyError, IndexError) as e:
            raise ExternalAPIError(
                service_name="T-Map Pedestrian API",
                status_code=500,
                detail=f"응답 데이터 파싱 중 오류가 발생했습니다: {str(e)}"
            )


# 싱글톤 인스턴스 제공
tmap_client = TMapClient()

