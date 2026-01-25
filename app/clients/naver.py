from typing import Any, Dict, Optional, Tuple, List
from app.clients.base import BaseClient
from app.core.config import settings
from app.core.exceptions import ExternalAPIError
import aiohttp

class NaverMapClient(BaseClient):
    """
    Naver Cloud Platform Maps Geocoding & Reverse Geocoding API 연동 클라이언트.
    주소-좌표 간 상호 변환 기능을 제공합니다.
    """

    def __init__(self):
        # NCP Maps API 기본 URL
        super().__init__(base_url="https://maps.apigw.ntruss.com")
        self.headers = {
            "x-ncp-apigw-api-key-id": settings.NAVER_MAP_CLIENT_ID,
            "x-ncp-apigw-api-key": settings.NAVER_MAP_CLIENT_SECRET,
            "Accept": "application/json"
        }

    async def geocode(
            self,
            query: str,
            coordinate: Optional[str] = None,
            filter_type: Optional[str] = None,
            count: int = 10,
            page: int = 1
    ) -> Dict[str, Any]:
        """
        [Geocoding] 주소 문자열을 좌표로 변환합니다./gc
        """
        endpoint = "/map-geocode/v2/geocode"
        params = {
            "query": query,
            "count": count,
            "page": page
        }

        if coordinate:
            params["coordinate"] = coordinate
        if filter_type:
            params["filter"] = filter_type

        return await self.request(
            "GET",
            endpoint,
            params=params,
            headers=self.headers
        )

    async def reverse_geocode(
            self,
            lat: float,
            lng: float,
            orders: str = "legalcode,admcode,addr,roadaddr"
    ) -> Dict[str, Any]:
        """
        [Reverse Geocoding] 위경도 좌표를 주소 정보로 변환합니다.

        Args:
            lat (float): 위도
            lng (float): 경도
            orders (str): 변환 타겟 타입 (법정동, 행정동, 지번, 도로명)
        """
        endpoint = "/map-reversegeocode/v2/gc"
        # 좌표 형식: "경도,위도"
        params = {
            "coords": f"{lng},{lat}",
            "orders": orders,
            "output": "json"
        }

        return await self.request(
            "GET",
            endpoint,
            params=params,
            headers=self.headers
        )

    async def get_coordinates(self, address: str) -> Optional[Tuple[float, float]]:
        """
        주소를 받아 (위도, 경도)를 반환하는 편의 메서드 (Geocoding)
        """
        try:
            result = await self.geocode(query=address, count=1)

            if result.get("status") == "OK" and result["meta"]["totalCount"] > 0:
                address_info = result["addresses"][0]
                # x: 경도(lng), y: 위도(lat)
                return float(address_info["y"]), float(address_info["x"])

            return None
        except Exception as e:
            raise ExternalAPIError(
                service_name="Naver Geocoding API",
                status_code=500,
                detail=f"좌표 변환 중 오류: {str(e)}"
            )

    async def get_address(self, lat: float, lng: float) -> Optional[str]:
        """
        위경도 좌표를 받아 읽기 쉬운 주소 문자열을 반환하는 편의 메서드 (Reverse Geocoding)
        예: "서울특별시 강남구 역삼동"
        """
        try:
            result = await self.reverse_geocode(lat, lng)

            if result.get("status", {}).get("code") == 0:
                results = result.get("results", [])
                if not results:
                    return None

                # 첫 번째 결과에서 지역 명칭 추출
                region = results[0].get("region", {})
                area1 = region.get("area1", {}).get("name", "")  # 시/도
                area2 = region.get("area2", {}).get("name", "")  # 시/군/구
                area3 = region.get("area3", {}).get("name", "")  # 읍/면/동
                area4 = region.get("area4", {}).get("name", "")  # 리

                address_parts = [area1, area2, area3, area4]
                return " ".join([p for p in address_parts if p]).strip()

            return None
        except Exception as e:
            raise ExternalAPIError(
                service_name="Naver Reverse Geocoding API",
                status_code=500,
                detail=f"주소 변환 중 오류: {str(e)}"
            )

from typing import Any, Dict, List, Optional
from app.clients.base import BaseClient
from app.core.config import settings
from app.core.exceptions import ExternalAPIError

class NaverSearchClient(BaseClient):
    """
    네이버 검색 API - 지역 검색(Local) 연동 클라이언트.
    업체명, 주소, 카테고리 및 위치(KATECH 좌표) 정보를 제공합니다.
    """

    def __init__(self):
        # 검색 API용 기본 URL
        super().__init__(base_url="https://openapi.naver.com")
        self.headers = {
            "X-Naver-Client-Id": settings.NAVER_DEV_CLIENT_ID, # 발급받은 ID
            "X-Naver-Client-Secret": settings.NAVER_DEV_CLIENT_SECRET, # 발급받은 Secret
            "Accept": "application/json"
        }

    async def search_local(
        self,
        query: str,
        display: int = 5,
        start: int = 1,
        sort: str = "random"
    ) -> List[Dict[str, Any]]:
        """
        키워드로 지역 업체를 검색하고 아이템 리스트를 반환합니다.

        Args:
            query (str): 검색어 (예: "판교 맛집")
            display (int): 한 번에 표시할 결과 개수 (최대 5)
            start (int): 검색 시작 위치
            sort (str): 정렬 방식 (random: 정확도, comment: 카페/블로그 리뷰 순)

        Returns:
            List[Dict[str, Any]]: 업체 정보 목록 (title, link, address, mapx, mapy 등)
        """
        endpoint = "/v1/search/local.json"
        params = {
            "query": query,
            "display": display,
            "start": start,
            "sort": sort
        }

        response = await self.request(
            "GET",
            endpoint,
            params=params,
            headers=self.headers
        )

        # 결과에서 실제 업체 목록(items)만 추출하여 반환
        return response.get("items", [])






# 싱글톤 인스턴스
naver_map_client = NaverMapClient()
# 싱글톤 인스턴스
naver_search_client = NaverSearchClient()

async def find_my_office():
    # 주소를 좌표로 변환
    coords = await naver_map_client.reverse_geocode("경기도 성남시 분당구 불정로 6")

    if coords:
        lat, lng = coords
        print(f"위도: {lat}, 경도: {lng}")
        # 이후 TMapClient.get_route_summary(start_coords=(lat, lng), ...) 호출 가능



