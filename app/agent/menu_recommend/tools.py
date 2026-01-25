from typing import List, Dict, Any, Type
import aiohttp
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from app.clients.naver import naver_search_client  # 이전에 만든 클라이언트
from pydantic.json_schema import SkipJsonSchema
from typing import Annotated, Type


# 1. 네이버 지역 검색 도구
class NaverSearchInput(BaseModel):
    query: str = Field(description="검색 키워드")
    start: int = Field(default=1, description="검색 시작 위치 (1~1000)")
    display: int = Field(default=5, description="결과 개수")


class NaverLocalSearchTool(BaseTool):
    name: str = "naver_local_search"
    description: str = "맛집을 검색합니다. 결과가 없으면 start를 늘려 재호출하세요."
    args_schema: Annotated[Type[BaseModel], SkipJsonSchema()] = NaverSearchInput

    async def _arun(self, query: str, start: int = 1, display: int = 5) -> List[Dict[str, Any]]:
        return await naver_search_client.search_local(query=query, start=start, display=display)

    def _run(self, *args, **kwargs): raise NotImplementedError


# 2. 캐치테이블 상세 정보 추출 도구 (Jina Reader 활용)
class CatchtableInput(BaseModel):
    url: str = Field(description="조회할 캐치테이블 상세 페이지 URL")


class CatchtableDetailTool(BaseTool):
    name: str = "get_catchtable_details"
    description: str = "캐치테이블 상세 정보를 가져옵니다."
    args_schema: Annotated[Type[BaseModel], SkipJsonSchema()] = CatchtableInput  # 이전 정의 참고

    async def _arun(self, url: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"https://r.jina.ai/{url}") as resp:
                return (await resp.text())[:3000] if resp.status == 200 else "실패"

    def _run(self, *args, **kwargs): raise NotImplementedError
