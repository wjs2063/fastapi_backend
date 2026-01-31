import asyncio
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from fastapi.routing import APIRoute
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from starlette.middleware.cors import CORSMiddleware
from core.db import connection_pool
from app.agent.menu_recommend.neo4j_db import Neo4jManager
from app.api.main import api_router
from app.core.config import settings


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


if settings.SENTRY_DSN and settings.ENVIRONMENT != "local":
    sentry_sdk.init(dsn=str(settings.SENTRY_DSN), enable_tracing=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # [Startup] 서버 시작 시 드라이버 풀 생성
    await Neo4jManager.init()
    async with connection_pool.connection() as conn:
        saver = AsyncPostgresSaver(conn)
        store = AsyncPostgresStore(conn)
        await asyncio.gather(saver.setup(), store.setup())
        print("LangGraph Storage Setup Completed.")

    yield
    # 앱 종료 시 풀 닫기
    await connection_pool.close()
    # [Shutdown] 서버 종료 시 드라이버 풀 해제
    await Neo4jManager.close()


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan,
)

# Set all CORS enabled origins
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.API_V1_STR)
