from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool
from sqlmodel import Session, create_engine, select

from app import crud
from app.core.config import settings
from app.models import User, UserCreate

engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))


# make sure all SQLModel models are imported (app.models) before initializing DB
# otherwise, SQLModel might fail to initialize relationships properly
# for more details: https://github.com/fastapi/full-stack-fastapi-template/issues/28
# DB 연결 설정 (환경변수 관리 권장)
DB_URI = "postgresql://postgres:changethis@localhost:5432"
connection_pool = AsyncConnectionPool(
    conninfo=DB_URI, max_size=20, kwargs={"autocommit": True}
)


async def init_storage():
    # 1. Checkpointer (Short-term / HITL)
    saver = AsyncPostgresSaver(connection_pool)
    # 2. Store (Long-term / Semantic Memory)
    store = AsyncPostgresStore(connection_pool)

    # 처음 실행 시 테이블 생성
    await saver.setup()
    await store.setup()

    return saver, store


def init_db(session: Session) -> None:
    # Tables should be created with Alembic migrations
    # But if you don't want to use migrations, create
    # the tables un-commenting the next lines
    # from sqlmodel import SQLModel

    # This works because the models are already imported and registered from app.models
    # SQLModel.metadata.create_all(engine)

    user = session.exec(
        select(User).where(User.email == settings.FIRST_SUPERUSER)
    ).first()
    if not user:
        user_in = UserCreate(
            email=settings.FIRST_SUPERUSER,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            is_superuser=True,
        )
        user = crud.create_user(session=session, user_create=user_in)
