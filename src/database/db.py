import os
from sqlalchemy import create_engine, Engine
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from alembic import command
from alembic.config import Config
from config import settings

engine: Engine | None = None

# Use scoped_session to ensure thread safety
SessionLocal: scoped_session | None = None

Base = declarative_base()

# SQLite database URL
db_url = f"sqlite:///{settings.db_file_path}"
db_url_async = f"sqlite+aiosqlite:///{settings.db_file_path}"


def get_engine():
    global engine
    if engine is None:
        engine = create_engine(db_url, connect_args={"check_same_thread": False})
    return engine


def get_async_engine():
    return create_async_engine(db_url_async, echo=True)


def get_session() -> scoped_session:
    global SessionLocal
    if SessionLocal is None:
        SessionLocal = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
        )
    return SessionLocal()


#
# MODELS
#

from database.models import DocumentIndex, FileIndex  # noqa


async def init_db():
    """Initialize database"""
    # await migrate()


def run_upgrade(connection, cfg):
    cfg.attributes["connection"] = connection
    command.upgrade(cfg, "head")


async def migrate():
    """Run database migrations"""
    eng = get_async_engine()
    alembic_cfg = Config(settings.alembic_ini_path)
    alembic_cfg.set_main_option(
        "script_location", os.path.join(os.path.dirname(__file__), "../alembic")
    )
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    async with eng.begin() as conn:
        await conn.run_sync(run_upgrade, alembic_cfg)
