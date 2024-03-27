import os
from pydantic_settings import BaseSettings
from xdg_base_dirs import xdg_data_home

config_dir = os.path.dirname(os.path.abspath(__file__))

kra_dir = "kra"


class Settings(BaseSettings):
    data_dir: str = os.path.join(xdg_data_home(), kra_dir)

    db_file_path: str = os.path.join(data_dir, "kra.db")

    cache_dir: str = os.path.join(data_dir, "ingestion_cache")
    cache_path: str = os.path.join(cache_dir, "cache")

    api_base: str = "http://localhost:8080/v1/"

    alembic_ini_path: str = os.path.join(config_dir, "alembic.ini")
    logging_conf_path: str = os.path.join(config_dir, "log_conf.yaml")

    vector_store_dir: str = os.path.join(data_dir, "vector_store")

    # get xdg data home

    debug: bool = False

    class Config:
        env_prefix = "KRA_"


settings = Settings()
