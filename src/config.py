from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "pgvector"
    db_password: str = "pgvector"
    db_dbname: str = "pgvector"

    api_base: str = "https://api.openai.com/v1/"

    class Config:
        env_prefix = "KRA_"


settings = Settings()
