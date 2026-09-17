from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = ""

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    chroma_dir: str = "./chroma_db"

    chunk_size: int = 700
    chunk_overlap: int = 120

    top_k: int = 5

    relevance_threshold: float = 0.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
