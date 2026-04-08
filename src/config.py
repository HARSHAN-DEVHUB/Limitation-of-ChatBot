from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_backend: str = os.getenv("LLM_BACKEND", "auto")
    llm_model_path: str = os.getenv("LLM_MODEL_PATH", "models/model.gguf")
    llm_ctx_size: int = int(os.getenv("LLM_CTX_SIZE", "4096"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "256"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    top_k: int = int(os.getenv("TOP_K", "5"))
    min_retrieval_score: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.35"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    index_path: Path = Path(os.getenv("INDEX_PATH", "data/index/faiss.index"))
    index_meta_path: Path = Path(
        os.getenv("INDEX_META_PATH", "data/index/chunks.jsonl")
    )
    sqlite_path: Path = Path(os.getenv("SQLITE_PATH", "data/chatbot.db"))
    raw_data_dir: Path = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
    processed_data_dir: Path = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))


settings = Settings()
