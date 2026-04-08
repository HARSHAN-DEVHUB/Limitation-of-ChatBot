from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_backend: str = os.getenv("LLM_BACKEND", "auto")
    llm_strict_backend: bool = os.getenv("LLM_STRICT_BACKEND", "false").lower() == "true"
    llm_model_path: str = os.getenv("LLM_MODEL_PATH", "models/model.gguf")
    llm_ctx_size: int = int(os.getenv("LLM_CTX_SIZE", "4096"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "256"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    hf_base_model: str = os.getenv("HF_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    hf_active_adapter_path: str = os.getenv("HF_ACTIVE_ADAPTER_PATH", "")
    lora_output_dir: Path = Path(os.getenv("LORA_OUTPUT_DIR", "models/adapters"))
    training_output_dir: Path = Path(os.getenv("TRAINING_OUTPUT_DIR", "models/training"))
    weight_update_state_path: Path = Path(
        os.getenv("WEIGHT_UPDATE_STATE_PATH", "data/processed/weekly_weight_state.json")
    )
    weight_update_min_new_chats: int = int(os.getenv("WEIGHT_UPDATE_MIN_NEW_CHATS", "20"))
    weight_update_max_steps: int = int(os.getenv("WEIGHT_UPDATE_MAX_STEPS", "120"))
    weight_update_learning_rate: float = float(os.getenv("WEIGHT_UPDATE_LEARNING_RATE", "2e-4"))
    weight_update_batch_size: int = int(os.getenv("WEIGHT_UPDATE_BATCH_SIZE", "1"))
    weight_gate_min_specificity: float = float(os.getenv("WEIGHT_GATE_MIN_SPECIFICITY", "0.5"))
    weight_gate_min_refusal_quality: float = float(
        os.getenv("WEIGHT_GATE_MIN_REFUSAL_QUALITY", "0.5")
    )
    weight_gate_min_citation_count: float = float(
        os.getenv("WEIGHT_GATE_MIN_CITATION_COUNT", "0.5")
    )
    weight_gate_max_eval_loss_increase: float = float(
        os.getenv("WEIGHT_GATE_MAX_EVAL_LOSS_INCREASE", "0.03")
    )
    weight_eval_suite_path: Path = Path(
        os.getenv("WEIGHT_EVAL_SUITE_PATH", "data/eval/university_support_questions.json")
    )
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
