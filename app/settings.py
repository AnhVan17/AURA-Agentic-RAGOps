from pydantic import BaseModel
from dotenv import load_dotenv
import os
import yaml

load_dotenv()


class AppConfig(BaseModel):
    name: str = "academic-ocr-rag-chatbot"
    version: str = "1.0.0"
    default_llm: str = "gemini-2.5-flash"  # Tập trung model tại đây


class ToyConfig(BaseModel):
    enabled: bool = True
    model: str = "gemini-2.5-flash" # Chỉnh lại model thực tế
    max_output_tokens: int = 1024

class GuardrailConfig(BaseModel):
    min_docs: int = 2
    min_ctx_tokens: int = 150
    fallback_lang: str = "vi"

class SummaryConfig(BaseModel):
    max_output_tokens: int = 2048

class QdrantConfig(BaseModel):
    url: str = "http://localhost:6333"
    api_key: str | None = None
    distance: str = "Cosine"
    prefer_grpc: bool = False
    hnsw: dict = {"m":16,"ef_construct":128,"ef_search":128}

class IngestConfig(BaseModel):
    batch_size: int = 64
    ocr: dict = {"dpi":300,"lang":"vie+eng","fmt":"PNG","max_pages":200,"oem":1,"psm":3}
    chunk: dict = {"target_tokens":700,"overlap_sentences":2,"min_chars":300}

class HydeConfig(BaseModel):
    enable: bool = True
    trigger: dict = {"min_hits": 2, "bm25_top1_below": 0.30}
    draft_len_sentences: int = 2

class CompressionConfig(BaseModel):
    enable: bool = True
    min_reduction_ratio: float = 0.30
    max_output_tokens: int = 1024

class ReorderConfig(BaseModel):
    enable: bool = True
    strategy: str = "short_to_long_group_by_heading"

class RetrievalConfig(BaseModel):
    top_k: int = 8
    ensemble: dict = {"dense": 0.6, "bm25": 0.4}
    bm25: dict = {"k1": 1.5, "b": 0.75, "min_query_len": 2}
    hyde: HydeConfig = HydeConfig()

class AppSettings(BaseModel):
    app: AppConfig
    toy: ToyConfig
    google_api_key: str = ""
    log_level: str = "INFO"
    app_env: str = "dev"
    qdrant: QdrantConfig
    ingest: IngestConfig
    retrieval: RetrievalConfig
    compression: CompressionConfig = CompressionConfig()
    reorder: ReorderConfig = ReorderConfig()
    summary: SummaryConfig = SummaryConfig()
    guardrail: GuardrailConfig = GuardrailConfig()
    qdrant_collection_prefix: str = "acad_sess_"
    api: dict = {}
    retry: dict = {"max_attempts": 3, "base_delay": 0.5, "max_delay": 5.0}

    @staticmethod
    def load() -> "AppSettings":
        with open("configs/app.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        
        # Retrieval now includes hyde
        ret_cfg = cfg.get("retrieval", {})
        hyde_cfg = HydeConfig(**ret_cfg.get("hyde", {}))
        retrieval = RetrievalConfig(
            top_k=ret_cfg.get("top_k", 8),
            ensemble=ret_cfg.get("ensemble", {"dense": 0.6, "bm25": 0.4}),
            bm25=ret_cfg.get("bm25", {"k1": 1.5, "b": 0.75, "min_query_len": 2}),
            hyde=hyde_cfg
        )

        return AppSettings(
            app=AppConfig(**cfg.get("app", {})),
            toy=ToyConfig(**cfg.get("toy", {})),
            qdrant=QdrantConfig(**cfg.get("qdrant", {})),
            ingest=IngestConfig(**cfg.get("ingest", {})),
            retrieval=retrieval,
            compression=CompressionConfig(**cfg.get("compression", {})),
            reorder=ReorderConfig(**cfg.get("reorder", {})),
            summary=SummaryConfig(**cfg.get("summary", {})),
            guardrail=GuardrailConfig(**cfg.get("guardrail", {})),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            qdrant_collection_prefix=os.getenv("QDRANT_COLLECTION_PREFIX","acad_sess_"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            app_env=os.getenv("APP_ENV", "dev"),
            api=cfg.get("api", {}),
            retry=cfg.get("retry", {"max_attempts": 3, "base_delay": 0.5, "max_delay": 5.0}),
        )


APPSETTINGS = AppSettings.load()
