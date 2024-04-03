import logging
from dataclasses import dataclass
from enum import Enum


class ExtractionMethod(str, Enum):
    # Baseline: extraction using LLM + entire report + output schema
    BASELINE = "baseline"
    # LLM retriever: extraction using LLM + LLM-retrieved relevant sections of the report + output schema
    LLM_RETRIEVER = "llm_retriever"
    # Vector retriever: extraction using LLM + vector-retrieved relevant sections of the report + output schema
    VECTOR_RETRIEVER = "vector_retriever"


class LLMModel(str, Enum):
    # For more, see https://platform.openai.com/docs/guides/text-generation
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


class EmbeddingFunction(str, Enum):
    # For more, see https://platform.openai.com/docs/guides/embeddings/embedding-models
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"


class LangSmithEvalDataset(str, Enum):
    MINMOD_EXTRACTION = "MinMod Extraction Dataset"
    MINMOD_EXTRACTION_TEST = "MinMod Extraction Dataset Test"
    MINMOD_EXTRACTION_2 = "MinMod Extraction Dataset 2"
    MINMOD_EXTRACTION_2_TEST = "MinMod Extraction Dataset 2 Test"


@dataclass
class Config:
    ##############################################################
    # General settings
    ##############################################################
    # Raw data directories
    RAW_REPORTS_DIR: str = "data/raw/mvt_zinc/reports"  # Remaining Raw reports
    RAW_REPORTS_DIR_FAILED: str = (
        "data/raw/mvt_zinc/reports_failed"  # Reports failed to be extracted using PDF extractor
    )
    RAW_REPORTS_DIR_PROCESSED: str = (
        "data/raw/mvt_zinc/reports_processed"  # Reports that have been processed
    )

    # Data asset directories
    PDF_EXTRACTION_DIR: str = "data/asset/extraction_pdf"  # Extraction from PDF
    GROUND_TRUTH_INFERLINK_DIR: str = (
        "data/asset/ground_truth/inferlink"  # Ground truth from Inferlink
    )
    GROUND_TRUTH_SIMPLIFIED_DIR: str = (
        "data/asset/ground_truth/simplified"  # Simplified ground truth that conforms to MineralSite schema
    )
    PARSED_PDF_DIR_ADOBE: str = (
        "data/asset/parsed_pdf_adobe"  # Parsed PDF result using Adobe
    )
    PARSED_PDF_DIR_AZURE: str = (
        "data/asset/parsed_pdf_azure"  # Parsed PDF result using azure
    )
    PARSED_PDF_W_GT_DIR: str = (
        "data/asset/parsed_pdf_w_gt"  # Parsed PDF (Adobe) with ground truth
    )
    MINMOD_EXTRACTION_BASE_DIR: str = (
        "data/asset/extraction_minmod"  # MinMod extraction result
    )

    ##############################################################
    # PDF extractor settings
    ##############################################################
    PDF_PARSER_OVERWRITE: bool = False

    ##############################################################
    # Logging settings
    ##############################################################
    LOGGING_DIR: str = "logs"
    LOGGING_LEVEL: any = logging.INFO

    ##############################################################
    # MinMod extractor settings
    ##############################################################
    MODEL_NAME: str = LLMModel.GPT_4_TURBO.value
    TEMPERATURE: float = 0
    MAX_TOKENS: int = 2048
    MINMOD_BULK_EXTRACTION_OVERWRITE: bool = True
    EMBEDDING_FUNCTION: EmbeddingFunction = EmbeddingFunction.TEXT_EMBEDDING_3_SMALL

    def minmod_extraction_dir(self, method: ExtractionMethod) -> str:
        """Return the directory for the MinMod extraction result of the specified method."""
        return f"{self.MINMOD_EXTRACTION_BASE_DIR}/{method.value}"

    ##############################################################
    # Evaluation settings
    ##############################################################
    EVAL_METHOD: ExtractionMethod = ExtractionMethod.VECTOR_RETRIEVER
    EVAL_MODEL_NAME: str = LLMModel.GPT_4_TURBO.value
    EVAL_DATASET: str = LangSmithEvalDataset.MINMOD_EXTRACTION_2.value
    CONCURRENCY_LEVEL: int = 5
