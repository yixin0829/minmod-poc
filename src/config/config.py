import logging
from dataclasses import dataclass
from enum import Enum


class ExtractionMethod(str, Enum):
    # Baseline: extraction using long-context LLM + entire output schema
    BASELINE = "baseline"
    LLM_RETRIEVER = "llm_retriever"
    VECTOR_RETRIEVER = "vector_retriever"


class LLMModel(str, Enum):
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


@dataclass
class Config:
    ##### General settings #####
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
    PARSED_PDF_DIR: str = "data/asset/parsed_pdf"  # Parsed PDF txt result
    PARSED_PDF_MOCK_DIR: str = (
        "data/asset/parsed_pdf_mock"  # Mock parsed PDF txt result
    )
    PARSED_PDF_W_GT_DIR: str = (
        "data/asset/parsed_pdf_w_gt"  # Parsed PDF txt with ground truth
    )
    MINMOD_EXTRACTION_BASE_DIR: str = (
        "data/asset/extraction_minmod"  # MinMod extraction result
    )

    ##### PDF extractor settings #####
    PDF_PARSER_OVERWRITE: bool = False

    ##### Logging settings #####
    LOGGING_DIR: str = "logs"
    LOGGING_LEVEL: any = logging.INFO

    ##### MinMod extractor settings #####
    MODEL_NAME: str = LLMModel.GPT_4_TURBO.value
    # MODEL_NAME: str = LLMModel.GPT_3_5_TURBO.value
    TEMPERATURE: float = 0
    MAX_TOKENS: int = 2048
    MINMOD_BULK_EXTRACTION_OVERWRITE: bool = True

    def minmod_extraction_dir(self, method: ExtractionMethod) -> str:
        """Return the directory for the MinMod extraction result of the specified method."""
        return f"{self.MINMOD_EXTRACTION_BASE_DIR}/{method.value}"

    ##### Evaluation settings #####
    EVAL_DATASET: str = "MinMod Extraction Dataset"
    EVAL_DATASET_TEST: str = "MinMod Extraction Test"
