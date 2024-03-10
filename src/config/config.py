import logging
from dataclasses import dataclass
from enum import Enum


class ExtractionMethod(str, Enum):
    BASELINE = "baseline"
    LLM_RETRIEVAL = "llm_retrieval"


class LLMModel(str, Enum):
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_3_5_TURBO = "gpt-3.5-turbo-0125"


@dataclass
class Config:
    # Raw data directories
    RAW_REPORTS_DIR: str = "data/raw/mvt_zinc/reports"
    RAW_REPORTS_DIR_FAILED: str = "data/raw/mvt_zinc/reports_failed"
    RAW_REPORTS_DIR_PROCESSED: str = "data/raw/mvt_zinc/reports_processed"

    # Data asset directories
    EXTRACTION_DIR: str = "data/asset/extraction_pdf"  # Extraction from PDF
    PARSED_RESULT_DIR: str = (
        "data/asset/parsed_result"  # Parsed result from PDF extraction
    )
    GROUND_TRUTH_DIR: str = "data/asset/ground_truth"

    ##### PDF extractor settings #####
    PDF_PARSER_OVERWRITE: bool = False

    ##### Logging settings #####
    LOGGING_DIR: str = "logs"
    LOGGING_LEVEL: any = logging.INFO

    ##### MinMod extractor settings #####
    MODEL_NAME: str = LLMModel.GPT_3_5_TURBO.value
    TEMPERATURE: float = 0.5
    MAX_TOKENS: int = 2048
    MINMOD_EXTRACTION_BASE_DIR: str = "data/asset/extraction_minmod"
    MINMOD_BULK_EXTRACTION_OVERWRITE: bool = False

    def minmod_extraction_dir(self, method: ExtractionMethod) -> str:
        return f"{self.MINMOD_EXTRACTION_BASE_DIR}/{method.value}"
