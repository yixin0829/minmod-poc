import logging
from dataclasses import dataclass
from enum import Enum


class ExtractionMethod(str, Enum):
    BASELINE = "baseline"
    LLM_RETRIEVAL = "llm_retrieval"


@dataclass
class Config:
    # Raw data directories
    RAW_REPORTS_DIR: str = "data/raw/mvt_zinc/reports"
    RAW_REPORTS_DIR_FAILED: str = "data/raw/mvt_zinc/reports_failed"
    RAW_REPORTS_DIR_PROCESSED: str = "data/raw/mvt_zinc/reports_processed"

    # Data asset directories
    EXTRACTION_DIR: str = "data/asset/pdf_extraction"
    PARSED_RESULT_DIR: str = "data/asset/parsed_result"

    # PDF extractor settings
    PDF_PARSER_OVERWRITE: bool = False

    # Logging
    LOGGING_DIR: str = "logs"
    LOGGING_LEVEL: any = logging.INFO

    # MinMod extractor settings
    MODEL: str = "gpt-4-turbo-preview"
    MINMOD_EXTRACTION_BASE_DIR: str = "data/asset/minmod_extraction"

    def minmod_extraction_dir(self, method: ExtractionMethod) -> str:
        return f"{self.MINMOD_EXTRACTION_BASE_DIR}/{method.value}"
