import logging
from dataclasses import dataclass


@dataclass
class Config:
    # Raw data directories
    RAW_REPORTS_DIR: str = "data/raw/mvt_zinc/reports"
    RAW_REPORTS_DIR_FAILED: str = "data/raw/mvt_zinc/reports_failed"
    RAW_REPORTS_DIR_PROCESSED: str = "data/raw/mvt_zinc/reports_processed"

    # Data asset directories
    EXTRACTION_DIR: str = "data/asset/extraction"
    PARSED_RESULT_DIR: str = "data/asset/parsed_result"

    # PDF extraction
    PDF_PARSER_OVERWRITE: bool = False

    # Logging
    LOGGING_DIR: str = "logs"
    LOGGING_LEVEL: any = logging.INFO

    # MinModExtractor
    MODEL: str = "gpt-3.5-turbo"
