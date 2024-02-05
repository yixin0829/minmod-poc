from dataclasses import dataclass


@dataclass
class Config:
    sample_reports_dir: str = "data/raw/reports"
    extraction_dir: str = "data/asset/extraction"
    parsed_result_dir: str = "data/asset/parsed_result"
