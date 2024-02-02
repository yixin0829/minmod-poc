from dataclasses import dataclass


@dataclass
class Config:
    input_path: str = "data/raw/reports"
    output_path: str = "data/asset/extraction"
