from dataclasses import dataclass


@dataclass
class Prompts:
    sys_prompt: str = 'You will be provided with a PDF document. Your task is to answer the question using only the provided document and to cite the passage(s) of the document used to answer the question. If the document does not contain the information needed to answer this question then simply write: "Insufficient information." If an answer to the question is provided, it must be annotated with a citation. Use the following format for to cite relevant passages ({"citation": …}).'
    user_1: str = "What's the name of the mineral site in the report?"
    user_2: str = 'What\'s the location of the mineral site? In your answer, include coordinate information in format of "POINT(latitude longitude)" or "POINT(easting northing)", the coordinate reference system used, the country where the mineral site is located, the state or province where the mineral site is located.'


@dataclass
class Config:
    sample_reports_dir: str = "data/raw/reports"
    extraction_dir: str = "data/asset/extraction"
    parsed_result_dir: str = "data/asset/parsed_result"
