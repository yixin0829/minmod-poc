import json
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from loguru import logger
from openai import Client, OpenAI
from pydantic import BaseModel, Field

load_dotenv()

GPT_MODEL = "gpt-4-turbo-preview"


class WeightUnits(str, Enum):
    tonnes = "tonnes"
    m_tonnes = "million tonnes"
    kg = "kilograms"
    other = "Other"


class GradeUnits(str, Enum):
    percent = "percent"
    g_tonnes = "grams per tonne"
    copper_eq_percent = "copper equivalence percent"
    lead_eq_percent = "lead equivalence percent"
    us_dollar_per_tonne = "US dollar per tonne"
    zn_eq_percent = "zinc equivalence percent"
    other = "Other"


class Commodity(str, Enum):
    zinc = "Zinc"
    other = "Other"


class MineralCategory(str, Enum):
    estimated = "Estimated"
    inferred = "Inferred"
    indicated = "Indicated"
    measured = "Measured"
    probable = "Probable"
    proven = "Proven"
    other = "Other"


class DepositType(str, Enum):
    supergene_zinc = "Supergene zinc"
    siliciclastic = "Siliciclastic-mafic zinc-lead"
    mvt_zinc_lead = "MVT zinc-lead"
    other = "Other"


class LocationInfo(BaseModel):
    location: str = Field(
        default="Unknown",
        # Relaxed the location description to include easting and northing.
        description="The coordinates of the mineral site represented as `POINT(<latitude> <longitude>)` or `POINT(<easting>, <northing>)` in `EPSG:4326` format.",
    )
    crs: str = Field(
        default="Unknown",
        description="The coordinate reference system (CRS) of the location.",
    )
    country: Optional[str] = Field(
        default="Unknown",
        description="The country where the mineral site is located.",
    )
    state_or_province: Optional[str] = Field(
        default="Unknown",
        description="The state or province where the mineral site is located.",
    )


class MineralInventory(BaseModel):
    commodity: Commodity = Field(description="The type of critical mineral.")
    category: Optional[MineralCategory] = Field(
        default="Unknown",
        description="The category of the mineral.",
    )
    ore_unit: Optional[WeightUnits] = Field(
        default="Unknown", description="The unit of the ore."
    )
    ore_value: Optional[float] = Field(
        default=0, description="The value of the ore in the unit of ore_unit"
    )
    grade_unit: Optional[GradeUnits] = Field(
        default="Unknown", description="The unit of the grade."
    )
    grade_value: Optional[float] = Field(
        default=0, description="The value of the grade in the unit of grade_unit"
    )
    cutoff_grade_unit: Optional[GradeUnits] = Field(
        default="Unknown",
        description="The unit of the cutoff grade. Example: percent (%)",
    )
    cutoff_grade_value: Optional[float] = Field(
        default=0,
        description="The value of the cutoff grade in the unit of cutoff_grade_unit",
    )
    contained_metal: Optional[float] = Field(
        default=0,
        description="Quantity of a contained metal in an inventory item, float.",
    )
    date: Optional[str] = Field(
        default="Unknown",
        description="The date of the mineral inventory in the 'dd-mm-YYYY' format.",
    )
    zone: Optional[str] = Field(
        default="Unknown",
        description="The zone of mineral site where inventory item was discovered",
    )


class MineralSite(BaseModel):
    name: str = Field(description="The name of the mineral site.")
    location_info: LocationInfo
    mineral_inventory: list[MineralInventory]
    deposit_type: DepositType = Field(description="The type of mineral deposit.")


class MinModExtractor(object):
    def __init__(self, GPT_MODEL: str) -> None:
        self.client = OpenAI()
        self.assistant = None
        self.thread = None

    def generate_json_schema(self, model: BaseModel) -> str:
        """
        Generate a JSON schema from a Pydantic model.
        https://docs.pydantic.dev/latest/concepts/json_schema/
        """
        output_schema = model.model_json_schema()
        output_schema_str = json.dumps(output_schema)
        return output_schema_str

    def create_assistant(self):
        self.assistant = self.client.beta.assistants.create(
            name="Mineral Data Extractor",
            instructions="Your task is to extract information from the mineral reports in structured format.",
            tools=[{"type": "retrieval"}],
            model=GPT_MODEL,
        )
        logger.info(f"Assistant created: {self.assistant.id}")

    def attach_file_to_assistant(self, assistant_id: str, file_id: str):
        """
        Create an assistant file by attaching a File to an assistant.
        """
        assistant_file = self.client.beta.assistants.files.create(
            assistant_id=assistant_id, file_id=file_id
        )

    def extract(self):
        self.thread = self.client.beta.threads.create()
        logger.info(f"Thread created: {thread.id}")
        message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=f"Given a PDF mineral report, your task is to extract structured data from the mineral report. Fill in the following JSON schema:\n# JSON Schema\n{self.generate_json_schema(MineralSite)}",
        )
        logger.info(f"Message added to the thread {thread.id}: {message.id}")

        run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
        )

        run = self.client.beta.threads.runs.retrieve(
            thread_id=self.thread.id, run_id=run.id
        )


if __name__ == "__main__":
    solution = MinModExtractor(GPT_MODEL=GPT_MODEL)
    print(solution.generate_json_schema(MineralInventory))
