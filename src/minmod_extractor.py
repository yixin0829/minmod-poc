import json
from enum import Enum
from typing import Optional

import instructor
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


class WeightUnits(str, Enum):
    tonnes = "tonnes"
    m_tonnes = "million tonnes"
    kg = "kilograms"


class GradeUnits(str, Enum):
    percent = "percent"
    g_tonnes = "grams per tonne"
    copper_eq_percent = "copper equivalence percent"
    lead_eq_percent = "lead equivalence percent"
    us_dollar_per_tonne = "US dollar per tonne"
    zn_eq_percent = "zinc equivalence percent"


class Commodity(str, Enum):
    zinc = "Zinc"
    other = "Other"


class LocationInfo(BaseModel):
    location: str = Field(
        default="Unknown",
        description="The latitude and longitude of the mineral site represented as `POINT(<latitude> <longitude>)` in `EPSG:4326` format.",
    )
    crs: str = Field(
        default="Unknown",
        description="The coordinate reference system (CRS) of the location. Example: WGS 84, Mercator and so on.",
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
    commodity: Commodity = Field(
        description="The type of critical mineral. Example: Zinc, Tungsten, and Nickel."
    )
    category: Optional[str] = Field(
        default="Unknown",
        description="The category of the mineral. Example: Inferred, Indicated, Measured",
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


class DepositType(str, Enum):
    supergene_zinc = "Supergene zinc"
    siliciclastic = "Siliciclastic-mafic zinc-lead"
    mvt_zinc_lead = "MVT zinc-lead"
    other = "Other"


class MineralSite(BaseModel):
    name: str = Field(description="The name of the mineral site.")
    location_info: LocationInfo
    mineral_inventory: list[MineralInventory]
    deposit_type: DepositType = Field(description="The type of mineral deposit.")


class JSONSchema(object):
    """
    One of the solutions for extracting structured data from long PDF text
    """

    def __init__(self) -> None:
        pass

    def generate_json_schema(self, model: BaseModel) -> str:
        """
        Generate a JSON schema from a Pydantic model.
        """
        # https://docs.pydantic.dev/latest/concepts/json_schema/
        output_schema = model.model_json_schema()
        output_schema_str = json.dumps(output_schema)
        return output_schema_str

    def extract(self) -> dict:
        """
        Extract structured data from long PDF text.
        """
        pass


class Instructor(object):
    """
    One of the solutions for extracting structured data from long PDF text
    """

    def __init__(self) -> None:
        # Enables `response_model`
        self.client = instructor.patch(OpenAI())

    def extract(self, file_path: str, response_model: BaseModel) -> BaseModel:
        with open(file_path, "r") as f:
            text = f.read()
        extract_model = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_model=response_model,
            messages=[
                {
                    "role": "system",
                    "content": "Assistant is a large language model designed to extract structured mineral data from long mineral reports.",
                },
                {"role": "user", "content": f"{text}"},
            ],
        )

        assert isinstance(extract_model, response_model)
        return extract_model


if __name__ == "__main__":
    # Generate the JSON schema of MineralSite
    # solution = JSONSchema()
    # output_schema = solution.generate_json_schema(MineralSite)
    # print(output_schema)

    # Extract structured data from long PDF text.
    solution = Instructor()
    model = solution.extract(
        file_path="data/asset/parsed_result/Bongará_Zn_3-2019/result.txt",
        response_model=MineralSite,
    )

    print(model.model_dump_json(indent=2))
