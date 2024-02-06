import json
from enum import Enum
from typing import Optional

import instructor
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


class LocationInfo(BaseModel):
    location: str = Field(
        description="The location of the mineral site in the format of POINT(latitude longitude). Example: POINT(-107.983333 27.083333), POINT(171425 9367875)"
    )
    crs: str = Field(
        description="The coordinate reference system (CRS) of the location. Example: WGS84"
    )
    country: Optional[str] = Field(
        description="The country where the mineral site is located.",
    )
    state_or_province: Optional[str] = Field(
        description="The state or province where the mineral site is located.",
    )


class MineralInventory(BaseModel):
    commodity: str = Field(
        description="The type of critical mineral. Example: Zinc, Tungsten, and Nickel."
    )
    category: Optional[str] = Field(
        description="The category of the mineral. Example: Inferred, Indicated, Measured"
    )
    ore_unit: Optional[str] = Field(
        description="The unit of the ore. Example: tonnes (t)"
    )
    ore_value: Optional[float] = Field(
        default=0, description="The value of the ore in the unit of ore_unit"
    )
    grade_unit: Optional[str] = Field(
        description="The unit of the grade. Example: percent (%)"
    )
    grade_value: Optional[float] = Field(
        default=0, description="The value of the grade in the unit of grade_unit"
    )
    cutoff_grade_unit: Optional[str] = Field(
        default="Unknown",
        description="The unit of the cutoff grade. Example: percent (%)",
    )
    cutoff_grade_value: Optional[float] = Field(
        default=0,
        description="The value of the cutoff grade in the unit of cutoff_grade_unit",
    )
    contained_metal: Optional[float] = Field(
        description="Quantity of a contained metal in an inventory item, float.",
    )
    date: Optional[str] = Field(
        description="The date of the mineral inventory in the 'dd-mm-YYYY' format. Example: 2021-01-01"
    )
    zone: Optional[str] = Field(
        description="The zone of mineral site where inventory item was discovered"
    )


class DepositType(str, Enum):
    supergene_zinc = "Supergene zinc"
    siliciclastic = "Siliciclastic-mafic zinc-lead"
    mvt_zinc_lead = "MVT zinc-lead"


class MineralSite(BaseModel):
    name: str = Field(description="The name of the mineral site.")
    mineral_inventory: list[MineralInventory]
    location_info: LocationInfo
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
        output_schema = model.model_json_schema()
        output_schema_str = json.dumps(output_template)
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

    def extract(self, file_path: str) -> dict:
        with open(file_path, "r") as f:
            text = f.read()
        mineral_site = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            response_model=MineralSite,
            messages=[
                {
                    "role": "system",
                    "content": "Assistant is a large language model designed to extract structured mineral data from long mineral reports.",
                },
                {"role": "user", "content": f"# PDF text\n{text}"},
            ],
        )

        assert isinstance(mineral_site, MineralSite)
        # log mineral_site in JSON format with indent = 2
        logger.info(f"Mineral site: {mineral_site.model_dump_json(indent=2)}")
        return mineral_site.model_dump()


if __name__ == "__main__":
    # Extract structured data from long PDF text.
    instructor = Instructor()
    mineral_site_dict = instructor.extract(
        "data/asset/parsed_result/Bongará_Zn_3-2019/result.txt"
    )
