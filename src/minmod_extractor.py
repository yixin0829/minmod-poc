import json
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class LocationInfo(BaseModel):
    location: str = Field(
        description="The location of the critical mineral site in the format of POINT(latitude longitude). Example: POINT(-107.983333 27.083333), POINT(171425 9367875)"
    )
    crs: str = Field(
        description="The coordinate reference system of the location. Example: WGS84"
    )
    country: Optional[str] = Field(
        default=None,
        description="The country where the critical mineral site is located.",
    )
    state_or_province: Optional[str] = Field(
        default=None,
        description="The state or province where the critical mineral site is located.",
    )


class MineralInventory(BaseModel):
    commodity: str = Field(description="The type of critical mineral. Example: Zinc")
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


class DepositType(str, Enum):
    supergene_zinc = "Supergene zinc"
    siliciclastic = "Siliciclastic-mafic zinc-lead"
    mvt_zinc_lead = "MVT zinc-lead"


class MineralSite(BaseModel):
    name: str = Field(description="The name of the mineral site.")
    mineral_inventory: list[MineralInventory]
    location_info: LocationInfo
    deposit_type: DepositType = Field(description="The type of mineral deposit.")


output_template = MineralSite.model_json_schema()
output_template_str = json.dumps(output_template)

# Ingest the output JSON schema in the LLM prompt
print(output_template_str)
