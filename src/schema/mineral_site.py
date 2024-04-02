from enum import Enum
from typing import Optional

from pydantic.v1 import BaseModel, Field


class WeightUnits(str, Enum):
    tonnes = "tonnes"
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
    tungsten = "Tungsten"
    nickel = "Nickel"


class MineralCategory(str, Enum):
    estimated = "estimated"
    inferred = "inferred"
    indicated = "indicated"
    measured = "measured"
    probable = "probable"
    proven = "proven"


class DepositType(str, Enum):
    supergene_zinc = "Supergene zinc"
    siliciclastic_mafic_zinc_lead = "Siliciclastic-mafic zinc-lead"
    mvt_zinc_lead = "MVT zinc-lead"
    irish_type_zinc = "Irish-type sediment-hosted zinc-lead"


class BasicInfo(BaseModel):
    name: str = Field(description="The name of the mineral site.")


class LocationInfo(BaseModel):
    location: Optional[str] = Field(
        description='latitude and longitude represented as "POINT (Lat Long)" in EPSG:4326 format.',
    )
    crs: Optional[str] = Field(
        description="The coordinate reference system (CRS) used. For example, WGS84, UTM etc.",
    )
    country: Optional[str] = Field(
        description="The country where the mineral site is located.",
    )
    state_or_province: Optional[str] = Field(
        description="The state or province where the mineral site is located.",
    )


class MineralCommodity(BaseModel):
    commodity: Commodity = Field(
        description="The commodity of an mineral inventory item."
    )
    category: Optional[MineralCategory] = Field(
        description="The category of an mineral inventory item.",
    )
    ore_unit: Optional[WeightUnits] = Field(
        description="The unit in which ore quantity is measured, eg, tonnes.",
    )
    ore_value: Optional[float] = Field(
        description="The value of ore quantity measured in ore unit."
    )
    grade_unit: Optional[GradeUnits] = Field(
        description="The unit in which grade is measured, eg, percent.",
    )
    grade_value: Optional[float] = Field(
        description="The value of grade measured in grade unit."
    )
    cutoff_grade_unit: Optional[GradeUnits] = Field(
        description="Cut-off grade unit of an inventory item.",
    )
    cutoff_grade_value: Optional[float] = Field(
        description="Cut-off grade value of an inventory item measured in cut-off grade unit.",
    )
    date: Optional[str] = Field(
        description='Effective date of mineral inventory, in "dd-mm-YYYY" format. For example, "01-01-2022".',
    )
    zone: Optional[str] = Field(
        description="Zone of mineral site where the mineral commodity was discovered.",
    )


class MineralInventory(BaseModel):
    mineral_inventory: list[MineralCommodity]


class DepositTypeCandidate(BaseModel):
    observed_name: DepositType = Field(
        description="The name of the mineral deposit type."
    )
    confidence: Optional[float] = Field(
        description="The confidence level of the mineral deposit type from 0 to 1.",
    )


class DepositTypeCandidates(BaseModel):
    candidates: list[DepositTypeCandidate]


class MineralSite(BaseModel):
    basic_info: BasicInfo
    location_info: LocationInfo
    mineral_inventory: list[MineralCommodity] = Field(
        description="Mineral inventory of the site include indicated/inferred/measured mineral resources and probable/proven mineral reserves in each mineral zone."
    )
    deposit_type_candidate: list[DepositTypeCandidate] = Field(
        description="A list of possible deposit type candidates extracted from the document."
    )
