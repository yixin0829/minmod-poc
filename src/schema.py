from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


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
    # tungsten = "Tungsten"
    # nickel = "Nickel"


class MineralCategory(str, Enum):
    estimated = "Estimated"
    inferred = "Inferred"
    indicated = "Indicated"
    measured = "Measured"
    probable = "Probable"
    proven = "Proven"


class DepositType(str, Enum):
    supergene_zinc = "Supergene zinc"
    siliciclastic = "Siliciclastic-mafic zinc-lead"
    mvt_zinc_lead = "MVT zinc-lead"
    irish_type_zinc = "Irish-type sediment- hosted zinc- lead"


class BasicInfo(BaseModel):
    name: str = Field(description="The name of the mineral site.")


class LocationInfo(BaseModel):
    location: str = Field(
        default="Unknown",
        # Relaxed the location description to include easting and northing.
        description="The coordinates of the mineral site represented as `POINT(<latitude> <longitude>)` or `POINT(<easting>, <northing>)`",
    )
    crs: str = Field(
        default="Unknown",
        description="The coordinate reference system (CRS) used for representing the location.",
    )
    country: Optional[str] = Field(
        default="Unknown",
        description="The country where the mineral site is located.",
    )
    state_or_province: Optional[str] = Field(
        default="Unknown",
        description="The state or province where the mineral site is located.",
    )


class MineralCommodity(BaseModel):
    commodity: Commodity = Field(description="The type of critical mineral commodity.")
    category: Optional[MineralCategory] = Field(
        default="Unknown",
        description="The category of the mineral commodity.",
    )
    ore_unit: Optional[WeightUnits] = Field(
        default="Unknown", description="The unit of the ore."
    )
    ore_value: Optional[float] = Field(
        default=-1, description="The value of the ore in the unit of ore_unit."
    )
    grade_unit: Optional[GradeUnits] = Field(
        default="Unknown", description="The unit of the grade."
    )
    grade_value: Optional[float] = Field(
        default=-1, description="The value of the grade in the unit of grade_unit."
    )
    cutoff_grade_unit: Optional[GradeUnits] = Field(
        default="Unknown",
        description="The unit of the cutoff grade.",
    )
    cutoff_grade_value: Optional[float] = Field(
        default=-1,
        description="The value of the cutoff grade in the unit of cutoff_grade_unit.",
    )
    date: Optional[str] = Field(
        default="Unknown",
        description="The date of the mineral commodity in the 'dd-mm-YYYY' format.",
    )
    zone: Optional[str] = Field(
        default="Unknown",
        description="The mineral zone where the mineral resources or reserves are located.",
    )


class MineralInventory(BaseModel):
    mineral_inventory: list[MineralCommodity]


class DepositType(BaseModel):
    deposit_type_name: DepositType = Field(
        "Other", description="The observed name of the mineral deposit type."
    )
    confidence: float = Field(
        0.0, description="Confidence level of the deposit type extracted.", ge=0, le=1
    )


class DepositTypeCandidate(BaseModel):
    deposit_type: list[DepositType] = Field(
        default="Other", description="All mineral deposit types in the mineral site."
    )


class MineralSite(BaseModel):
    basic_info: BasicInfo
    location_info: LocationInfo
    mineral_inventory: MineralCommodity
    deposit_type_candidate: DepositTypeCandidate
