from enum import Enum
from typing import Optional

from pydantic.v1 import BaseModel, Field


class WeightUnits(str, Enum):
    tonnes = "tonnes"
    m_tonnes = "million tonnes"
    kg = "kilograms"
    unknown = "unknown"


class GradeUnits(str, Enum):
    percent = "percent"
    g_tonnes = "grams per tonne"
    copper_eq_percent = "copper equivalence percent"
    lead_eq_percent = "lead equivalence percent"
    us_dollar_per_tonne = "US dollar per tonne"
    zn_eq_percent = "zinc equivalence percent"
    unknown = "unknown"


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
    unknown = "unknown"


class DepositType(str, Enum):
    supergene_zinc = "Supergene zinc"
    siliciclastic_mafic_zinc_lead = "Siliciclastic-mafic zinc-lead"
    mvt_zinc_lead = "MVT zinc-lead"
    irish_type_zinc = "Irish-type sediment-hosted zinc-lead"
    unknown = "unknown"


class CRS(str, Enum):
    WGS84 = "WGS84"
    UTM = "UTM"
    unknown = "unknown"


class BasicInfo(BaseModel):
    name: str = Field(description="The name of the mineral site.")


class LocationInfo(BaseModel):
    location: Optional[str] = Field(
        default="unknown",
        # Relaxed the location description to include easting and northing.
        description="Polygon or Point, value indicates the geolocation of the mineral site, represented as `POINT(<latitude> <longitude>)`.",
    )
    crs: Optional[CRS] = Field(
        default="unknown",
        description="The coordinate reference system (CRS) used for the mineral site's location.",
    )
    country: Optional[str] = Field(
        default="unknown",
        description="The country where the mineral site is located.",
    )
    state_or_province: Optional[str] = Field(
        default="unknown",
        description="The state or province where the mineral site is located.",
    )


class MineralCommodity(BaseModel):
    commodity: Commodity = Field(
        description="The commodity of an mineral inventory item."
    )
    category: Optional[MineralCategory] = Field(
        default="unknown",
        description="The category of an mineral inventory item.",
    )
    ore_unit: Optional[WeightUnits] = Field(
        default="unknown",
        description="The unit in which ore quantity is measured, eg, tonnes.",
    )
    ore_value: Optional[float] = Field(
        default=-1, description="The value of ore quantity measured in ore unit."
    )
    grade_unit: Optional[GradeUnits] = Field(
        default="unknown",
        description="The unit in which grade is measured, eg, percent.",
    )
    grade_value: Optional[float] = Field(
        default=-1, description="The value of grade measured in grade unit."
    )
    cutoff_grade_unit: Optional[GradeUnits] = Field(
        default="unknown",
        description="Cut-off grade unit of an inventory item.",
    )
    cutoff_grade_value: Optional[float] = Field(
        default=-1,
        description="Cut-off grade value of an inventory item measured in cut-off grade unit.",
    )
    contained_metal: Optional[float] = Field(
        default=-1,
        description="The quantity of a contained metal in an inventory item.",
    )
    date: Optional[str] = Field(
        default="unknown",
        description='When in the point of time mineral inventory valid. Format as "YYYY-mm".',
    )
    zone: Optional[str] = Field(
        default="unknown",
        description="Zone of mineral site where the mineral commodity was discovered.",
    )


class MineralInventory(BaseModel):
    mineral_inventory: list[MineralCommodity]


class DepositTypeCandidate(BaseModel):
    observed_name: DepositType = Field(
        description="The name of the possible mineral deposit type."
    )


class DepositTypeCandidates(BaseModel):
    deposit_type_candidate: list[DepositTypeCandidate]


class MineralSite(BaseModel):
    basic_info: BasicInfo
    location_info: LocationInfo
    mineral_inventory: list[MineralCommodity] = Field(
        description="Mineral inventory of the site include indicated/inferred/measured mineral resources and probable/proven mineral reserves in each mineral zone."
    )
    deposit_type_candidate: list[DepositTypeCandidate] = Field(
        description="A list of possible deposit type candidates extracted from the document."
    )
