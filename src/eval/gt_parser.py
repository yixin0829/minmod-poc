"""
Ground truth parser that parses a more detailed JSON structure (TA2 schema) to simplified schema defined in MineralSite.
"""

import json
import os

import requests
from bs4 import BeautifulSoup
from loguru import logger

from config.config import Config
from schema.mineral_site import (
    Commodity,
    DepositType,
    GradeUnits,
    MineralCategory,
    WeightUnits,
)

uri_to_str_mapping = {
    "https://minmod.isi.edu/resource/Q589": Commodity.zinc.value,
    "https://minmod.isi.edu/resource/Q200": WeightUnits.tonnes.value,
    "https://minmod.isi.edu/resource/Q201": GradeUnits.percent.value,
    "https://minmod.isi.edu/resource/INDICATED": MineralCategory.indicated.value,
    "https://minmod.isi.edu/resource/INFERRED": MineralCategory.inferred.value,
    "https://minmod.isi.edu/resource/MEASURED": MineralCategory.measured.value,
    "https://minmod.isi.edu/resource/PROBABLE": MineralCategory.probable.value,
    "https://minmod.isi.edu/resource/PROVEN": MineralCategory.proven.value,
    "https://minmod.isi.edu/resource/Q380": DepositType.mvt_zinc_lead.value,
    "https://minmod.isi.edu/resource/Q330": DepositType.supergene_zinc.value,
    "https://minmod.isi.edu/resource/Q378": DepositType.irish_type_zinc.value,
    "https://minmod.isi.edu/resource/Q375": DepositType.siliciclastic_mafic_zinc_lead.value,
}


def get_v(d: dict, k: str, default):
    """
    Wrapper method to get the value of a key from a dictionary with a default value.
    if the value is a URI, map it to a string.
    """
    if not isinstance(default, int) and k in d and d[k].startswith("http"):
        # map the uri to string
        return uri_to_str_mapping[d[k]]
    else:
        return d.get(k, default)


def transform_inferlink_gt(input_data: dict):
    """
    Transform inferlink's ground truth JSON to the simplified schema defined in MineralSite.
    """

    if isinstance(input_data["MineralSite"], list):
        mineral_site = input_data["MineralSite"][0]
    elif isinstance(input_data["MineralSite"], dict):
        mineral_site = input_data["MineralSite"]

    # Initialize the output JSON structure
    output_data = {
        "basic_info": {
            "name": mineral_site["name"],
        },
        "location_info": mineral_site["location_info"],
        "mineral_inventory": [],
        "deposit_type_candidate": [],
    }

    # Populate mineral inventory in output JSON
    for item in mineral_site["MineralInventory"]:
        parsed_item = {
            "commodity": get_v(item, "commodity", "unknown"),
            "category": get_v(item, "category", "unknown"),
            "ore_unit": get_v(item["ore"], "ore_unit", "unknown"),
            "ore_value": get_v(item["ore"], "ore_value", -1),
            "grade_unit": get_v(item["grade"], "grade_unit", "unknown"),
            "grade_value": get_v(item["grade"], "grade_value", -1),
        }

        if item.get("cutoff_grade") is None:
            parsed_item["cutoff_grade_unit"] = "unknown"
            parsed_item["cutoff_grade_value"] = -1
        else:
            parsed_item["cutoff_grade_unit"] = get_v(
                item["cutoff_grade"], "grade_unit", "unknown"
            )
            parsed_item["cutoff_grade_value"] = get_v(
                item["cutoff_grade"], "grade_value", -1
            )

        parsed_item["date"] = item.get("date", "unknown")
        parsed_item["zone"] = item.get("zone", "unknown")

        output_data["mineral_inventory"].append(parsed_item)

    # Transform deposit type
    for deposit in input_data["MineralSite"][0]["deposit_type"]:
        output_data["deposit_type_candidate"].append(
            {"observed_name": get_v(deposit, "id", "unknown"), "confidence": 1.0}
        )

    return output_data


def write_json_to_new_directory(file_name, output_dir, data):
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=2)


if __name__ == "__main__":
    # Read all JSON from inferlink_gt_dir
    for file_name in sorted(os.listdir(Config.GROUND_TRUTH_INFERLINK_DIR)):
        logger.info(f"Transforming {file_name}")
        if file_name.endswith(".json"):
            with open(
                os.path.join(Config.GROUND_TRUTH_INFERLINK_DIR, file_name), "r"
            ) as f:
                input_json = json.load(f)

        # Transform the input JSON to the output JSON
        output_json = transform_inferlink_gt(input_json)

        # Write the output JSON to a new directory
        write_json_to_new_directory(
            file_name, Config.GROUND_TRUTH_SIMPLIFIED_DIR, output_json
        )
